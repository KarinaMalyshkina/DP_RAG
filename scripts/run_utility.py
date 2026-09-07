"""Measure answer quality at each calibrated privacy budget — the other half of the trade-off.

For each cell — (seed x dataset variant x epsilon x pipeline) — it indexes the
corpus, answers held-out patient questions under the calibrated guarantee, and
scores the answers against the real doctor replies.

The task comes from the corpus itself: ``HealthCareMagicLoader`` keeps the patient's
question in ``metadata["query"]`` and uses the doctor's reply as the document text,
so the reply is both a corpus document and the reference answer. That is the RAG
setting — the answer lives in the retrieval store — and it means a system that
retrieves well and generates faithfully scores well.

**This must be run with the same ``--n-tokens`` and ``--delta`` as the attacks.**
A privacy-utility curve built from cells that were configured differently compares
two different systems and means nothing.

On metrics: token F1 is primary. Exact match is reported but will sit at ~0 for
every system including the non-private baseline, because the references are
free-form replies of tens of words — that is a property of the corpus, not a
finding about DP. ROUGE-L is the order-sensitive cross-check.

Outputs (under --output-dir): results.json, results.csv, predictions.jsonl, report.md.

Usage:
    python scripts/run_utility.py --epsilons inf,0.5,1,2,4,8 --n-tokens 32 \
        --corpus-size 1000 --n-queries 200 --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dp_rag_eval.core.types import PrivacyTarget
from dp_rag_eval.datasets.loaders import HealthCareMagicLoader
from dp_rag_eval.evaluation import aggregate_scores, score_answer
from dp_rag_eval.pipelines import Embedder, load_generator, resolve_device
from dp_rag_eval.runner.sweep import (
    ALL_PIPELINES,
    NON_PRIVATE,
    VARIANT_COLUMNS,
    VARIANT_SEED_OFFSET,
    CheckpointedRun,
    PipelineParams,
    build_pipeline,
    cell_config_suffix,
    eps_label,
    grid_from_args,
    make_variant_cell,
    parse_floats,
    parse_ints,
    parse_strs,
    truncate_doc,
    variant_configs,
)

logger = logging.getLogger("run_utility")

CSV_FIELDS = [
    "pipeline",
    # The generator, per row rather than only in meta: reporting merges several run
    # directories, and once two models are in play a row that cannot name its own is
    # unattributable. CheckpointedRun refuses to mix them within one directory.
    "generator",
    *VARIANT_COLUMNS,
    "epsilon",
    "realized_epsilon",
    "realized_delta",
    "seed",
    "status",
    "token_f1_mean",
    "rouge_l_mean",
    "exact_match_mean",
    "n_answers",
    "n_empty",
    "tokens_per_answer_mean",
    "n_indexed",
    "n_split",
    # Koga diagnostics. An empty answer is either "the selection refused" or "the
    # model said nothing"; only these columns tell the two apart, and only the last
    # three say whether refusal was marginal or hopeless.
    "ld_failure_rate",
    "vote_threshold",
    "first_step_lead_mean",
    "refused_at_first_step",
    "seconds",
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("runs/utility"), help="Where to write results")
    p.add_argument("--pipelines", default=",".join(ALL_PIPELINES), help="Comma-separated: naive,koga,grislain")

    p.add_argument(
        "--generator",
        default="Qwen/Qwen3.5-9B",
        help="HF causal-LM id (shared). The default is the model the matrix is run on; pass a "
        "small one (e.g. Qwen/Qwen2.5-0.5B-Instruct) with --device cpu for a local dry run.",
    )
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--dtype", default=None, help="bfloat16 / float16 / float32")

    p.add_argument("--corpus-size", type=int, default=200, help="Documents indexed")
    p.add_argument("--n-queries", type=int, default=25, help="Questions answered per cell")
    p.add_argument("--max-doc-chars", type=int, default=300, help="Truncate documents")
    p.add_argument("--max-answer-chars", type=int, default=400, help="Truncate the reference answer")

    p.add_argument("--epsilons", default="inf", help="Target total budgets; 'inf' is the non-private reference")
    p.add_argument("--delta", type=float, default=1e-6, help="Target delta; must be << 1/corpus size")
    p.add_argument("--n-tokens", type=int, default=32, help="Answer length the guarantee covers")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds")

    p.add_argument("--variant-grid", choices=["base", "duplication", "all"], default="base")
    p.add_argument("--variants", default=None, help="Restrict the grid to these variant names")
    p.add_argument("--dup-rates", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9", help="full_duplicate rate sweep")
    p.add_argument("--dup-copies", default="1,2,3,5,10", help="full_duplicate extra-copies sweep")
    p.add_argument("--overlap-rates", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9", help="partial_overlap sweep")
    p.add_argument("--canary-repeats", default="1,2,5,10", help="canary n_repeats sweep")
    p.add_argument("--base-dup-rate", type=float, default=0.3)
    p.add_argument("--base-dup-copies", type=int, default=1)
    p.add_argument("--base-overlap-rate", type=float, default=0.3)
    p.add_argument("--overlap-hosts", type=int, default=3)
    p.add_argument("--base-canary-repeats", type=int, default=1)
    p.add_argument("--n-canaries", type=int, default=10)

    p.add_argument(
        "--n-split",
        type=int,
        default=20,
        help="Koga: floor on the voter count. Auto-sizing raises it per cell to whatever the "
        "privacy target needs, so this only sets the minimum for budgets that need fewer.",
    )
    p.add_argument(
        "--auto-n-split-cap",
        type=int,
        default=128,
        help="Size Koga's voter count per cell to the minimum that makes the target feasible, up to this cap. "
        "The default 128 is what buys an overlap with Grislain's reachable range (eps 4 and 8 at "
        "n_tokens=32) while keeping retrieval selective on a 1000-document corpus. Pass 0 to pin the "
        "count to --n-split instead, which leaves every low-epsilon cell INFEASIBLE.",
    )
    p.add_argument(
        "--n-split-headroom",
        type=float,
        default=1.0,
        help="Koga: multiply the minimum feasible voter count by this. At 1.0 the selection succeeds only "
        "under near-unanimity, which with one document per voter is why every answer comes back empty. "
        "Costs GPU time and nothing else — the vote cutoff and the accounting are both independent of "
        "the voter count.",
    )
    p.add_argument(
        "--n-split-fixed",
        type=int,
        default=None,
        help="Koga: use exactly this many voters at every budget, overriding the sizing and the cap. "
        "Required for an answer-rate curve: otherwise the voter count moves with epsilon and the two "
        "effects cannot be separated.",
    )
    p.add_argument(
        "--n-docs",
        type=int,
        default=1,
        help="Koga: documents per voter. Groups stay disjoint, so the guarantee is unchanged; agreement "
        "rises because voters' contexts start to overlap.",
    )
    p.add_argument("--retrieval-share", type=float, default=0.1, help="Grislain: share of epsilon spent retrieving")
    p.add_argument("--top-k", type=int, default=5, help="Documents retrieved per query")
    return p.parse_args()


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# Utility under calibrated privacy budgets", ""]
    lines.append(f"- generator: `{meta['generator']}`  |  device: `{meta['device']}`")
    lines.append(
        f"- corpus: {meta['corpus_size']} docs  |  queries/cell: {meta['n_queries']}  "
        f"|  answer length: {meta['n_tokens']} tokens  |  seeds: {meta['seeds']}"
    )
    lines.append(f"- target epsilons: {meta['epsilons']}  |  delta: {meta['delta']}")
    lines.append("")
    lines.append(
        "**Token F1 is the primary metric.** Exact match is near zero for every system, "
        "including the non-private baseline, because the references are free-form doctor "
        "replies — that reflects the corpus, not the effect of DP. ROUGE-L adds word order."
    )
    lines.append("")
    lines.append(
        "Plot against `realized_epsilon`, not the requested value: it is what each "
        "pipeline's own accountant reports for the configuration actually run. Rows with "
        "status `INFEASIBLE` are configurations the mechanism cannot deliver at all — "
        "they are a result about the mechanism's reachable range, not a failed run."
    )
    lines.append("")
    lines.append(
        "| pipeline | variant | params | target eps | realized eps | seed | status | F1 | ROUGE-L | EM | empty | secs |"
    )
    lines.append("|" + "---|" * 12)
    for r in rows:

        def fmt(key: str) -> str:
            v = r.get(key, float("nan"))
            if isinstance(v, str):
                return v
            return "nan" if v != v else f"{v:.3f}"

        lines.append(
            f"| {r['pipeline']} | {r['variant']} | {r['params']} | {r['epsilon']} | "
            f"{fmt('realized_epsilon')} | {r['seed']} | {r['status']} | {fmt('token_f1_mean')} | "
            f"{fmt('rouge_l_mean')} | {fmt('exact_match_mean')} | {r.get('n_empty', 0)} | {r['seconds']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    device = resolve_device(args.device)
    pipeline_names = parse_strs(args.pipelines)
    epsilons = parse_floats(args.epsilons)
    seeds = parse_ints(args.seeds)
    variants = variant_configs(args.variant_grid, grid_from_args(args), parse_strs(args.variants or "") or None)

    if args.delta >= 1.0 / max(args.corpus_size, 1):
        logger.warning(
            "delta=%g is not below 1/corpus_size=%g. A delta at or above 1/n permits a "
            "mechanism that simply publishes a random record while remaining 'private'.",
            args.delta,
            1.0 / max(args.corpus_size, 1),
        )

    # A system allowed n_tokens tokens cannot match a reference many times longer:
    # recall, and therefore F1, is capped by the length ratio no matter how good the
    # answer is. Left unchecked this reads as "DP destroyed utility" when it is
    # actually the measurement being mis-scaled.
    reference_words = args.max_answer_chars / 5.5  # ~5.5 chars per English word
    if reference_words > 1.5 * args.n_tokens:
        ceiling = min(1.0, args.n_tokens / reference_words)
        logger.warning(
            "References average ~%.0f words but answers are capped at %d tokens: token-F1 "
            "cannot exceed ~%.2f even for a perfect system. Lower --max-answer-chars to "
            "~%d or raise --n-tokens so the two are comparable.",
            reference_words,
            args.n_tokens,
            2 * ceiling / (1 + ceiling),
            int(args.n_tokens * 5.5),
        )

    params = PipelineParams(
        delta=args.delta,
        n_tokens=args.n_tokens,
        top_k=args.top_k,
        n_split=args.n_split,
        n_docs=args.n_docs,
        retrieval_share=args.retrieval_share,
        auto_n_split_cap=args.auto_n_split_cap,
        n_split_headroom=args.n_split_headroom,
        n_split_fixed=args.n_split_fixed,
    )
    total_cells = (
        len(seeds) * len(variants) * sum(1 if name in NON_PRIVATE else len(epsilons) for name in pipeline_names)
    )

    meta = {
        "task": "HealthCareMagic patient question -> doctor reply",
        "planned_cells": total_cells,
        "generator": args.generator,
        "embedder": args.embedder,
        "device": device,
        "corpus_size": args.corpus_size,
        "n_queries": args.n_queries,
        "n_tokens": args.n_tokens,
        "seeds": seeds,
        "epsilons": [eps_label(e) for e in epsilons],
        "delta": args.delta,
        "top_k": args.top_k,
        "n_split": args.n_split,
        "n_split_headroom": args.n_split_headroom,
        "n_split_fixed": args.n_split_fixed,
        "n_docs": args.n_docs,
        "variant_seed_offset": VARIANT_SEED_OFFSET,
    }
    run = CheckpointedRun(args.output_dir, meta, CSV_FIELDS)
    run.report_fn = lambda: render_markdown(run.rows, meta)
    logger.info(
        "Planned: %d cells (%d seeds x %d variants x %d epsilons)",
        total_cells,
        len(seeds),
        len(variants),
        len(epsilons),
    )

    generator, tokenizer = load_generator(args.generator, device, dtype=args.dtype)
    embedder = Embedder(args.embedder, device)

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        loader = HealthCareMagicLoader(n_docs=args.corpus_size, seed=seed)
        base = [truncate_doc(d, args.max_doc_chars) for d in loader.load()]

        # Evaluate on documents that carry a question, using the document text as
        # the reference answer.
        answerable = [d for d in base if d.metadata.get("query")]
        if not answerable:
            raise SystemExit("No documents carry metadata['query'] — cannot build a utility task")
        queries = random.Random(seed).sample(answerable, min(args.n_queries, len(answerable)))
        logger.info("[seed %d] %d evaluation questions over %d documents", seed, len(queries), len(base))

        for cell_spec in variants:
            variant_columns = cell_spec.columns()
            corpus = make_variant_cell(cell_spec, seed).apply(base)

            for epsilon in epsilons:
                target = PrivacyTarget(epsilon=epsilon, delta=args.delta, n_tokens=args.n_tokens)
                for name in pipeline_names:
                    if name in NON_PRIVATE and target.is_private:
                        continue
                    cell = (
                        f"{name}|{cell_spec.name}|{variant_columns['params']}|{target.label()}"
                        f"{cell_config_suffix(name, target, params)}|seed={seed}"
                    )
                    if run.is_done(cell):
                        continue

                    start = time.perf_counter()
                    row: dict[str, Any] = {
                        "cell": cell,
                        "pipeline": name,
                        "generator": args.generator,
                        **variant_columns,
                        "epsilon": eps_label(epsilon),
                        "seed": seed,
                        "n_indexed": len(corpus),
                        "status": "OK",
                        "error": "",
                    }
                    predictions: list[dict[str, Any]] = []
                    try:
                        pipe, calibration = build_pipeline(
                            name,
                            target,
                            params,
                            generator=generator,
                            tokenizer=tokenizer,
                            embedder=embedder,
                            device=device,
                        )
                        row["realized_epsilon"] = calibration.realized.epsilon
                        row["realized_delta"] = calibration.realized.delta
                        # The non-private calibration carries no diagnostics, so without the
                        # fallback an eps=inf cell records no voter count — and it is the
                        # reference point every other cell in a fixed-count sweep is compared
                        # against, so it is exactly the one that must not be blank.
                        row["n_split"] = calibration.diagnostics.get("n_split") or getattr(pipe, "n_split", "")
                        if not calibration.feasible:
                            row["status"] = "INFEASIBLE"
                            row["error"] = calibration.reason
                            logger.warning("Skipping %s: %s", cell, calibration.reason)
                        else:
                            logger.info("Running %s ...", cell)
                            pipe.index(corpus)
                            if hasattr(pipe, "reset_ld_stats"):
                                pipe.reset_ld_stats()
                            per_answer: list[dict[str, float]] = []
                            n_empty = 0
                            n_tokens_out: list[int] = []
                            for doc in queries:
                                question = str(doc.metadata["query"])
                                reference = doc.text[: args.max_answer_chars]
                                answer = pipe.answer(question)
                                generated = str(answer.get("answer", ""))
                                n_tokens_out.append(int(answer.get("n_generated", 0)))
                                if not generated.strip():
                                    n_empty += 1
                                scores = score_answer(generated, reference)
                                per_answer.append(scores)
                                predictions.append(
                                    {
                                        "doc_id": doc.id,
                                        "question": question,
                                        "reference": reference,
                                        "generated": generated,
                                        **scores,
                                    }
                                )
                            row.update(aggregate_scores(per_answer))
                            row["n_empty"] = n_empty
                            row["tokens_per_answer_mean"] = (
                                sum(n_tokens_out) / len(n_tokens_out) if n_tokens_out else 0.0
                            )
                            # Without these an empty-answer cell is indistinguishable
                            # from one where the mechanism worked and the model had
                            # nothing to say.
                            if hasattr(pipe, "ld_failure_rate"):
                                row["ld_failure_rate"] = pipe.ld_failure_rate
                            if hasattr(pipe, "vote_summary"):
                                summary = pipe.vote_summary()
                                row["first_step_lead_mean"] = summary.get("first_step_lead_mean", "")
                                row["vote_threshold"] = summary.get("vote_threshold", "")
                                row["refused_at_first_step"] = summary.get("refused_at_first_step", "")
                        del pipe
                    except Exception as exc:  # noqa: BLE001 - record failures per cell
                        row["status"] = "ERROR"
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        logger.error("FAILED %s: %s\n%s", cell, exc, traceback.format_exc())
                    row["seconds"] = time.perf_counter() - start
                    run.record(row, predictions)
                    logger.info("Checkpoint saved (%d/%d cells)", len(run.done), total_cells)

    run.finalise()

    print()
    for r in run.rows:
        f1 = r.get("token_f1_mean", float("nan"))
        f1_s = "nan" if f1 != f1 else f"{f1:.3f}"
        print(f"{r['cell']:<60} {r['status']:<11} F1={f1_s}  ({r['seconds']:.1f}s)")
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
