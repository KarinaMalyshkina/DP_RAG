"""Run the canary extraction attack (Secret Sharer; Carlini et al., 2019) across a sweep.

For each cell — (seed x canary repeats x epsilon x pipeline x mode) — it:
  1. Loads a HealthCareMagic corpus and injects ``n_canaries`` distinct secrets,
     each duplicated into ``n_repeats`` documents.
  2. Indexes that corpus in the pipeline.
  3. Queries for each secret and ranks the true code against random distractors.
  4. Records Secret-Sharer exposure and the verbatim extraction rate.

This is the experiment RQ1 rests on most directly: ``n_repeats`` is the number of
copies of *the same secret* in the corpus. Under a per-document DP guarantee,
removing one copy leaves the others, so exposure is expected to climb with
``n_repeats`` even at a fixed epsilon.

Modes:
  exposure    Rank the true code by log-probability against ``n_distractors``
              random codes; report ``log2(N) - log2(rank)``. Continuous and
              comparable across epsilon. Needs ``completion_logprob``.
              For ``koga`` this bypasses LD-Gumbel noise (an upper bound — Koga
              has no scorable distribution); for ``grislain`` it is DP-faithful.
  generation  Ask for the code and check for a verbatim leak. Black-box, works
              against any pipeline, but coarse: usually 0 for every DP setting.

Cost note: exposure mode scores ``n_distractors + 1`` candidates per canary. The
Grislain pipeline scores a whole candidate set against one shared prompt prefix,
so raising ``--n-distractors`` is much cheaper there than the count suggests.

Outputs (under --output-dir): results.json, results.csv, predictions.jsonl, report.md.

Usage:
    python scripts/run_canary.py --epsilons inf,1,4 --canary-repeats 1,2,5,10 \
        --corpus-size 1000 --n-canaries 20 --n-distractors 255 --seeds 42,43,44
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

from dp_rag_eval.attacks import CanaryExtractionAttack
from dp_rag_eval.core.types import PrivacyTarget
from dp_rag_eval.datasets.loaders import HealthCareMagicLoader
from dp_rag_eval.datasets.variants import CanaryInjectionVariant
from dp_rag_eval.pipelines import Embedder, load_generator, resolve_device
from dp_rag_eval.runner.sweep import (
    ALL_PIPELINES,
    NON_PRIVATE,
    CheckpointedRun,
    PipelineParams,
    build_pipeline,
    eps_label,
    parse_floats,
    parse_ints,
    parse_strs,
    truncate_doc,
)

logger = logging.getLogger("run_canary")

ALL_MODES = ["exposure", "generation"]
MODE_REQUIREMENTS = {"exposure": "completion_logprob", "generation": "answer"}

CSV_FIELDS = [
    "pipeline",
    "mode",
    "n_repeats",
    "epsilon",
    "realized_epsilon",
    "realized_delta",
    "seed",
    "status",
    "mean_exposure",
    "median_exposure",
    "max_exposure",
    "max_possible_exposure",
    "extraction_rate",
    "mean_rank",
    "n_canaries",
    "n_candidates",
    "n_indexed",
    "n_split",
    "seconds",
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("runs/canary"), help="Where to write results")
    p.add_argument("--pipelines", default=",".join(ALL_PIPELINES), help="Comma-separated: naive,koga,grislain")
    p.add_argument("--modes", default="exposure", help=f"Comma-separated subset of {ALL_MODES}")

    p.add_argument("--generator", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF causal-LM id (shared)")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--dtype", default=None, help="bfloat16 / float16 / float32 (default: bf16 on cuda, fp32 on cpu)")

    p.add_argument("--corpus-size", type=int, default=200, help="Documents in the indexed corpus")
    p.add_argument("--max-doc-chars", type=int, default=300, help="Truncate documents")
    p.add_argument("--n-canaries", type=int, default=10, help="Distinct secrets injected per cell")
    p.add_argument("--canary-repeats", default="1", help="Sweep: documents each secret is inserted into")
    p.add_argument("--n-distractors", type=int, default=255, help="Random reference codes per canary")

    p.add_argument("--epsilons", default="inf", help="Target total budgets; 'inf' is the non-private reference")
    p.add_argument("--delta", type=float, default=1e-6, help="Target delta; must be << 1/corpus size")
    p.add_argument("--n-tokens", type=int, default=32, help="Answer length the guarantee covers")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds; each is an independent replicate")

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
    p.add_argument("--retrieval-share", type=float, default=0.1, help="Grislain: share of epsilon spent retrieving")
    p.add_argument("--top-k", type=int, default=5, help="Documents retrieved per query")
    return p.parse_args()


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# Canary extraction (Secret Sharer; Carlini et al., 2019)", ""]
    lines.append(
        f"- generator: `{meta['generator']}`  |  embedder: `{meta['embedder']}`  |  device: `{meta['device']}`"
    )
    lines.append(
        f"- corpus: {meta['corpus_size']} docs  |  canaries: {meta['n_canaries']}  "
        f"|  distractors: {meta['n_distractors']}  |  seeds: {meta['seeds']}"
    )
    lines.append(f"- epsilons: {meta['epsilons']}  |  delta: {meta['delta']}  |  modes: {meta['modes']}")
    lines.append("")
    lines.append(
        "Exposure = `log2(n_candidates) - log2(rank)`. The maximum is "
        "`log2(n_distractors + 1)`; a secret sitting mid-pack scores ~0. "
        "`extraction_rate` is the fraction of secrets ranked first."
    )
    lines.append("")
    lines.append(
        "**The RQ1 reading:** hold epsilon fixed and vary `n_repeats`. A per-document "
        "DP guarantee protects one copy at a time, so exposure rising with `n_repeats` "
        "is the gap between the formal guarantee and actual protection."
    )
    lines.append("")
    lines.append(
        "Caveat: `exposure` for `koga` scores raw voter log-probabilities, bypassing "
        "LD-Gumbel noise (the mechanism selects tokens and has no scorable "
        "distribution for an arbitrary string) — read it as an upper bound. For "
        "`grislain` the score passes through `DPLogitsAggregator` and is faithful."
    )
    lines.append("")
    lines.append(
        "| pipeline | mode | n_repeats | eps | seed | status | mean exposure | max possible | "
        "extraction rate | mean rank | secs |"
    )
    lines.append("|" + "---|" * 11)
    for r in rows:

        def fmt(key: str) -> str:
            v = r.get(key, float("nan"))
            return "nan" if v != v else f"{v:.3f}"

        lines.append(
            f"| {r['pipeline']} | {r['mode']} | {r['n_repeats']} | {r['epsilon']} | {r['seed']} | "
            f"{r['status']} | {fmt('mean_exposure')} | {fmt('max_possible_exposure')} | "
            f"{fmt('extraction_rate')} | {fmt('mean_rank')} | {r['seconds']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    torch.set_num_threads(os.cpu_count() or 4)

    device = resolve_device(args.device)
    pipeline_names = parse_strs(args.pipelines)
    modes = parse_strs(args.modes)
    epsilons = parse_floats(args.epsilons)
    seeds = parse_ints(args.seeds)
    repeats_grid = parse_ints(args.canary_repeats)

    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}; choose from {ALL_MODES}")

    params = PipelineParams(
        delta=args.delta,
        n_tokens=args.n_tokens,
        top_k=args.top_k,
        n_split=args.n_split,
        retrieval_share=args.retrieval_share,
        auto_n_split_cap=args.auto_n_split_cap,
    )
    meta = {
        "attack": "Carlini et al. 2019 Secret Sharer, RAG-adapted",
        "generator": args.generator,
        "embedder": args.embedder,
        "device": device,
        "dtype": args.dtype or ("bfloat16" if device.startswith("cuda") else "float32"),
        "corpus_size": args.corpus_size,
        "n_canaries": args.n_canaries,
        "n_distractors": args.n_distractors,
        "canary_repeats": repeats_grid,
        "seeds": seeds,
        "epsilons": [eps_label(e) for e in epsilons],
        "delta": args.delta,
        "n_tokens": args.n_tokens,
        "modes": modes,
        "top_k": args.top_k,
    }
    run = CheckpointedRun(args.output_dir, meta, CSV_FIELDS)

    generator, tokenizer = load_generator(args.generator, device, dtype=args.dtype)
    embedder = Embedder(args.embedder, device)

    total_cells = sum(
        len(seeds) * len(repeats_grid) * (1 if name in NON_PRIVATE else len(epsilons)) * len(modes)
        for name in pipeline_names
    )
    logger.info("Planned: %d cells (%d seeds x %d repeat levels)", total_cells, len(seeds), len(repeats_grid))

    for seed in seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        loader = HealthCareMagicLoader(n_docs=args.corpus_size, seed=seed)
        base = [truncate_doc(d, args.max_doc_chars) for d in loader.load()]
        if len(base) < args.corpus_size:
            logger.warning("Loaded %d documents, requested %d", len(base), args.corpus_size)

        for n_repeats in repeats_grid:
            variant = CanaryInjectionVariant(n_canaries=args.n_canaries, n_repeats=n_repeats, seed=seed)
            corpus = variant.apply(base)
            canaries = variant.injected_canaries
            logger.info(
                "[seed %d][n_repeats %d] %d canaries injected into %d documents",
                seed,
                n_repeats,
                len(canaries),
                len(corpus),
            )

            for epsilon in epsilons:
                target = PrivacyTarget(epsilon=epsilon, delta=args.delta, n_tokens=args.n_tokens)
                for name in pipeline_names:
                    if name in NON_PRIVATE and target.is_private:
                        continue
                    label = eps_label(epsilon)
                    prefix = f"{name}|%s|repeats={n_repeats}|{target.label()}|seed={seed}"
                    pending = [m for m in modes if not run.is_done(prefix % m)]
                    if not pending:
                        continue

                    pipe, calibration = build_pipeline(
                        name,
                        target,
                        params,
                        generator=generator,
                        tokenizer=tokenizer,
                        embedder=embedder,
                        device=device,
                    )
                    calib_row = {
                        "realized_epsilon": calibration.realized.epsilon,
                        "realized_delta": calibration.realized.delta,
                        "n_split": calibration.diagnostics.get("n_split", ""),
                    }
                    if not calibration.feasible:
                        # The mechanism cannot deliver this budget at any setting. Record
                        # the gap rather than running a cell whose numbers would describe
                        # the mechanism failing to operate.
                        logger.warning("INFEASIBLE %s: %s", prefix % pending[0], calibration.reason)
                        for mode in pending:
                            run.record(
                                {
                                    **calib_row,
                                    "cell": prefix % mode,
                                    "pipeline": name,
                                    "mode": mode,
                                    "n_repeats": n_repeats,
                                    "epsilon": label,
                                    "seed": seed,
                                    "n_indexed": len(corpus),
                                    "status": "INFEASIBLE",
                                    "error": calibration.reason,
                                    "seconds": 0.0,
                                }
                            )
                        del pipe
                        continue

                    pipe.index(corpus)

                    for mode in pending:
                        cell = prefix % mode
                        required = MODE_REQUIREMENTS.get(mode)
                        if required and not hasattr(pipe, required):
                            logger.info("Skipping %s (pipeline has no %s)", cell, required)
                            continue

                        logger.info("Running %s ...", cell)
                        start = time.perf_counter()
                        row: dict[str, Any] = {
                            **calib_row,
                            "cell": cell,
                            "pipeline": name,
                            "mode": mode,
                            "n_repeats": n_repeats,
                            "epsilon": label,
                            "seed": seed,
                            "n_indexed": len(corpus),
                            "status": "OK",
                            "error": "",
                        }
                        predictions: list[dict[str, Any]] = []
                        try:
                            attack = CanaryExtractionAttack(
                                pipe,
                                canaries,
                                n_distractors=args.n_distractors,
                                mode=mode,
                                seed=seed,
                            )
                            result = attack.run()
                            row.update(result.metrics)
                            predictions = result.predictions
                        except Exception as exc:  # noqa: BLE001 - record failures per cell
                            row["status"] = "ERROR"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            logger.error("FAILED %s: %s\n%s", cell, exc, traceback.format_exc())
                        row["seconds"] = time.perf_counter() - start
                        run.record(row, predictions)
                        logger.info("Checkpoint saved (%d/%d cells)", len(run.done), total_cells)

                    del pipe

    run.finalise(render_markdown(run.rows, meta))

    print()
    for r in run.rows:
        exp = r.get("mean_exposure", float("nan"))
        exp_s = "nan" if exp != exp else f"{exp:.3f}"
        print(f"{r['cell']:<60} {r['status']:<6} mean_exposure={exp_s}  ({r['seconds']:.1f}s)")
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
