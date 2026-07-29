"""Run the RAG membership inference attack (Anderson et al., 2024) across a sweep.

For each cell — (seed x dataset variant x epsilon x pipeline x scoring mode) — it:
  1. Splits HealthCareMagic into members (indexed) and held-out non-members.
  2. Applies the variant to the member documents.
  3. **Calibrates** the pipeline to the target total (epsilon, delta), or records the
     cell as INFEASIBLE if the mechanism cannot deliver that guarantee at all.
  4. Indexes the corpus and runs the MIA over balanced member/non-member targets.
  5. Records AUC, TPR@FPR=1%, TPR@FPR=5% and threshold metrics.

``--epsilons`` are **target total budgets for one answer of ``--n-tokens`` tokens**,
including retrieval. Each pipeline solves its own knobs to meet that target using
its own accountant, so the axis means the same thing for all of them. Plot against
the recorded ``realized_epsilon``. See ``docs/METHODOLOGY.md``.

Scoring modes (``--modes``) are not interchangeable:

  unified      P(the deployed system answers 'Yes'), the framework's common
               estimand — exact where the output distribution is closed-form
               (naive, Grislain), sampled where it is not (Koga). **Compare
               pipelines with this one.**
  upper_bound  What an attacker reading pre-noise internals could extract. Koga
               only; flat in epsilon by construction. Never plot alongside unified.
  black_box    Parse Yes/No from generated text. The paper's threat model and what
               a real API exposes, but binary, so TPR@FPR=1% is degenerate.

TPR@FPR=1% is only resolvable with at least 100 non-member targets; below that it
is structurally zero regardless of how much membership leaks.

Outputs (under --output-dir): results.json, results.csv, predictions.jsonl, report.md.

Usage:
    # groups 1+2: baseline and calibrated epsilon sweep over the four variants
    python scripts/run_attack.py --variant-grid base --epsilons inf,0.5,1,2,4,8 \
        --n-tokens 32 --delta 1e-6 --members 1000 --non-members 1000 \
        --targets-per-class 250 --seeds 42,43,44

    # group 3: duplication-strength sweep at two representative budgets
    python scripts/run_attack.py --variant-grid duplication --epsilons inf,1,4 \
        --n-tokens 32 --members 1000 --non-members 1000 \
        --targets-per-class 250 --seeds 42,43,44
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

from dp_rag_eval.attacks import RAGMembershipInferenceAttack
from dp_rag_eval.attacks.membership import ALL_MODES, MODE_METHODS
from dp_rag_eval.core.types import Document, PrivacyTarget
from dp_rag_eval.datasets.loaders import HealthCareMagicLoader
from dp_rag_eval.pipelines import Embedder, load_generator, resolve_device
from dp_rag_eval.runner.sweep import (
    ALL_PIPELINES,
    NON_PRIVATE,
    VARIANT_COLUMNS,
    CheckpointedRun,
    PipelineParams,
    build_pipeline,
    eps_label,
    grid_from_args,
    make_variant_cell,
    parse_floats,
    parse_ints,
    parse_strs,
    seeded_sample,
    split_members,
    variant_configs,
)

logger = logging.getLogger("run_attack")

# Which scores each pipeline can produce. Consulted before a pipeline is built so
# that an unsupported (pipeline, mode) pair costs nothing.
PIPELINE_MODES = {
    "naive": {"unified", "black_box"},
    "koga": {"unified", "upper_bound", "black_box"},
    "grislain": {"unified", "black_box"},
}

CSV_FIELDS = [
    "pipeline",
    "mode",
    *VARIANT_COLUMNS,
    "epsilon",
    "realized_epsilon",
    "realized_delta",
    "seed",
    "status",
    "auc",
    "tpr_at_fpr_01",
    "tpr_at_fpr_05",
    "accuracy",
    "precision",
    "tpr",
    "fpr",
    "n_members",
    "n_non_members",
    "n_indexed",
    "n_split",
    "ld_threshold",
    "ld_failure_rate",
    "seconds",
    "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("runs/attack"), help="Where to write results")
    p.add_argument("--pipelines", default=",".join(ALL_PIPELINES), help="Comma-separated: naive,koga,grislain")
    p.add_argument("--modes", default="unified", help=f"Comma-separated subset of {ALL_MODES}")

    p.add_argument(
        "--generator",
        default="Qwen/Qwen3.5-9B",
        help="HF causal-LM id (shared). The default is the model the matrix is run on; pass a "
        "small one (e.g. Qwen/Qwen2.5-0.5B-Instruct) with --device cpu for a local dry run.",
    )
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--dtype", default=None, help="bfloat16 / float16 / float32")

    p.add_argument("--members", type=int, default=40, help="Documents indexed as members")
    p.add_argument("--non-members", type=int, default=40, help="Held-out documents (never indexed)")
    p.add_argument(
        "--targets-per-class",
        type=int,
        default=10,
        help="Member and non-member targets queried each. TPR@FPR=1%% needs >=100 to be resolvable.",
    )
    p.add_argument("--max-doc-chars", type=int, default=300, help="Truncate documents")
    p.add_argument("--sample-chars", type=int, default=300, help="Target sample length placed in the attack prompt")

    p.add_argument("--epsilons", default="inf", help="Target total budgets; 'inf' is the non-private reference")
    p.add_argument("--delta", type=float, default=1e-6, help="Target delta; must be << 1/corpus size")
    p.add_argument("--n-tokens", type=int, default=32, help="Answer length the guarantee covers")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds; each is an independent replicate")

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
    p.add_argument("--mc-samples", type=int, default=200, help="Koga: draws used to estimate P('Yes')")
    p.add_argument("--retrieval-share", type=float, default=0.1, help="Grislain: share of epsilon spent retrieving")
    p.add_argument("--top-k", type=int, default=5, help="Documents retrieved per query")
    return p.parse_args()


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# RAG membership inference attack (Anderson et al., 2024)", ""]
    lines.append(f"- generator: `{meta['generator']}`  |  device: `{meta['device']}`")
    lines.append(
        f"- members: {meta['members']}  |  non-members: {meta['non_members']}  "
        f"|  targets/class: {meta['targets_per_class']}  |  seeds: {meta['seeds']}"
    )
    lines.append(
        f"- target epsilons: {meta['epsilons']}  |  delta: {meta['delta']}  "
        f"|  answer length: {meta['n_tokens']} tokens  |  modes: {meta['modes']}"
    )
    lines.append("")
    lines.append("AUC 0.5 = no better than chance; higher = membership leaks.")
    lines.append("")
    lines.append(
        "**Plot against `realized_epsilon`.** It is the total guarantee each pipeline's "
        "own accountant reports for the configuration actually run, covering retrieval "
        "and generation of an entire answer. The requested value is kept only for "
        "traceability."
    )
    lines.append("")
    lines.append(
        "**Rows with status `INFEASIBLE`** are targets the mechanism cannot deliver at "
        "any parameter setting — for Koga, budgets so tight that the Limited-Domain "
        "selection needs more agreeing voters than exist. They were not run. Had they "
        "been, they would have produced an AUC of 0.5 that looks exactly like a "
        "successful defence. The reachable epsilon range is itself a result."
    )
    lines.append("")
    lines.append(
        "**Mode `unified`** is the only one comparable across pipelines: P(the deployed "
        "system answers 'Yes'), computed exactly where the output distribution is "
        "closed-form and sampled where it is not. `upper_bound` describes an attacker "
        "reading pre-noise internals and is flat in epsilon by construction; the gap "
        "between the two is how much leakage the mechanism removes."
    )
    lines.append("")
    lines.append(
        "| pipeline | mode | variant | params | target eps | realized eps | seed | status | "
        "AUC | TPR@1% | TPR@5% | n_split | secs |"
    )
    lines.append("|" + "---|" * 13)
    for r in rows:

        def fmt(key: str) -> str:
            v = r.get(key, float("nan"))
            if isinstance(v, str):
                return v or "-"
            return "nan" if v != v else f"{v:.3f}"

        lines.append(
            f"| {r['pipeline']} | {r['mode']} | {r['variant']} | {r['params']} | {r['epsilon']} | "
            f"{fmt('realized_epsilon')} | {r['seed']} | {r['status']} | {fmt('auc')} | "
            f"{fmt('tpr_at_fpr_01')} | {fmt('tpr_at_fpr_05')} | {r.get('n_split', '-')} | {r['seconds']:.1f} |"
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
    variants = variant_configs(args.variant_grid, grid_from_args(args), parse_strs(args.variants or "") or None)

    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        raise SystemExit(f"Unknown modes: {sorted(unknown)}; choose from {ALL_MODES}")
    if args.targets_per_class < 100:
        logger.warning(
            "targets-per-class=%d: TPR@FPR=0.01 needs >=100 non-member targets to be "
            "resolvable and will read 0 regardless of leakage",
            args.targets_per_class,
        )
    if args.delta >= 1.0 / max(args.members, 1):
        logger.warning(
            "delta=%g is not below 1/members=%g. At or above 1/n, a mechanism may publish "
            "a whole random record and still satisfy the definition.",
            args.delta,
            1.0 / max(args.members, 1),
        )

    params = PipelineParams(
        delta=args.delta,
        n_tokens=args.n_tokens,
        top_k=args.top_k,
        n_split=args.n_split,
        mc_samples=args.mc_samples,
        retrieval_share=args.retrieval_share,
        auto_n_split_cap=args.auto_n_split_cap,
    )
    # Which (pipeline, mode) pairs will actually be scheduled — needed before meta so
    # the planned cell count is exact rather than an upper bound.
    pipeline_modes = {name: [m for m in modes if m in PIPELINE_MODES.get(name, set())] for name in pipeline_names}
    for name, supported in pipeline_modes.items():
        dropped = sorted(set(modes) - set(supported))
        if dropped:
            logger.info("%s does not implement %s — those cells are not scheduled", name, dropped)

    total_cells = sum(
        len(seeds) * len(variants) * (1 if name in NON_PRIVATE else len(epsilons)) * len(pipeline_modes[name])
        for name in pipeline_names
    )

    meta = {
        "attack": "Anderson et al. 2024 (arXiv:2405.20446) MIA",
        "planned_cells": total_cells,
        "generator": args.generator,
        "embedder": args.embedder,
        "device": device,
        "members": args.members,
        "non_members": args.non_members,
        "targets_per_class": args.targets_per_class,
        "seeds": seeds,
        "epsilons": [eps_label(e) for e in epsilons],
        "delta": args.delta,
        "n_tokens": args.n_tokens,
        "modes": modes,
        "n_split": args.n_split,
        "top_k": args.top_k,
        "mc_samples": args.mc_samples,
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

        # The member / non-member split is re-drawn per seed, so seeds are independent
        # replicates of the whole experiment rather than of the DP noise alone.
        loader = HealthCareMagicLoader(n_docs=args.members + args.non_members + 50, seed=seed)
        members_base, non_members = split_members(loader.load(), args.members, args.non_members, args.max_doc_chars)
        targets: list[tuple[Document, bool]] = [
            (d, True) for d in seeded_sample(members_base, args.targets_per_class, seed)
        ] + [(d, False) for d in seeded_sample(non_members, args.targets_per_class, seed + 1)]
        random.Random(seed).shuffle(targets)
        logger.info("[seed %d] %d targets (%d per class)", seed, len(targets), args.targets_per_class)

        for cell_spec in variants:
            variant_columns = cell_spec.columns()
            indexed_corpus = make_variant_cell(cell_spec, seed).apply(members_base)
            logger.info(
                "[seed %d][%s %s] corpus: %d documents",
                seed,
                cell_spec.name,
                variant_columns["params"],
                len(indexed_corpus),
            )

            for epsilon in epsilons:
                target = PrivacyTarget(epsilon=epsilon, delta=args.delta, n_tokens=args.n_tokens)
                for name in pipeline_names:
                    if name in NON_PRIVATE and target.is_private:
                        continue  # no mechanism: only the eps=inf cell is meaningful
                    prefix = f"{name}|%s|{cell_spec.name}|{variant_columns['params']}|{target.label()}|seed={seed}"
                    pending = [m for m in pipeline_modes[name] if not run.is_done(prefix % m)]
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
                    base_row = {
                        "pipeline": name,
                        **variant_columns,
                        "epsilon": eps_label(epsilon),
                        "realized_epsilon": calibration.realized.epsilon,
                        "realized_delta": calibration.realized.delta,
                        "seed": seed,
                        "n_indexed": len(indexed_corpus),
                        "n_split": calibration.diagnostics.get("n_split", ""),
                        "ld_threshold": calibration.diagnostics.get("ld_threshold", ""),
                    }

                    if not calibration.feasible:
                        # Record once per mode so the gap is visible in the results
                        # rather than silently absent, then move on without running.
                        logger.warning("INFEASIBLE %s: %s", prefix % pending[0], calibration.reason)
                        for mode in pending:
                            run.record(
                                {
                                    **base_row,
                                    "cell": prefix % mode,
                                    "mode": mode,
                                    "status": "INFEASIBLE",
                                    "error": calibration.reason,
                                    "seconds": 0.0,
                                }
                            )
                        del pipe
                        continue

                    pipe.index(indexed_corpus)
                    for mode in pending:
                        cell = prefix % mode
                        required = MODE_METHODS.get(mode)
                        if required and not hasattr(pipe, required):
                            logger.info("Skipping %s (pipeline has no %s)", cell, required)
                            continue

                        logger.info("Running %s ...", cell)
                        start = time.perf_counter()
                        row: dict[str, Any] = {**base_row, "cell": cell, "mode": mode, "status": "OK", "error": ""}
                        predictions: list[dict[str, Any]] = []
                        try:
                            if hasattr(pipe, "reset_ld_stats"):
                                pipe.reset_ld_stats()
                            attack = RAGMembershipInferenceAttack(pipe, sample_chars=args.sample_chars, mode=mode)
                            result = attack.run(targets)
                            row.update(result.metrics)
                            predictions = result.predictions
                            if hasattr(pipe, "ld_failure_rate"):
                                row["ld_failure_rate"] = pipe.ld_failure_rate
                        except Exception as exc:  # noqa: BLE001 - record failures per cell
                            row["status"] = "ERROR"
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            logger.error("FAILED %s: %s\n%s", cell, exc, traceback.format_exc())
                        row["seconds"] = time.perf_counter() - start
                        run.record(row, predictions)
                        logger.info("Checkpoint saved (%d/%d cells)", len(run.done), total_cells)

                    del pipe

    run.finalise()

    print()
    for r in run.rows:
        auc = r.get("auc", float("nan"))
        auc_s = "nan" if auc != auc else f"{auc:.3f}"
        print(f"{r['cell']:<64} {r['status']:<11} AUC={auc_s}  ({r['seconds']:.1f}s)")
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
