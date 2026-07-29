"""Compare duplicated-corpus cells against their zero-duplication control.

Reads finished sweep results back — from one output directory or several — pairs
each duplicated cell with the ``original`` cell at the same pipeline, scoring mode,
epsilon target and seed, and reports ``delta`` and ``ratio`` plus a mean / std /
95% CI pooled over the seed axis.

This is the RQ1 arithmetic. Everything upstream produces one number per
configuration; this is where "how much worse does duplication make it" becomes a
computed value rather than a reading exercise over a flat table.

Pairing uses the typed variant columns (``variant``, ``dup_axis``, ``dup_strength``,
``duplication_rate``, ...) that the runners write, so nothing here parses the
human-readable ``params`` string. Results produced before those columns existed will
have no ``variant`` column and cannot be paired.

Several ``--results`` paths are merged and deduplicated on the cell key, so a base
grid and a duplication grid that were run into separate directories can still be
compared. Prefer running them into one directory in the first place.

Outputs (under --output-dir): degradation.md, degradation_pairs.csv,
degradation_pooled.csv.

Usage:
    python scripts/report_degradation.py --results runs/mia_main --metric auc
    python scripts/report_degradation.py --results runs/mia_main runs/mia_duplication \
        --metric auc --metric tpr_at_fpr_01 --output-dir runs/analysis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dp_rag_eval.evaluation.reporter import (
    ANCHOR_VARIANT,
    PAIR_FIELDS,
    POOLED_FIELDS,
    PairedCell,
    PooledStat,
    detect_metrics,
    load_rows,
    pair_against_anchor,
    pool_over_seeds,
    render_markdown,
    write_csv,
)

logger = logging.getLogger("report_degradation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--results",
        nargs="+",
        required=True,
        type=Path,
        help="Run directories or results.json/.csv files. Several are merged and deduplicated on cell key.",
    )
    p.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric column to compare (repeatable). Default: every known metric present in the rows.",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Where to write the report (default: first source)")
    p.add_argument("--anchor-variant", default=ANCHOR_VARIANT, help="Variant treated as the zero-duplication control")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"No result rows found under {[str(s) for s in args.results]}")

    metrics = args.metric or detect_metrics(rows)
    if not metrics:
        raise SystemExit("No comparable metric columns found in these results")

    variants = {str(r.get("variant", "")) for r in rows}
    if args.anchor_variant not in variants:
        raise SystemExit(
            f"No '{args.anchor_variant}' rows in these results (found variants: {sorted(v for v in variants if v)}). "
            "The control and the duplicated cells must be in the same results, or passed as several --results paths."
        )

    pairs: list[PairedCell] = []
    warnings: list[str] = []
    for metric in metrics:
        metric_pairs, metric_warnings = pair_against_anchor(rows, metric, anchor_variant=args.anchor_variant)
        pairs += metric_pairs
        warnings += [f"[{metric}] {w}" for w in metric_warnings]
        logger.info("%s: %d paired cells", metric, len(metric_pairs))

    pooled: list[PooledStat] = pool_over_seeds(pairs)

    output_dir = args.output_dir or Path(args.results[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "sources": [str(s) for s in args.results],
        "metrics": metrics,
        "anchor": args.anchor_variant,
        "n_rows": len(rows),
    }
    write_csv(output_dir / "degradation_pairs.csv", pairs, PAIR_FIELDS)
    write_csv(output_dir / "degradation_pooled.csv", pooled, POOLED_FIELDS)
    (output_dir / "degradation.md").write_text(render_markdown(pairs, pooled, warnings, meta), encoding="utf-8")

    print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not pairs:
        print("No cells could be paired — check that duplicated and control rows share a pipeline/mode/epsilon/seed.")
    for s in pooled:
        if s.quantity != "delta":
            continue
        repeats = f" repeats={s.canary_n_repeats:.0f}" if s.canary_n_repeats else ""
        print(
            f"{s.metric:<14} {s.pipeline:<9} {s.mode:<11} eps={s.epsilon:<5} "
            f"{s.variant}({s.params}){repeats} delta={s.mean:+.4f} +/- {s.std:.4f} (n={s.n_seeds})"
        )
    print(f"\nWrote report to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
