"""Build the four controlled dataset variants and cache them to disk.

Loads a base corpus (HealthCareMagic) and writes, under ``--output-dir``:

    original.jsonl
    full_duplicate.jsonl
    partial_overlap.jsonl
    canary.jsonl
    canary_ground_truth.json   (the injected secrets — ground truth for attacks)
    summary.json               (params + per-variant document counts)

Usage:
    python scripts/prepare_dataset.py --n-docs 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dp_rag_eval.datasets import (
    CanaryInjectionVariant,
    FullDuplicateVariant,
    HealthCareMagicLoader,
    OriginalVariant,
    PartialOverlapVariant,
    save_documents_jsonl,
)

logger = logging.getLogger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-docs", type=int, default=1000, help="Base corpus size (None-like: pass -1 for full split)")
    p.add_argument("--seed", type=int, default=42, help="Global seed for loading and all variants")
    p.add_argument("--output-dir", type=Path, default=Path("data/variants"), help="Where to write the variants")
    p.add_argument("--cache-dir", type=Path, default=None, help="HuggingFace datasets cache dir")
    # full_duplicate
    p.add_argument("--dup-rate", type=float, default=0.3, help="Fraction of docs duplicated verbatim")
    p.add_argument("--dup-copies", type=int, default=1, help="Extra verbatim copies per duplicated doc")
    # partial_overlap
    p.add_argument("--overlap-rate", type=float, default=0.3, help="Fraction of docs used as fragment donors")
    p.add_argument("--overlap-hosts", type=int, default=3, help="Host docs receiving each donor fragment")
    # canary
    p.add_argument("--n-canaries", type=int, default=20, help="Number of distinct secrets to inject")
    p.add_argument("--canary-repeats", type=int, default=1, help="Documents each secret is inserted into")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_docs = None if args.n_docs is not None and args.n_docs < 0 else args.n_docs
    loader = HealthCareMagicLoader(n_docs=n_docs, seed=args.seed, cache_dir=args.cache_dir)
    base = loader.load()

    variants = {
        "original": OriginalVariant(),
        "full_duplicate": FullDuplicateVariant(args.dup_rate, args.dup_copies, args.seed),
        "partial_overlap": PartialOverlapVariant(args.overlap_rate, args.overlap_hosts, args.seed),
        "canary": CanaryInjectionVariant(args.n_canaries, args.canary_repeats, seed=args.seed),
    }

    summary: dict = {
        "hf_name": loader.hf_name,
        "base_documents": len(base),
        "seed": args.seed,
        "variants": {},
    }

    for name, variant in variants.items():
        corpus = variant.apply(base)
        path = save_documents_jsonl(corpus, out_dir / f"{name}.jsonl")
        summary["variants"][name] = {"documents": len(corpus), "file": path.name}
        logger.info("wrote %s (%d documents)", path, len(corpus))

        if isinstance(variant, CanaryInjectionVariant):
            gt_path = out_dir / "canary_ground_truth.json"
            gt_path.write_text(json.dumps(variant.injected_canaries, indent=2, ensure_ascii=False), encoding="utf-8")
            summary["variants"][name]["n_canaries"] = len(variant.injected_canaries)
            summary["variants"][name]["n_repeats"] = variant.n_repeats
            summary["variants"][name]["ground_truth_file"] = gt_path.name
            logger.info("wrote %s (%d canaries)", gt_path, len(variant.injected_canaries))

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", summary_path)
    logger.info("Done. Variants in %s", out_dir.resolve())


if __name__ == "__main__":
    main()
