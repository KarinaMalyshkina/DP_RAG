"""Run the RAG membership inference attack (Anderson et al., 2024) across variants.

For each (pipeline x dataset variant) it:
  1. Splits HealthCareMagic into members (indexed) and held-out non-members.
  2. Applies the variant to the member documents and indexes them in the pipeline.
  3. Runs the black-box MIA over a balanced set of member/non-member targets.
  4. Records AUC and threshold metrics.

This is a smoke-scale demonstration (tiny CPU model, few targets): it shows the
attack and the privacy-vs-duplication trend, not research-grade numbers. Scale up
with --members / --targets-per-class / --generator for real runs (ideally on GPU).

Outputs (under --output-dir): results.json, results.csv, report.md

Usage:
    python scripts/run_attack.py
    python scripts/run_attack.py --pipelines koga,grislain --variants original,full_duplicate
    python scripts/run_attack.py --members 200 --targets-per-class 100 --generator <gpu-model>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from dp_rag_eval.attacks import RAGMembershipInferenceAttack
from dp_rag_eval.core.types import Document
from dp_rag_eval.datasets.loaders import HealthCareMagicLoader
from dp_rag_eval.datasets.variants import (
    CanaryInjectionVariant,
    FullDuplicateVariant,
    OriginalVariant,
    PartialOverlapVariant,
)
from dp_rag_eval.pipelines import Embedder, GrislainPipeline, KogaPipeline, NaivePipeline, load_generator, resolve_device

logger = logging.getLogger("run_attack")

ALL_VARIANTS = ["original", "full_duplicate", "partial_overlap", "canary"]
ALL_PIPELINES = ["naive", "koga", "grislain"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("runs/attack"), help="Where to write results")
    p.add_argument("--pipelines", default=",".join(ALL_PIPELINES), help="Comma-separated subset of: koga,grislain")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS), help="Comma-separated subset of the four variants")
    # Gray-box MIA needs a model that can follow the Yes/No instruction; 135M is too
    # weak (collapses to chance), so the attack defaults to a small-but-capable model.
    p.add_argument("--generator", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF causal-LM id (shared)")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--members", type=int, default=40, help="Documents indexed as members")
    p.add_argument("--non-members", type=int, default=40, help="Held-out documents (never indexed)")
    p.add_argument("--targets-per-class", type=int, default=10, help="Member and non-member targets queried each")
    p.add_argument("--max-doc-chars", type=int, default=300, help="Truncate documents (CPU speed)")
    p.add_argument("--sample-chars", type=int, default=300, help="Target sample length placed in the attack prompt")
    p.add_argument(
        "--mode",
        choices=["gray_box", "black_box"],
        default="gray_box",
        help="gray_box: continuous P(Yes) from first-token logits; black_box: parse Yes/No text",
    )
    p.add_argument("--max-new-tokens", type=int, default=4, help="Tokens generated per query (black-box only)")
    p.add_argument("--n-split", type=int, default=8, help="Koga: number of voters")
    p.add_argument("--seed", type=int, default=42, help="Global seed")
    return p.parse_args()


def make_variant(name: str, seed: int) -> Any:
    """Instantiate a dataset variant with defaults matching scripts/prepare_dataset.py."""
    if name == "original":
        return OriginalVariant()
    if name == "full_duplicate":
        return FullDuplicateVariant(duplication_rate=0.3, n_copies=1, seed=seed)
    if name == "partial_overlap":
        return PartialOverlapVariant(overlap_rate=0.3, n_hosts=3, seed=seed)
    if name == "canary":
        return CanaryInjectionVariant(n_canaries=10, n_repeats=1, seed=seed)
    raise ValueError(f"Unknown variant: {name}")


def truncate_doc(doc: Document, max_chars: int) -> Document:
    if max_chars <= 0 or len(doc.text) <= max_chars:
        return doc.copy()
    return doc.copy(text=doc.text[:max_chars])


def build_pipelines(names: list[str], args: argparse.Namespace, device: str) -> dict[str, Any]:
    generator, tokenizer = load_generator(args.generator, device)
    embedder = Embedder(args.embedder, device)
    pipelines: dict[str, Any] = {}
    if "naive" in names:
        pipelines["naive"] = NaivePipeline(
            generator,
            tokenizer,
            embedder,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
    if "koga" in names:
        pipelines["koga"] = KogaPipeline(
            generator,
            tokenizer,
            embedder,
            n_split=args.n_split,
            n_docs=1,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
    if "grislain" in names:
        pipelines["grislain"] = GrislainPipeline(
            generator,
            tokenizer,
            embedder,
            max_new_tokens=args.max_new_tokens,
        )
    return pipelines


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    lines = ["# RAG membership inference attack (Anderson et al., 2024)", ""]
    lines.append(f"- mode: `{meta['mode']}`")
    lines.append(
        f"- generator: `{meta['generator']}`  |  embedder: `{meta['embedder']}`  |  device: `{meta['device']}`"
    )
    lines.append(
        f"- members indexed: {meta['members']}  |  non-members held out: {meta['non_members']}  "
        f"|  targets/class: {meta['targets_per_class']}  |  seed: {meta['seed']}"
    )
    lines.append("")
    lines.append("AUC 0.5 = no better than chance; higher = membership leaks. Each row = one (pipeline, variant).")
    lines.append("")
    lines.append(
        "**Caveat (gray_box mode):** `grislain` applies `DPLogitsAggregator` before scoring — "
        "the score reflects the actual DP output distribution. `koga` averages raw voter logits "
        "before DP-Gumbel noise — the score is an adversarial upper bound, not the DP-protected output. "
        "`naive` has no DP and serves as the epsilon=inf baseline."
    )
    lines.append("")
    lines.append("| pipeline | variant | status | AUC | TPR@FPR=0.01 | accuracy | TPR | FPR | secs |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:

        def fmt(key: str) -> str:
            v = r.get(key, float("nan"))
            return "nan" if v != v else f"{v:.3f}"

        lines.append(
            f"| {r['pipeline']} | {r['variant']} | {r['status']} | {fmt('auc')} | "
            f"{fmt('tpr_at_fpr_01')} | {fmt('accuracy')} | {fmt('tpr')} | {fmt('fpr')} | {r['seconds']:.1f} |"
        )
    lines.append("")
    lines.append(
        "Note: smoke-scale (tiny CPU model, few targets) — AUC is noisy. The intended reading is the "
        "*trend*: members should be easier to detect than non-members, and duplication "
        "(`full_duplicate`) should raise AUC relative to `original`."
    )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)

    device = resolve_device(args.device)
    pipeline_names = [n.strip() for n in args.pipelines.split(",") if n.strip()]
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]

    # Member / non-member split from a single HealthCareMagic load.
    loader = HealthCareMagicLoader(n_docs=args.members + args.non_members + 50, seed=args.seed)
    base = loader.load()
    if len(base) < args.members + args.non_members:
        raise SystemExit(f"Loaded only {len(base)} docs; need {args.members + args.non_members}")
    members_base = [truncate_doc(d, args.max_doc_chars) for d in base[: args.members]]
    non_members = [truncate_doc(d, args.max_doc_chars) for d in base[args.members : args.members + args.non_members]]

    rng = random.Random(args.seed)
    member_targets = rng.sample(members_base, min(args.targets_per_class, len(members_base)))
    non_member_targets = rng.sample(non_members, min(args.targets_per_class, len(non_members)))
    targets: list[tuple[Document, bool]] = [(d, True) for d in member_targets] + [
        (d, False) for d in non_member_targets
    ]
    rng.shuffle(targets)
    logger.info("Targets: %d members + %d non-members", len(member_targets), len(non_member_targets))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "attack": "Anderson et al. 2024 (arXiv:2405.20446) MIA",
        "mode": args.mode,
        "generator": args.generator,
        "embedder": args.embedder,
        "device": device,
        "members": args.members,
        "non_members": args.non_members,
        "targets_per_class": args.targets_per_class,
        "seed": args.seed,
    }

    # Load checkpoint: skip (pipeline, variant) cells already completed in a prior run.
    checkpoint_path = args.output_dir / "results.json"
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    done: set[tuple[str, str]] = set()
    if checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            rows = saved.get("results", [])
            detail = saved.get("predictions", {})
            done = {(r["pipeline"], r["variant"]) for r in rows}
            logger.info("Checkpoint: %d cells already done — skipping %s", len(done), sorted(done))
        except Exception:
            logger.warning("Could not read checkpoint; starting fresh")

    def _save() -> None:
        """Atomically write current results so a mid-run kill loses at most one cell."""
        tmp = checkpoint_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"meta": meta, "results": rows, "predictions": detail}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(checkpoint_path)

    pipelines = build_pipelines(pipeline_names, args, device)

    for variant in variant_names:
        indexed_corpus = make_variant(variant, args.seed).apply(members_base)
        logger.info(
            "[%s] indexed corpus: %d documents (from %d members)", variant, len(indexed_corpus), len(members_base)
        )
        for name in pipeline_names:
            if (name, variant) in done:
                logger.info("Skipping %s/%s (already in checkpoint)", name, variant)
                continue
            pipe = pipelines[name]
            logger.info("Attacking %s on %s ...", name, variant)
            start = time.perf_counter()
            record: dict[str, Any] = {"pipeline": name, "variant": variant, "status": "OK", "error": ""}
            try:
                pipe.index(indexed_corpus)
                attack = RAGMembershipInferenceAttack(pipe, sample_chars=args.sample_chars, mode=args.mode)
                result = attack.run(targets)
                record.update(result.metrics)
                detail[f"{name}/{variant}"] = result.predictions
            except Exception as exc:  # noqa: BLE001 - record failures per cell
                record["status"] = "ERROR"
                record["error"] = f"{type(exc).__name__}: {exc}"
                logger.error("FAILED %s/%s: %s\n%s", name, variant, exc, traceback.format_exc())
            record["seconds"] = time.perf_counter() - start
            rows.append(record)
            done.add((name, variant))
            _save()
            logger.info("Checkpoint saved (%d/%d cells done)", len(done), len(pipeline_names) * len(variant_names))

    (args.output_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": rows, "predictions": detail}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    csv_fields = [
        "pipeline",
        "variant",
        "status",
        "auc",
        "tpr_at_fpr_01",
        "accuracy",
        "tpr",
        "fpr",
        "n_members",
        "n_non_members",
        "seconds",
    ]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "report.md").write_text(render_markdown(rows, meta), encoding="utf-8")

    print()
    for r in rows:
        auc = r.get("auc", float("nan"))
        auc_s = "nan" if auc != auc else f"{auc:.3f}"
        print(f"{r['pipeline']:>9} / {r['variant']:<16} {r['status']:<6} AUC={auc_s}  ({r['seconds']:.1f}s)")
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
