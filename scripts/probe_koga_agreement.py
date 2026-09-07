#!/usr/bin/env python3
"""Measure how much Koga's voters agree, and derive the budget at which it can answer.

Koga's Limited-Domain selection emits a token only when some candidate holds a vote
lead of roughly ``1 + log(2/delta_step)/eps_step``. Two things decide whether that
happens, and they are independent:

* **how many voters agree** — a property of the corpus, the question, the model and
  the voter count, and *not* of epsilon: votes are argmax next tokens, and no noise
  is added to them;
* **how many are required** — a property of epsilon, delta and the answer length,
  and not of the data.

So the agreement side can be measured once, with the mechanism switched off, and the
requirement side evaluated afterwards for any budget. One probe yields the whole
epsilon axis instead of one point per GPU-hour.

**At the first step this is exact.** Nothing has been sampled yet, so the histogram
is the same one a run at any finite budget would see, and "which budgets let Koga
emit at least one token" is answered without approximation. From the second step the
DP run may select a different token and the trajectories diverge, so the projection
for a *complete* answer is an estimate and the full curve still has to be measured
directly.

Output (under --output-dir): ``agreement.jsonl`` with one row per step per question,
``projection.csv`` with P(first token) against epsilon for every voter count probed,
and ``report.md``.

Example::

    uv run python scripts/probe_koga_agreement.py \\
      --generator Qwen/Qwen3.5-9B --output-dir runs/agreement_probe \\
      --n-split 105,200 --n-queries 100 --n-tokens 32 --corpus-size 1000 --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from dp_rag_eval.core.types import PrivacyTarget
from dp_rag_eval.datasets.loaders import HealthCareMagicLoader
from dp_rag_eval.pipelines import Embedder, KogaPipeline, load_generator, resolve_device
from dp_rag_eval.privacy.calibration import (
    _koga_split_deltas,
    _solve_koga_step_epsilon,
    ld_vote_threshold,
)
from dp_rag_eval.runner.sweep import parse_floats, parse_ints, truncate_doc

logger = logging.getLogger("probe_koga_agreement")

# Budgets the projection is evaluated at. Deliberately reaches far past anything
# defensible as a privacy guarantee: the point is to locate where the mechanism
# starts working, and reporting that it only does so at eps=100 is the finding.
DEFAULT_PROJECTION_EPS = "4,5,8,16,32,50,100,250,500"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("runs/agreement_probe"))
    p.add_argument("--generator", default="Qwen/Qwen3.5-9B", help="HF causal-LM id")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--dtype", default=None, help="bfloat16 / float16 / float32")

    p.add_argument("--n-split", default="105,200", help="Comma-separated voter counts to probe")
    p.add_argument("--n-docs", type=int, default=1, help="Documents per voter")
    p.add_argument("--n-queries", type=int, default=100, help="Questions per voter count")
    p.add_argument("--n-tokens", type=int, default=32, help="Answer length the guarantee would cover")
    p.add_argument("--corpus-size", type=int, default=1000, help="Documents indexed")
    p.add_argument("--max-doc-chars", type=int, default=300, help="Truncate documents")
    p.add_argument("--seeds", default="42", help="Comma-separated seeds")
    p.add_argument("--delta", type=float, default=1e-6, help="Target delta the projection assumes")
    p.add_argument(
        "--projection-epsilons",
        default=DEFAULT_PROJECTION_EPS,
        help="Budgets the measured histograms are projected onto. No GPU cost — these are evaluated "
        "from the recorded votes, not run.",
    )
    return p.parse_args()


def vote_threshold_for(epsilon: float, delta: float, n_tokens: int) -> tuple[float, float]:
    """``(eps_step, required vote lead)`` for one total budget, via Koga's own accountant."""
    eps_step, _ = _solve_koga_step_epsilon(epsilon, delta, n_tokens, 0.5)
    delta_step, _ = _koga_split_deltas(PrivacyTarget(epsilon=epsilon, delta=delta, n_tokens=n_tokens), 0.5)
    return eps_step, ld_vote_threshold(eps_step, delta_step)


def success_probability(max_votes: int, eps_step: float, threshold: float) -> float:
    """P(the selection emits a token) given the leading vote count.

    The mechanism compares ``max_i(count_i + Gumbel_i)`` against
    ``threshold + Gumbel``. Taking the leader as the only serious contender, the
    comparison is between two Gumbel variates of scale ``1/eps_step``, whose
    difference is logistic — so the transition is not a step but a band roughly
    ``1/eps_step`` votes wide. At eps=5 and a 32-token answer that band is about 5
    votes, which is why a voter count sitting exactly on the threshold gives a coin
    flip rather than a guarantee.
    """
    return 1.0 / (1.0 + math.exp(-eps_step * (max_votes - threshold)))


@torch.no_grad()
def probe_one(pipeline: KogaPipeline, questions: list[str]) -> list[dict[str, Any]]:
    """Run plurality voting (no DP) over ``questions`` and return the per-step trace."""
    rows: list[dict[str, Any]] = []
    for i, question in enumerate(questions):
        # Cleared per question so vote_trace holds exactly this answer's steps.
        pipeline.reset_ld_stats()
        pipeline.answer(question, seed=42 + i)
        rows.extend({"question_index": i, **record} for record in pipeline.vote_trace)
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d questions", i + 1, len(questions))
    return rows


def build_report(
    traces: dict[int, list[dict[str, Any]]],
    projection: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    lines = ["# Koga voter agreement probe", ""]
    lines.append(f"- generator: `{meta['generator']}`  |  device: `{meta['device']}`")
    lines.append(
        f"- voter counts: {meta['n_split']}  |  questions: {meta['n_queries']}  "
        f"|  answer length: {meta['n_tokens']} tokens  |  delta: {meta['delta']}"
    )
    lines.append("")
    lines.append(
        "Votes are argmax next tokens and carry no noise, so these histograms are the "
        "ones a run at **any** budget would see. The projection below is therefore exact "
        "for the first token and an estimate for a complete answer."
    )
    lines.append("")

    lines.append("## Agreement by step")
    lines.append("")
    lines.append("| voters | step | mean leading votes | mean share | min | max |")
    lines.append("|" + "---|" * 6)
    for n_split, rows in sorted(traces.items()):
        by_step: dict[int, list[int]] = {}
        for row in rows:
            by_step.setdefault(int(row["step"]), []).append(int(row["max_votes"]))
        for step in sorted(by_step):
            if step > 0 and step % 4 and step != max(by_step):
                continue  # every 4th step plus the last one: the full curve is in the jsonl
            votes = by_step[step]
            mean = sum(votes) / len(votes)
            lines.append(f"| {n_split} | {step} | {mean:.1f} | {mean / n_split:.2f} | {min(votes)} | {max(votes)} |")
    lines.append("")

    lines.append("## Projected P(first token) against budget")
    lines.append("")
    lines.append("**Exact**, not extrapolated: the first step precedes any sampling.")
    lines.append("")
    epsilons = sorted({row["epsilon"] for row in projection})
    lines.append("| voters | " + " | ".join(f"eps={e:g}" for e in epsilons) + " |")
    lines.append("|" + "---|" * (len(epsilons) + 1))
    for n_split in sorted(traces):
        cells = []
        for e in epsilons:
            match = [r for r in projection if r["n_split"] == n_split and r["epsilon"] == e]
            cells.append(f"{match[0]['p_first_token']:.2f}" if match else "-")
        lines.append(f"| {n_split} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| eps | required vote lead | eps_step |")
    lines.append("|---|---|---|")
    seen: set[float] = set()
    for row in projection:
        if row["epsilon"] not in seen:
            seen.add(row["epsilon"])
            lines.append(f"| {row['epsilon']:g} | {row['threshold']:.1f} | {row['eps_step']:.4f} |")
    lines.append("")
    lines.append(
        "Read the first table where P crosses ~0.5 and put the answer-rate sweep's "
        "epsilon grid around it. A grid chosen without this measurement risks spending "
        "every cell either wholly empty or wholly saturated."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    n_splits = parse_ints(args.n_split)
    seeds = parse_ints(args.seeds)
    projection_eps = parse_floats(args.projection_epsilons)

    meta = {
        "probe": "Koga voter agreement (plurality vote, mechanism off)",
        "generator": args.generator,
        "embedder": args.embedder,
        "device": device,
        "n_split": n_splits,
        "n_docs": args.n_docs,
        "n_queries": args.n_queries,
        "n_tokens": args.n_tokens,
        "corpus_size": args.corpus_size,
        "delta": args.delta,
        "seeds": seeds,
    }

    generator, tokenizer = load_generator(args.generator, device, dtype=args.dtype)
    embedder = Embedder(args.embedder, device)

    seed = seeds[0]
    random.seed(seed)
    torch.manual_seed(seed)
    loader = HealthCareMagicLoader(n_docs=args.corpus_size, seed=seed)
    base = [truncate_doc(d, args.max_doc_chars) for d in loader.load()]
    answerable = [d for d in base if d.metadata.get("query")]
    if not answerable:
        raise SystemExit("No documents carry metadata['query'] — cannot build the probe task")
    sampled = random.Random(seed).sample(answerable, min(args.n_queries, len(answerable)))
    questions = [str(d.metadata["query"]) for d in sampled]
    logger.info("%d questions over %d documents", len(questions), len(base))

    traces: dict[int, list[dict[str, Any]]] = {}
    trace_path = args.output_dir / "agreement.jsonl"
    with trace_path.open("w", encoding="utf-8") as fh:
        for n_split in n_splits:
            logger.info("Probing %d voters ...", n_split)
            start = time.perf_counter()
            pipeline = KogaPipeline(
                generator,
                tokenizer,
                embedder,
                n_split=n_split,
                n_docs=args.n_docs,
                max_new_tokens=args.n_tokens,
                dp=False,
                device=device,
            )
            pipeline.index(base)
            rows = probe_one(pipeline, questions)
            traces[n_split] = rows
            for row in rows:
                fh.write(json.dumps({"n_split": n_split, **row}, ensure_ascii=False) + "\n")
            logger.info("  %d voters: %d step records in %.0f s", n_split, len(rows), time.perf_counter() - start)
            del pipeline

    projection: list[dict[str, Any]] = []
    for n_split, rows in traces.items():
        first_step = [int(r["max_votes"]) for r in rows if int(r["step"]) == 0]
        for epsilon in projection_eps:
            eps_step, threshold = vote_threshold_for(epsilon, args.delta, args.n_tokens)
            probabilities = [success_probability(v, eps_step, threshold) for v in first_step]
            projection.append(
                {
                    "n_split": n_split,
                    "epsilon": epsilon,
                    "eps_step": eps_step,
                    "threshold": threshold,
                    "mean_first_step_votes": sum(first_step) / len(first_step) if first_step else 0.0,
                    "p_first_token": sum(probabilities) / len(probabilities) if probabilities else 0.0,
                }
            )

    with (args.output_dir / "projection.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(projection[0]))
        writer.writeheader()
        writer.writerows(projection)

    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = build_report(traces, projection, meta)
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
