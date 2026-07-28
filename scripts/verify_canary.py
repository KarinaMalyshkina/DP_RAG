"""Smoke verification of the canary extraction attack (Secret Sharer; Carlini et al., 2019).

Drives the full path end-to-end on a small CPU model:

    synthetic corpus -> CanaryInjectionVariant -> NaivePipeline.index
      -> CanaryExtractionAttack (exposure + generation modes)

and writes a reproducible artifact (``results.json``) plus a console summary. This
is a *smoke* check (tiny model, few canaries): it proves the attack machinery and
the ``completion_logprob`` scorer are wired correctly, not research-grade numbers.

For the naive (epsilon=infinity) pipeline the injected secret should be fully
exposed (rank 1, exposure == log2(n_candidates)); DP pipelines should push it down.

Usage:
    python scripts/verify_canary.py
    python scripts/verify_canary.py --n-canaries 5 --n-distractors 255 --generator <gpu-model>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

# Prefer the local HF cache; do not hit the network for a smoke check.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from dp_rag_eval.attacks import CanaryExtractionAttack
from dp_rag_eval.core.types import Document
from dp_rag_eval.datasets.variants import CanaryInjectionVariant
from dp_rag_eval.pipelines import Embedder, NaivePipeline, load_generator, resolve_device

logger = logging.getLogger("verify_canary")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("runs/canary_smoke"), help="Where to write results.json")
    p.add_argument("--generator", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF causal-LM id")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--n-base-docs", type=int, default=6, help="Size of the synthetic base corpus")
    p.add_argument("--n-canaries", type=int, default=2, help="Distinct secrets to inject")
    p.add_argument("--n-repeats", type=int, default=1, help="Documents each secret is inserted into (duplication)")
    p.add_argument("--n-distractors", type=int, default=15, help="Random reference codes per canary")
    p.add_argument("--max-new-tokens", type=int, default=16, help="Tokens generated per query (generation mode)")
    p.add_argument("--seed", type=int, default=1, help="Global seed")
    return p.parse_args()


def build_base_corpus(n_docs: int) -> list[Document]:
    """A tiny, dependency-free base corpus of look-alike patient notes."""
    return [
        Document(id=f"d{i}", text=f"Patient {i} came in with a mild cough and was advised rest and fluids.")
        for i in range(n_docs)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    device = resolve_device(args.device)

    variant = CanaryInjectionVariant(n_canaries=args.n_canaries, n_repeats=args.n_repeats, seed=args.seed)
    corpus = variant.apply(build_base_corpus(args.n_base_docs))
    logger.info("Injected canaries: %s", [(c["name"], c["secret_code"]) for c in variant.injected_canaries])

    generator, tokenizer = load_generator(args.generator, device)
    embedder = Embedder(args.embedder, device)
    pipe = NaivePipeline(generator, tokenizer, embedder, max_new_tokens=args.max_new_tokens, device=device)
    pipe.index(corpus)

    exposure_res = CanaryExtractionAttack(
        pipe, variant.injected_canaries, n_distractors=args.n_distractors, mode="exposure", seed=args.seed
    ).run()

    gen_attack = CanaryExtractionAttack(pipe, variant.injected_canaries, mode="generation", seed=args.seed)
    gen_attack.mode = "generation"  # force even though completion_logprob exists
    generation_res = gen_attack.run()

    meta: dict[str, Any] = {
        "attack": "Carlini et al. 2019 (arXiv:1802.08232) Secret Sharer / canary extraction",
        "pipeline": "naive (epsilon=inf reference)",
        "generator": args.generator,
        "embedder": args.embedder,
        "device": device,
        "n_base_docs": args.n_base_docs,
        "n_canaries": args.n_canaries,
        "n_repeats": args.n_repeats,
        "n_distractors": args.n_distractors,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "exposure": {"metrics": exposure_res.metrics, "predictions": exposure_res.predictions},
                "generation": {"metrics": generation_res.metrics, "predictions": generation_res.predictions},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Canary extraction (Secret Sharer) — naive eps=inf reference ===")
    m = exposure_res.metrics
    print(
        f"exposure mode: mean_exposure={m['mean_exposure']:.3f} / max_possible={m['max_possible_exposure']:.3f}  "
        f"extraction_rate={m['extraction_rate']:.3f}  mean_rank={m['mean_rank']:.2f}"
    )
    for p in exposure_res.predictions:
        print(
            f"  {p['name']:<20} secret={p['secret_code']} rank={p['rank']}/{p['n_candidates']} "
            f"exposure={p['exposure']:.3f} extracted={p['extracted']}"
        )
    print(f"generation mode: extraction_rate={generation_res.metrics['extraction_rate']:.3f} (verbatim regurgitation)")
    print(f"\nWrote results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
