"""Check that the GPU speed/memory optimisations did not change the numbers.

Three of the changes made for the 9B run are the kind that fail silently — they
produce plausible values that are subtly wrong rather than raising:

  1. Grislain scores a whole candidate set against one shared prompt prefix, with
     the prefix KV cache rewound between candidates. A bad rewind leaks one
     candidate's tokens into the next, and the only symptom is drifting scores.
  2. Koga forwards all voters as one left-padded batch instead of one at a time.
     Wrong padding alignment shifts which position the logits come from, which
     changes every vote without ever erroring.
  3. Thinking mode is disabled through the chat template. If the flag fails to
     reach the template, the first generated token is `<think>` and every
     gray-box score measures the model's urge to reason instead of membership.

Run it once locally against the smoke model, and once on the GPU box against the
real generator before starting the matrix.

Usage:
    python scripts/verify_optimizations.py
    python scripts/verify_optimizations.py --generator Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

import torch

from dp_rag_eval.core.types import Document
from dp_rag_eval.pipelines import Embedder, GrislainPipeline, KogaPipeline, load_generator, resolve_device
from dp_rag_eval.pipelines.loading import chat_prompt, forward_logits, supports_thinking

logger = logging.getLogger("verify_optimizations")

CORPUS = [
    Document(
        id=f"doc-{i:02d}",
        text=(
            f"Patient {i} reports a persistent cough lasting {i + 2} weeks. "
            f"The doctor recommends rest, fluids and a follow-up in {i + 1} days. "
            f"No fever was recorded during examination number {i}."
        ),
        metadata={"source": "synthetic"},
    )
    for i in range(12)
]
QUERY = "What did the doctor recommend for the persistent cough?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF causal-LM id")
    p.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id")
    p.add_argument("--device", default=None, help="cpu / cuda (auto-detected if unset)")
    p.add_argument("--dtype", default=None, help="Override the compute dtype")
    p.add_argument("--n-split", type=int, default=6, help="Koga voters used in the batching check")
    return p.parse_args()


def check_prefix_cache_reuse(pipe: GrislainPipeline) -> bool:
    """Scoring the same candidate twice in one batch must give the same number.

    The candidate list is ``[a, b, a]``: positions 0 and 2 are identical strings
    scored before and after another candidate has passed through the shared cache.
    If the rewind is wrong, the second occurrence is conditioned on leftover tokens
    and the scores diverge.
    """
    random.seed(0)
    torch.manual_seed(0)
    scores = pipe.completion_logprobs(QUERY, ["rest and fluids", "a chest x-ray", "rest and fluids"])
    first, other, repeat = scores
    delta = abs(first - repeat)
    ok = delta < 1e-3
    print(f"  candidate A (1st occurrence): {first:.6f}")
    print(f"  candidate B                 : {other:.6f}")
    print(f"  candidate A (2nd occurrence): {repeat:.6f}")
    print(f"  |A1 - A2| = {delta:.3e}  -> {'OK' if ok else 'MISMATCH: prefix cache is leaking state'}")
    return ok


def check_voter_batching(pipe: KogaPipeline) -> bool:
    """Batched voter logits must pick the same tokens as one-prompt-at-a-time forwards.

    Exact logits differ slightly (padded attention is not bit-identical), so the
    check is on the argmax — that is the only thing the vote consumes.
    """
    retrieved = pipe._retrieve(QUERY, pipe.n_split * pipe.n_docs)
    prompts = pipe._build_voter_prompts(retrieved, QUERY)

    batched, _, _ = pipe._voter_prefill(prompts)
    batched_tokens = torch.argmax(batched, dim=-1).cpu().tolist()

    individual: list[int] = []
    for prompt in prompts:
        input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(pipe.device)
        _, kept = forward_logits(pipe.model, keep=1, input_ids=input_ids)
        individual.append(int(torch.argmax(kept[0, -1])))

    agree = sum(1 for a, b in zip(batched_tokens, individual) if a == b)
    ok = agree == len(individual)
    print(f"  voters: {len(individual)}   agreeing argmax tokens: {agree}/{len(individual)}")
    if not ok:
        print(f"  batched   : {batched_tokens}")
        print(f"  individual: {individual}")
        print("  -> MISMATCH: left-padding alignment in _voter_prefill is wrong")
    else:
        print("  -> OK")
    return ok


def check_thinking_disabled(tokenizer, model, device: str) -> bool:
    """The first generated token must be answer content, not a reasoning marker."""
    messages = [{"role": "user", "content": 'Does this: "a persistent cough" appear in the context? Answer Yes or No.'}]
    prompt = chat_prompt(tokenizer, messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    _, kept = forward_logits(model, keep=1, input_ids=input_ids)
    token_id = int(torch.argmax(kept[0, -1]))
    decoded = tokenizer.decode([token_id])
    thinking_model = supports_thinking(tokenizer)
    ok = "think" not in decoded.lower()
    print(f"  template supports enable_thinking: {thinking_model}")
    print(f"  first token: {token_id} -> {decoded!r}")
    print(f"  -> {'OK' if ok else 'MISMATCH: model still opens with a thinking marker'}")
    if not thinking_model:
        print("  (note: this model has no thinking mode, so the check is vacuous here —")
        print("   re-run it against the real generator on the GPU box)")
    return ok


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    device = resolve_device(args.device)

    print(f"generator: {args.generator}   device: {device}")
    generator, tokenizer = load_generator(args.generator, device, dtype=args.dtype)
    embedder = Embedder(args.embedder, device)
    print(f"dtype: {generator.dtype}   vocab: {generator.config.vocab_size}")
    print()

    results: dict[str, bool] = {}

    print("[1/3] Grislain prefix-cache reuse across candidates")
    grislain = GrislainPipeline(generator, tokenizer, embedder, max_new_tokens=8, top_k=3, dp=True)
    grislain.index(CORPUS)
    results["grislain prefix cache"] = check_prefix_cache_reuse(grislain)
    print()

    print("[2/3] Koga batched voter forward vs per-voter forward")
    koga = KogaPipeline(generator, tokenizer, embedder, n_split=args.n_split, max_new_tokens=4, device=device)
    koga.index(CORPUS)
    results["koga voter batching"] = check_voter_batching(koga)
    print()

    print("[3/3] Thinking mode disabled")
    results["thinking disabled"] = check_thinking_disabled(tokenizer, generator, device)
    print()

    failed = [name for name, ok in results.items() if not ok]
    for name, ok in results.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        print(f"\n{len(failed)} check(s) failed — do not start the matrix.")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
