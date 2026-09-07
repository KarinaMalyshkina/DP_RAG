#!/usr/bin/env python3
"""Pre-flight for a generator the matrix has not been run with before.

Answers, in one minute on the target box, the four questions that decide whether a
checkpoint can carry this framework's attacks -- and that all fail *silently* if
they are not asked:

1. **Which auto class loaded it, and does it emit vocabulary logits?**
   Multimodal wrappers can hand back hidden states. ``load_generator`` now rejects
   that, so reaching this script at all is already the answer; the width is printed
   so it can be compared against the tokenizer.
2. **What does the chat template put at the end of the prompt?**
   Gemma 4 pre-fills an empty, already-closed thought channel
   (``<|channel>thought\\n<channel|>``) when thinking is disabled. That is the
   difference between "the tags are in the prompt" and "the model generates them".
3. **What is the first token the model actually generates?**
   Every gray-box score in this framework reads it. If it is a control token
   rather than the answer, every AUC in the run is meaningless but well-formed.
4. **Are the Yes/No ids distinct and reachable?**
   The membership estimand is P(the system answers 'Yes').

Usage::

    python scripts/probe_generator.py --generator google/gemma-4-12B-it
    python scripts/probe_generator.py --generator Qwen/Qwen3.5-9B --device cuda

Prints a PASS/FAIL summary and exits non-zero on failure, so it can gate a sweep.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dp_rag_eval.pipelines import load_generator, resolve_device  # noqa: E402
from dp_rag_eval.pipelines.loading import (  # noqa: E402
    chat_prompt,
    supports_thinking,
    yes_no_token_ids,
    yes_probability,
)

QUESTION = (
    "Context: Vienna is the capital of Austria.\n\n"
    "Question: Is Vienna mentioned in the context above? Answer Yes or No."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", required=True, help="HF causal-LM id")
    p.add_argument("--device", default=None, help='"cuda" / "cpu"; auto-detected when omitted')
    p.add_argument("--dtype", default=None, help="Override the per-device default dtype")
    p.add_argument("--n-tokens", type=int, default=5, help="How many tokens to generate for inspection")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    device = resolve_device(args.device)

    model, tokenizer = load_generator(args.generator, device, dtype=args.dtype)
    failures: list[str] = []

    print("\n" + "=" * 72)
    print(f"generator      : {args.generator}")
    print(f"loaded class   : {type(model).__name__}")
    print(f"dtype / device : {model.dtype} on {device}")
    print(f"thinking model : {supports_thinking(tokenizer)} (chat_prompt disables it when True)")
    print(f"tokenizer size : {len(tokenizer)}")

    # --- 2. what the template appends ------------------------------------------------
    messages = [{"role": "user", "content": QUESTION}]
    rendered = chat_prompt(tokenizer, messages, tokenize=False, add_generation_prompt=True)
    tail = rendered[-160:]
    print("\n--- rendered prompt, last 160 chars " + "-" * 36)
    print(repr(tail))

    # --- 1 + 3. logits and the first generated tokens --------------------------------
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1, :].float()
    print(f"\nlogit width    : {int(logits.shape[-1])}")

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_ids = generated[0, input_ids.shape[1] :].tolist()
    print("\n--- first generated tokens " + "-" * 45)
    for position, token_id in enumerate(new_ids):
        print(f"  [{position}] id={token_id:<8} {tokenizer.decode([token_id])!r}")

    first = tokenizer.decode([new_ids[0]]).strip() if new_ids else ""
    if not first:
        failures.append("the first generated token is whitespace or empty")
    elif not any(first.lower().startswith(w) for w in ("yes", "no")):
        failures.append(
            f"the first generated token is {first!r}, not an answer -- a control or reasoning "
            "token here invalidates every gray-box score"
        )

    # --- 4. Yes/No reachability ------------------------------------------------------
    yes_ids, no_ids = yes_no_token_ids(tokenizer)
    print(f"\nYes ids        : {yes_ids}")
    print(f"No ids         : {no_ids}")
    if not yes_ids or not no_ids:
        failures.append("Yes or No has no first-token id in this tokenizer")
    elif set(yes_ids) & set(no_ids):
        failures.append("Yes and No share a first-token id; the two classes are not separable")
    else:
        score = yes_probability(logits, yes_ids, no_ids)
        print(f"P(Yes) on a question whose answer is Yes: {score:.4f}")
        if not 0.0 <= score <= 1.0:
            failures.append(f"P(Yes) = {score} is outside [0, 1]")
        elif score < 0.5:
            failures.append(
                f"P(Yes) = {score:.4f} on a question whose answer is plainly Yes -- the prompt or "
                "the first-token position is wrong for this model"
            )

    print("\n" + "=" * 72)
    if failures:
        print("FAIL")
        for line in failures:
            print("  - " + line)
        return 1
    print("PASS -- this generator can carry the attacks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
