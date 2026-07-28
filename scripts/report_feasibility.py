"""Report which privacy targets each mechanism can actually deliver.

The reachable epsilon range is a property of a mechanism, not of a run, and it is
worth knowing *before* renting a GPU: at realistic answer lengths, Koga's
LD-Gumbel selection needs more agreeing voters than a practical configuration has,
and cells below that point cannot produce a meaningful result at any cost.

Emits a markdown table so the numbers in ``docs/METHODOLOGY.md`` stay computed
rather than typed. Regenerate it after any change to calibration.

Usage:
    python scripts/report_feasibility.py
    python scripts/report_feasibility.py --delta 1e-6 --epsilons 0.5,1,2,4,8 \
        --n-tokens 1,4,8,16,32 --n-splits 50,100,200
"""

from __future__ import annotations

import argparse
import logging

from dp_rag_eval.core.types import PrivacyTarget
from dp_rag_eval.privacy import calibrate_grislain, calibrate_koga, min_voters_for
from dp_rag_eval.runner.sweep import parse_floats, parse_ints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epsilons", default="0.5,1,2,4,8,16,32,64", help="Target total budgets")
    p.add_argument("--delta", type=float, default=1e-6, help="Target delta")
    p.add_argument("--n-tokens", default="1,4,8,16,32", help="Answer lengths to tabulate")
    p.add_argument("--n-splits", default="50,100,200", help="Voter counts to check feasibility at")
    p.add_argument("--corpus-size", type=int, default=10000, help="Largest corpus, for the delta sanity check")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    args = parse_args()
    epsilons = parse_floats(args.epsilons)
    token_counts = parse_ints(args.n_tokens)
    voter_counts = parse_ints(args.n_splits)

    print(f"# Mechanism feasibility at delta={args.delta:g}\n")
    limit = 1.0 / max(args.corpus_size, 1)
    verdict = "ok" if args.delta < limit / 10 else "TOO LARGE"
    print(f"delta must be well below 1/n = {limit:g} for the largest corpus ({args.corpus_size} docs): {verdict}\n")

    print("## Koga: votes the LD-Gumbel selection needs\n")
    print("A vote histogram over `n_split` voters cannot exceed `n_split`, so a")
    print("requirement above the voter count means the mechanism can never select a")
    print("token — every query fails and AUC lands on 0.5 for reasons that have")
    print("nothing to do with privacy.\n")
    header = "| eps_total | " + " | ".join(f"n_tokens={t}" for t in token_counts) + " |"
    print(header)
    print("|" + "---|" * (len(token_counts) + 1))
    for epsilon in epsilons:
        cells = []
        for n_tokens in token_counts:
            needed = min_voters_for(PrivacyTarget(epsilon, args.delta, n_tokens))
            cells.append(str(needed))
        print(f"| {epsilon:g} | " + " | ".join(cells) + " |")

    for n_tokens in token_counts:
        print(f"\n## Reachable at n_tokens={n_tokens}\n")
        print("| eps_total | " + " | ".join(f"koga n_split={n}" for n in voter_counts) + " | grislain |")
        print("|" + "---|" * (len(voter_counts) + 2))
        for epsilon in epsilons:
            target = PrivacyTarget(epsilon, args.delta, n_tokens)
            cells = []
            for n_split in voter_counts:
                cells.append("ok" if calibrate_koga(target, n_split=n_split).feasible else "no")
            grislain = "ok" if calibrate_grislain(target).feasible else "no"
            print(f"| {epsilon:g} | " + " | ".join(cells) + f" | {grislain} |")

    print("\n## Calibration accuracy\n")
    print("| eps_total | n_tokens | koga realized | grislain realized |")
    print("|---|---|---|---|")
    for n_tokens in token_counts:
        for epsilon in epsilons:
            target = PrivacyTarget(epsilon, args.delta, n_tokens)
            koga = calibrate_koga(target, n_split=10_000).realized.epsilon
            gris = calibrate_grislain(target).realized.epsilon
            print(f"| {epsilon:g} | {n_tokens} | {koga:.4f} | {gris:.4f} |")


if __name__ == "__main__":
    main()
