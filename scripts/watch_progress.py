"""Show how far a running sweep has got, without attaching to its tmux session.

Reads each run directory's ``results.json`` — rewritten atomically after every
finished cell — so it is safe to run against a live sweep, and safe to run from a
second shell or straight over SSH. Nothing here touches the sweep.

What it answers: is it still moving, how much is left, how long that will take, and
is anything going wrong that is not visible as a crash. The last one matters most:
a Koga cell whose Limited-Domain selection never fires reports ``status=OK`` and an
AUC of 0.5, which reads as a successful defence rather than a configuration that
measured nothing. That shows up here as a flag on ``ld_failure_rate``.

Usage:
    # one look
    python scripts/watch_progress.py runs/mia_main

    # several sweeps at once, refreshing
    python scripts/watch_progress.py runs/mia_main runs/canary_main --interval 60

    # from your laptop, without opening tmux
    ssh pod 'cd /workspace/DPRAG && uv run python scripts/watch_progress.py runs/mia_main'
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from dp_rag_eval.evaluation.progress import read_progress, render_progress


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path, help="Sweep output directories to watch")
    p.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds between refreshes. 0 (default) prints once and exits — which is what you want "
        "over SSH or in a cron; use a value like 60 for a live pane.",
    )
    p.add_argument("--recent", type=int, default=5, help="How many recently finished cells to show")
    p.add_argument("--no-clear", action="store_true", help="Do not clear the screen between refreshes")
    return p.parse_args()


def render_all(run_dirs: list[Path], recent: int) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blocks = [render_progress(read_progress(d, recent=recent)) for d in run_dirs]
    return f"dp-rag-eval progress @ {stamp}\n\n" + "\n\n".join(blocks)


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        print(render_all(args.run_dirs, args.recent))
        return

    while True:
        if not args.no_clear:
            print("\033[2J\033[H", end="")
        print(render_all(args.run_dirs, args.recent), flush=True)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:  # Ctrl-C here must not look like a failed sweep
            print()
            return


if __name__ == "__main__":
    main()
