"""Answer Telegram commands about a running sweep, and shout if it stalls.

Runs in its own tmux pane next to the sweep and long-polls Telegram, so you can ask
for progress or pull the current results from your phone without opening SSH:

    /status         progress, ETA and flags for every watched run
    /flags          only the problems — errors and dead Limited-Domain cells
    /report [run]   report.md as a file, current as of the last finished cell
    /csv [run]      results.csv as a file
    /degradation    recompute duplicated-vs-control deltas now and send the report
    /runs           which directories are being watched
    /help

Every sweep rewrites ``results.json``, ``results.csv`` and ``report.md`` after each
cell, so these always reflect finished work — there is no "wait for it to complete"
state where the files are stale or absent.

**The stall watchdog** alerts when no new cell has been recorded for
``--stall-minutes``. That covers the failure modes that produce no exit code: a
wedged CUDA call, a killed process, a sweep quietly making no progress. It does
**not** cover the pod dying, because this daemon dies with it — nothing running on
the machine can report the machine's own death. A run that has reached its planned
cell count is treated as complete and never alerts.

**Only ``TELEGRAM_CHAT_ID`` is answered.** A bot username is discoverable, so
anything from another chat is logged and ignored; file access is restricted to the
run directories named on the command line, and never accepts an arbitrary path.

Usage:
    # in its own tmux pane
    uv run python scripts/telegram_bot.py runs/mia_main runs/canary_main

    # window is per-run: group 3 cells are slow, canary cells are not
    uv run python scripts/telegram_bot.py runs/mia_main --stall-minutes 300
"""

from __future__ import annotations

import argparse
import logging
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dp_rag_eval.evaluation.progress import read_progress, render_progress
from dp_rag_eval.evaluation.reporter import (
    ANCHOR_VARIANT,
    detect_metrics,
    load_rows,
    pair_against_anchor,
    pool_over_seeds,
    render_markdown,
)
from dp_rag_eval.notifications import TelegramCredentials, get_updates, send_document, send_message

logger = logging.getLogger("telegram_bot")

HELP = """dp-rag-eval bot

/status         progress, ETA and flags for every watched run
/flags          only the problems
/report [run]   report.md as a file
/csv [run]      results.csv as a file
/degradation [run]  recompute duplicated-vs-control deltas and send them
/runs           which directories are watched
/help           this message

[run] is optional when only one directory is watched."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path, help="Sweep output directories to watch")
    p.add_argument(
        "--stall-minutes",
        type=float,
        default=60.0,
        help="Alert when no new cell has been recorded for this long. A cell is written only "
        "when it finishes, so this must EXCEED the slowest cell or it fires on every one of "
        "them. Measured on an A6000: Grislain ~4 min, Koga at eps=4 with 128 voters ~3.5 h. "
        "0 disables the watchdog.",
    )
    p.add_argument("--poll-seconds", type=int, default=30, help="Telegram long-poll window")
    p.add_argument("--quiet-start", action="store_true", help="Do not announce startup")
    return p.parse_args()


class Bot:
    """Command handling and stall detection over a fixed set of run directories."""

    def __init__(self, creds: TelegramCredentials, run_dirs: list[Path], stall_minutes: float) -> None:
        self.creds = creds
        self.run_dirs = run_dirs
        self.stall_seconds = stall_minutes * 60.0
        self.offset = 0
        # One alert per stall episode, cleared when the run moves again — a watchdog
        # that repeats every 30 seconds is a watchdog you learn to ignore.
        self.alerted: set[Path] = set()

    # -- helpers ----------------------------------------------------------

    def resolve(self, argument: str | None) -> Path | None:
        """Map a user-supplied run name to a watched directory, or ``None``.

        Matching is by name against the whitelist rather than by path, so no message
        can make the bot read a file outside the directories it was started with.
        """
        if argument is None:
            return self.run_dirs[0] if len(self.run_dirs) == 1 else None
        wanted = argument.strip().strip("/").replace("\\", "/")
        for run_dir in self.run_dirs:
            candidate = str(run_dir).replace("\\", "/")
            if wanted in (candidate, run_dir.name):
                return run_dir
        return None

    def say(self, text: str) -> None:
        send_message(self.creds, text)

    def _need_run(self, argument: str | None) -> Path | None:
        run_dir = self.resolve(argument)
        if run_dir is None:
            names = ", ".join(d.name for d in self.run_dirs)
            self.say(f"Which run? Watching: {names}")
        return run_dir

    def _send_file(self, run_dir: Path, name: str) -> None:
        path = run_dir / name
        if not path.is_file():
            self.say(f"{run_dir.name}/{name} does not exist yet — no cell has finished.")
            return
        stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
        send_document(self.creds, path, caption=f"{run_dir.name}/{name} (as of {stamp})")

    # -- commands ---------------------------------------------------------

    def cmd_status(self) -> None:
        self.say("\n\n".join(render_progress(read_progress(d)) for d in self.run_dirs))

    def cmd_flags(self) -> None:
        lines: list[str] = []
        for run_dir in self.run_dirs:
            progress = read_progress(run_dir)
            if progress.error:
                lines.append(f"{run_dir.name}: {progress.error}")
            elif progress.flags:
                lines.append(f"{run_dir.name}:\n" + "\n".join(f"  ! {f}" for f in progress.flags))
            else:
                lines.append(f"{run_dir.name}: no flags ({progress.n_done} cells)")
        self.say("\n\n".join(lines))

    def cmd_degradation(self, argument: str | None) -> None:
        """Recompute the duplicated-vs-control comparison from current results."""
        run_dir = self._need_run(argument)
        if run_dir is None:
            return
        rows = load_rows([run_dir])
        if not rows:
            self.say(f"{run_dir.name}: no results yet.")
            return
        metrics = detect_metrics(rows)
        if not metrics:
            self.say(f"{run_dir.name}: no comparable metric columns yet.")
            return
        if not any(str(r.get("variant", "")) == ANCHOR_VARIANT for r in rows):
            self.say(
                f"{run_dir.name}: no '{ANCHOR_VARIANT}' control rows yet, so there is nothing "
                "to compare duplicated cells against."
            )
            return

        pairs, warnings = [], []
        for metric in metrics:
            metric_pairs, metric_warnings = pair_against_anchor(rows, metric)
            pairs += metric_pairs
            warnings += [f"[{metric}] {w}" for w in metric_warnings]
        if not pairs:
            self.say(f"{run_dir.name}: control rows exist but no duplicated cell matches one yet.")
            return

        meta = {"sources": [str(run_dir)], "metrics": metrics, "anchor": ANCHOR_VARIANT, "n_rows": len(rows)}
        report = render_markdown(pairs, pool_over_seeds(pairs), warnings, meta)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"degradation_{run_dir.name}.md"
            path.write_text(report, encoding="utf-8")
            send_document(self.creds, path, caption=f"{run_dir.name}: {len(pairs)} paired cells")

    def handle(self, text: str) -> None:
        parts = text.strip().split()
        if not parts:
            return
        # Group chats deliver commands as /status@botname.
        command = parts[0].lower().lstrip("/").split("@")[0]
        argument = parts[1] if len(parts) > 1 else None

        if command in {"start", "help"}:
            self.say(HELP)
        elif command == "runs":
            self.say("Watching:\n" + "\n".join(f"  {d}" for d in self.run_dirs))
        elif command == "status":
            self.cmd_status()
        elif command == "flags":
            self.cmd_flags()
        elif command == "report":
            run_dir = self._need_run(argument)
            if run_dir:
                self._send_file(run_dir, "report.md")
        elif command == "csv":
            run_dir = self._need_run(argument)
            if run_dir:
                self._send_file(run_dir, "results.csv")
        elif command == "degradation":
            self.cmd_degradation(argument)
        else:
            self.say(f"Unknown command: {command}\n\n{HELP}")

    # -- watchdog ---------------------------------------------------------

    def check_stalls(self) -> None:
        if self.stall_seconds <= 0:
            return
        now = datetime.now(tz=timezone.utc)
        for run_dir in self.run_dirs:
            progress = read_progress(run_dir)
            # A finished run stops writing forever; that is not a stall.
            complete = progress.planned is not None and progress.n_done >= progress.planned
            if progress.error or progress.updated is None or complete:
                self.alerted.discard(run_dir)
                continue

            age = (now - progress.updated).total_seconds()
            if age > self.stall_seconds:
                if run_dir not in self.alerted:
                    self.alerted.add(run_dir)
                    self.say(
                        f"STALLED: {run_dir.name} has recorded no new cell for "
                        f"{age / 60:.0f} minutes.\n\n{render_progress(progress)}"
                    )
            elif run_dir in self.alerted:
                self.alerted.discard(run_dir)
                self.say(f"{run_dir.name} is moving again ({progress.n_done} cells).")

    # -- main loop --------------------------------------------------------

    def run(self, poll_seconds: int) -> None:
        while True:
            for update in get_updates(self.creds, self.offset, timeout=poll_seconds):
                self.offset = max(self.offset, int(update.get("update_id", 0)) + 1)
                message = update.get("message") or {}
                chat_id = str((message.get("chat") or {}).get("id", ""))
                text = message.get("text")
                if not text:
                    continue
                if chat_id != self.creds.chat_id:
                    logger.warning("Ignoring message from unauthorised chat %s: %r", chat_id, text[:80])
                    continue
                logger.info("Command: %s", text.strip())
                try:
                    self.handle(text)
                except Exception:  # noqa: BLE001 - one bad command must not kill the watchdog
                    logger.exception("Command failed: %r", text)
                    self.say(f"That command failed: {text.strip()} (see the bot log)")
            self.check_stalls()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    creds = TelegramCredentials.from_env()
    if creds is None:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are not set. Unlike notify.py, the bot "
            "cannot do anything useful without them, so this is fatal rather than a warning."
        )

    bot = Bot(creds, args.run_dirs, args.stall_minutes)
    logger.info("Watching %s (stall alert after %g min)", [str(d) for d in args.run_dirs], args.stall_minutes)
    if not args.quiet_start:
        bot.say(f"Bot up on {socket.gethostname()}, watching {', '.join(d.name for d in args.run_dirs)}.\n\n{HELP}")

    while True:
        try:
            bot.run(args.poll_seconds)
        except KeyboardInterrupt:
            logger.info("Stopped")
            return
        except Exception:  # noqa: BLE001 - the daemon outlives transient network failures
            logger.exception("Poll loop crashed; retrying in 30s")
            time.sleep(30)


if __name__ == "__main__":
    main()
