"""Send a Telegram message — and optionally the result files — when a sweep stops.

A sweep runs for days on a rented box. Knowing *that* it stopped, and whether it
stopped because it finished or because something broke, is worth more than any live
dashboard: it is the difference between paying for idle GPU hours and not.

Credentials come from the environment (put them in ``.env.gpu``):

    export TELEGRAM_BOT_TOKEN=123456:ABC...      # from @BotFather
    export TELEGRAM_CHAT_ID=987654321            # from @userinfobot

**This never fails a sweep.** Missing credentials, no network, a Telegram outage —
all are logged and exit 0. A notifier that can take down a three-day run by raising
is worse than no notifier.

**What it does not do:** listen. This is one-shot and outbound only — it runs when
you run it. To ask for results from your phone (``/status``, ``/report``), run
``scripts/telegram_bot.py``, which polls for commands.

**It also cannot report a dead pod.** Chaining this after a sweep relies on the
shell surviving to run it. If the machine is destroyed or the process is SIGKILLed
along with its shell, nothing is sent — the silence is the only signal. The bot's
stall watchdog covers a hung or killed *sweep*, but nothing on the pod can report
the pod itself dying.

Usage:
    # chained after a sweep: status, summary, and the report as a file
    uv run python scripts/run_attack.py ... ; \
      uv run python scripts/notify.py --status $? --label "group 2" \
        --summary runs/mia_main --file runs/mia_main/report.md

    # ad hoc
    uv run python scripts/notify.py --text "pod is about to be destroyed"

    # check the credentials work before relying on them
    uv run python scripts/notify.py --text "test" --verbose
"""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

from dp_rag_eval.evaluation.progress import read_progress, render_progress
from dp_rag_eval.notifications import TelegramCredentials, send_document, send_message

logger = logging.getLogger("notify")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", default=None, help="Message body. Combined with --summary if both are given.")
    p.add_argument(
        "--summary",
        nargs="*",
        type=Path,
        default=None,
        help="Run directories whose progress summary to append (same output as watch_progress.py)",
    )
    p.add_argument(
        "--file",
        nargs="*",
        type=Path,
        default=None,
        help="Files to upload after the message, e.g. runs/mia_main/report.md. Missing or "
        "oversized files are reported, not fatal.",
    )
    p.add_argument(
        "--status",
        type=int,
        default=None,
        help="Exit code of the command that just ran; 0 reports success, anything else reports failure",
    )
    p.add_argument("--label", default=None, help="Short name for this run, e.g. 'group 2'")
    p.add_argument("--verbose", action="store_true", help="Log the message and the API response")
    return p.parse_args()


def build_message(args: argparse.Namespace) -> str:
    host = socket.gethostname()
    label = args.label or "dp-rag-eval"

    if args.status is None:
        header = f"[{label}] {host}"
    elif args.status == 0:
        header = f"[{label}] FINISHED on {host}"
    else:
        header = f"[{label}] FAILED (exit {args.status}) on {host}"

    parts = [header]
    if args.text:
        parts.append(args.text)
    for run_dir in args.summary or []:
        parts.append(render_progress(read_progress(run_dir)))
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not args.text and not args.summary and not args.file and args.status is None:
        raise SystemExit("Nothing to send: pass --text, --summary, --file and/or --status")

    creds = TelegramCredentials.from_env()
    if creds is None:
        logger.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — nothing will be sent")

    message = build_message(args)
    if args.verbose:
        logger.info("Message:\n%s", message)
    if send_message(creds, message):
        logger.info("Notification sent")

    for path in args.file or []:
        if send_document(creds, path, caption=path.name):
            logger.info("Sent %s", path)


if __name__ == "__main__":
    main()
