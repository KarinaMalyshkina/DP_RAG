"""Tests for the bot's command routing and stall watchdog.

Loaded from ``scripts/`` because the bot is a script, not part of the package. The
cases worth pinning down are the ones that would be discovered at 3am on a rented
box: a finished run alerting forever, a stall alerting every 30 seconds, and a
message from a stranger being answered.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dp_rag_eval.notifications import TelegramCredentials

_SPEC = importlib.util.spec_from_file_location(
    "telegram_bot", Path(__file__).resolve().parents[1] / "scripts" / "telegram_bot.py"
)
telegram_bot = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(telegram_bot)


@pytest.fixture
def bot(tmp_path, monkeypatch):
    """A bot over two run dirs, with every outbound call captured."""
    sent: list[str] = []
    documents: list[Path] = []

    monkeypatch.setattr(telegram_bot, "send_message", lambda creds, text, **kw: sent.append(text) or True)
    monkeypatch.setattr(telegram_bot, "send_document", lambda creds, path, **kw: documents.append(Path(path)) or True)

    dirs = [tmp_path / "mia_main", tmp_path / "canary_main"]
    for d in dirs:
        d.mkdir()
    instance = telegram_bot.Bot(TelegramCredentials("t", "42"), dirs, stall_minutes=60)
    instance.sent = sent  # type: ignore[attr-defined]
    instance.documents = documents  # type: ignore[attr-defined]
    return instance


def write_results(run_dir: Path, n_done: int, *, planned: int | None = None, age_minutes: float = 0.0):
    rows = [{"cell": f"c{i}", "pipeline": "koga", "status": "OK", "seconds": 10.0} for i in range(n_done)]
    meta = {"planned_cells": planned} if planned is not None else {}
    path = run_dir / "results.json"
    path.write_text(json.dumps({"meta": meta, "results": rows}), encoding="utf-8")
    if age_minutes:
        old = (datetime.now(tz=timezone.utc) - timedelta(minutes=age_minutes)).timestamp()
        import os

        os.utime(path, (old, old))
    return path


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def test_resolve_matches_a_watched_directory_by_name(bot):
    assert bot.resolve("mia_main") == bot.run_dirs[0]
    assert bot.resolve("canary_main") == bot.run_dirs[1]


def test_resolve_refuses_a_path_outside_the_whitelist(bot):
    # No message may make the bot read arbitrary files off the pod.
    assert bot.resolve("/etc/passwd") is None
    assert bot.resolve("../../secrets") is None
    assert bot.resolve("runs/some_other_run") is None


def test_resolve_needs_a_name_when_several_runs_are_watched(bot):
    assert bot.resolve(None) is None


def test_resolve_defaults_to_the_only_run_when_there_is_one(tmp_path):
    only = telegram_bot.Bot(TelegramCredentials("t", "42"), [tmp_path / "solo"], stall_minutes=0)

    assert only.resolve(None) == tmp_path / "solo"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_status_reports_every_watched_run(bot):
    write_results(bot.run_dirs[0], 3, planned=10)
    write_results(bot.run_dirs[1], 1, planned=4)

    bot.handle("/status")

    assert "3/10" in bot.sent[-1] and "1/4" in bot.sent[-1]


def test_report_sends_the_file(bot):
    (bot.run_dirs[0] / "report.md").write_text("# results", encoding="utf-8")

    bot.handle("/report mia_main")

    assert bot.documents == [bot.run_dirs[0] / "report.md"]


def test_report_for_a_run_with_no_finished_cell_explains_instead_of_failing(bot):
    bot.handle("/report mia_main")

    assert bot.documents == []
    assert "does not exist yet" in bot.sent[-1]


def test_group_chat_command_suffix_is_stripped(bot):
    write_results(bot.run_dirs[0], 1, planned=2)
    write_results(bot.run_dirs[1], 0, planned=2)

    bot.handle("/status@dprag_bot")

    assert "1/2" in bot.sent[-1]


def test_unknown_command_gets_the_help_text(bot):
    bot.handle("/nonsense")

    assert "Unknown command" in bot.sent[-1]


def test_degradation_without_control_rows_says_so(bot):
    rows = [
        {"cell": "c1", "pipeline": "naive", "mode": "unified", "variant": "full_duplicate", "status": "OK", "auc": 0.7}
    ]
    (bot.run_dirs[0] / "results.json").write_text(json.dumps({"meta": {}, "results": rows}), encoding="utf-8")

    bot.handle("/degradation mia_main")

    assert bot.documents == []
    assert "control rows" in bot.sent[-1]


def test_degradation_sends_a_report_once_a_pair_exists(bot):
    rows = [
        {
            "cell": "c1",
            "pipeline": "naive",
            "mode": "unified",
            "epsilon": "inf",
            "seed": 42,
            "variant": "original",
            "params": "-",
            "status": "OK",
            "auc": 0.6,
        },
        {
            "cell": "c2",
            "pipeline": "naive",
            "mode": "unified",
            "epsilon": "inf",
            "seed": 42,
            "variant": "full_duplicate",
            "params": "d=0.5",
            "status": "OK",
            "auc": 0.8,
        },
    ]
    (bot.run_dirs[0] / "results.json").write_text(json.dumps({"meta": {}, "results": rows}), encoding="utf-8")

    bot.handle("/degradation mia_main")

    assert len(bot.documents) == 1
    assert bot.documents[0].name == "degradation_mia_main.md"


# ---------------------------------------------------------------------------
# Stall watchdog
# ---------------------------------------------------------------------------


def test_a_stalled_run_alerts(bot):
    write_results(bot.run_dirs[0], 5, planned=100, age_minutes=90)
    write_results(bot.run_dirs[1], 1, planned=4)

    bot.check_stalls()

    assert any("STALLED" in m for m in bot.sent)


def test_a_stall_alerts_once_not_on_every_poll(bot):
    write_results(bot.run_dirs[0], 5, planned=100, age_minutes=90)
    write_results(bot.run_dirs[1], 1, planned=4)

    bot.check_stalls()
    bot.check_stalls()
    bot.check_stalls()

    assert sum("STALLED" in m for m in bot.sent) == 1


def test_recovery_clears_the_alert_and_says_so(bot):
    write_results(bot.run_dirs[0], 5, planned=100, age_minutes=90)
    write_results(bot.run_dirs[1], 1, planned=4)
    bot.check_stalls()

    write_results(bot.run_dirs[0], 6, planned=100)  # fresh mtime
    bot.check_stalls()

    assert "moving again" in bot.sent[-1]


def test_a_completed_run_never_alerts(bot):
    # results.json stops changing forever once a run finishes; without this the
    # watchdog would alert every night for the rest of the pod's life.
    write_results(bot.run_dirs[0], 100, planned=100, age_minutes=600)
    write_results(bot.run_dirs[1], 4, planned=4, age_minutes=600)

    bot.check_stalls()

    assert not any("STALLED" in m for m in bot.sent)


def test_a_run_that_has_not_started_does_not_alert(bot):
    bot.check_stalls()

    assert bot.sent == []


def test_the_watchdog_can_be_disabled(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    write_results(run, 5, planned=100, age_minutes=600)
    sent: list[str] = []
    off = telegram_bot.Bot(TelegramCredentials("t", "42"), [run], stall_minutes=0)
    off.say = lambda text: sent.append(text)  # type: ignore[method-assign]

    off.check_stalls()

    assert sent == []
