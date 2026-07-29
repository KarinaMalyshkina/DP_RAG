"""Tests for the Telegram transport.

The contract that matters is negative: none of this may raise. It is chained onto
sweeps that run for days, so a failure here has to degrade to a log line, never to a
non-zero exit that looks like the sweep itself failing.
"""

from __future__ import annotations

import urllib.error

import pytest

from dp_rag_eval import notifications
from dp_rag_eval.notifications import (
    MAX_CHARS,
    TelegramCredentials,
    _chunks,
    _multipart,
    send_document,
    send_message,
)


@pytest.fixture
def creds():
    return TelegramCredentials(token="123:ABC", chat_id="42")


@pytest.fixture
def calls(monkeypatch):
    """Record every _post call instead of hitting the network."""
    recorded = []

    def fake_post(creds, method, params, *, body=None, content_type=None, timeout=20.0):
        recorded.append({"method": method, "params": params, "body": body, "content_type": content_type})
        return {"ok": True}

    monkeypatch.setattr(notifications, "_post", fake_post)
    return recorded


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_credentials_missing_from_env_return_none(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert TelegramCredentials.from_env() is None


def test_half_configured_credentials_are_treated_as_missing(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "   ")

    assert TelegramCredentials.from_env() is None


# ---------------------------------------------------------------------------
# Never raising
# ---------------------------------------------------------------------------


def test_sending_without_credentials_is_a_no_op_not_an_error(caplog):
    assert send_message(None, "hello") is False
    assert "not sending" in caplog.text.lower()


def test_a_network_failure_is_swallowed(monkeypatch, creds):
    def explode(*args, **kwargs):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(notifications.urllib.request, "urlopen", explode)

    assert send_message(creds, "hello") is False  # and does not raise


def test_an_http_error_is_swallowed(monkeypatch, creds):
    def explode(*args, **kwargs):
        raise urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(notifications.urllib.request, "urlopen", explode)

    assert send_message(creds, "hello") is False


def test_missing_file_is_reported_not_raised(tmp_path, creds):
    assert send_document(creds, tmp_path / "nope.md") is False


def test_empty_file_is_not_sent(tmp_path, creds, calls):
    empty = tmp_path / "results.csv"
    empty.write_text("", encoding="utf-8")

    assert send_document(creds, empty) is False
    assert calls == []


def test_oversized_file_is_refused_with_an_explanation(tmp_path, creds, calls, monkeypatch):
    monkeypatch.setattr(notifications, "MAX_UPLOAD_BYTES", 10)
    big = tmp_path / "predictions.jsonl"
    big.write_text("x" * 100, encoding="utf-8")

    assert send_document(creds, big) is False
    assert [c["method"] for c in calls] == ["sendMessage"]
    assert "rsync" in calls[0]["params"]["text"]


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------


def test_short_messages_are_sent_in_one_piece(creds, calls):
    send_message(creds, "short")

    assert len(calls) == 1
    assert calls[0]["params"]["text"] == "short"


def test_long_messages_split_on_line_boundaries(creds, calls):
    text = "\n".join(f"| row {i} | value |" for i in range(1000))

    send_message(creds, text)

    assert len(calls) > 1
    for call in calls:
        assert len(call["params"]["text"]) <= MAX_CHARS
        # A table row must never be cut in half.
        assert not call["params"]["text"].startswith(" value |")
    assert "".join(c["params"]["text"] for c in calls) == text


def test_a_single_overlong_line_is_still_split():
    parts = _chunks("x" * (MAX_CHARS * 2 + 5))

    assert len(parts) == 3
    assert all(len(p) <= MAX_CHARS for p in parts)
    assert "".join(parts) == "x" * (MAX_CHARS * 2 + 5)


# ---------------------------------------------------------------------------
# Upload encoding
# ---------------------------------------------------------------------------


def test_multipart_body_carries_the_filename_and_content():
    body, content_type = _multipart("report.md", b"# hello")

    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=")[1]
    assert boundary.encode() in body
    assert b'name="document"; filename="report.md"' in body
    assert b"# hello" in body
    assert body.endswith(f"--{boundary}--\r\n".encode())


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("report.md", "text/markdown"), ("results.csv", "text/csv"), ("results.json", "application/json")],
)
def test_text_files_get_a_readable_mime_type(filename, expected):
    # Left to mimetypes these are application/octet-stream on Windows, which makes
    # Telegram offer a blind download instead of a preview.
    body, _ = _multipart(filename, b"x")

    assert f"Content-Type: {expected}".encode() in body


def test_send_document_posts_the_file_with_a_caption(tmp_path, creds, calls):
    report = tmp_path / "report.md"
    report.write_text("# results", encoding="utf-8")

    assert send_document(creds, report, caption="mia_main/report.md") is True

    assert calls[0]["method"] == "sendDocument"
    assert calls[0]["params"]["caption"] == "mia_main/report.md"
    assert calls[0]["params"]["chat_id"] == "42"
    assert b"# results" in calls[0]["body"]
