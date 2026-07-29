"""Telegram transport: messages, file uploads and inbound command polling.

Used by ``scripts/notify.py`` (one-shot, chained after a sweep) and
``scripts/telegram_bot.py`` (a daemon answering commands). It exists as a module
rather than living in one of those scripts because both need the same three things
and a second copy of a retry-and-swallow-errors helper is a second place to get it
wrong.

**Nothing here raises.** Every function reports failure by return value. A sweep
runs for days on paid hardware; a notifier that can take it down — because a token
expired, or Telegram had an outage, or the pod briefly lost DNS — is worse than no
notifier at all. The cost of that choice is that a misconfigured token fails
silently, so :func:`TelegramCredentials.from_env` logs loudly when it finds nothing
and the runbook tells you to send a test message before relying on it.

No new dependency: ``requests`` is not in ``pyproject.toml``, so the multipart body
for file uploads is built by hand against ``urllib``.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import secrets
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/{method}"

#: Telegram rejects text messages above 4096 characters; leave room for a suffix.
MAX_CHARS = 3900

#: Bot API upload ceiling is 50 MB. Refuse earlier and say so, rather than spending
#: minutes uploading a predictions.jsonl that will be rejected.
MAX_UPLOAD_BYTES = 45 * 1024 * 1024

#: ``mimetypes`` does not know ``.md`` on every platform (notably Windows), and the
#: fallback of ``application/octet-stream`` makes Telegram offer a blind download
#: instead of an in-app preview. Every file this sends is text.
_TEXT_TYPES = {
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".jsonl": "application/json",
    ".txt": "text/plain",
    ".log": "text/plain",
}


@dataclass(frozen=True)
class TelegramCredentials:
    """Bot token and the single chat allowed to talk to it."""

    token: str
    chat_id: str

    @classmethod
    def from_env(cls) -> TelegramCredentials | None:
        """Read ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``, or return ``None``."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return None
        return cls(token=token, chat_id=chat_id)


def _post(
    creds: TelegramCredentials,
    method: str,
    params: dict[str, Any],
    *,
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any] | None:
    """Call one Bot API method. Returns the ``result`` payload, or ``None`` on any failure."""
    url = API_URL.format(token=creds.token, method=method)
    if body is None:
        data = urllib.parse.urlencode(params).encode()
        request = urllib.request.Request(url, data=data)
    else:
        request = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", data=body)
        if content_type:
            request.add_header("Content-Type", content_type)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300] if hasattr(exc, "read") else ""
        logger.warning("Telegram %s failed: HTTP %s %s", method, exc.code, detail)
        return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        # Deliberately broad: this must never be able to fail a sweep.
        logger.warning("Telegram %s failed (%s: %s)", method, type(exc).__name__, exc)
        return None

    if not payload.get("ok"):
        logger.warning("Telegram %s rejected: %s", method, str(payload)[:300])
        return None
    return payload.get("result")


def _chunks(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split on line boundaries so a progress table is never cut mid-row."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        # A single line longer than the limit still has to be hard-split.
        while len(line) > limit:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) > limit:
            parts.append(current)
            current = line
        else:
            current += line
    if current:
        parts.append(current)
    return parts


def send_message(creds: TelegramCredentials | None, text: str, *, chat_id: str | None = None) -> bool:
    """Send ``text``, split across several messages if it is too long."""
    if creds is None:
        logger.warning("No Telegram credentials — not sending. Message was:\n%s", text)
        return False
    target = chat_id or creds.chat_id
    sent = True
    for chunk in _chunks(text):
        params = {"chat_id": target, "text": chunk, "disable_web_page_preview": "true"}
        sent = _post(creds, "sendMessage", params) is not None and sent
    return sent


def _multipart(filename: str, content: bytes, *, field: str = "document") -> tuple[bytes, str]:
    """Build a minimal multipart/form-data body for one file part."""
    boundary = f"----dprag{secrets.token_hex(16)}"
    suffix = Path(filename).suffix.lower()
    mime = _TEXT_TYPES.get(suffix) or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + content + tail, f"multipart/form-data; boundary={boundary}"


def send_document(
    creds: TelegramCredentials | None,
    path: str | Path,
    *,
    caption: str | None = None,
    chat_id: str | None = None,
) -> bool:
    """Upload a file. Missing or oversized files are reported, not raised."""
    path = Path(path)
    if creds is None:
        logger.warning("No Telegram credentials — not sending %s", path)
        return False
    if not path.is_file():
        logger.warning("Not a file, not sending: %s", path)
        return False

    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        logger.warning(
            "%s is %.1f MB, above the %d MB upload limit — not sending", path, size / 1e6, MAX_UPLOAD_BYTES // 1024**2
        )
        send_message(
            creds,
            f"{path.name} is {size / 1e6:.0f} MB and too large for Telegram — fetch it with rsync instead.",
            chat_id=chat_id,
        )
        return False
    if size == 0:
        logger.warning("%s is empty — not sending", path)
        return False

    body, content_type = _multipart(path.name, path.read_bytes())
    params: dict[str, Any] = {"chat_id": chat_id or creds.chat_id}
    if caption:
        params["caption"] = caption[:1000]
    # Uploads are slower than messages; a 60s ceiling covers a large results.csv.
    return _post(creds, "sendDocument", params, body=body, content_type=content_type, timeout=60.0) is not None


def get_updates(creds: TelegramCredentials, offset: int, *, timeout: int = 30) -> list[dict[str, Any]]:
    """Long-poll for inbound messages.

    ``timeout`` is Telegram's server-side long-poll window: the request blocks until
    a message arrives or the window expires, so the caller loops without a sleep and
    without hammering the API. The HTTP read timeout is deliberately larger.
    """
    params = {"offset": offset, "timeout": timeout, "allowed_updates": json.dumps(["message"])}
    result = _post(creds, "getUpdates", params, timeout=timeout + 15)
    return result if isinstance(result, list) else []
