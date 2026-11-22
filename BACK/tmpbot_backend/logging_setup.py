# JSON logging for the TMPBOT backend.
#
# Render's log search and any later log drain read one JSON object per
# line more reliably than free text. Every record carries an ISO-8601 UTC
# timestamp (TMP standard: all messages timestamped) and any extra fields
# the caller passes with ``extra={...}``.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""One-line JSON log formatter and root logger setup."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes every LogRecord has. Anything else on a record came from
# ``extra=`` and is copied into the JSON object.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()
) | {"message", "asctime", "taskName", "color_message"}
# color_message is uvicorn's copy of the message with terminal colour
# codes; the plain message is already in "msg".


class JsonFormatter(logging.Formatter):
    """Format a log record as a single line of JSON.

    Keys: ``ts`` (UTC, ISO 8601, milliseconds), ``level``, ``logger``,
    ``msg``, then any ``extra`` fields, then ``exc`` when an exception
    is attached. Values that JSON cannot encode fall back to ``str()``.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render one record.

        Args:
            record: The record to format.

        Returns:
            str: A JSON object on one line.
        """
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload: dict[str, Any] = {
            "ts": ts.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Send all logging to stdout as JSON lines.

    Replaces the root logger's handlers so repeated calls (tests, uvicorn
    reloads) do not duplicate output. Uvicorn's server loggers propagate
    to the root, so their lines come out as JSON too. Uvicorn's access
    logger is silenced: the app's access-log middleware writes one JSON
    line per request with a request id, so uvicorn's would be a
    duplicate.

    Args:
        level: Logging level name, for example ``"INFO"``.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for old in list(root.handlers):
        root.removeHandler(old)
    root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = []
        uv_logger.propagate = True

    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
