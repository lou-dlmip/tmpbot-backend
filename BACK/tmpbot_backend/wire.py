# SSE wire-format encoder for POST /api/v1/chat.
#
# Implements DOCS/STND/streaming_wire_format.md (TMPBOT v1). Every event
# the backend sends is built by one helper here, so the wire format lives
# in one file. Lifted from NINOBOT origin/main BACK/ninobot_backend/wire.py
# (the SSE encoder, meta, tool_call, tool_result, token, done, error) with
# NINOBOT's citation and ENSO material removed, and three TMPBOT events
# added for the designed cards: ``cases``, ``gate`` and ``download``
# (spec section 3b).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Server-sent event encoders for the chat stream."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Iterable, Optional

log = logging.getLogger("tmpbot.wire")

# ----------------------------------------------------------------------
# Event names
# ----------------------------------------------------------------------

EVENT_META = "meta"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_TOKEN = "token"
EVENT_CASES = "cases"
EVENT_GATE = "gate"
EVENT_DOWNLOAD = "download"
EVENT_DONE = "done"
EVENT_ERROR = "error"

ALL_EVENTS = (
    EVENT_META,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    EVENT_TOKEN,
    EVENT_CASES,
    EVENT_GATE,
    EVENT_DOWNLOAD,
    EVENT_DONE,
    EVENT_ERROR,
)
TERMINAL_EVENTS = (EVENT_DONE, EVENT_ERROR)

# Stop reasons allowed in ``done``.
STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_ABSTENTION = "abstention"
STOP_REFUSAL = "refusal"
STOP_REASONS = (STOP_END_TURN, STOP_MAX_TOKENS, STOP_ABSTENTION, STOP_REFUSAL)

# Error codes allowed in ``error``.
ERR_UPSTREAM_TIMEOUT = "upstream_timeout"
ERR_UPSTREAM_UNAVAILABLE = "upstream_unavailable"
ERR_RATE_LIMITED = "rate_limited"
ERR_INTERNAL = "internal_error"
ERR_INVALID_REQUEST = "invalid_request"
ERROR_CODES = (
    ERR_UPSTREAM_TIMEOUT,
    ERR_UPSTREAM_UNAVAILABLE,
    ERR_RATE_LIMITED,
    ERR_INTERNAL,
    ERR_INVALID_REQUEST,
)

# Optional machine-readable ``error.detail`` (wire format v1.1). The
# frontend picks a more specific localised message from it.
DETAIL_FAIR_USE_DAILY = "fair_use_daily"  # free account's daily cap
DETAIL_BURST = "burst"  # too many questions in a minute
DETAIL_TURNSTILE_REQUIRED = "turnstile_required"  # get a token, retry
ERROR_DETAILS = (
    DETAIL_FAIR_USE_DAILY,
    DETAIL_BURST,
    DETAIL_TURNSTILE_REQUIRED,
)

# Why a ``gate`` card is shown.
GATE_SUBSCRIBER_ONLY = "subscriber_only"  # paywall card after a teaser
GATE_FREE_LIMIT = "free_limit"  # free-limit card, 11th signed-out question
GATE_REASONS = (GATE_SUBSCRIBER_ONLY, GATE_FREE_LIMIT)

# The handoff draws 0 to 3 case cards under an answer.
MAX_CASE_CARDS = 3

# Case-study and publication ids are lowercase slugs, as in the scraped
# export (DOCS/DSGN/webflow_export_2026-11/case_studies/*.json).
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Assistant message ids: "TMP-" and 12 hex digits.
MESSAGE_ID_RE = re.compile(r"^TMP-[0-9a-f]{12}$")


def new_message_id() -> str:
    """Return a fresh id for an assistant turn, for example ``TMP-3f2a…``.

    Returns:
        str: ``"TMP-"`` followed by 12 random hex digits.
    """
    return f"TMP-{uuid.uuid4().hex[:12]}"


# ----------------------------------------------------------------------
# Low-level SSE encoder
# ----------------------------------------------------------------------


def encode_event(name: str, payload: dict[str, Any]) -> str:
    """Encode one SSE record.

    The JSON sits on one ``data:`` line and keeps non-ASCII text as is
    (``ensure_ascii=False``), so Japanese and accented text pass through
    unmangled. A blank line ends the record.

    Args:
        name: Event name, one of ``ALL_EVENTS``.
        payload: JSON-serialisable payload.

    Returns:
        str: The encoded record, ending with a blank line.

    Raises:
        ValueError: If ``name`` is not a known event.
    """
    if name not in ALL_EVENTS:
        raise ValueError(f"unknown SSE event name: {name!r}")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {data}\n\n"


# ----------------------------------------------------------------------
# Per-event helpers
# ----------------------------------------------------------------------


def event_meta(message_id: str, language: str) -> str:
    """Encode the opening ``meta`` event.

    Args:
        message_id: Id of the assistant turn being streamed.
        language: Locale the answer is written in.

    Returns:
        str: The encoded event.

    Raises:
        ValueError: If ``message_id`` does not match ``MESSAGE_ID_RE``.
    """
    if not MESSAGE_ID_RE.match(message_id):
        raise ValueError(f"bad message id: {message_id!r}")
    return encode_event(
        EVENT_META, {"message_id": message_id, "language": language}
    )


def event_tool_call(name: str, args: dict[str, Any]) -> str:
    """Encode a ``tool_call`` event: the model is calling a tool.

    Args:
        name: Tool name.
        args: Arguments the model passed.

    Returns:
        str: The encoded event.
    """
    return encode_event(EVENT_TOOL_CALL, {"name": name, "args": args})


def event_tool_result(
    name: str, ok: bool, detail: Optional[str] = None
) -> str:
    """Encode a ``tool_result`` event, the companion of a ``tool_call``.

    Args:
        name: Tool name, matching the ``tool_call``.
        ok: Whether the tool succeeded.
        detail: Short reason when ``ok`` is False.

    Returns:
        str: The encoded event.
    """
    payload: dict[str, Any] = {"name": name, "ok": ok}
    if detail is not None:
        payload["detail"] = detail
    return encode_event(EVENT_TOOL_RESULT, payload)


def event_token(content: str) -> str:
    """Encode one ``token`` event carrying a piece of answer text.

    Args:
        content: The text delta. Boundaries carry no meaning.

    Returns:
        str: The encoded event.
    """
    return encode_event(EVENT_TOKEN, {"content": content})


def event_cases(ids: Iterable[str]) -> Optional[str]:
    """Encode a ``cases`` event: case-study cards under the answer.

    The caller must already have checked each id against published
    content (Phase 5 does this against ``COPS/CASE/``). This helper
    enforces the format: ids that are not slugs are dropped with a
    warning, repeats are removed (first wins), and at most
    ``MAX_CASE_CARDS`` are kept.

    Args:
        ids: Case-study ids in display order.

    Returns:
        Optional[str]: The encoded event, or None when no valid id is
        left, in which case nothing should be sent.
    """
    kept: list[str] = []
    for raw in ids:
        if not isinstance(raw, str) or not SLUG_RE.match(raw):
            log.warning("cases event: dropping malformed id",
                        extra={"case_id": repr(raw)})
            continue
        if raw not in kept:
            kept.append(raw)
    if len(kept) > MAX_CASE_CARDS:
        log.warning("cases event: truncating to %d cards", MAX_CASE_CARDS,
                    extra={"dropped": kept[MAX_CASE_CARDS:]})
        kept = kept[:MAX_CASE_CARDS]
    if not kept:
        return None
    return encode_event(EVENT_CASES, {"ids": kept})


def event_gate(reason: str) -> str:
    """Encode a ``gate`` event: show the paywall or free-limit card.

    Args:
        reason: ``subscriber_only`` (paywall card after a 1 to 2
            sentence teaser) or ``free_limit`` (the free-limit card in
            place of an answer).

    Returns:
        str: The encoded event.

    Raises:
        ValueError: If ``reason`` is not one of ``GATE_REASONS``.
    """
    if reason not in GATE_REASONS:
        raise ValueError(f"unknown gate reason: {reason!r}")
    return encode_event(EVENT_GATE, {"reason": reason})


def event_download(publication_id: str, title: str, year: int) -> str:
    """Encode a ``download`` event: a publication card (D17).

    The card carries no file link. Clicking it asks the backend for a
    short-lived signed URL, issued only after the subscription is
    checked at that moment (spec section 3b, Phase 7).

    Args:
        publication_id: Published publication id (a slug).
        title: Title shown on the card.
        year: Publication year, shown in the eyebrow.

    Returns:
        str: The encoded event.

    Raises:
        ValueError: If the id is not a slug, the title is empty, or the
            year is outside 1900 to 2100.
    """
    if not SLUG_RE.match(publication_id):
        raise ValueError(f"bad publication id: {publication_id!r}")
    if not title.strip():
        raise ValueError("download title is empty")
    if not 1900 <= int(year) <= 2100:
        raise ValueError(f"implausible publication year: {year!r}")
    return encode_event(
        EVENT_DOWNLOAD,
        {"publication_id": publication_id, "title": title, "year": int(year)},
    )


def event_done(
    stop_reason: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    conversation_id: Optional[str] = None,
) -> str:
    """Encode the terminal ``done`` event.

    Args:
        stop_reason: One of ``STOP_REASONS``.
        input_tokens: Model input tokens for this turn.
        output_tokens: Model output tokens for this turn.
        conversation_id: The stored conversation this turn joined
            (v1.1, optional); the frontend sends it with the next turn.

    Returns:
        str: The encoded event.

    Raises:
        ValueError: If ``stop_reason`` is not one of ``STOP_REASONS``.
    """
    if stop_reason not in STOP_REASONS:
        raise ValueError(f"unknown stop reason: {stop_reason!r}")
    payload: dict[str, Any] = {
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
        },
    }
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    return encode_event(EVENT_DONE, payload)


def event_error(code: str, message: str,
                detail: Optional[str] = None) -> str:
    """Encode the terminal ``error`` event.

    Args:
        code: One of ``ERROR_CODES``. The frontend picks its localised
            fallback reply from this, never from ``message``.
        message: Short description for logs and debugging.
        detail: Optional sub-reason, one of ``ERROR_DETAILS`` (v1.1).

    Returns:
        str: The encoded event.

    Raises:
        ValueError: If ``code`` is not one of ``ERROR_CODES`` or
            ``detail`` is not one of ``ERROR_DETAILS``.
    """
    if code not in ERROR_CODES:
        raise ValueError(f"unknown error code: {code!r}")
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        if detail not in ERROR_DETAILS:
            raise ValueError(f"unknown error detail: {detail!r}")
        payload["detail"] = detail
    return encode_event(EVENT_ERROR, payload)


# ----------------------------------------------------------------------
# Decoder (tests, eval scripts and smoke checks)
# ----------------------------------------------------------------------


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE body back into ``(event, payload)`` pairs.

    Handles only what this module writes: one ``event:`` line and one
    ``data:`` line per record. Used by tests and smoke checks.

    Args:
        text: The full response body.

    Returns:
        list[tuple[str, dict[str, Any]]]: Events in order.

    Raises:
        ValueError: If a record lacks a name or data, or the data is not
            valid JSON.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    for record in text.split("\n\n"):
        if not record.strip():
            continue
        name: Optional[str] = None
        data: Optional[str] = None
        for line in record.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = line[len("data: "):]
        if name is None or data is None:
            raise ValueError(f"malformed SSE record: {record!r}")
        try:
            events.append((name, json.loads(data)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"bad JSON in SSE record: {data!r}") from exc
    return events
