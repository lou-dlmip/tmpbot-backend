# Tests for the stream on POST /api/v1/chat: response headers, event
# order, token-by-token text, the demo card switches and request
# validation (tmpbot_backend.main, tmpbot_backend.canned). The writer is
# the scripted fake from conftest; no provider is called.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Chat stream event-sequence tests."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest
from chat_helpers import build_runtime
from fastapi.testclient import TestClient

from tmpbot_backend import canned, wire
from tmpbot_backend.llm import ScriptedLLM, ScriptedReply
from tmpbot_backend.locales import LOCALES
from tmpbot_backend.main import _guarded_stream, create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_EXPORT = REPO_ROOT / "DOCS" / "DSGN" / "webflow_export_2026-11"


def _events(client, body, headers):
    """POST a chat request and return the parsed events.

    Args:
        client: Test client.
        body: Request body.
        headers: Request headers.

    Returns:
        list[tuple[str, dict]]: Events in order.
    """
    resp = client.post("/api/v1/chat", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return wire.parse_sse(resp.text)


def _check_frame(events):
    """Assert the frame every stream must have: meta first, done last.

    Args:
        events: Parsed events.
    """
    names = [n for n, _ in events]
    assert names[0] == "meta"
    assert names.count("meta") == 1
    assert names[-1] == "done"
    assert sum(n in wire.TERMINAL_EVENTS for n in names) == 1


def test_stream_headers(client, make_body, auth_headers):
    """SSE content type, no caching, no proxy buffering."""
    resp = client.post("/api/v1/chat", json=make_body(), headers=auth_headers)
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"


def test_plain_sequence_is_meta_tokens_done(client, make_body, auth_headers):
    """meta, then several tokens, then done; no card events."""
    events = _events(client, make_body(), auth_headers)
    _check_frame(events)
    middle = [n for n, _ in events[1:-1]]
    assert middle and set(middle) == {"token"}
    assert len(middle) > 1, "text should arrive in pieces"


def test_meta_payload(client, make_body, auth_headers):
    """meta carries a well-formed message id and the language."""
    events = _events(client, make_body("de"), auth_headers)
    meta = events[0][1]
    assert wire.MESSAGE_ID_RE.match(meta["message_id"])
    assert meta["language"] == "de"


def test_message_ids_differ_per_turn(client, make_body, auth_headers):
    """Each turn gets its own id."""
    a = _events(client, make_body(), auth_headers)[0][1]["message_id"]
    b = _events(client, make_body(), auth_headers)[0][1]["message_id"]
    assert a != b


def test_done_payload(client, make_body, auth_headers):
    """done ends the turn, reports the writer's usage and the stored
    conversation id."""
    done = _events(client, make_body(), auth_headers)[-1][1]
    assert done["stop_reason"] == "end_turn"
    assert done["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert len(done["conversation_id"]) == 36


ANSWERS = {
    "en": "We start by listening.",
    "ja": "まずはお話を伺います。",
    "es": "Empezamos escuchando.",
    "pt": "Começamos ouvindo.",
    "fr": "Nous commençons par écouter.",
    "de": "Wir hören zuerst zu.",
    "it": "Iniziamo ascoltando.",
}


@pytest.mark.parametrize("lang", LOCALES)
def test_tokens_rebuild_the_reply_in_each_language(
    settings, make_body, auth_headers, lang
):
    """Joined tokens equal the writer's text, in all seven languages,
    and the prompt asks for that language."""
    llm = ScriptedLLM([ScriptedReply(text=ANSWERS[lang])])
    client = TestClient(create_app(settings, build_runtime(settings, llm)))
    events = _events(client, make_body(lang), auth_headers)
    text = "".join(p["content"] for n, p in events if n == "token")
    assert text == ANSWERS[lang]
    assert "Always reply in" in llm.calls[0].prompt_text()


def test_japanese_streams_unescaped_utf8(settings, make_body, auth_headers):
    """Japanese travels as UTF-8, not as \\u escapes."""
    llm = ScriptedLLM([ScriptedReply(text=ANSWERS["ja"])])
    client = TestClient(create_app(settings, build_runtime(settings, llm)))
    resp = client.post(
        "/api/v1/chat", json=make_body("ja"), headers=auth_headers
    )
    assert '{"content":"まずはお話を伺います。"}' in resp.content.decode(
        "utf-8")
    assert "\\u" not in resp.text


def test_demo_cases(client, make_body, auth_headers):
    """cases demo: reply text, then one cases event after the last token."""
    events = _events(client, make_body(demo="cases"), auth_headers)
    _check_frame(events)
    names = [n for n, _ in events]
    assert names[-2] == "cases"
    assert names.count("cases") == 1
    assert events[-2][1] == {"ids": list(canned.DEMO_CASE_IDS)}


def test_demo_gate(client, make_body, auth_headers):
    """gate demo: short teaser, then the paywall card."""
    events = _events(client, make_body("fr", demo="gate"), auth_headers)
    _check_frame(events)
    assert events[-2] == ("gate", {"reason": "subscriber_only"})
    text = "".join(p["content"] for n, p in events if n == "token")
    assert text == canned.CANNED_TEASER["fr"]


def test_demo_limit(client, make_body, auth_headers):
    """limit demo: no answer text at all, only the free-limit card."""
    events = _events(client, make_body(demo="limit"), auth_headers)
    assert [n for n, _ in events] == ["meta", "gate", "done"]
    assert events[1][1] == {"reason": "free_limit"}


def test_demo_download(client, make_body, auth_headers):
    """download demo: reply text, then the publication card."""
    events = _events(client, make_body(demo="download"), auth_headers)
    _check_frame(events)
    assert events[-2] == ("download", canned.DEMO_PUBLICATION)


def test_demo_ignored_in_production(settings, make_body, auth_headers):
    """Production never sends a forced card."""
    prod = dataclasses.replace(settings, env="production")
    client = TestClient(create_app(prod, build_runtime(prod)))
    for demo in canned.DEMO_MODES:
        events = _events(client, make_body(demo=demo), auth_headers)
        assert {n for n, _ in events} == {"meta", "token", "done"}, demo


def test_demo_case_ids_are_real_published_slugs():
    """The demo cards point at case studies that exist on asktmp.com."""
    for slug in canned.DEMO_CASE_IDS:
        path = CASE_EXPORT / "case_studies" / f"{slug}.json"
        assert path.is_file(), slug
        assert json.loads(path.read_text(encoding="utf-8"))["slug"] == slug


@pytest.mark.parametrize(
    "body",
    [
        {"language": "xx",
         "messages": [{"role": "user", "content": "hi"}]},
        {"language": "en", "messages": []},
        {"language": "en",
         "messages": [{"role": "assistant", "content": "hi"}]},
        {"language": "en",
         "messages": [{"role": "system", "content": "hi"}]},
        {"language": "en",
         "messages": [{"role": "user", "content": ""}]},
        {"language": "en", "demo": "explode",
         "messages": [{"role": "user", "content": "hi"}]},
        {"language": "en",
         "messages": [{"role": "user", "content": "x" * 4001}]},
        {"language": "en",
         "messages": [{"role": "user", "content": "x" * 4000}] * 26},
    ],
    ids=[
        "bad-locale", "no-messages", "last-not-user", "system-role",
        "empty-content", "bad-demo", "message-too-long", "total-too-long",
    ],
)
def test_invalid_requests_are_422(client, auth_headers, body):
    """Bad bodies are rejected before any stream starts."""
    resp = client.post("/api/v1/chat", json=body, headers=auth_headers)
    assert resp.status_code == 422


def test_history_is_accepted(client, auth_headers):
    """A multi-turn history ending in a user message streams normally."""
    body = {
        "language": "en",
        "messages": [
            {"id": "USR-1", "role": "user", "content": "Hello"},
            {"id": "TMP-000000000001", "role": "assistant",
             "content": "Hello."},
            {"role": "user", "content": "And land tenure?"},
        ],
    }
    _check_frame(_events(client, body, auth_headers))


async def _collect(agen):
    """Drain an async generator into a list.

    Args:
        agen: The async generator.

    Returns:
        list: Everything it yielded.
    """
    return [item async for item in agen]


def test_guarded_stream_turns_failure_into_error_event():
    """A generator that dies mid-answer ends with an error event, not a
    dropped connection."""
    async def broken():
        yield wire.event_meta("TMP-0123456789ab", "en")
        yield wire.event_token("Half an ")
        raise RuntimeError("model fell over")

    out = asyncio.run(
        _collect(_guarded_stream(broken(), "TMP-0123456789ab"))
    )
    events = wire.parse_sse(b"".join(out).decode("utf-8"))
    assert [n for n, _ in events] == ["meta", "token", "error"]
    assert events[-1][1]["code"] == "internal_error"


def test_guarded_stream_no_second_terminal_after_done():
    """If done was already sent, a later failure adds no error event."""
    async def fails_after_done():
        yield wire.event_meta("TMP-0123456789ab", "en")
        yield wire.event_done("end_turn")
        raise RuntimeError("cleanup failed")

    out = asyncio.run(
        _collect(_guarded_stream(fails_after_done(), "TMP-0123456789ab"))
    )
    names = [n for n, _ in wire.parse_sse(b"".join(out).decode("utf-8"))]
    assert names == ["meta", "done"]
