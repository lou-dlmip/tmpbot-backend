# Free-question count, fair-use cap, burst limits, usage log and stored
# conversations against a real Postgres (M5.4, S14, S18).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Server-side limits and records, on the Postgres stores."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import psycopg
import pytest
from chat_helpers import TEST_SECRET, build_runtime, db_user, make_token
from fastapi.testclient import TestClient

from tmpbot_backend import wire
from tmpbot_backend.config import Settings
from tmpbot_backend.db import Database
from tmpbot_backend.llm import LLMError, ScriptedLLM, ScriptedReply
from tmpbot_backend.main import create_app
from tmpbot_backend.stores import (
    PgConversationStore,
    PgRateLimiter,
    PgUsageLog,
    PgVisitorStore,
    day_window,
    hash_ip,
)
from tmpbot_backend.turnstile import FakeTurnstile

SALT = "limits-test-salt-0123"


def _settings(url, **changes):
    """Local settings over the test database.

    Args:
        url: Database URL.
        **changes: Fields to override.

    Returns:
        Settings: The settings.
    """
    base = Settings(env="local", shared_secret=TEST_SECRET,
                    canned_token_delay_s=0.0, database_url=url,
                    ip_hash_salt=SALT)
    return dataclasses.replace(base, **changes)


def _client(settings, llm=None, **overrides):
    """A chat client on the Postgres stores.

    Args:
        settings: Settings with ``database_url``.
        llm: Writer; a ScriptedLLM when None.
        **overrides: Runtime fields to replace.

    Returns:
        TestClient: The client.
    """
    db = Database(settings.database_url)
    stores = dict(visitors=PgVisitorStore(db), limiter=PgRateLimiter(db),
                  usage=PgUsageLog(db),
                  conversations=PgConversationStore(db))
    stores.update(overrides)
    runtime = build_runtime(settings, llm or ScriptedLLM(), **stores)
    return TestClient(create_app(settings, runtime))


def _ask(client, *, cookie=None, ip="203.0.113.5", token=None,
         turnstile=None, conversation_id=None, question="Hello"):
    """Send one question.

    Args:
        client: Test client.
        cookie: Visitor id the proxy read from the cookie.
        ip: Client IP.
        token: Access token.
        turnstile: Turnstile token.
        conversation_id: Conversation to continue.
        question: The question.

    Returns:
        tuple[list, httpx.Response]: Events and the response.
    """
    headers = {"X-TMPBOT-Proxy-Secret": TEST_SECRET,
               "X-TMPBOT-Client-IP": ip}
    if cookie:
        headers["X-TMPBOT-Visitor-Id"] = cookie
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if turnstile:
        headers["X-TMPBOT-Turnstile-Token"] = turnstile
    body = {"language": "en",
            "messages": [{"role": "user", "content": question}]}
    if conversation_id:
        body["conversation_id"] = conversation_id
    resp = client.post("/api/v1/chat", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return wire.parse_sse(resp.text), resp


def _names(events):
    """Event names.

    Args:
        events: Parsed events.

    Returns:
        list[str]: Names.
    """
    return [n for n, _ in events]


def test_eleventh_signed_out_question_gets_the_free_limit_card(
        migrated_db):
    """Ten answers, then the free-limit card with no model call."""
    llm = ScriptedLLM()
    client = _client(_settings(migrated_db, rate_per_minute=100), llm)
    events, resp = _ask(client, ip="203.0.113.10")
    cookie = resp.headers["X-TMPBOT-Visitor-Id"]
    for _ in range(9):
        events, _ = _ask(client, cookie=cookie, ip="203.0.113.10")
        assert "token" in _names(events)
    assert len(llm.calls) == 10
    events, resp = _ask(client, cookie=cookie, ip="203.0.113.10")
    assert _names(events) == ["meta", "gate", "done"]
    assert events[1][1] == {"reason": "free_limit"}
    assert len(llm.calls) == 10, "the 11th question reached the model"
    assert resp.headers["X-TMPBOT-Visitor-Id"] == cookie
    with psycopg.connect(migrated_db) as conn:
        used = conn.execute(
            "SELECT questions_used FROM anonymous_visitors "
            "WHERE visitor_id = %s", (cookie,)).fetchone()[0]
    assert used == 10


def test_clearing_cookies_does_not_reset_the_count(migrated_db):
    """A cookieless (or forged-cookie) visitor from the same IP is the
    same visitor."""
    client = _client(_settings(migrated_db, rate_per_minute=100))
    ip = "203.0.113.20"
    _, resp = _ask(client, ip=ip)
    first = resp.headers["X-TMPBOT-Visitor-Id"]
    for _ in range(9):
        _ask(client, cookie=first, ip=ip)
    events, resp = _ask(client, ip=ip)  # cookie cleared
    assert events[1][1] == {"reason": "free_limit"}
    assert resp.headers["X-TMPBOT-Visitor-Id"] == first
    events, _ = _ask(client, ip=ip,
                     cookie="00000000-0000-4000-8000-00000000dead")
    assert events[1][1] == {"reason": "free_limit"}
    other, _ = _ask(client, ip="203.0.113.21")
    assert "token" in _names(other), "another address starts afresh"


def test_failed_answer_gives_the_question_back(migrated_db):
    """A writer failure before any text does not use a free question."""
    down = LLMError("upstream_unavailable", "down")
    # The writer and its fallback both fail.
    llm = ScriptedLLM([ScriptedReply(error=down),
                       ScriptedReply(error=down)])
    client = _client(_settings(migrated_db), llm)
    events, resp = _ask(client, ip="203.0.113.30")
    assert events[-1][0] == "error"
    cookie = resp.headers["X-TMPBOT-Visitor-Id"]
    with psycopg.connect(migrated_db) as conn:
        used = conn.execute(
            "SELECT questions_used FROM anonymous_visitors "
            "WHERE visitor_id = %s", (cookie,)).fetchone()[0]
    assert used == 0


def test_hundred_and_first_free_account_question_is_refused(migrated_db):
    """S18: 100 questions a day for a free account, then a refusal."""
    llm = ScriptedLLM()
    client = _client(_settings(migrated_db, rate_per_minute=1000,
                               rate_per_ip_per_minute=1000), llm)
    token = make_token(user_id=db_user(migrated_db))
    for k in range(100):
        events, _ = _ask(client, token=token, ip="203.0.113.40")
        assert events[-1][0] == "done", k
    events, _ = _ask(client, token=token, ip="203.0.113.40")
    assert _names(events) == ["meta", "error"]
    assert events[1][1]["code"] == "rate_limited"
    assert events[1][1]["detail"] == "fair_use_daily"
    assert len(llm.calls) == 100
    # Another free account is unaffected; a subscriber has no cap.
    other = make_token(user_id=db_user(migrated_db))
    assert _ask(client, token=other, ip="203.0.113.41")[0][-1][0] == "done"
    with psycopg.connect(migrated_db) as conn:
        today = day_window(datetime.now(timezone.utc))
        count = conn.execute(
            "SELECT count FROM request_rate_limits WHERE scope = "
            "'fair_use_daily' AND window_start_at = %s "
            "ORDER BY count DESC LIMIT 1", (today,)).fetchone()[0]
    assert count == 100


def test_subscribers_have_no_daily_cap(migrated_db):
    """The cap applies to free accounts only."""
    client = _client(_settings(migrated_db, fair_use_daily_cap=2,
                               rate_per_minute=100))
    token = make_token(plan="access", user_id=db_user(migrated_db))
    for _ in range(4):
        events, _ = _ask(client, token=token, ip="203.0.113.50")
        assert events[-1][0] == "done"


def test_burst_limit(migrated_db):
    """Questions per minute per visitor are capped."""
    client = _client(_settings(migrated_db, rate_per_minute=3))
    _, resp = _ask(client, ip="203.0.113.60")
    cookie = resp.headers["X-TMPBOT-Visitor-Id"]
    _ask(client, cookie=cookie, ip="203.0.113.60")
    _ask(client, cookie=cookie, ip="203.0.113.60")
    events, _ = _ask(client, cookie=cookie, ip="203.0.113.60")
    assert events[-1] == ("error", {"code": "rate_limited",
                                    "message": "Too many questions in a "
                                               "minute.",
                                    "detail": "burst"})


def test_turnstile_once_per_visitor(migrated_db):
    """The first question needs a valid token; later ones do not."""
    turnstile = FakeTurnstile(["good-token"])
    client = _client(_settings(migrated_db), turnstile=turnstile)
    events, resp = _ask(client, ip="203.0.113.70")
    assert events[-1][1]["detail"] == "turnstile_required"
    cookie = resp.headers["X-TMPBOT-Visitor-Id"]
    events, _ = _ask(client, cookie=cookie, ip="203.0.113.70",
                     turnstile="bad-token")
    assert events[-1][1]["detail"] == "turnstile_required"
    events, _ = _ask(client, cookie=cookie, ip="203.0.113.70",
                     turnstile="good-token")
    assert events[-1][0] == "done"
    events, _ = _ask(client, cookie=cookie, ip="203.0.113.70")
    assert events[-1][0] == "done"
    with psycopg.connect(migrated_db) as conn:
        row = conn.execute(
            "SELECT questions_used, turnstile_verified_at IS NOT NULL "
            "FROM anonymous_visitors WHERE visitor_id = %s",
            (cookie,)).fetchone()
    assert row == (2, True), "refused turns must not use questions"


def test_usage_log_and_conversation_rows(migrated_db):
    """One usage row per turn, no text in it; turns stored in order;
    a conversation id from another owner is not reused."""
    client = _client(_settings(migrated_db, rate_per_minute=100),
                     ScriptedLLM([ScriptedReply(text="First answer."),
                                  ScriptedReply(text="Second answer.")]))
    events, resp = _ask(client, ip="203.0.113.80", question="One?")
    cid = events[-1][1]["conversation_id"]
    cookie = resp.headers["X-TMPBOT-Visitor-Id"]
    events, _ = _ask(client, cookie=cookie, ip="203.0.113.80",
                     conversation_id=cid, question="Two?")
    assert events[-1][1]["conversation_id"] == cid
    with psycopg.connect(migrated_db) as conn:
        rows = conn.execute(
            "SELECT seq, role, content FROM messages "
            "WHERE conversation_id = %s ORDER BY seq", (cid,)).fetchall()
        usage = conn.execute(
            "SELECT plan, model, input_tokens, output_tokens, cost_usd, "
            "stop_reason, visitor_id::TEXT, conversation_id::TEXT "
            "FROM usage_log WHERE conversation_id = %s", (cid,)).fetchall()
        ip_hash = conn.execute(
            "SELECT ip_hash FROM anonymous_visitors WHERE visitor_id = %s",
            (cookie,)).fetchone()[0]
    assert rows == [(0, "user", "One?"), (1, "assistant", "First answer."),
                    (2, "user", "Two?"), (3, "assistant", "Second answer.")]
    assert len(usage) == 2
    plan, model, tin, tout, cost, stop, visitor, conv = usage[0]
    assert (plan, tin, tout, stop, visitor, conv) == (
        "anonymous", 100, 20, "end_turn", cookie, cid)
    assert model == "anthropic/claude-haiku-5-5" and float(cost) > 0
    assert ip_hash == hash_ip("203.0.113.80", SALT)
    assert "203.0.113.80" not in ip_hash
    # Someone else's conversation id starts a new conversation.
    events, _ = _ask(client, ip="203.0.113.81", conversation_id=cid)
    assert events[-1][1]["conversation_id"] != cid


@pytest.mark.parametrize("cookie", ["not-a-uuid", "", "' OR 1=1 --"])
def test_bad_cookie_values_are_harmless(migrated_db, cookie):
    """Odd cookie values make a new or IP-matched visitor, never an
    error."""
    client = _client(_settings(migrated_db, rate_per_minute=100))
    events, resp = _ask(client, cookie=cookie, ip="203.0.113.90")
    assert events[-1][0] == "done"
    assert len(resp.headers["X-TMPBOT-Visitor-Id"]) == 36
