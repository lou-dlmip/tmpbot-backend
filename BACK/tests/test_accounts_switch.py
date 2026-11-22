# The accounts launch switch and attaching signed-out conversations at
# sign-in (M7.1, M5.7; launch phase L2).
#
#   * TMPBOT_ACCOUNTS_ENABLED: accepted words, the free-question default
#     it implies (10 on, 20 off), production's SUPABASE_URL rule, and the
#     verifier the runtime picks (off: every token is ignored);
#   * the in-memory conversation store's claim;
#   * the chat route attaches the visitor's conversations when a signed-in
#     turn still carries the visitor cookie, and the conversation goes on;
#   * POST /api/v1/claim: proxy secret, sign-in required, count returned;
#   * the Postgres store's claim on a real database.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Accounts switch and conversation claim."""

from __future__ import annotations

import dataclasses

import psycopg
import pytest
from chat_helpers import TEST_SECRET, build_runtime, db_user, make_token
from fastapi.testclient import TestClient

from tmpbot_backend import wire
from tmpbot_backend.config import ConfigError, Settings, load_settings
from tmpbot_backend.db import Database
from tmpbot_backend.identity import (
    JwksVerifier,
    NoTokenVerifier,
    resolve_identity,
)
from tmpbot_backend.llm import ScriptedLLM, ScriptedReply
from tmpbot_backend.main import create_app
from tmpbot_backend.runtime import build_runtime as real_build_runtime
from tmpbot_backend.stores import (
    MemoryConversationStore,
    Owner,
    PgConversationStore,
    StoredTurn,
)

PROD = {
    "TMPBOT_ENV": "production",
    "DATABASE_URL": "postgresql://x@localhost/x",
    "TMPBOT_LLM_PROVIDER": "litellm",
    "TMPBOT_EMBEDDER": "openai",
    "OPENAI_API_KEY": "sk-test",
    "TMPBOT_IP_HASH_SALT": "0123456789abcdef0123",
    "TURNSTILE_SECRET_KEY": "secret",
}


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (None, False), ("", False), ("0", False), ("false", False),
    ("No", False), ("off", False), ("1", True), ("true", True),
    ("YES", True), (" on ", True),
])
def test_switch_words(raw, expected):
    env = {} if raw is None else {"TMPBOT_ACCOUNTS_ENABLED": raw}
    assert load_settings(env).accounts_enabled is expected


@pytest.mark.parametrize("raw", ["enabled", "2", "tru"])
def test_switch_typo_is_an_error(raw):
    with pytest.raises(ConfigError):
        load_settings({"TMPBOT_ACCOUNTS_ENABLED": raw})


def test_free_limit_default_follows_switch():
    assert load_settings({}).free_question_limit == 20
    assert load_settings(
        {"TMPBOT_ACCOUNTS_ENABLED": "true"}).free_question_limit == 10


@pytest.mark.parametrize("switch", ["true", "false"])
def test_explicit_free_limit_wins(switch):
    s = load_settings({"TMPBOT_ACCOUNTS_ENABLED": switch,
                       "TMPBOT_FREE_QUESTION_LIMIT": "7"})
    assert s.free_question_limit == 7


def test_production_needs_supabase_only_with_accounts_on():
    load_settings(PROD).check()
    with pytest.raises(ConfigError, match="SUPABASE_URL"):
        load_settings({**PROD, "TMPBOT_ACCOUNTS_ENABLED": "1"}).check()
    load_settings({**PROD, "TMPBOT_ACCOUNTS_ENABLED": "1",
                   "SUPABASE_URL": "https://abc.supabase.co"}).check()


def test_runtime_ignores_tokens_with_accounts_off():
    s = Settings(supabase_url="https://abc.supabase.co",
                 accounts_enabled=False)
    runtime = real_build_runtime(s)
    assert isinstance(runtime.verifier, NoTokenVerifier)
    identity = resolve_identity(f"Bearer {make_token()}", runtime.verifier)
    assert not identity.signed_in


def test_runtime_checks_tokens_with_accounts_on():
    s = Settings(supabase_url="https://abc.supabase.co",
                 accounts_enabled=True)
    assert isinstance(real_build_runtime(s).verifier, JwksVerifier)


def test_runtime_without_supabase_url_is_signed_out_even_when_on():
    s = Settings(accounts_enabled=True)
    assert isinstance(real_build_runtime(s).verifier, NoTokenVerifier)


# ----------------------------------------------------------------------
# Memory store
# ----------------------------------------------------------------------
def _turn(text="Hi"):
    return StoredTurn(language="en", audience=None, question=text,
                      answer="Answer.", wire_message_id="msg-1")


def test_memory_claim_moves_only_that_visitors_threads():
    store = MemoryConversationStore()
    mine = store.save_turn(Owner(visitor_id="v1"), None, _turn())
    other = store.save_turn(Owner(visitor_id="v2"), None, _turn())
    assert store.claim("v1", "u1") == 1
    assert store.owners[mine] == Owner(user_id="u1")
    assert store.owners[other] == Owner(visitor_id="v2")
    assert store.claim("v1", "u1") == 0


# ----------------------------------------------------------------------
# Routes, in memory
# ----------------------------------------------------------------------
def _client(settings, llm=None):
    runtime = build_runtime(settings, llm or ScriptedLLM())
    return TestClient(create_app(settings, runtime)), runtime


def _ask(client, *, cookie=None, token=None, conversation_id=None):
    headers = {"X-TMPBOT-Proxy-Secret": TEST_SECRET,
               "X-TMPBOT-Client-IP": "203.0.113.20"}
    if cookie:
        headers["X-TMPBOT-Visitor-Id"] = cookie
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"language": "en",
            "messages": [{"role": "user", "content": "Hello"}]}
    if conversation_id:
        body["conversation_id"] = conversation_id
    resp = client.post("/api/v1/chat", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return wire.parse_sse(resp.text), resp


def test_sign_in_continues_the_signed_out_conversation(settings):
    client, runtime = _client(
        dataclasses.replace(settings, rate_per_minute=100),
        ScriptedLLM([ScriptedReply(text="One."), ScriptedReply(text="Two.")]))
    events, resp = _ask(client)
    cid = events[-1][1]["conversation_id"]
    cookie = resp.headers["X-TMPBOT-Visitor-Id"]
    user = "11111111-2222-4333-8444-555555555555"
    events, _ = _ask(client, cookie=cookie, token=make_token(user_id=user),
                     conversation_id=cid)
    assert events[-1][1]["conversation_id"] == cid
    store = runtime.deps.conversations
    assert store.owners[cid] == Owner(user_id=user)
    assert [m["content"] for m in store.messages[cid]] == [
        "Hello", "One.", "Hello", "Two."]


def test_claim_endpoint(settings):
    client, runtime = _client(settings)
    _, resp = _ask(client)
    cookie = resp.headers["X-TMPBOT-Visitor-Id"]
    base = {"X-TMPBOT-Proxy-Secret": TEST_SECRET,
            "X-TMPBOT-Visitor-Id": cookie}
    assert client.post("/api/v1/claim", headers={
        "X-TMPBOT-Visitor-Id": cookie,
        "Authorization": f"Bearer {make_token()}"}).status_code == 401
    assert client.post("/api/v1/claim", headers=base).status_code == 401
    ok = client.post("/api/v1/claim", headers={
        **base, "Authorization": f"Bearer {make_token()}"})
    assert ok.status_code == 200 and ok.json() == {"claimed": 1}
    again = client.post("/api/v1/claim", headers={
        **base, "Authorization": f"Bearer {make_token()}"})
    assert again.json() == {"claimed": 0}


def test_claim_without_visitor_cookie_is_zero(settings):
    client, _ = _client(settings)
    resp = client.post("/api/v1/claim", headers={
        "X-TMPBOT-Proxy-Secret": TEST_SECRET,
        "Authorization": f"Bearer {make_token()}"})
    assert resp.json() == {"claimed": 0}


def test_claim_failure_never_blocks_the_turn(settings):
    class Broken(MemoryConversationStore):
        def claim(self, visitor_id, user_id):
            raise RuntimeError("database down")

    runtime = build_runtime(settings, ScriptedLLM(),
                            conversations=Broken())
    client = TestClient(create_app(settings, runtime))
    events, _ = _ask(client, cookie="11111111-2222-4333-8444-555555555555",
                     token=make_token())
    assert events[-1][0] == "done"


# ----------------------------------------------------------------------
# Postgres store
# ----------------------------------------------------------------------
def test_pg_claim(migrated_db):
    db = Database(migrated_db)
    store = PgConversationStore(db)
    with psycopg.connect(migrated_db, autocommit=True) as conn:
        vid = str(conn.execute(
            "INSERT INTO anonymous_visitors (ip_hash) VALUES (%s) "
            "RETURNING visitor_id", ("ab" * 32,)).fetchone()[0])
    cid = store.save_turn(Owner(visitor_id=vid), None, _turn())
    user = db_user(migrated_db)
    assert store.claim(vid, user) == 1
    assert store.claim(vid, user) == 0
    assert store.claim("not-a-uuid", user) == 0
    with psycopg.connect(migrated_db) as conn:
        owner = conn.execute(
            "SELECT user_id::TEXT, visitor_id FROM conversations "
            "WHERE conversation_id = %s", (cid,)).fetchone()
        claimed = conn.execute(
            "SELECT claimed_by::TEXT FROM anonymous_visitors "
            "WHERE visitor_id = %s", (vid,)).fetchone()[0]
    assert owner == (user, None)
    assert claimed == user
