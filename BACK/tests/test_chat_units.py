# Unit tests for the chat's building blocks: routing, cost, identity,
# Turnstile, catalog, embedding parity with TOOL/, prompt rendering,
# in-memory retrieval and stores, the admission checks, config checks,
# the LiteLLM chunk normaliser, the local stand-in writer, wire
# additions and the downloads route.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Unit tests for the Phase 5 backend modules."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from chat_helpers import TEST_SECRET, build_runtime, make_token
from fastapi.testclient import TestClient

from tmpbot_backend import wire
from tmpbot_backend.catalog import load_catalog
from tmpbot_backend.config import (
    REPO_ROOT,
    ConfigError,
    Settings,
    load_settings,
)
from tmpbot_backend.cost import TurnUsage, turn_cost_usd
from tmpbot_backend.embedding import FakeEmbedder, OpenAIQueryEmbedder
from tmpbot_backend.gatekeeper import admit
from tmpbot_backend.identity import (
    ANONYMOUS,
    NoTokenVerifier,
    StaticKeyVerifier,
    resolve_identity,
)
from tmpbot_backend.llm import LocalFakeLLM, extract_delta
from tmpbot_backend.main import create_app
from tmpbot_backend.prompt import (
    load_layers,
    reference_block,
    system_message,
)
from tmpbot_backend.retrieval import (
    MemoryRetriever,
    Passage,
    RetrievalBundle,
    Teaser,
    enforce_tiers,
    merge_passages,
)
from tmpbot_backend.routing import RoutingError, load_policy, parse_policy
from tmpbot_backend.stores import (
    MemoryRateLimiter,
    MemoryVisitorStore,
    day_window,
    hash_ip,
)
from tmpbot_backend.turnstile import (
    CloudflareTurnstile,
    DisabledTurnstile,
    FakeTurnstile,
)

KEY = "test-jwt-signing-key-0123456789abcdef"
POLICY = load_policy(REPO_ROOT / "BOTS" / "POLC" / "llm_routing.yaml")
FREE = resolve_identity(f"Bearer {make_token()}", StaticKeyVerifier(KEY))
ACCESS = resolve_identity(f"Bearer {make_token(plan='access')}",
                          StaticKeyVerifier(KEY))


def run(coro):
    """Run a coroutine to completion.

    Args:
        coro: The coroutine.

    Returns:
        Any: Its result.
    """
    return asyncio.run(coro)


# ---------------------------------------------------------------- routing


def test_routing_file_resolves_every_locale():
    """French on Mistral, everything else on the default, all priced."""
    assert POLICY.model_for("fr").startswith("mistral/")
    for lang in ("en", "ja", "es", "pt", "de", "it"):
        assert POLICY.model_for(lang) == POLICY.locale_models["*"]
    for lang in ("en", "fr"):
        assert POLICY.price_for(POLICY.model_for(lang)) is not None
    assert POLICY.price_for(POLICY.fallback_model) is not None
    assert POLICY.max_tokens == 400 and POLICY.history_messages == 12


@pytest.mark.parametrize("data", [
    {}, {"defaults": {"locales": {"en": "a/b"}}},
    {"defaults": {"locales": {"*": "a/b", "xx": "c/d"}}},
    {"defaults": {"locales": {"*": "a/b"}, "temperature": "hot"}},
    {"defaults": {"locales": {"*": "a/b"}},
     "prices": {"a/b": {"input": -1, "output": 1}}},
])
def test_routing_rejects_bad_policies(data):
    """Missing default, unknown locale, bad numbers fail loudly."""
    with pytest.raises(RoutingError):
        parse_policy(data)


def test_price_lookup_accepts_bare_ids():
    """A bare model id finds the prefixed price row."""
    assert POLICY.price_for("claude-haiku-5-5") == POLICY.price_for(
        "anthropic/claude-haiku-5-5")
    assert POLICY.price_for("no/such-model") is None


# ------------------------------------------------------------------- cost


def test_turn_cost_includes_cache_tokens():
    """Input, output, cache writes (1.25x) and reads (0.1x) all count."""
    usage = TurnUsage(input_tokens=1_000_000, output_tokens=1_000_000,
                      cache_read_tokens=1_000_000,
                      cache_write_tokens=1_000_000)
    cost = turn_cost_usd(POLICY, "anthropic/claude-haiku-5-5", usage)
    assert cost == pytest.approx(0.10 + 0.50 + 0.125 + 0.01)


def test_unpriced_model_costs_zero():
    """An unknown model logs a warning and costs nothing."""
    assert turn_cost_usd(POLICY, "x/y", TurnUsage(10, 10)) == 0.0


# --------------------------------------------------------------- identity


def test_identity_from_valid_tokens():
    """Free and subscriber tokens resolve to their plans."""
    assert FREE.plan == "free" and FREE.signed_in
    assert ACCESS.plan == "access" and ACCESS.subscriber


@pytest.mark.parametrize("header", [
    None, "", "Basic abc", "Bearer", "Bearer not.a.jwt",
    f"Bearer {make_token(plan='access', key='wrong-key-000000000000')}",
    f"Bearer {make_token(plan='access', expires_in=-10)}",
])
def test_bad_tokens_mean_anonymous(header):
    """Anything unproven is anonymous, never more."""
    assert resolve_identity(header, StaticKeyVerifier(KEY)) == ANONYMOUS


def test_non_uuid_subject_or_role_is_anonymous():
    """A token for a non-user role or odd subject gives no access."""
    import time

    import jwt
    base = {"aud": "authenticated", "exp": int(time.time()) + 60,
            "plan": "access"}
    for extra in ({"sub": "admin"},
                  {"sub": "00000000-0000-4000-8000-000000000001",
                   "role": "service_role"}):
        token = jwt.encode({**base, **extra}, KEY, algorithm="HS256")
        assert resolve_identity(f"Bearer {token}",
                                StaticKeyVerifier(KEY)) == ANONYMOUS


def test_no_token_verifier_accepts_nothing():
    """Before sign-in exists every caller is anonymous."""
    token = make_token(plan="access")
    assert resolve_identity(f"Bearer {token}", NoTokenVerifier()) == \
        ANONYMOUS


# -------------------------------------------------------------- turnstile


def _cf(body, status=200):
    """A Cloudflare verifier over a mocked HTTP transport.

    Args:
        body: JSON body to return.
        status: HTTP status.

    Returns:
        tuple[CloudflareTurnstile, list]: Verifier and captured requests.
    """
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CloudflareTurnstile("secret-x", client), seen


def test_cloudflare_turnstile_success_and_form():
    """A success reply passes; secret, token and IP are posted."""
    verifier, seen = _cf({"success": True})
    assert run(verifier.verify("tok", "192.0.2.9")).ok
    form = seen[0].content.decode()
    assert "secret=secret-x" in form and "response=tok" in form
    assert "remoteip=192.0.2.9" in form


@pytest.mark.parametrize("body,status,token", [
    ({"success": False, "error-codes": ["invalid-input-response"]}, 200,
     "tok"),
    ({}, 500, "tok"),
    ({"success": True}, 200, ""),
    ({"success": True}, 200, "x" * 3000),
])
def test_cloudflare_turnstile_failures(body, status, token):
    """Rejections, server errors, missing and huge tokens fail."""
    verifier, _ = _cf(body, status)
    assert not run(verifier.verify(token, None)).ok


def test_fake_and_disabled_turnstile():
    """The fake passes listed tokens; disabled is not enforced."""
    fake = FakeTurnstile(["ok"])
    assert run(fake.verify("ok", None)).ok
    assert not run(fake.verify("no", None)).ok
    assert fake.enforced and not DisabledTurnstile.enforced


# ---------------------------------------------------------------- catalog


def test_catalog_has_published_case_studies_only(tmp_path):
    """Active case studies count; drafts and other types do not."""
    catalog = load_catalog(REPO_ROOT / "COPS")
    assert len(catalog.case_studies) == 16
    assert catalog.is_case_study("el-nino-risk-analysis")
    assert catalog.is_case_study("dryad")
    assert not catalog.is_case_study("Dryad")
    assert not catalog.is_case_study(["dryad"])
    (tmp_path / "CASE").mkdir()
    (tmp_path / "PUBL").mkdir()
    (tmp_path / "CASE" / "x_en.md").write_text(
        '---\nid: "x"\ntype: "case_study"\nsource_lifecycle: "draft"\n'
        'year: 2020\n---\nBody', encoding="utf-8")
    (tmp_path / "PUBL" / "p_en.md").write_text(
        '---\nid: "p"\ntype: "publication"\nlanguage: "en"\n'
        'title: "P"\nyear: 2025\nsource_lifecycle: "active"\n---\nB',
        encoding="utf-8")
    (tmp_path / "PUBL" / "p_ja.md").write_text(
        '---\nid: "p"\ntype: "publication"\nlanguage: "ja"\n'
        'title: "ピー"\nyear: 2025\nsource_lifecycle: "draft"\n---\nB',
        encoding="utf-8")
    small = load_catalog(tmp_path)
    assert not small.case_studies
    pub = small.publication("p")
    assert pub.title_in("ja") == "P", "draft titles are not used"
    assert small.publication("q") is None


# -------------------------------------------------------------- embedding


def test_fake_embedder_matches_the_index_tool():
    """The backend's fake embeds exactly as TOOL/embedding.py does, so a
    corpus indexed by the tool is searchable here."""
    spec = importlib.util.spec_from_file_location(
        "tool_embedding_for_parity", REPO_ROOT / "TOOL" / "embedding.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    texts = ["Lithium supply chains in Côte d'Ivoire", "土地の権利について",
             "", "Forest Carbon Project 2009"]
    ours = FakeEmbedder()
    theirs = tool.FakeEmbedder()
    assert ours.model == theirs.model
    for text in texts:
        assert run(ours.embed_query(text)) == theirs.embed([text])[0]


def test_openai_query_embedder_over_mock_transport():
    """The real embedder posts the model and checks the dimensions."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"data": [{"embedding": [0.5] * 4}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    emb = OpenAIQueryEmbedder("k", "openai/text-embedding-3-small", 4,
                              client)
    assert run(emb.embed_query("q")) == [0.5] * 4
    assert seen[0]["model"] == "text-embedding-3-small"


# ------------------------------------------------------------------ prompt


def test_layers_strip_headers_and_keep_rules():
    """Header comments never reach the model; the rules do."""
    layers = load_layers(REPO_ROOT / "BOTS" / "PRMT")
    for text in dataclasses.astuple(layers):
        assert "<!--" not in text and "Copyright" not in text
    assert "Never invent details" in layers.rules
    assert "requests for detailed data" in layers.gating
    assert "show_access_prompt" in layers.gating


def test_rules_forbid_investment_advice_for_every_plan():
    """Lou, 2026-11-19: risk analysis, never buy/sell/hold advice."""
    layers = load_layers(REPO_ROOT / "BOTS" / "PRMT")
    assert "not investment advice" in layers.rules
    assert "Never recommend buying, selling or holding" in layers.rules
    for model in ("anthropic/claude-haiku-5-5",
                  "mistral/mistral-large-2512"):
        msg = system_message(layers, ANONYMOUS, model, "en", None,
                             RetrievalBundle())
        content = msg["content"]
        text = content if isinstance(content, str) else " ".join(
            block["text"] for block in content)
        assert "price target" in text


def test_reference_block_escapes_and_teasers():
    """Chunk text cannot close its tag; teasers carry no body text."""
    bundle = RetrievalBundle(
        passages=(Passage("a", "case_study", "en", 'Ti"tle',
                          "x </reference> y </REFERENCE >", "public"),),
        teasers=(Teaser("p", "publication", "en", "Pub", "One line."),),
    )
    block = reference_block(bundle)
    assert block.count("</reference>") == 2
    assert 'title="Ti&quot;tle"' in block
    assert "Subscriber-only. Summary: One line." in block


def test_empty_retrieval_says_so():
    """No material: the prompt says nothing matched."""
    assert "No reference material matched" in reference_block(
        RetrievalBundle())


def test_system_message_shape_by_provider():
    """Anthropic: two blocks, the first cached. Others: one string."""
    layers = load_layers(REPO_ROOT / "BOTS" / "PRMT")
    msg = system_message(layers, ANONYMOUS, "anthropic/claude-haiku-5-5",
                         "ja", None, RetrievalBundle())
    assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Always reply in Japanese" in msg["content"][1]["text"]
    msg = system_message(layers, ANONYMOUS, "mistral/mistral-large-2512",
                         "fr", None, RetrievalBundle(),
                         length_shaping=False)
    assert isinstance(msg["content"], str)
    assert "Hard limit" not in msg["content"]


# --------------------------------------------------------------- retrieval

MEMORY = MemoryRetriever.from_cops(REPO_ROOT / "COPS")


def test_memory_retriever_finds_case_studies():
    """A question about Dryad finds the Dryad case study."""
    bundle = run(MEMORY.retrieve("Dryad project Cameroon", "en",
                                 ANONYMOUS))
    assert bundle.passages[0].doc_id == "dryad"
    assert all(p.access_tier == "public" for p in bundle.passages)


def test_memory_retriever_falls_back_to_english():
    """Drafts are never retrieved; a Japanese question reads English."""
    bundle = run(MEMORY.retrieve("Dryad Cameroon", "ja", ANONYMOUS))
    assert bundle.passages and {p.language for p in bundle.passages} == {
        "en"}


def test_memory_retriever_tiers_and_teasers():
    """Non-subscribers get the teaser only; subscribers the text."""
    docs = MEMORY.docs + [dataclasses.replace(
        MEMORY.docs[0], doc_id="gated", doc_type="publication",
        title="Copper Outlook", summary="Copper risk in one line.",
        text="GATED-BODY copper ranking", access_tier="subscriber",
        words=frozenset({"copper", "outlook", "ranking", "gated", "body"}),
        language="en")]
    retriever = MemoryRetriever(docs)
    for who in (ANONYMOUS, FREE):
        bundle = run(retriever.retrieve("copper outlook ranking", "en",
                                        who))
        assert "GATED-BODY" not in repr(bundle.passages)
        assert [t.doc_id for t in bundle.teasers] == ["gated"]
    bundle = run(retriever.retrieve("copper outlook ranking", "en",
                                    ACCESS))
    assert bundle.passages[0].doc_id == "gated" and not bundle.teasers


def test_enforce_tiers_is_the_last_line():
    """A subscriber passage for a non-subscriber is dropped."""
    gated = Passage("g", "publication", "en", "G", "secret", "subscriber")
    public = Passage("p", "about", "en", "P", "open", "public")
    assert enforce_tiers([gated, public], FREE) == [public]
    assert enforce_tiers([gated, public], ACCESS) == [gated, public]


def test_merge_prefers_the_visitor_language():
    """One passage language per document, the visitor's first."""
    en = Passage("d", "about", "en", "D", "en text", "public", 0.9)
    ja = Passage("d", "about", "ja", "D", "ja text", "public", 0.5)
    assert merge_passages([en, ja], "ja") == [ja]
    assert merge_passages([en, ja], "en") == [en]


# ------------------------------------------------------------------ stores


def test_memory_visitor_store_matches_by_ip_within_window():
    """Cookie first; else the busiest recent visitor from the IP."""
    store = MemoryVisitorStore()
    a = store.resolve(None, "h1", 7)
    store.consume_question(a.visitor_id, 10)
    assert store.resolve(None, "h1", 7).visitor_id == a.visitor_id
    assert store.resolve(a.visitor_id, "h2", 7).visitor_id == a.visitor_id
    store.visitors[a.visitor_id].last_seen -= timedelta(days=8)
    assert store.resolve(None, "h2", 7).visitor_id != a.visitor_id


def test_memory_rate_limiter_windows():
    """A window fills; the next window starts empty."""
    limiter = MemoryRateLimiter()
    day = day_window(datetime(2026, 11, 14, 23, 59, tzinfo=timezone.utc))
    assert [limiter.hit("s", "k", day, 2) for _ in range(3)] == [
        True, True, False]
    assert limiter.hit("s", "k", day + timedelta(days=1), 2)


def test_hash_ip_is_salted_and_hex():
    """Different salts, different hashes; 64 hex digits."""
    a, b = hash_ip("192.0.2.1", "salt-a"), hash_ip("192.0.2.1", "salt-b")
    assert a != b and len(a) == 64 and int(a, 16) >= 0


def test_day_window_is_midnight_jst():
    """The fair-use day runs midnight to midnight Japan time."""
    jst = timezone(timedelta(hours=9))
    # 08:00 JST and 23:59 UTC the day before fall in the same JST day.
    assert day_window(datetime(2026, 11, 15, 8, 0, tzinfo=jst)) == \
        datetime(2026, 11, 15, tzinfo=jst)
    assert day_window(datetime(2026, 11, 14, 23, 59, tzinfo=timezone.utc)) \
        == datetime(2026, 11, 15, tzinfo=jst)
    # 14:59 UTC is still the previous JST day; 15:00 UTC starts a new one.
    assert day_window(datetime(2026, 11, 14, 14, 59, tzinfo=timezone.utc)) \
        == datetime(2026, 11, 14, tzinfo=jst)
    assert day_window(datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc)) \
        == datetime(2026, 11, 15, tzinfo=jst)


class _BrokenStore:
    """A visitor store whose database is down."""

    def resolve(self, *args):
        raise OSError("db down")


def test_admission_fails_closed_when_the_store_fails(settings):
    """A store failure refuses the turn instead of serving it uncounted."""
    out = run(admit(settings=settings, identity=ANONYMOUS,
                    visitors=_BrokenStore(), limiter=MemoryRateLimiter(),
                    turnstile=DisabledTurnstile(), cookie_id=None,
                    client_ip="192.0.2.1", turnstile_token=None))
    assert out.refusal == "unavailable"


def test_unavailable_refusal_streams_an_error(settings):
    """The refusal becomes an upstream_unavailable error event."""
    runtime = build_runtime(settings, visitors=_BrokenStore())
    client = TestClient(create_app(settings, runtime))
    resp = client.post("/api/v1/chat", json={
        "language": "en", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-TMPBOT-Proxy-Secret": TEST_SECRET})
    events = wire.parse_sse(resp.text)
    assert [n for n, _ in events] == ["meta", "error"]
    assert events[1][1]["code"] == "upstream_unavailable"


# ------------------------------------------------------------------ config


def test_production_refuses_unsafe_settings():
    """Production needs a database, real writer, salt, Turnstile and,
    with accounts on, sign-in checks."""
    with pytest.raises(ConfigError) as err:
        Settings(env="production", database_url="postgres://x",
                 llm_provider="fake", accounts_enabled=True).check()
    message = str(err.value)
    for name in ("TMPBOT_LLM_PROVIDER=litellm", "TMPBOT_EMBEDDER=openai",
                 "TMPBOT_IP_HASH_SALT", "TURNSTILE_SECRET_KEY",
                 "SUPABASE_URL"):
        assert name in message
    with pytest.raises(ConfigError):
        Settings(env="staging").check()
    Settings(env="local").check()


def test_production_settings_pass_when_complete():
    """A complete production environment loads and checks."""
    s = load_settings({
        "TMPBOT_ENV": "production", "DATABASE_URL": "postgres://x",
        "OPENAI_API_KEY": "k", "TMPBOT_IP_HASH_SALT": "s" * 32,
        "TURNSTILE_SECRET_KEY": "t", "SUPABASE_URL": "https://p.supa/",
        "TMPBOT_FAIR_USE_DAILY_CAP": "150",
    })
    s.check()
    assert (s.llm_provider, s.embedder) == ("litellm", "openai")
    assert s.fair_use_daily_cap == 150
    assert s.supabase_url == "https://p.supa"


@pytest.mark.parametrize("env", [
    {"TMPBOT_FAIR_USE_DAILY_CAP": "lots"},
    {"TMPBOT_LLM_PROVIDER": "openai"},
])
def test_bad_limit_and_provider_values(env):
    """Typos fail at start-up."""
    with pytest.raises(ConfigError):
        load_settings(env).check()


# --------------------------------------------------------------- LLM layer


def test_extract_delta_normalises_litellm_chunks():
    """Text, tool pieces, finish reason and usage come through; the
    LiteLLM filler delta is dropped."""
    chunk = {"choices": [{"delta": {"content": "Hi", "tool_calls": [
        {"index": 0, "id": "c1", "function": {"name": "show_case_studies",
                                              "arguments": '{"ids"'}},
        {"index": 0, "id": None, "function": {"name": None,
                                              "arguments": "{}"}},
    ]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                  "cache_read_input_tokens": 7}}
    delta = extract_delta(chunk)
    assert delta.text == "Hi" and len(delta.tool_calls) == 1
    assert delta.tool_calls[0].name == "show_case_studies"
    assert delta.finish_reason == "tool_calls"
    assert delta.usage == TurnUsage(5, 2, 7, 0)
    assert extract_delta({"choices": []}).text == ""


def test_local_fake_writer_uses_reference_ids(settings):
    """The keyless local writer labels itself and shows real cards."""
    runtime = build_runtime(settings, LocalFakeLLM())
    client = TestClient(create_app(settings, runtime))
    resp = client.post("/api/v1/chat", json={
        "language": "en",
        "messages": [{"role": "user", "content": "Dryad Cameroon forest"}]},
        headers={"X-TMPBOT-Proxy-Secret": TEST_SECRET})
    events = wire.parse_sse(resp.text)
    text = "".join(p["content"] for n, p in events if n == "token")
    assert text.startswith("This is a local test answer, not a model.")
    cases = [p for n, p in events if n == "cases"]
    assert cases and "dryad" in cases[0]["ids"]
    resp = client.post("/api/v1/chat", json={
        "language": "en",
        "messages": [{"role": "user",
                      "content": "Send me the forecast report PDF"}]},
        headers={"X-TMPBOT-Proxy-Secret": TEST_SECRET})
    assert ("gate", {"reason": "subscriber_only"}) in wire.parse_sse(
        resp.text)


# -------------------------------------------------------------------- wire


def test_wire_optional_fields():
    """error.detail and done.conversation_id are optional and checked."""
    rec = wire.parse_sse(wire.event_error("rate_limited", "m",
                                          "fair_use_daily"))
    assert rec[0][1]["detail"] == "fair_use_daily"
    with pytest.raises(ValueError):
        wire.event_error("rate_limited", "m", "made_up")
    done = wire.parse_sse(wire.event_done("end_turn", 1, 2, "cid"))[0][1]
    assert done["conversation_id"] == "cid"
    assert "conversation_id" not in wire.parse_sse(
        wire.event_done("end_turn"))[0][1]


# --------------------------------------------------------------- downloads


@pytest.mark.parametrize("token,pid,status", [
    (None, "anything", 403),
    ("free", "anything", 403),
    ("access", "no-such-publication", 404),
])
def test_download_route_checks_plan_and_id(client, token, pid, status):
    """Downloads need TMP Access now and a published id."""
    headers = {"X-TMPBOT-Proxy-Secret": TEST_SECRET}
    if token:
        plan = "access" if token == "access" else None
        headers["Authorization"] = f"Bearer {make_token(plan=plan)}"
    resp = client.post(f"/api/v1/downloads/{pid}", headers=headers)
    assert resp.status_code == status


def test_download_route_not_configured_yet(settings, tmp_path):
    """A published id for a subscriber: 503 until Storage exists."""
    (tmp_path / "CASE").mkdir()
    (tmp_path / "PUBL").mkdir()
    (tmp_path / "PUBL" / "p_en.md").write_text(
        '---\nid: "p"\ntype: "publication"\nlanguage: "en"\n'
        'title: "P"\nyear: 2025\nsource_lifecycle: "active"\n---\nB',
        encoding="utf-8")
    runtime = build_runtime(settings, cops_dir=tmp_path)
    client = TestClient(create_app(settings, runtime))
    resp = client.post("/api/v1/downloads/p", headers={
        "X-TMPBOT-Proxy-Secret": TEST_SECRET,
        "Authorization": f"Bearer {make_token(plan='access')}"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "downloads_not_configured"
    resp = client.post("/api/v1/downloads/p")
    assert resp.status_code == 401, "the proxy secret comes first"
