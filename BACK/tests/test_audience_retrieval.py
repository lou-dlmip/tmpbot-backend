# Audience pages (COPS/AUDI) in retrieval, without a database.
#
# The "Asking as" audience reaches the retriever on every search of a
# turn; a visitor with an audience reads their own audience's pages and
# never another audience's; with no audience (or "just curious") at most
# one page per topic is kept, so the five versions of a topic cannot
# crowd case studies out. The rules never widen the paywall, the pages
# never become case-study cards, and the prompt names whose page it is.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Audience-aware retrieval over the in-memory COPS retriever."""

from __future__ import annotations

import asyncio
import dataclasses
import typing

import pytest
from chat_helpers import TEST_SECRET, build_runtime, make_token
from fastapi.testclient import TestClient

from tmpbot_backend import wire
from tmpbot_backend.catalog import load_catalog
from tmpbot_backend.config import REPO_ROOT
from tmpbot_backend.identity import (
    ANONYMOUS,
    StaticKeyVerifier,
    resolve_identity,
)
from tmpbot_backend.llm import ScriptedLLM, ScriptedReply
from tmpbot_backend.main import ChatRequest, create_app
from tmpbot_backend.orchestrator import retrieval_query, retrieve_for_turn
from tmpbot_backend.prompt import AUDIENCES, reference_block
from tmpbot_backend.retrieval import (
    AUDIENCE_BOOST,
    AUDIENCE_PAGE_KEYS,
    AUDIENCE_PAGE_LIMIT,
    MemoryRetriever,
    Passage,
    RetrievalBundle,
    apply_audience,
    page_audience,
)
from tmpbot_backend.tools import CardCollector

COPS = REPO_ROOT / "COPS"
MEMORY = MemoryRetriever.from_cops(COPS)
KEY = "test-jwt-signing-key-0123456789abcdef"
FREE = resolve_identity(f"Bearer {make_token()}", StaticKeyVerifier(KEY))
ACCESS = resolve_identity(f"Bearer {make_token(plan='access')}",
                          StaticKeyVerifier(KEY))
HEADERS = {"X-TMPBOT-Proxy-Secret": TEST_SECRET,
           "X-Forwarded-For": "203.0.113.50"}
DEBT = "What does TMP do on sovereign debt?"


def run(coro):
    """Run a coroutine to completion.

    Args:
        coro: The coroutine.

    Returns:
        Any: Its result.
    """
    return asyncio.run(coro)


def _page_ids(bundle):
    """Doc ids of the audience pages in a bundle.

    Args:
        bundle: A RetrievalBundle.

    Returns:
        list[str]: Ids, in passage order.
    """
    return [p.doc_id for p in bundle.passages if p.audience is not None]


def _passage(doc_id, score, audience=None, topic=None, chunk=0):
    """A small passage for the unit tests.

    Args:
        doc_id: Document id.
        score: Score.
        audience: Audience key for an audience page.
        topic: Topic id for an audience page.
        chunk: Distinguishes chunks of one document.

    Returns:
        Passage: The passage.
    """
    return Passage(doc_id, "audience_page" if audience else "case_study",
                   "en", doc_id, f"{doc_id} text {chunk}", "public", score,
                   None, audience, topic)


# ------------------------------------------------------------ the keys


def test_page_keys_are_the_backend_audiences():
    """Audience pages exist for every picker key except just_curious,
    and the request model accepts exactly the picker keys."""
    assert set(AUDIENCE_PAGE_KEYS) == set(AUDIENCES) - {"just_curious"}
    literal = typing.get_args(
        ChatRequest.model_fields["audience"].annotation)[0]
    assert set(typing.get_args(literal)) == set(AUDIENCES)


@pytest.mark.parametrize("audience, expected", [
    ("investor", "investor"), ("non_profit", "non_profit"),
    ("just_curious", None), (None, None), ("nonprofit", None), ("", None),
])
def test_page_audience(audience, expected):
    """Only real page audiences select pages."""
    assert page_audience(audience) == expected


def test_real_cops_has_every_audience_page_loaded():
    """The in-memory retriever loads all 24 active audience pages."""
    pages = [d for d in MEMORY.docs if d.doc_type == "audience_page"]
    assert len(pages) == 24
    assert {d.audience for d in pages} == set(AUDIENCE_PAGE_KEYS)
    assert all(d.language == "en" and d.access_tier == "public"
               for d in pages)
    assert all(d.doc_id == f"{d.audience.replace('_', '-')}-{d.topic}"
               for d in pages)


# ----------------------------------------------------- apply_audience


def test_with_an_audience_other_pages_go_and_own_pages_rise():
    """Another audience's page is dropped; the visitor's page is boosted
    above a slightly better case study; plain documents are kept."""
    passages = [
        _passage("case-a", 0.030),
        _passage("investor-debt", 0.029, "investor", "debt"),
        _passage("government-debt", 0.031, "government", "debt"),
    ]
    kept = apply_audience(passages, "investor")
    assert [p.doc_id for p in kept] == ["investor-debt", "case-a"]
    assert kept[0].score == pytest.approx(0.029 * AUDIENCE_BOOST)
    assert kept[1].score == 0.030


@pytest.mark.parametrize("audience", [None, "just_curious"])
def test_without_an_audience_one_page_per_topic(audience):
    """The best page per topic stays with all its chunks; the others on
    that topic go; other topics and plain documents are untouched."""
    passages = [
        _passage("case-a", 0.020),
        _passage("funder-debt", 0.031, "funder", "debt", 0),
        _passage("funder-debt", 0.012, "funder", "debt", 1),
        _passage("investor-debt", 0.030, "investor", "debt"),
        _passage("government-food", 0.010, "government", "food"),
    ]
    kept = apply_audience(passages, audience)
    assert [(p.doc_id, p.score) for p in kept] == [
        ("funder-debt", 0.031), ("case-a", 0.020),
        ("funder-debt", 0.012), ("government-food", 0.010)]


@pytest.mark.parametrize("audience", [None, "funder"])
def test_at_most_two_pages_per_search(audience):
    """However many pages match, the best AUDIENCE_PAGE_LIMIT stay, so
    case studies keep their places."""
    passages = [_passage("case-a", 0.001)] + [
        _passage(f"funder-t{i}", 0.05 - i / 1000, "funder", f"t{i}")
        for i in range(5)]
    kept = apply_audience(passages, audience)
    assert AUDIENCE_PAGE_LIMIT == 2
    assert [p.doc_id for p in kept] == ["funder-t0", "funder-t1", "case-a"]


def test_apply_audience_never_adds_or_changes_tiers():
    """The rules only remove or reorder: a subscriber passage stays a
    subscriber passage, and nothing new appears."""
    gated = dataclasses.replace(_passage("pub", 0.5),
                                access_tier="subscriber")
    for audience in (None, "investor"):
        kept = apply_audience([gated, _passage("case-a", 0.1)], audience)
        assert {(p.doc_id, p.access_tier) for p in kept} == {
            ("pub", "subscriber"), ("case-a", "public")}


# ------------------------------------------------ the memory retriever


@pytest.mark.parametrize("audience", list(AUDIENCE_PAGE_KEYS))
def test_visitor_sees_only_their_own_audience_pages(audience):
    """A sovereign-debt question as any audience retrieves only that
    audience's pages, never another audience's."""
    bundle = run(MEMORY.retrieve(DEBT, "en", ANONYMOUS, audience=audience))
    pages = [p for p in bundle.passages if p.audience is not None]
    assert all(p.audience == audience for p in pages)


@pytest.mark.parametrize("audience", ["investor", "government", "funder",
                                      "non_profit"])
def test_own_sovereign_debt_page_is_retrieved(audience):
    """Every audience with a sovereign-debt page gets that page."""
    bundle = run(MEMORY.retrieve(DEBT, "en", ANONYMOUS, audience=audience))
    prefix = audience.replace("_", "-")
    assert f"{prefix}-sovereign-debt" in _page_ids(bundle)


def test_corporate_has_no_debt_page_and_gets_no_one_elses():
    """Corporates have no sovereign-debt page; they get none of the
    others' either."""
    bundle = run(MEMORY.retrieve(DEBT, "en", ANONYMOUS,
                                 audience="corporate"))
    assert not any("sovereign-debt" in d for d in _page_ids(bundle))


@pytest.mark.parametrize("audience", [None, "just_curious"])
def test_no_audience_keeps_one_page_per_topic(audience):
    """Without an audience the debt question still reads one debt page,
    and no topic appears twice."""
    bundle = run(MEMORY.retrieve(DEBT, "en", ANONYMOUS, audience=audience))
    pages = [p for p in bundle.passages if p.audience is not None]
    assert any(p.topic == "sovereign-debt" for p in pages)
    topics = [p.topic for p in pages]
    assert len(topics) == len(set(topics))


def test_audience_does_not_open_the_paywall():
    """With an audience set, a non-subscriber still gets a teaser only;
    a subscriber gets the text."""
    gated = dataclasses.replace(
        MEMORY.docs[0], doc_id="gated", doc_type="publication",
        title="Sovereign Debt Outlook", summary="Debt risk in one line.",
        text="GATED-BODY sovereign debt ranking", access_tier="subscriber",
        words=frozenset({"what", "does", "tmp", "do", "on", "sovereign",
                         "debt", "outlook", "gated", "body"}),
        language="en", audience=None, topic=None)
    retriever = MemoryRetriever(MEMORY.docs + [gated])
    for who in (ANONYMOUS, FREE):
        for audience in AUDIENCES:
            bundle = run(retriever.retrieve(DEBT, "en", who,
                                            audience=audience))
            assert "GATED-BODY" not in repr(bundle.passages)
            assert "gated" in [t.doc_id for t in bundle.teasers]
    bundle = run(retriever.retrieve(DEBT, "en", ACCESS, audience="investor"))
    assert "gated" in [p.doc_id for p in bundle.passages]


# ---------------------------------------------------------- threading


class _RecordingRetriever:
    """Records the audience of every search; returns nothing."""

    def __init__(self):
        """Create the recorder."""
        self.calls = []

    async def retrieve(self, query, language, identity, audience=None):
        """Record one search.

        Args:
            query: Search text.
            language: Locale.
            identity: Caller.
            audience: "Asking as" key.

        Returns:
            RetrievalBundle: Empty.
        """
        self.calls.append((query, audience))
        return RetrievalBundle()


def test_retrieve_for_turn_passes_the_audience_to_every_search():
    """The current search and both carry-over searches use the
    audience."""
    history = [("user", "one"), ("assistant", "a"), ("user", "two"),
               ("assistant", "b"), ("user", "three")]
    stub = _RecordingRetriever()
    run(retrieve_for_turn(stub, history, "en", ANONYMOUS,
                          audience="funder"))
    assert {q for q, _ in stub.calls} == {retrieval_query(history), "two",
                                          "one"}
    assert {a for _, a in stub.calls} == {"funder"}
    stub = _RecordingRetriever()
    run(retrieve_for_turn(stub, history, "en", ANONYMOUS))
    assert {a for _, a in stub.calls} == {None}


def _system_text(llm):
    """The system prompt text of the writer's first call.

    Args:
        llm: The ScriptedLLM.

    Returns:
        str: All system text joined.
    """
    content = llm.calls[0].messages[0]["content"]
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content)
    return content


def _ask(settings, audience):
    """Ask the debt question through the app and return the prompt.

    Args:
        settings: Settings fixture.
        audience: "Asking as" key, or None to leave it out.

    Returns:
        str: The writer's system prompt.
    """
    llm = ScriptedLLM([ScriptedReply(text="We work on debt.")])
    client = TestClient(create_app(settings, build_runtime(settings, llm)))
    body = {"language": "en",
            "messages": [{"role": "user", "content": DEBT}]}
    if audience is not None:
        body["audience"] = audience
    resp = client.post("/api/v1/chat", json=body, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    assert wire.parse_sse(resp.text)[-1][0] == "done"
    return _system_text(llm)


def test_stream_turn_threads_the_audience_into_the_prompt(settings):
    """Through the app: an investor's prompt holds the investor debt page
    and no other audience's; a government's holds the government page."""
    investor = _ask(settings, "investor")
    assert 'id="investor-sovereign-debt"' in investor
    assert 'audience="investor"' in investor
    for other in ("government", "funder", "non_profit", "corporate"):
        assert f'audience="{other}"' not in investor
    government = _ask(settings, "government")
    assert 'id="government-sovereign-debt"' in government
    assert 'audience="investor"' not in government


def test_no_audience_prompt_holds_one_debt_page(settings):
    """Without an audience the prompt still carries sovereign-debt copy,
    from one audience page only."""
    prompt = _ask(settings, None)
    assert prompt.count("-sovereign-debt\"") == 1


# ------------------------------------------------- prompt and cards


def test_reference_block_names_the_pages_audience():
    """An audience page carries audience="..."; other documents do not,
    and the stand-in writer's fixed attribute prefix is unchanged."""
    text = reference_block(RetrievalBundle((
        _passage("funder-debt", 0.1, "funder", "sovereign-debt"),
        _passage("case-a", 0.1),
    )))
    assert ('<reference id="funder-debt" doc_type="audience_page" '
            'access="public" title="funder-debt" language="en" '
            'audience="funder">') in text
    assert ('<reference id="case-a" doc_type="case_study" '
            'access="public" title="case-a" language="en">') in text


def test_audience_pages_are_never_case_study_cards():
    """The catalog holds no audience page, so a card call naming one is
    dropped."""
    catalog = load_catalog(COPS)
    page_ids = {d.doc_id for d in MEMORY.docs
                if d.doc_type == "audience_page"}
    assert page_ids and not page_ids & set(catalog.case_studies)
    cards = CardCollector(catalog, ANONYMOUS, "en")
    outcome = cards.apply("show_case_studies",
                          {"ids": ["investor-sovereign-debt"]})
    assert not outcome.ok and cards.case_ids == []
    outcome = cards.apply("show_case_studies",
                          {"ids": ["investor-transition-minerals",
                                   "transition-minerals"]})
    assert outcome.ok and cards.case_ids == ["transition-minerals"]
