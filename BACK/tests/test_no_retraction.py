# Tests for the "bot retracts its own earlier answers" fix: the rule in
# BOTS/PRMT/rules.md, the carry-over of earlier turns' reference
# material (retrieval.merge_bundles, orchestrator.retrieve_for_turn) and
# the whole turn through the app with the scripted writer.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Earlier answers stay grounded: prompt rule and retrieval carry-over."""

from __future__ import annotations

import asyncio
import dataclasses

import pytest
from chat_helpers import TEST_SECRET, build_runtime
from fastapi.testclient import TestClient

from tmpbot_backend import wire
from tmpbot_backend.config import REPO_ROOT
from tmpbot_backend.identity import ANONYMOUS
from tmpbot_backend.llm import ScriptedLLM, ScriptedReply
from tmpbot_backend.main import create_app
from tmpbot_backend.orchestrator import (
    carry_over_queries,
    retrieval_query,
    retrieve_for_turn,
)
from tmpbot_backend.prompt import load_layers
from tmpbot_backend.retrieval import (
    CARRY_SLOTS,
    MAX_PASSAGES,
    MemoryRetriever,
    Passage,
    RetrievalBundle,
    Teaser,
    merge_bundles,
)

HEADERS = {"X-TMPBOT-Proxy-Secret": TEST_SECRET,
           "X-TMPBOT-Client-IP": "192.0.2.1"}


def _p(doc_id: str, score: float = 0.5) -> Passage:
    """A public passage.

    Args:
        doc_id: Document id.
        score: Relevance score.

    Returns:
        Passage: The passage.
    """
    return Passage(doc_id, "about", "en", doc_id.upper(), f"text {doc_id}",
                   "public", score)


def _bundle(*ids: str) -> RetrievalBundle:
    """A bundle of public passages in the given order.

    Args:
        *ids: Document ids, best first.

    Returns:
        RetrievalBundle: The bundle.
    """
    return RetrievalBundle(tuple(_p(i) for i in ids))


def _ids(bundle: RetrievalBundle) -> list[str]:
    """Passage ids in order.

    Args:
        bundle: A bundle.

    Returns:
        list[str]: Ids.
    """
    return [p.doc_id for p in bundle.passages]


# ------------------------------------------------------------------ rules


def test_rules_tell_the_bot_not_to_retract_earlier_answers():
    """The rule text is pinned; the never-invent rule stays."""
    rules = load_layers(REPO_ROOT / "BOTS" / "PRMT").rules
    assert "Do not retract, correct, disclaim" in rules
    assert "the visitor says it is wrong" in rules
    assert "directly contradicts it" in rules
    assert 'Never volunteer a line such as "I need to correct my earlier ' \
        'answer"' in rules
    assert "Never invent details, clients, numbers or outcomes" in rules


# --------------------------------------------------------- merge_bundles


def test_current_question_comes_first_and_duplicates_go():
    """Current results lead; a document seen twice appears once."""
    merged = merge_bundles(_bundle("a", "b"), [_bundle("b", "c")])
    assert _ids(merged) == ["a", "b", "c"]


def test_budget_is_kept_and_carry_slots_are_reserved():
    """A full current bundle still leaves room for earlier material."""
    current = _bundle("c1", "c2", "c3", "c4", "c5", "c6")
    merged = merge_bundles(current, [_bundle("p1", "p2"),
                                     _bundle("q1", "q2")])
    assert len(merged.passages) == MAX_PASSAGES
    assert _ids(merged)[:MAX_PASSAGES - CARRY_SLOTS] == [
        "c1", "c2", "c3"]
    # Round-robin, most recent turn first.
    assert _ids(merged)[MAX_PASSAGES - CARRY_SLOTS:] == ["p1", "q1", "p2"]


def test_spare_room_goes_to_whoever_needs_it():
    """Few earlier passages: the current question keeps the rest."""
    current = _bundle("c1", "c2", "c3", "c4", "c5", "c6")
    merged = merge_bundles(current, [_bundle("p1")])
    assert _ids(merged) == ["c1", "c2", "c3", "c4", "c5", "p1"]
    # Few current passages: earlier ones fill past the reserve.
    older = _bundle("p1", "p2", "p3", "p4", "p5")
    merged = merge_bundles(_bundle("c1"), [older])
    assert _ids(merged) == ["c1", "p1", "p2", "p3", "p4", "p5"]


def test_nothing_earlier_changes_nothing():
    """A first turn is unchanged."""
    current = _bundle("a", "b")
    assert merge_bundles(current, []) == current
    assert merge_bundles(RetrievalBundle(), []) == RetrievalBundle()


def test_teasers_merge_within_their_budget():
    """Teasers from earlier turns join, deduplicated, at most three."""
    def teaser(i):
        return Teaser(i, "publication", "en", i.upper(), "summary")

    primary = RetrievalBundle((), (teaser("t1"), teaser("t2")))
    older = RetrievalBundle((), (teaser("t2"), teaser("t3"), teaser("t4")))
    merged = merge_bundles(primary, [older])
    assert [t.doc_id for t in merged.teasers] == ["t1", "t2", "t3"]


# ------------------------------------------------------- carry_over_queries


def test_carry_over_queries_are_the_two_previous_questions():
    """Most recent first; assistant turns, blanks and repeats are
    skipped."""
    history = [("user", "q1"), ("assistant", "a1"), ("user", "q2"),
               ("assistant", "a2"), ("user", "  "), ("assistant", "a3"),
               ("user", "q2"), ("assistant", "a4"), ("user", "q3")]
    assert carry_over_queries(history) == ["q2", "q1"]
    assert carry_over_queries([("user", "only")]) == []
    assert carry_over_queries([("user", "same"), ("assistant", "x"),
                               ("user", "same")]) == []


# ------------------------------------------------------ retrieve_for_turn


class _StubRetriever:
    """Returns a fixed bundle per query and records the queries."""

    def __init__(self, table, fail_on=()):
        """Create the stub.

        Args:
            table: Mapping of query text to document ids.
            fail_on: Queries that raise.
        """
        self.table = table
        self.fail_on = set(fail_on)
        self.queries = []
        self.audiences = []

    async def retrieve(self, query, language, identity, audience=None):
        """Return the bundle for ``query``.

        Args:
            query: Search text.
            language: Locale.
            identity: Caller.
            audience: "Asking as" key, recorded for the threading test.

        Returns:
            RetrievalBundle: The bundle.

        Raises:
            RuntimeError: For a query listed in ``fail_on``.
        """
        self.queries.append(query)
        self.audiences.append(audience)
        if query in self.fail_on:
            raise RuntimeError("boom")
        return _bundle(*self.table.get(query, ()))


HISTORY = [("user", "where do you work"), ("assistant", "In 70 countries."),
           ("user", "what did you do this year"),
           ("assistant", "El Nino work."),
           ("user", "what do you do in south east asia")]
TABLE = {
    "where do you work": ["about", "mission"],
    "what did you do this year": ["elnino", "about"],
    # The current search text joins the last two questions.
    "what did you do this year what do you do in south east asia":
        ["growasia", "elnino"],
}


def test_earlier_turns_material_is_retrieved_again():
    """Turn three still carries what turns one and two relied on."""
    stub = _StubRetriever(TABLE)
    bundle = asyncio.run(retrieve_for_turn(stub, HISTORY, "en", ANONYMOUS))
    assert _ids(bundle)[:2] == ["growasia", "elnino"]
    assert {"about", "mission"} <= set(_ids(bundle))
    assert set(stub.queries) == {
        retrieval_query(HISTORY), "what did you do this year",
        "where do you work"}


def test_a_failing_carry_over_search_is_skipped():
    """Only the current question's search may end the turn."""
    stub = _StubRetriever(TABLE, fail_on=["where do you work"])
    bundle = asyncio.run(retrieve_for_turn(stub, HISTORY, "en", ANONYMOUS))
    assert "mission" not in _ids(bundle) and "about" in _ids(bundle)
    stub = _StubRetriever(TABLE, fail_on=[retrieval_query(HISTORY)])
    with pytest.raises(RuntimeError):
        asyncio.run(retrieve_for_turn(stub, HISTORY, "en", ANONYMOUS))


def test_first_turn_searches_once():
    """No history, no extra searches."""
    stub = _StubRetriever(TABLE)
    asyncio.run(retrieve_for_turn(stub, [("user", "where do you work")],
                                  "en", ANONYMOUS))
    assert stub.queries == ["where do you work"]


# --------------------------------------------------------- whole turn


def _docs_for(retriever: MemoryRetriever):
    """Three synthetic documents, each matching one rare word.

    Args:
        retriever: A retriever to borrow a document shape from.

    Returns:
        list: Documents.
    """
    base = retriever.docs[0]
    docs = []
    for doc_id, word in (("doc-alpha", "alphaword"),
                         ("doc-beta", "betaword"),
                         ("doc-gamma", "gammaword")):
        docs.append(dataclasses.replace(
            base, doc_id=doc_id, doc_type="about", title=doc_id,
            summary="", text=f"Body of {doc_id}.", access_tier="public",
            language="en", words=frozenset({word})))
    return docs


def test_third_turn_prompt_still_holds_the_first_turns_material(settings):
    """Through the app: the writer's reference block for turn three
    holds turn one's document too, after the current ones, and the
    stable prompt block carries the no-retraction rule."""
    retriever = MemoryRetriever(_docs_for(MemoryRetriever.from_cops(
        REPO_ROOT / "COPS")))
    llm = ScriptedLLM([ScriptedReply(text="We work in many places.")])
    runtime = build_runtime(settings, llm, retriever=retriever)
    client = TestClient(create_app(settings, runtime))
    messages = [
        {"role": "user", "content": "tell me alphaword"},
        {"role": "assistant", "content": "Answer one."},
        {"role": "user", "content": "tell me betaword"},
        {"role": "assistant", "content": "Answer two."},
        {"role": "user", "content": "tell me gammaword"},
    ]
    resp = client.post("/api/v1/chat", json={"language": "en",
                                             "messages": messages},
                       headers=HEADERS)
    assert resp.status_code == 200, resp.text
    assert wire.parse_sse(resp.text)[-1][0] == "done"
    content = llm.calls[0].messages[0]["content"]
    stable, turn = content[0]["text"], content[1]["text"]
    assert "Do not retract, correct, disclaim" in stable
    assert "doc-" not in stable
    # The search text joins the last two questions (beta, gamma); alpha
    # can only come from the carry-over, so it must come last.
    pos = {n: turn.find(f'<reference id="doc-{n}"')
           for n in ("alpha", "beta", "gamma")}
    assert all(v >= 0 for v in pos.values())
    assert pos["alpha"] > max(pos["beta"], pos["gamma"])
    # The cacheable block does not change when the references do.
    assert content[0].get("cache_control") == {"type": "ephemeral"}
