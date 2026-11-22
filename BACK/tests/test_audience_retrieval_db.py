# Audience pages in the Postgres retriever, against a real Postgres.
#
# The real COPS tree (with its 24 audience pages) plus one
# subscriber-only publication on sovereign debt is indexed with the
# Phase 4 tool and the fake embedder. The SQL wrapper around
# match_corpus_chunks must drop other audiences' pages before the
# candidate limit, keep the visitor's own, keep one page per topic when
# there is no audience, and leave the paywall exactly as it was: the
# publication's text reaches subscribers only, whatever the audience.
# Skipped without a database, as test_paywall_db.py is.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Audience-aware retrieval over Postgres with the fake embedder."""

from __future__ import annotations

import asyncio
import shutil

import pytest
from chat_helpers import make_token

from tmpbot_backend.config import REPO_ROOT
from tmpbot_backend.db import Database
from tmpbot_backend.embedding import FakeEmbedder
from tmpbot_backend.identity import (
    ANONYMOUS,
    StaticKeyVerifier,
    resolve_identity,
)
from tmpbot_backend.orchestrator import retrieve_for_turn
from tmpbot_backend.retrieval import (
    AUDIENCE_PAGE_KEYS,
    AUDIENCE_PAGE_LIMIT,
    CANDIDATES_PER_LANGUAGE,
    MAX_PASSAGES,
    PgRetriever,
)

KEY = "test-jwt-signing-key-0123456789abcdef"
FREE = resolve_identity(f"Bearer {make_token()}", StaticKeyVerifier(KEY))
ACCESS = resolve_identity(f"Bearer {make_token(plan='access')}",
                          StaticKeyVerifier(KEY))
DEBT = "What does TMP do on sovereign debt and climate debt crises?"
SECRET_MARK = "SECRET-DEBT-FINDING"
# Close to the publication's own words, so the fake embedder ranks it
# for a subscriber and the test is about tiers, not ranking.
GATED_QUESTION = ("Sovereign Debt Outlook 2026: which sovereign debt of "
                  "drought-exposed producers is most exposed to climate "
                  "shocks?")

PUBLICATION = f"""---
id: "debt-outlook"
type: "publication"
language: "en"
title: "Sovereign Debt Outlook 2026"
version: "1.0"
last_updated: 2026-11-13
source_lifecycle: "active"
access_tier: "subscriber"
year: 2026
topics:
  - "sovereign-debt"
summary: "Which sovereign debt is most exposed to climate shocks."
file: "debt-outlook-2026.pdf"
---

## Findings

{SECRET_MARK}: sovereign debt of drought-exposed producers faces the
largest climate debt crises we model.
"""


@pytest.fixture(scope="module")
def retriever(migrated_db, tmp_path_factory):
    """A Postgres retriever over the indexed real content plus one
    subscriber-only publication.

    Returns:
        PgRetriever: Retriever using the fake embedder.
    """
    import index_corpus

    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REPO_ROOT / "COPS", root / "COPS")
    (root / "FRNT" / "i18n").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "FRNT" / "i18n" / "routing.ts",
                root / "FRNT" / "i18n" / "routing.ts")
    shutil.copytree(REPO_ROOT / "FRNT" / "public" / "images",
                    root / "FRNT" / "public" / "images")
    (root / "COPS" / "PUBL").mkdir(exist_ok=True)
    (root / "COPS" / "PUBL" / "debt-outlook_en.md").write_text(
        PUBLICATION, encoding="utf-8")
    status = index_corpus.main([
        "--database-url", migrated_db, "--embedder", "fake",
        "--root", str(root),
    ])
    assert status == 0
    return PgRetriever(Database(migrated_db), FakeEmbedder())


def _search(retriever, identity=ANONYMOUS, audience=None, query=DEBT):
    """Run one search.

    Args:
        retriever: The PgRetriever.
        identity: The caller.
        audience: "Asking as" key.
        query: Search text.

    Returns:
        RetrievalBundle: The material.
    """
    return asyncio.run(retriever.retrieve(query, "en", identity,
                                          audience=audience))


def test_audience_pages_carry_audience_and_topic(retriever):
    """Rows come back with the audience and topic from their metadata."""
    bundle = _search(retriever, audience="government")
    pages = [p for p in bundle.passages if p.doc_type == "audience_page"]
    assert pages
    for p in pages:
        assert p.audience == "government"
        assert p.doc_id == f"government-{p.topic}"


@pytest.mark.parametrize("audience", list(AUDIENCE_PAGE_KEYS))
def test_only_the_visitors_audience_pages(retriever, audience):
    """No other audience's page reaches any visitor."""
    for query in (DEBT, "transition minerals supply chains",
                  "food security and climate risk"):
        bundle = _search(retriever, audience=audience, query=query)
        others = [p.doc_id for p in bundle.passages
                  if p.audience not in (None, audience)]
        assert others == []
        pages = {p.doc_id for p in bundle.passages if p.audience}
        assert len(pages) <= AUDIENCE_PAGE_LIMIT
        assert len(bundle.passages) == MAX_PASSAGES


@pytest.mark.parametrize("audience", ["investor", "government", "funder",
                                      "non_profit"])
def test_own_debt_page_is_retrieved(retriever, audience):
    """The visitor's own sovereign-debt page is among the passages."""
    bundle = _search(retriever, audience=audience)
    prefix = audience.replace("_", "-")
    assert f"{prefix}-sovereign-debt" in [p.doc_id for p in bundle.passages]


@pytest.mark.parametrize("audience", [None, "just_curious"])
def test_no_audience_keeps_one_page_per_topic(retriever, audience):
    """Without an audience a debt page is still read, once per topic,
    and case studies keep their places."""
    bundle = _search(retriever, audience=audience,
                     query="transition minerals supply chains")
    pages = [p for p in bundle.passages if p.audience is not None]
    topics = [p.topic for p in pages]
    assert len(topics) == len(set(topics))
    assert "transition-minerals" in [p.doc_id for p in bundle.passages]
    # Pages and other documents are ranked apart in SQL, so dropping
    # pages never leaves the passage list short.
    assert len(bundle.passages) == MAX_PASSAGES
    debt = _search(retriever, audience=audience)
    assert any(p.topic == "sovereign-debt" for p in debt.passages)


def test_filter_runs_before_the_candidate_limit(retriever):
    """A minerals question as a government finds a full set of
    candidates even though four other audiences' minerals pages rank
    near the top of the raw search."""
    bundle = _search(retriever, audience="government",
                     query="lithium and copper supply chains for the "
                           "energy transition")
    ids = [p.doc_id for p in bundle.passages]
    assert "government-transition-minerals" in ids
    assert "transition-minerals" in ids
    assert len(ids) >= min(6, CANDIDATES_PER_LANGUAGE)


@pytest.mark.parametrize("audience", [None, "just_curious", "investor",
                                      "government", "funder"])
def test_paywall_holds_with_any_audience(retriever, audience):
    """Non-subscribers get the publication as a teaser only; subscribers
    read it; the audience changes neither."""
    for who in (ANONYMOUS, FREE):
        bundle = _search(retriever, who, audience, GATED_QUESTION)
        assert SECRET_MARK not in repr(bundle.passages)
        assert all(p.access_tier == "public" for p in bundle.passages)
        assert "debt-outlook" in [t.doc_id for t in bundle.teasers]
    bundle = _search(retriever, ACCESS, audience, GATED_QUESTION)
    assert "debt-outlook" in [p.doc_id for p in bundle.passages]
    assert SECRET_MARK in repr(bundle.passages)
    assert not bundle.teasers


def test_carry_over_searches_follow_the_audience(retriever):
    """Every search of a multi-turn retrieval keeps the audience: no
    other audience's page arrives through carried material."""
    history = [("user", "Tell me about your food security work"),
               ("assistant", "We look at food systems."),
               ("user", "And transition minerals?"),
               ("assistant", "We work on minerals."),
               ("user", DEBT)]
    merged = asyncio.run(retrieve_for_turn(retriever, history, "en",
                                           ANONYMOUS, audience="funder"))
    assert merged.passages
    assert all(p.audience in (None, "funder") for p in merged.passages)
