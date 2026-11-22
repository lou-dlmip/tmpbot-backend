# Carry-over retrieval against a real Postgres with the fake embedder:
# the real content is indexed with the Phase 4 tool, and the third turn
# of a conversation must still see the document the first turn relied
# on. Skipped without a database, as test_paywall_db.py is.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Multi-turn retrieval over Postgres with the fake embedder."""

from __future__ import annotations

import asyncio
import shutil

import pytest

from tmpbot_backend.config import REPO_ROOT
from tmpbot_backend.db import Database
from tmpbot_backend.embedding import FakeEmbedder
from tmpbot_backend.identity import ANONYMOUS
from tmpbot_backend.orchestrator import retrieval_query, retrieve_for_turn
from tmpbot_backend.retrieval import MAX_PASSAGES, PgRetriever

HISTORY = [
    ("user", "What was the Dryad project in Cameroon?"),
    ("assistant", "We worked with Dryad in Cameroon."),
    ("user", "Tell me about your forest carbon project"),
    ("assistant", "We assessed forest carbon."),
    ("user", "How do you handle land tenure risk?"),
]


@pytest.fixture(scope="module")
def retriever(migrated_db, tmp_path_factory):
    """A Postgres retriever over the indexed real content.

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
    status = index_corpus.main([
        "--database-url", migrated_db, "--embedder", "fake",
        "--root", str(root),
    ])
    assert status == 0
    return PgRetriever(Database(migrated_db), FakeEmbedder())


def test_third_turn_keeps_the_first_turns_documents(retriever):
    """Searching the current text alone loses the first question's
    document; the carry-over search brings it back, inside the budget,
    with the current results first."""
    alone = asyncio.run(retriever.retrieve(
        retrieval_query(HISTORY), "en", ANONYMOUS))
    first = asyncio.run(retriever.retrieve(HISTORY[0][1], "en", ANONYMOUS))
    lost = [p.doc_id for p in first.passages
            if p.doc_id not in {q.doc_id for q in alone.passages}]
    assert lost, "fixture no longer shows a loss; pick other questions"

    merged = asyncio.run(retrieve_for_turn(
        retriever, HISTORY, "en", ANONYMOUS))
    ids = [p.doc_id for p in merged.passages]
    assert len(ids) <= MAX_PASSAGES and len(ids) == len(set(ids))
    assert lost[0] in ids
    assert ids[0] == alone.passages[0].doc_id
    assert all(p.access_tier == "public" for p in merged.passages)
