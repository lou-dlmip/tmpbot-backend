# Retrieval with the paywall enforced before anything reaches the prompt.
#
# ADR-0007 (spec 3b): for a non-subscriber, retrieval never passes the
# text of `access_tier: subscriber` content to the model. It passes only
# the titles and one-line summaries of matching gated documents, marked
# subscriber-only: exactly enough for a teaser.
#
# Three layers hold that line for the Postgres retriever:
#   1. The search asks for the public tier only.
#   2. The transaction runs as the visitor's app role, so row level
#      security on corpus_chunks refuses subscriber rows even if the
#      tier list were wrong (Phase 4 decision 1).
#   3. Every returned passage is checked again here; a non-public one
#      for a non-subscriber is dropped and logged as an error.
# Teasers come from the corpus_teasers view, which has no text column.
#
# Languages: the visitor's language is searched first, then English,
# because most translations are still drafts (never retrieved) and the
# model writes in the visitor's language whatever the source language.
#
# MemoryRetriever implements the same rules over the COPS/ files, for
# local work without a database and for unit tests.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Hybrid retrieval with access-tier filtering and teasers."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from .catalog import read_frontmatter
from .db import Database, as_app_role
from .embedding import QueryEmbedder, normalise_text, vector_literal
from .identity import Identity

log = logging.getLogger("tmpbot.retrieval")

MAX_PASSAGES = 6
MAX_TEASERS = 3
# Passages of the MAX_PASSAGES budget kept for earlier turns' material
# (merge_bundles).
CARRY_SLOTS = 3
CANDIDATES_PER_LANGUAGE = 8

PUBLIC = "public"
SUBSCRIBER = "subscriber"


@dataclass(frozen=True)
class Passage:
    """A chunk of content the model may read.

    Attributes:
        doc_id: Document id (a case-study id for case studies).
        doc_type: ``case_study``, ``publication``, ``about`` and so on.
        language: Language of the text.
        title: Document title.
        text: The chunk text.
        access_tier: ``public`` or ``subscriber``.
        score: Fused relevance score, higher is better.
        year: Year from the frontmatter, when known.
    """

    doc_id: str
    doc_type: str
    language: str
    title: str
    text: str
    access_tier: str
    score: float = 0.0
    year: Optional[int] = None


@dataclass(frozen=True)
class Teaser:
    """Title and summary of a gated document, for a non-subscriber.

    Attributes:
        doc_id: Document id.
        doc_type: Document type.
        language: Language of the title and summary.
        title: Title.
        summary: One-line summary.
    """

    doc_id: str
    doc_type: str
    language: str
    title: str
    summary: str


@dataclass(frozen=True)
class RetrievalBundle:
    """What retrieval hands the prompt builder.

    Attributes:
        passages: Readable chunks, best first.
        teasers: Gated documents the visitor may only hear about.
    """

    passages: tuple[Passage, ...] = ()
    teasers: tuple[Teaser, ...] = field(default_factory=tuple)


class Retriever(Protocol):
    """Finds reference material for a question."""

    async def retrieve(self, query: str, language: str,
                       identity: Identity) -> RetrievalBundle:
        """Retrieve material for one question.

        Args:
            query: Search text.
            language: The visitor's locale.
            identity: The caller; decides the tiers.

        Returns:
            RetrievalBundle: Passages and teasers.
        """
        ...


def search_languages(language: str) -> list[str]:
    """Languages to search, the visitor's first.

    Args:
        language: The visitor's locale.

    Returns:
        list[str]: ``[language]``, plus ``en`` when different.
    """
    return [language] if language == "en" else [language, "en"]


def allowed_tiers(identity: Identity) -> list[str]:
    """Access tiers the caller may read.

    Args:
        identity: The caller.

    Returns:
        list[str]: Both tiers for subscribers, else public only.
    """
    return [PUBLIC, SUBSCRIBER] if identity.subscriber else [PUBLIC]


def enforce_tiers(passages: Sequence[Passage],
                  identity: Identity) -> list[Passage]:
    """Drop any passage the caller may not read (the last check).

    Args:
        passages: Candidate passages.
        identity: The caller.

    Returns:
        list[Passage]: Passages in an allowed tier.
    """
    tiers = set(allowed_tiers(identity))
    kept = []
    for p in passages:
        if p.access_tier in tiers:
            kept.append(p)
        else:
            log.error("gated passage reached the backend filter; dropped",
                      extra={"doc_id": p.doc_id, "plan": identity.plan})
    return kept


def merge_passages(passages: Sequence[Passage], language: str,
                   limit: int = MAX_PASSAGES) -> list[Passage]:
    """Order passages by score, one language per document.

    When a document matched in both the visitor's language and English,
    the visitor's language wins.

    Args:
        passages: Candidates from every searched language.
        language: The visitor's locale.
        limit: Most passages to keep.

    Returns:
        list[Passage]: Best first.
    """
    doc_lang: dict[str, str] = {}
    for p in passages:
        if p.language == language or p.doc_id not in doc_lang:
            doc_lang[p.doc_id] = p.language
    kept = [p for p in passages if doc_lang[p.doc_id] == p.language]
    kept.sort(key=lambda p: (-p.score, p.doc_id))
    return kept[:limit]


def merge_bundles(primary: RetrievalBundle,
                  carried: Sequence[RetrievalBundle],
                  limit: int = MAX_PASSAGES,
                  carry_slots: int = CARRY_SLOTS) -> RetrievalBundle:
    """Merge the current question's material with earlier turns' material.

    The reference block is rebuilt on every turn, so the writer sees its
    earlier answers but not their sources. Keeping some of those sources
    in the block stops it from doubting itself. The merge keeps the
    total inside ``limit``:

    * the current question's passages come first, in their own order;
    * up to ``carry_slots`` passages from earlier turns follow, taken
      round-robin (the most recent turn first), one per document;
    * if earlier turns have fewer to give, the current question uses the
      spare room, and the other way round.

    Scores from different queries are not comparable, so nothing is
    re-sorted. Teasers merge the same way, within ``MAX_TEASERS``.

    Args:
        primary: Material for the current question.
        carried: Material for earlier questions, most recent first.
        limit: Most passages in the result.
        carry_slots: Passages reserved for earlier turns.

    Returns:
        RetrievalBundle: Merged, deduplicated material.
    """
    own: list[Passage] = []
    seen: set[str] = set()
    for p in primary.passages:
        if p.doc_id not in seen:
            seen.add(p.doc_id)
            own.append(p)
    queues = [list(b.passages) for b in carried]
    extra: list[Passage] = []
    while any(queues):
        for queue in queues:
            while queue:
                p = queue.pop(0)
                if p.doc_id not in seen:
                    seen.add(p.doc_id)
                    extra.append(p)
                    break
    take_carry = min(len(extra), carry_slots)
    take_own = min(len(own), max(limit - take_carry, 0))
    take_carry = min(len(extra), max(limit - take_own, 0))
    passages = own[:take_own] + extra[:take_carry]

    teasers: list[Teaser] = []
    seen_teasers: set[str] = set()
    for t in [*primary.teasers, *(t for b in carried for t in b.teasers)]:
        if t.doc_id not in seen_teasers:
            seen_teasers.add(t.doc_id)
            teasers.append(t)
    return RetrievalBundle(tuple(passages), tuple(teasers[:MAX_TEASERS]))


def _year(metadata: Any) -> Optional[int]:
    """Read a year from chunk metadata.

    Args:
        metadata: The JSONB metadata.

    Returns:
        Optional[int]: The year, if a sensible one is present.
    """
    if isinstance(metadata, dict):
        value = metadata.get("year")
        if isinstance(value, int) and 1900 <= value <= 2100:
            return value
    return None


# ----------------------------------------------------------------------
# Postgres
# ----------------------------------------------------------------------

SEARCH_SQL = (
    "SELECT doc_id, doc_type, language, title, chunk_text, access_tier, "
    "score, metadata "
    "FROM public.match_corpus_chunks(%s::vector, %s, %s, %s, %s, "
    "match_embedding_model => %s)"
)

# Gated documents whose title, summary or topics share a word with the
# question. The OR-ed query mirrors match_corpus_chunks' lexical arm.
TEASER_SQL = """
WITH q AS (
    SELECT nullif(replace(plainto_tsquery('public.simple_unaccent', %s)
                          ::TEXT, ' & ', ' | '), '')::tsquery AS query
)
SELECT t.doc_id, t.doc_type, t.language, t.title, t.summary,
       ts_rank_cd(
           to_tsvector('public.simple_unaccent',
                       t.title || ' ' || coalesce(t.summary, '') || ' '
                       || array_to_string(t.topics, ' ')),
           q.query) AS rank
  FROM public.corpus_teasers t, q
 WHERE q.query IS NOT NULL
   AND t.language = ANY(%s)
   AND to_tsvector('public.simple_unaccent',
                   t.title || ' ' || coalesce(t.summary, '') || ' '
                   || array_to_string(t.topics, ' ')) @@ q.query
 ORDER BY rank DESC, t.doc_id
 LIMIT %s
"""


class PgRetriever:
    """Hybrid search over corpus_chunks, run as the visitor's role.

    Args:
        db: Database helper.
        embedder: Query embedder; its model must match the index.
    """

    def __init__(self, db: Database, embedder: QueryEmbedder) -> None:
        """Create the retriever.

        Args:
            db: Database helper.
            embedder: Query embedder.
        """
        self.db = db
        self.embedder = embedder

    async def retrieve(self, query: str, language: str,
                       identity: Identity) -> RetrievalBundle:
        """Retrieve passages and, for non-subscribers, teasers.

        Args:
            query: Search text.
            language: The visitor's locale.
            identity: The caller.

        Returns:
            RetrievalBundle: The material.
        """
        vector = await self.embedder.embed_query(query)
        literal = vector_literal(vector)
        return await asyncio.to_thread(
            self._search, literal, query, language, identity
        )

    def _search(self, literal: str, query: str, language: str,
                identity: Identity) -> RetrievalBundle:
        """Run the searches in one read-only transaction.

        Args:
            literal: pgvector literal of the query embedding.
            query: Search text.
            language: The visitor's locale.
            identity: The caller.

        Returns:
            RetrievalBundle: The material.
        """
        langs = search_languages(language)
        tiers = allowed_tiers(identity)
        found: list[Passage] = []
        teasers: list[Teaser] = []
        with self.db.connect() as conn:
            as_app_role(conn, identity)
            for lang in langs:
                rows = conn.execute(SEARCH_SQL, (
                    literal, query, lang, tiers, CANDIDATES_PER_LANGUAGE,
                    self.embedder.model,
                )).fetchall()
                found.extend(
                    Passage(doc_id=r[0], doc_type=r[1], language=r[2],
                            title=r[3], text=r[4], access_tier=r[5],
                            score=float(r[6]), year=_year(r[7]))
                    for r in rows
                )
            if not identity.subscriber:
                rows = conn.execute(
                    TEASER_SQL, (query, langs, MAX_TEASERS * 2)
                ).fetchall()
                teasers = _dedupe_teasers(
                    [Teaser(r[0], r[1], r[2], r[3], r[4] or "")
                     for r in rows], language)
            conn.rollback()  # read only; also ends SET LOCAL ROLE
        passages = merge_passages(enforce_tiers(found, identity), language)
        return RetrievalBundle(tuple(passages), tuple(teasers))


def _dedupe_teasers(teasers: Sequence[Teaser],
                    language: str) -> list[Teaser]:
    """One teaser per document, the visitor's language first.

    Args:
        teasers: Candidates, best first.
        language: The visitor's locale.

    Returns:
        list[Teaser]: At most ``MAX_TEASERS``.
    """
    best: dict[str, Teaser] = {}
    order: list[str] = []
    for t in teasers:
        if t.doc_id not in best:
            best[t.doc_id] = t
            order.append(t.doc_id)
        elif t.language == language:
            best[t.doc_id] = t
    return [best[d] for d in order][:MAX_TEASERS]


# ----------------------------------------------------------------------
# In memory (local work, unit tests)
# ----------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SECTIONS = ("CASE", "ABOU", "LEAD", "PUBL")
_TYPE_MAP = {"practice-areas": "practice_areas"}


@dataclass(frozen=True)
class _Doc:
    """A COPS document held in memory."""

    doc_id: str
    doc_type: str
    language: str
    title: str
    summary: str
    text: str
    access_tier: str
    year: Optional[int]
    words: frozenset[str]


def _words(text: str) -> frozenset[str]:
    """Distinct words of two or more characters, accent-insensitive.

    Args:
        text: Any text.

    Returns:
        frozenset[str]: The words.
    """
    return frozenset(w for w in _WORD_RE.findall(normalise_text(text))
                     if len(w) >= 2)


def _body(path: Path) -> str:
    """Return a Markdown file's body without the frontmatter.

    Args:
        path: The file.

    Returns:
        str: The body.
    """
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].strip()
    return text.strip()


class MemoryRetriever:
    """Word-overlap retrieval over the active COPS/ files.

    Same tier rules as the Postgres retriever: non-subscribers get
    public passages plus subscriber titles and summaries only.

    Args:
        docs: Documents to search; see ``from_cops``.
    """

    def __init__(self, docs: Sequence[_Doc]) -> None:
        """Create the retriever.

        Args:
            docs: Documents.
        """
        self.docs = list(docs)

    @classmethod
    def from_cops(cls, cops_dir: Path) -> "MemoryRetriever":
        """Load every active document under ``COPS/``.

        Args:
            cops_dir: The ``COPS/`` folder.

        Returns:
            MemoryRetriever: The retriever.
        """
        docs = []
        for section in _SECTIONS:
            for path in sorted((cops_dir / section).glob("*.md")):
                meta = read_frontmatter(path)
                if meta.get("source_lifecycle") != "active":
                    continue
                doc_type = str(meta.get("type", ""))
                doc_type = _TYPE_MAP.get(doc_type, doc_type)
                title = str(meta.get("title") or meta.get("name") or "")
                body = _body(path)
                summary = str(meta.get("summary") or "")
                year = meta.get("year")
                docs.append(_Doc(
                    doc_id=str(meta.get("id")),
                    doc_type=doc_type,
                    language=str(meta.get("language")),
                    title=title,
                    summary=summary,
                    text=body,
                    access_tier=str(meta.get("access_tier", PUBLIC)),
                    year=year if isinstance(year, int) else None,
                    words=_words(f"{title} {summary} {body}"),
                ))
        return cls(docs)

    async def retrieve(self, query: str, language: str,
                       identity: Identity) -> RetrievalBundle:
        """Retrieve by shared words.

        Args:
            query: Search text.
            language: The visitor's locale.
            identity: The caller.

        Returns:
            RetrievalBundle: The material.
        """
        q = _words(query)
        langs = set(search_languages(language))
        tiers = set(allowed_tiers(identity))
        scored = []
        teasers = []
        for d in self.docs:
            if d.language not in langs:
                continue
            overlap = len(q & d.words)
            if not overlap:
                continue
            score = overlap / (1 + len(q))
            if d.access_tier in tiers:
                scored.append(Passage(d.doc_id, d.doc_type, d.language,
                                      d.title, d.text, d.access_tier,
                                      score, d.year))
            elif d.access_tier == SUBSCRIBER:
                head = _words(d.title + " " + d.summary)
                if q & head:
                    teasers.append((score, Teaser(
                        d.doc_id, d.doc_type, d.language, d.title,
                        d.summary)))
        teasers.sort(key=lambda st: (-st[0], st[1].doc_id))
        passages = merge_passages(enforce_tiers(scored, identity), language)
        return RetrievalBundle(
            tuple(passages),
            tuple(_dedupe_teasers([t for _, t in teasers], language)),
        )
