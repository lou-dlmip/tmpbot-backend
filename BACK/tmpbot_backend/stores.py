# Server-side state for the chat: the free-question count, rate limits,
# the usage log and stored conversations.
#
# Each store has a Postgres version (tables from INFR/migrations/0003
# and 0004, plus 0007) and an in-memory version with the same rules, for
# local work without a database and for fast unit tests. The database
# tests in BACK/tests/test_*_db.py run the Postgres versions against a
# real server.
#
# Methods are synchronous; the chat route calls them through
# asyncio.to_thread so a slow query never blocks other streams.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Visitor counts, rate limits, usage log and conversation storage."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol

from .db import Database

log = logging.getLogger("tmpbot.stores")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def hash_ip(ip: str, salt: str) -> str:
    """Hash a client IP address with the deployment's salt.

    HMAC-SHA-256 keyed by the salt, so the hash cannot be reversed by
    trying every IPv4 address without the salt.

    Args:
        ip: The client address as text ("" when unknown).
        salt: ``TMPBOT_IP_HASH_SALT``.

    Returns:
        str: 64 lowercase hex digits.
    """
    return hmac.new(salt.encode("utf-8"), ip.strip().encode("utf-8"),
                    hashlib.sha256).hexdigest()


def parse_uuid(value: Optional[str]) -> Optional[str]:
    """Return a canonical UUID string, or None for anything else.

    Args:
        value: Candidate text.

    Returns:
        Optional[str]: The UUID, lower-case with hyphens.
    """
    if not value:
        return None
    try:
        return str(uuid.UUID(str(value).strip()))
    except ValueError:
        return None


# The fair-use day follows Japan time, where TMP is based (Lou, 2026-11-15).
# Japan has no daylight saving, so a fixed offset is exact.
FAIR_USE_DAY_TZ = timezone(timedelta(hours=9), "JST")


def day_window(now: datetime) -> datetime:
    """Start of the fair-use day (midnight Japan time).

    Args:
        now: Current time, timezone-aware.

    Returns:
        datetime: Midnight JST of that day, timezone-aware (equal to 15:00
        UTC the previous day).
    """
    local = now.astimezone(FAIR_USE_DAY_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def minute_window(now: datetime) -> datetime:
    """Start of the current minute (UTC).

    Args:
        now: Current time, timezone-aware.

    Returns:
        datetime: The minute's start.
    """
    return now.astimezone(timezone.utc).replace(second=0, microsecond=0)


# ----------------------------------------------------------------------
# Interfaces
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Visitor:
    """A signed-out visitor.

    Attributes:
        visitor_id: Their id (also their cookie value).
        questions_used: Free questions used so far.
        turnstile_verified: Whether they passed Turnstile once.
    """

    visitor_id: str
    questions_used: int
    turnstile_verified: bool = False


class VisitorStore(Protocol):
    """The free-question count per signed-out visitor."""

    def resolve(self, cookie_id: Optional[str], ip_hash: str,
                window_days: int) -> Visitor:
        """Find or create the visitor for this request.

        Args:
            cookie_id: Visitor id from the cookie, if any.
            ip_hash: Hashed client IP.
            window_days: How far back to match a cookieless visitor by
                hashed IP.

        Returns:
            Visitor: The visitor.
        """
        ...

    def consume_question(self, visitor_id: str, limit: int) -> bool:
        """Count one question if the visitor has any left.

        Args:
            visitor_id: The visitor.
            limit: Free questions allowed.

        Returns:
            bool: True if counted; False when the limit is reached.
        """
        ...

    def refund_question(self, visitor_id: str) -> None:
        """Give back a question whose answer failed.

        Args:
            visitor_id: The visitor.
        """
        ...

    def mark_verified(self, visitor_id: str) -> None:
        """Record that the visitor passed Turnstile.

        Args:
            visitor_id: The visitor.
        """
        ...


class RateLimiter(Protocol):
    """Fixed-window counters."""

    def hit(self, scope: str, key: str, window_start: datetime,
            limit: int) -> bool:
        """Count one request if the window has room.

        Args:
            scope: Counter family, for example ``fair_use_daily``.
            key: Who is counted (user id, visitor id, IP hash).
            window_start: Start of the current window.
            limit: Requests allowed per window.

        Returns:
            bool: True if counted; False when the window is full.
        """
        ...


@dataclass(frozen=True)
class UsageRow:
    """One usage_log row (no question or answer text).

    Attributes:
        model: Writer model id.
        language: Reply locale.
        plan: ``anonymous``, ``free`` or ``access``.
        message_id: Wire id of the assistant turn.
        conversation_id: Stored conversation, if any.
        visitor_id: Signed-out visitor, if any.
        user_id: Signed-in user, if any.
        input_tokens: Fresh input tokens.
        output_tokens: Output tokens.
        cache_read_tokens: Cache reads.
        cache_write_tokens: Cache writes.
        cost_usd: Turn cost.
        prompt_cache_enabled: Whether the stable block was marked.
        stop_reason: Wire stop reason or error code.
        fallback_used: Whether the fallback writer answered.
        source: ``chat`` or ``eval``.
    """

    model: str
    language: str
    plan: str
    message_id: str
    conversation_id: Optional[str] = None
    visitor_id: Optional[str] = None
    user_id: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    prompt_cache_enabled: bool = False
    stop_reason: Optional[str] = None
    fallback_used: bool = False
    source: str = "chat"


class UsageLog(Protocol):
    """Per-turn spend."""

    def record(self, row: UsageRow) -> None:
        """Write one row.

        Args:
            row: The row.
        """
        ...


@dataclass(frozen=True)
class Owner:
    """Who owns a conversation: exactly one of the two is set.

    Attributes:
        user_id: Signed-in user.
        visitor_id: Signed-out visitor.
    """

    user_id: Optional[str] = None
    visitor_id: Optional[str] = None


@dataclass(frozen=True)
class StoredTurn:
    """One question and answer to store.

    Attributes:
        language: Reply locale.
        audience: "Asking as" key, or None.
        question: The visitor's message.
        answer: The answer text (may be empty when cards carry it).
        wire_message_id: The assistant turn's wire id.
        events: Cards shown, or None.
    """

    language: str
    audience: Optional[str]
    question: str
    answer: str
    wire_message_id: str
    events: Optional[dict[str, Any]] = None


class ConversationStore(Protocol):
    """Stored conversations (ADR from NINOBOT 0048: history server-side).
    """

    def save_turn(self, owner: Owner, conversation_id: Optional[str],
                  turn: StoredTurn) -> str:
        """Append a turn, creating the conversation when needed.

        Args:
            owner: The owner.
            conversation_id: The conversation the client says it is in;
                ignored unless the owner owns it.
            turn: What to store.

        Returns:
            str: The conversation id the turn was stored in.
        """
        ...

    def claim(self, visitor_id: str, user_id: str) -> int:
        """Attach a signed-out visitor's conversations to an account.

        Called when a signed-in request still carries the visitor
        cookie, so the conversation begun before sign-in continues under
        the account (spec 3b). Idempotent: conversations that already
        belong to an account are left alone.

        Args:
            visitor_id: The visitor id from the cookie.
            user_id: The signed-in user.

        Returns:
            int: How many conversations moved.
        """
        ...


# ----------------------------------------------------------------------
# In memory
# ----------------------------------------------------------------------


@dataclass
class _MemVisitor:
    visitor_id: str
    ip_hash: str
    questions_used: int = 0
    verified: bool = False
    claimed: bool = False
    last_seen: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))


class MemoryVisitorStore:
    """In-memory free-question counts (same rules as Postgres)."""

    def __init__(self) -> None:
        """Create an empty store."""
        self.visitors: dict[str, _MemVisitor] = {}
        self._lock = threading.Lock()

    def resolve(self, cookie_id: Optional[str], ip_hash: str,
                window_days: int) -> Visitor:
        """See VisitorStore.resolve."""
        now = datetime.now(timezone.utc)
        with self._lock:
            row = self.visitors.get(parse_uuid(cookie_id) or "")
            if row is None or row.claimed:
                since = now - timedelta(days=window_days)
                matches = [v for v in self.visitors.values()
                           if v.ip_hash == ip_hash and not v.claimed
                           and v.last_seen > since]
                matches.sort(key=lambda v: (-v.questions_used,
                                            -v.last_seen.timestamp()))
                row = matches[0] if matches else None
            if row is None:
                row = _MemVisitor(str(uuid.uuid4()), ip_hash)
                self.visitors[row.visitor_id] = row
            row.last_seen = now
            row.ip_hash = ip_hash
            return Visitor(row.visitor_id, row.questions_used, row.verified)

    def consume_question(self, visitor_id: str, limit: int) -> bool:
        """See VisitorStore.consume_question."""
        with self._lock:
            row = self.visitors[visitor_id]
            if row.questions_used >= limit:
                return False
            row.questions_used += 1
            return True

    def refund_question(self, visitor_id: str) -> None:
        """See VisitorStore.refund_question."""
        with self._lock:
            row = self.visitors[visitor_id]
            row.questions_used = max(row.questions_used - 1, 0)

    def mark_verified(self, visitor_id: str) -> None:
        """See VisitorStore.mark_verified."""
        with self._lock:
            self.visitors[visitor_id].verified = True


class MemoryRateLimiter:
    """In-memory fixed-window counters."""

    def __init__(self) -> None:
        """Create an empty limiter."""
        self.counts: dict[tuple[str, str, datetime], int] = {}
        self._lock = threading.Lock()

    def hit(self, scope: str, key: str, window_start: datetime,
            limit: int) -> bool:
        """See RateLimiter.hit."""
        slot = (scope, key, window_start)
        with self._lock:
            count = self.counts.get(slot, 0)
            if count >= limit:
                return False
            self.counts[slot] = count + 1
            return True


class MemoryUsageLog:
    """Keeps usage rows in a list."""

    def __init__(self) -> None:
        """Create an empty log."""
        self.rows: list[UsageRow] = []

    def record(self, row: UsageRow) -> None:
        """See UsageLog.record."""
        self.rows.append(row)


class MemoryConversationStore:
    """Keeps conversations in a dict."""

    def __init__(self) -> None:
        """Create an empty store."""
        self.owners: dict[str, Owner] = {}
        self.messages: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def save_turn(self, owner: Owner, conversation_id: Optional[str],
                  turn: StoredTurn) -> str:
        """See ConversationStore.save_turn."""
        with self._lock:
            cid = parse_uuid(conversation_id)
            if cid is None or self.owners.get(cid) != owner:
                cid = str(uuid.uuid4())
                self.owners[cid] = owner
                self.messages[cid] = []
            rows = self.messages[cid]
            rows.append({"seq": len(rows), "role": "user",
                         "content": turn.question})
            if turn.answer or turn.events:
                rows.append({"seq": len(rows), "role": "assistant",
                             "content": turn.answer,
                             "wire_message_id": turn.wire_message_id,
                             "events": turn.events})
            return cid

    def claim(self, visitor_id: str, user_id: str) -> int:
        """See ConversationStore.claim."""
        before = Owner(visitor_id=visitor_id)
        after = Owner(user_id=user_id)
        with self._lock:
            moved = [cid for cid, owner in self.owners.items()
                     if owner == before]
            for cid in moved:
                self.owners[cid] = after
            return len(moved)


# ----------------------------------------------------------------------
# Postgres
# ----------------------------------------------------------------------


class PgVisitorStore:
    """Free-question counts in ``anonymous_visitors``.

    Args:
        db: Database helper (service connection; the table is
            operator-only).
    """

    def __init__(self, db: Database) -> None:
        """Create the store.

        Args:
            db: Database helper.
        """
        self.db = db

    def resolve(self, cookie_id: Optional[str], ip_hash: str,
                window_days: int) -> Visitor:
        """See VisitorStore.resolve.

        The cookie wins when it names an unclaimed visitor. Otherwise
        (no cookie, a cleared or forged one) the busiest unclaimed
        visitor seen from the same hashed IP within the window is
        reused, so clearing cookies does not reset the count.
        """
        cols = ("visitor_id::TEXT, questions_used, "
                "turnstile_verified_at IS NOT NULL")
        cid = parse_uuid(cookie_id)
        with self.db.connect() as conn:
            row = None
            if cid:
                row = conn.execute(
                    "UPDATE public.anonymous_visitors "
                    "SET last_seen_at = now(), ip_hash = %s "
                    "WHERE visitor_id = %s AND claimed_by IS NULL "
                    f"RETURNING {cols}",
                    (ip_hash, cid),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    "UPDATE public.anonymous_visitors "
                    "SET last_seen_at = now() "
                    "WHERE visitor_id = ("
                    "  SELECT visitor_id FROM public.anonymous_visitors "
                    "  WHERE ip_hash = %s AND claimed_by IS NULL "
                    "    AND last_seen_at > now() "
                    "        - make_interval(days => %s) "
                    "  ORDER BY questions_used DESC, last_seen_at DESC "
                    "  LIMIT 1 FOR UPDATE) "
                    f"RETURNING {cols}",
                    (ip_hash, window_days),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    "INSERT INTO public.anonymous_visitors (ip_hash) "
                    f"VALUES (%s) RETURNING {cols}",
                    (ip_hash,),
                ).fetchone()
        return Visitor(row[0], int(row[1]), bool(row[2]))

    def consume_question(self, visitor_id: str, limit: int) -> bool:
        """See VisitorStore.consume_question (one atomic UPDATE)."""
        with self.db.connect() as conn:
            row = conn.execute(
                "UPDATE public.anonymous_visitors "
                "SET questions_used = questions_used + 1, "
                "    last_question_at = now(), last_seen_at = now() "
                "WHERE visitor_id = %s AND questions_used < %s "
                "RETURNING questions_used",
                (visitor_id, limit),
            ).fetchone()
        return row is not None

    def refund_question(self, visitor_id: str) -> None:
        """See VisitorStore.refund_question."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE public.anonymous_visitors "
                "SET questions_used = greatest(questions_used - 1, 0) "
                "WHERE visitor_id = %s",
                (visitor_id,),
            )

    def mark_verified(self, visitor_id: str) -> None:
        """See VisitorStore.mark_verified."""
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE public.anonymous_visitors "
                "SET turnstile_verified_at = now() WHERE visitor_id = %s",
                (visitor_id,),
            )


class PgRateLimiter:
    """Fixed-window counters in ``request_rate_limits``.

    Args:
        db: Database helper.
    """

    def __init__(self, db: Database) -> None:
        """Create the limiter.

        Args:
            db: Database helper.
        """
        self.db = db

    def hit(self, scope: str, key: str, window_start: datetime,
            limit: int) -> bool:
        """See RateLimiter.hit (one atomic upsert)."""
        with self.db.connect() as conn:
            row = conn.execute(
                "INSERT INTO public.request_rate_limits "
                "(scope, key, window_start_at, count) "
                "VALUES (%s, %s, %s, 1) "
                "ON CONFLICT (scope, key, window_start_at) DO UPDATE "
                "SET count = public.request_rate_limits.count + 1 "
                "WHERE public.request_rate_limits.count < %s "
                "RETURNING count",
                (scope, key, window_start, limit),
            ).fetchone()
        return row is not None


class PgUsageLog:
    """Rows in ``usage_log``.

    Args:
        db: Database helper.
    """

    def __init__(self, db: Database) -> None:
        """Create the log.

        Args:
            db: Database helper.
        """
        self.db = db

    def record(self, row: UsageRow) -> None:
        """See UsageLog.record."""
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO public.usage_log (source, call_kind, model, "
                "language, plan, message_id, conversation_id, visitor_id, "
                "user_id, input_tokens, output_tokens, cache_read_tokens, "
                "cache_write_tokens, cost_usd, prompt_cache_enabled, "
                "stop_reason, fallback_used) VALUES (%s, 'writer', %s, "
                "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (row.source, row.model, row.language, row.plan,
                 row.message_id, row.conversation_id, row.visitor_id,
                 row.user_id, row.input_tokens, row.output_tokens,
                 row.cache_read_tokens, row.cache_write_tokens,
                 row.cost_usd, row.prompt_cache_enabled, row.stop_reason,
                 row.fallback_used),
            )


class PgConversationStore:
    """Rows in ``conversations`` and ``messages``.

    Args:
        db: Database helper.
    """

    def __init__(self, db: Database) -> None:
        """Create the store.

        Args:
            db: Database helper.
        """
        self.db = db

    def save_turn(self, owner: Owner, conversation_id: Optional[str],
                  turn: StoredTurn) -> str:
        """See ConversationStore.save_turn."""
        cid = parse_uuid(conversation_id)
        with self.db.connect() as conn:
            row = None
            if cid:
                row = conn.execute(
                    "SELECT conversation_id::TEXT FROM public.conversations "
                    "WHERE conversation_id = %s AND user_id "
                    "IS NOT DISTINCT FROM %s AND visitor_id "
                    "IS NOT DISTINCT FROM %s FOR UPDATE",
                    (cid, owner.user_id, owner.visitor_id),
                ).fetchone()
            if row is None:
                row = conn.execute(
                    "INSERT INTO public.conversations (user_id, "
                    "visitor_id, language, audience, title) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "RETURNING conversation_id::TEXT",
                    (owner.user_id, owner.visitor_id, turn.language,
                     turn.audience, turn.question.strip()[:120] or None),
                ).fetchone()
            cid = row[0]
            seq = conn.execute(
                "SELECT coalesce(max(seq) + 1, 0) FROM public.messages "
                "WHERE conversation_id = %s", (cid,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO public.messages (conversation_id, seq, role, "
                "content) VALUES (%s, %s, 'user', %s)",
                (cid, seq, turn.question),
            )
            if turn.answer or turn.events:
                conn.execute(
                    "INSERT INTO public.messages (conversation_id, seq, "
                    "role, content, wire_message_id, events) "
                    "VALUES (%s, %s, 'assistant', %s, %s, %s::jsonb)",
                    (cid, seq + 1, turn.answer, turn.wire_message_id,
                     json.dumps(turn.events) if turn.events else None),
                )
            conn.execute(
                "UPDATE public.conversations SET last_message_at = now() "
                "WHERE conversation_id = %s", (cid,),
            )
        return cid

    def claim(self, visitor_id: str, user_id: str) -> int:
        """See ConversationStore.claim (claim_visitor_conversations())."""
        vid, uid = parse_uuid(visitor_id), parse_uuid(user_id)
        if vid is None or uid is None:
            return 0
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT public.claim_visitor_conversations(%s, %s)",
                (vid, uid),
            ).fetchone()
        return int(row[0] or 0)
