# Admission checks run before any model spend (M5.4, BOOTSTRAP 9:
# "Limits and Turnstile go live before the URL is public").
#
# In order, for one chat turn:
#   1. Signed-out callers are matched to a visitor row (cookie, else
#      hashed IP), so clearing cookies does not reset the count.
#   2. Burst limits: questions per minute per visitor or account, and per
#      hashed IP.
#   3. Signed out with all free questions used: the free-limit card
#      (S14). Checked before Turnstile so the card always shows.
#   4. Signed out and not yet verified: a Turnstile token is required.
#   5. Signed out: one free question is counted (atomically).
#   6. Free account: the fair-use daily cap (S18), midnight to midnight
#      Japan time. Subscribers have no daily cap.
#
# Any database failure refuses the turn: an uncounted anonymous question
# is exactly the cost risk these checks exist for.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Decide whether a chat turn may run."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

from .config import Settings
from .identity import Identity
from .stores import (
    RateLimiter,
    Visitor,
    VisitorStore,
    day_window,
    hash_ip,
    minute_window,
)
from .turnstile import TurnstileVerifier

log = logging.getLogger("tmpbot.gatekeeper")

Refusal = Literal[
    "free_limit", "fair_use", "burst", "turnstile", "unavailable"
]

SCOPE_FAIR_USE = "fair_use_daily"
SCOPE_BURST_IDENTITY = "burst_identity"
SCOPE_BURST_IP = "burst_ip"


@dataclass(frozen=True)
class Admission:
    """The outcome of the checks.

    Attributes:
        visitor: The signed-out visitor, if any (its id goes back to the
            proxy for the cookie, even on a refusal).
        refusal: Why the turn may not run, or None when it may.
        counted_question: Whether a free question was counted (it is
            given back if the answer fails).
    """

    visitor: Optional[Visitor] = None
    refusal: Optional[Refusal] = None
    counted_question: bool = False


async def admit(
    *,
    settings: Settings,
    identity: Identity,
    visitors: VisitorStore,
    limiter: RateLimiter,
    turnstile: TurnstileVerifier,
    cookie_id: Optional[str],
    client_ip: str,
    turnstile_token: Optional[str],
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Admission:
    """Run the admission checks for one turn.

    Args:
        settings: Limits and salt.
        identity: The caller.
        visitors: Free-question store.
        limiter: Rate limiter.
        turnstile: Turnstile verifier.
        cookie_id: Visitor id from the cookie.
        client_ip: The visitor's IP ("" when unknown).
        turnstile_token: Token from the widget, if sent.
        now: Clock, for tests.

    Returns:
        Admission: The outcome.
    """
    ip_hash = hash_ip(client_ip, settings.ip_hash_salt)
    visitor: Optional[Visitor] = None
    try:
        if not identity.signed_in:
            visitor = await asyncio.to_thread(
                visitors.resolve, cookie_id, ip_hash,
                settings.visitor_ip_window_days,
            )
        key = identity.user_id or (visitor.visitor_id if visitor else "")
        minute = minute_window(now())
        if not await asyncio.to_thread(
            limiter.hit, SCOPE_BURST_IDENTITY, key, minute,
            settings.rate_per_minute,
        ) or not await asyncio.to_thread(
            limiter.hit, SCOPE_BURST_IP, ip_hash, minute,
            settings.rate_per_ip_per_minute,
        ):
            log.info("burst limit reached", extra={"plan": identity.plan})
            return Admission(visitor, "burst")

        if visitor is not None:
            if visitor.questions_used >= settings.free_question_limit:
                return Admission(visitor, "free_limit")
            if turnstile.enforced and not visitor.turnstile_verified:
                result = await turnstile.verify(turnstile_token,
                                                client_ip or None)
                if not result.ok:
                    log.info("turnstile check failed",
                             extra={"reason": result.reason})
                    return Admission(visitor, "turnstile")
                await asyncio.to_thread(visitors.mark_verified,
                                        visitor.visitor_id)
            counted = await asyncio.to_thread(
                visitors.consume_question, visitor.visitor_id,
                settings.free_question_limit,
            )
            if not counted:
                return Admission(visitor, "free_limit")
            return Admission(visitor, None, counted_question=True)

        if identity.plan == "free":
            allowed = await asyncio.to_thread(
                limiter.hit, SCOPE_FAIR_USE, identity.user_id or "",
                day_window(now()), settings.fair_use_daily_cap,
            )
            if not allowed:
                log.info("fair-use daily cap reached")
                return Admission(None, "fair_use")
        return Admission(visitor, None)
    except Exception:  # noqa: BLE001 - any store failure fails closed
        log.exception("admission checks failed; refusing the turn")
        return Admission(visitor, "unavailable")
