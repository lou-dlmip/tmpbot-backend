# Cloudflare Turnstile verification behind one interface.
#
# A signed-out visitor proves once that they are a person before their
# first question; the stamp is kept on their visitor row (migration
# 0007). Signed-in users are not asked: sign-up has its own check
# (Phase 7). Adapted from NINOBOT's FRNT/lib/turnstile.ts, moved into the
# backend so the free count, the limits and this check live together.
#
#   - CloudflareTurnstile: the real siteverify call (needs
#     TURNSTILE_SECRET_KEY). Never called in tests.
#   - FakeTurnstile: accepts a fixed set of tokens (tests).
#   - DisabledTurnstile: no check at all. Allowed in local and staging
#     only; production refuses to start without a secret (config.py).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Turnstile token verification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import httpx

log = logging.getLogger("tmpbot.turnstile")

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
MAX_TOKEN_CHARS = 2048


@dataclass(frozen=True)
class TurnstileResult:
    """Outcome of one verification.

    Attributes:
        ok: Whether the token is valid.
        reason: Why not, for logs.
    """

    ok: bool
    reason: str = ""


class TurnstileVerifier(Protocol):
    """Checks a Turnstile token."""

    enforced: bool

    async def verify(self, token: Optional[str],
                     remote_ip: Optional[str]) -> TurnstileResult:
        """Verify a token.

        Args:
            token: The widget's response token.
            remote_ip: The visitor's IP, passed to Cloudflare.

        Returns:
            TurnstileResult: The outcome.
        """
        ...


class DisabledTurnstile:
    """No check (local and staging without a secret)."""

    enforced = False

    async def verify(self, token: Optional[str],
                     remote_ip: Optional[str]) -> TurnstileResult:
        """Accept everything.

        Args:
            token: Ignored.
            remote_ip: Ignored.

        Returns:
            TurnstileResult: Always ok.
        """
        return TurnstileResult(True, "disabled")


class FakeTurnstile:
    """Accepts a fixed set of tokens.

    Args:
        valid_tokens: Tokens that pass.
    """

    enforced = True

    def __init__(self, valid_tokens: Iterable[str] = ("pass",)) -> None:
        """Create the fake.

        Args:
            valid_tokens: Tokens that pass.
        """
        self.valid = set(valid_tokens)
        self.seen: list[Optional[str]] = []

    async def verify(self, token: Optional[str],
                     remote_ip: Optional[str]) -> TurnstileResult:
        """Pass listed tokens only.

        Args:
            token: The token.
            remote_ip: Ignored.

        Returns:
            TurnstileResult: The outcome.
        """
        self.seen.append(token)
        if not token:
            return TurnstileResult(False, "missing_token")
        return TurnstileResult(token in self.valid,
                               "" if token in self.valid else "invalid")


class CloudflareTurnstile:
    """Cloudflare's siteverify endpoint.

    Args:
        secret: ``TURNSTILE_SECRET_KEY``.
        client: An ``httpx.AsyncClient``; tests pass a mock transport.
    """

    enforced = True

    def __init__(self, secret: str,
                 client: Optional[httpx.AsyncClient] = None) -> None:
        """Create the verifier.

        Args:
            secret: Secret key.
            client: HTTP client.
        """
        self._secret = secret
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def verify(self, token: Optional[str],
                     remote_ip: Optional[str]) -> TurnstileResult:
        """Ask Cloudflare whether the token is valid.

        Args:
            token: The token.
            remote_ip: The visitor's IP.

        Returns:
            TurnstileResult: The outcome; network failures fail closed.
        """
        if not token:
            return TurnstileResult(False, "missing_token")
        if len(token) > MAX_TOKEN_CHARS:
            return TurnstileResult(False, "token_too_long")
        form = {"secret": self._secret, "response": token}
        if remote_ip:
            form["remoteip"] = remote_ip
        try:
            resp = await self._client.post(SITEVERIFY_URL, data=form)
            body = resp.json() if resp.status_code == 200 else {}
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("turnstile siteverify failed",
                        extra={"error": type(exc).__name__})
            return TurnstileResult(False, "siteverify_error")
        if body.get("success") is True:
            return TurnstileResult(True)
        codes = ",".join(body.get("error-codes") or []) or "rejected"
        return TurnstileResult(False, codes)
