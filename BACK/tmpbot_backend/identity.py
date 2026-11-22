# Who is asking: an anonymous visitor, a free account, or a subscriber.
#
# The Next.js proxy forwards the visitor's Supabase access token as
# `Authorization: Bearer …`. A valid token makes the caller a signed-in
# user; its `plan` claim set to `access` makes them a subscriber (the
# claim Phase 7's access-token hook stamps, and the one the database's
# row level security reads through tmpbot_request_plan()). No token, or
# one that fails the check, means an anonymous visitor: never more
# access than the token proves.
#
# Verifiers:
#   - JwksVerifier: Supabase's published signing keys (SUPABASE_URL),
#     adapted from NINOBOT's supabase_auth.py. Used when SUPABASE_URL is
#     set; never called in tests.
#   - StaticKeyVerifier: a fixed key, for tests and local work.
#   - NoTokenVerifier: accepts nothing; everyone is anonymous. The
#     default until Supabase exists (Phase 7).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Resolve the caller's identity and plan from the access token."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional, Protocol

import jwt

log = logging.getLogger("tmpbot.identity")

Plan = Literal["anonymous", "free", "access"]

_BEARER_RE = re.compile(r"^Bearer ([A-Za-z0-9\-._~+/]+=*)$")


@dataclass(frozen=True)
class Identity:
    """The caller of one chat turn.

    Attributes:
        plan: ``anonymous``, ``free`` or ``access``.
        user_id: Supabase user id for signed-in callers.
    """

    plan: Plan
    user_id: Optional[str] = None

    @property
    def signed_in(self) -> bool:
        """Whether the caller has an account."""
        return self.user_id is not None

    @property
    def subscriber(self) -> bool:
        """Whether the caller has TMP Access."""
        return self.plan == "access"


ANONYMOUS = Identity(plan="anonymous")


class TokenVerifier(Protocol):
    """Checks an access token and returns its claims."""

    def verify(self, token: str) -> dict[str, Any]:
        """Verify a token.

        Args:
            token: The JWT.

        Returns:
            dict[str, Any]: The verified claims.

        Raises:
            jwt.PyJWTError: When the token is invalid.
        """
        ...


class NoTokenVerifier:
    """Accepts no token: every caller is anonymous."""

    def verify(self, token: str) -> dict[str, Any]:
        """Refuse every token.

        Args:
            token: Ignored.

        Raises:
            jwt.InvalidTokenError: Always.
        """
        raise jwt.InvalidTokenError("sign-in is not configured")


class StaticKeyVerifier:
    """Verifies tokens against one fixed key (tests, local work).

    Args:
        key: HMAC secret or public key.
        algorithms: Accepted algorithms.
        audience: Expected ``aud``.
    """

    def __init__(self, key: Any, algorithms: tuple[str, ...] = ("HS256",),
                 audience: str = "authenticated") -> None:
        """Create the verifier.

        Args:
            key: Key.
            algorithms: Algorithms.
            audience: Audience.
        """
        self._key = key
        self._algorithms = list(algorithms)
        self._audience = audience

    def verify(self, token: str) -> dict[str, Any]:
        """Verify signature, expiry and audience.

        Args:
            token: The JWT.

        Returns:
            dict[str, Any]: Claims.
        """
        return jwt.decode(token, self._key, algorithms=self._algorithms,
                          audience=self._audience,
                          options={"require": ["exp", "sub"]})


class JwksVerifier:
    """Verifies Supabase access tokens against the project's JWKS.

    Args:
        supabase_url: Project URL, for example
            ``https://abc.supabase.co``.
        audience: Expected ``aud`` (``authenticated``).
    """

    def __init__(self, supabase_url: str,
                 audience: str = "authenticated") -> None:
        """Create the verifier.

        Args:
            supabase_url: Project URL.
            audience: Audience.
        """
        base = supabase_url.rstrip("/")
        self._issuer = f"{base}/auth/v1"
        self._audience = audience
        self._jwks = jwt.PyJWKClient(
            f"{self._issuer}/.well-known/jwks.json", cache_keys=True,
            lifespan=600,
        )

    def verify(self, token: str) -> dict[str, Any]:
        """Verify a Supabase access token.

        Args:
            token: The JWT.

        Returns:
            dict[str, Any]: Claims.
        """
        key = self._jwks.get_signing_key_from_jwt(token)
        return jwt.decode(
            token, key.key, algorithms=["ES256", "RS256"],
            audience=self._audience, issuer=self._issuer,
            options={"require": ["exp", "sub"]},
        )


def identity_from_claims(claims: dict[str, Any]) -> Identity:
    """Turn verified claims into an identity.

    Args:
        claims: Verified JWT claims.

    Returns:
        Identity: ``access`` when the ``plan`` claim says so, else
        ``free``; anonymous when ``sub`` is not a UUID or the role is
        not ``authenticated``.
    """
    sub = claims.get("sub")
    try:
        user_id = str(uuid.UUID(str(sub)))
    except (ValueError, TypeError):
        return ANONYMOUS
    if claims.get("role", "authenticated") != "authenticated":
        return ANONYMOUS
    plan: Plan = "access" if claims.get("plan") == "access" else "free"
    return Identity(plan=plan, user_id=user_id)


def resolve_identity(authorization: Optional[str],
                     verifier: TokenVerifier) -> Identity:
    """Resolve the caller from the Authorization header.

    Args:
        authorization: The header value, if any.
        verifier: Token verifier.

    Returns:
        Identity: The caller; anonymous when the header is absent,
        malformed or fails verification.
    """
    match = _BEARER_RE.match((authorization or "").strip())
    if not match:
        return ANONYMOUS
    try:
        claims = verifier.verify(match.group(1))
    except jwt.PyJWTError as exc:
        log.info("access token rejected; treating caller as anonymous",
                 extra={"reason": type(exc).__name__})
        return ANONYMOUS
    except Exception:  # noqa: BLE001 - JWKS fetch failures and the like
        log.exception("access token check failed; caller anonymous")
        return ANONYMOUS
    return identity_from_claims(claims)
