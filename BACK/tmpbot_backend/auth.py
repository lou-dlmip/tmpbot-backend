# Shared-secret check between the Next.js chat proxy and this backend.
#
# The browser never calls the backend directly. FRNT/app/api/chat/route.ts
# (Phase 2b) reads TMPBOT_SHARED_SECRET on the server and sends it in the
# X-TMPBOT-Proxy-Secret header. Adapted from NINOBOT's auth.py
# (X-Demo-Auth / DEMO_API_SECRET). User identity (Supabase JWT) is a
# separate check added in Phase 7; this one only proves the caller is
# our own proxy.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""FastAPI dependency enforcing the proxy shared secret."""

from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, Request, status

log = logging.getLogger("tmpbot.auth")

PROXY_SECRET_HEADER = "X-TMPBOT-Proxy-Secret"


def require_proxy_secret(
    request: Request,
    x_tmpbot_proxy_secret: Optional[str] = Header(
        default=None, alias=PROXY_SECRET_HEADER
    ),
) -> None:
    """Reject a request that does not carry the proxy's shared secret.

    Compares in constant time (``hmac.compare_digest``). Fails closed: if
    the backend has no secret configured, every request is refused, so a
    deploy that forgot the variable never serves the chat openly.

    Args:
        request: The incoming request; the expected secret is read from
            ``request.app.state.settings``.
        x_tmpbot_proxy_secret: Value of the ``X-TMPBOT-Proxy-Secret``
            header, if present.

    Raises:
        HTTPException: 401 when the secret is unset on the backend, the
            header is missing, or the values differ.
    """
    expected = request.app.state.settings.shared_secret
    if not expected:
        log.warning(
            "TMPBOT_SHARED_SECRET is not set; refusing chat request",
            extra={"path": request.url.path},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Proxy secret not configured on backend.",
        )
    supplied = x_tmpbot_proxy_secret or ""
    if not hmac.compare_digest(
        supplied.encode("utf-8"), expected.encode("utf-8")
    ):
        log.info(
            "Proxy secret missing or wrong",
            extra={"path": request.url.path, "header_present": bool(supplied)},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing proxy secret.",
        )
