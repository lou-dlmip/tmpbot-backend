# FastAPI application for the TMPBOT backend.
#
# GET /health, POST /api/v1/chat (the chat, streamed as server-sent
# events), POST /api/v1/claim (attach a visitor's conversations to the
# account after sign-in) and POST /api/v1/downloads/{id} (download links
# for subscribers), behind the proxy shared secret. JSON logs on stdout,
# Sentry only when SENTRY_DSN_BACKEND is set. The app is built by a
# factory so importing this module has no side effects; run it with
#
#   BACK/.venv/bin/uvicorn --app-dir BACK --factory \
#       tmpbot_backend.main:create_app --host 127.0.0.1 --port 8100
#
# No CORS middleware: browsers never call this service. They call the
# Next.js proxy (/api/chat), which calls here server to server and passes
# the visitor cookie, client IP, Turnstile token and access token in
# headers (DOCS/STND/streaming_wire_format.md).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""TMPBOT backend application factory and routes."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
import uuid
from typing import AsyncIterator, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import __version__, wire
from .auth import require_proxy_secret
from .canned import stream_canned_reply
from .config import Settings, load_settings
from .gatekeeper import Admission, admit
from .identity import Identity, resolve_identity
from .locales import LOCALE_PATTERN
from .logging_setup import configure_logging
from .observability import init_sentry
from .orchestrator import Turn, stream_turn
from .prompt import AUDIENCES
from .runtime import Runtime, build_runtime
from .stores import parse_uuid

log = logging.getLogger("tmpbot.main")

REQUEST_ID_HEADER = "X-Request-Id"
# Set by the Next.js proxy (lib/chat-proxy.ts). Trusted only because the
# proxy secret has already been checked.
VISITOR_HEADER = "X-TMPBOT-Visitor-Id"
CLIENT_IP_HEADER = "X-TMPBOT-Client-IP"
TURNSTILE_HEADER = "X-TMPBOT-Turnstile-Token"

# Request size limits. The handoff caps answers at about 90 words, so
# real conversations sit far below these; they stop a caller sending a
# multi-megabyte body.
MAX_MESSAGES = 200
MAX_MESSAGE_CHARS = 4000
MAX_REQUEST_CONTENT_CHARS = 100_000

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    # Stops nginx-style proxies (Render's edge included) from buffering
    # the stream, which would turn token-by-token output into one lump.
    "X-Accel-Buffering": "no",
}


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------


class ChatMessage(BaseModel):
    """One message of the conversation sent with a chat request."""

    id: Optional[str] = Field(default=None, max_length=64)
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``.

    The shape follows NINOBOT's v1 request (language plus inline
    history), plus the optional audience and conversation id (v1.1).
    """

    language: str = Field(..., pattern=LOCALE_PATTERN)
    messages: list[ChatMessage] = Field(
        ..., min_length=1, max_length=MAX_MESSAGES
    )
    # "Asking as" picker (D10): one of prompt.AUDIENCES' keys.
    audience: Optional[
        Literal["investor", "corporate", "government", "funder",
                "non_profit", "just_curious"]
    ] = None
    # The stored conversation this turn continues (from the last `done`).
    conversation_id: Optional[str] = Field(default=None, max_length=64)
    # Forces a card event so the frontend can draw every card before the
    # real chat exists. Ignored in production.
    demo: Optional[Literal["cases", "gate", "limit", "download"]] = None

    @model_validator(mode="after")
    def _check_conversation(self) -> "ChatRequest":
        """Require a final user message and bound the total size.

        Returns:
            ChatRequest: The validated request.

        Raises:
            ValueError: If the last message is not from the user, or the
                messages together exceed MAX_REQUEST_CONTENT_CHARS.
                FastAPI turns this into a 422 response.
        """
        if self.messages[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        total = sum(len(m.content) for m in self.messages)
        if total > MAX_REQUEST_CONTENT_CHARS:
            raise ValueError(
                f"messages total {total} characters, over the "
                f"{MAX_REQUEST_CONTENT_CHARS} limit"
            )
        return self


# ----------------------------------------------------------------------
# Access log middleware
# ----------------------------------------------------------------------


class AccessLogMiddleware:
    """Log one JSON line per request and tag responses with a request id.

    Written as plain ASGI middleware rather than Starlette's
    ``BaseHTTPMiddleware``, which wraps the response body and can hold
    back a streamed response.

    Args:
        app: The wrapped ASGI app.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``.

        Args:
            app: The ASGI application to wrap.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive,
                       send: Send) -> None:
        """Handle one ASGI call.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append(
                    (REQUEST_ID_HEADER.lower().encode(), request_id.encode())
                )
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            log.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status_code,
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000, 1
                    ),
                },
            )


def _incoming_request_id(scope: Scope) -> Optional[str]:
    """Return a safe ``X-Request-Id`` sent by the proxy, if any.

    Args:
        scope: ASGI connection scope.

    Returns:
        Optional[str]: The header value when it is 8 to 64 characters of
        letters, digits and hyphens; otherwise None.
    """
    for key, value in scope.get("headers", []):
        if key == REQUEST_ID_HEADER.lower().encode():
            text = value.decode("latin-1")
            if 8 <= len(text) <= 64 and all(
                c.isalnum() or c == "-" for c in text
            ):
                return text
    return None


# ----------------------------------------------------------------------
# Application factory
# ----------------------------------------------------------------------


def create_app(settings: Optional[Settings] = None,
               runtime: Optional[Runtime] = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Settings to use; read from the environment when None.
        runtime: Prebuilt collaborators (tests); built from settings
            when None, after ``Settings.check()``.

    Returns:
        FastAPI: The configured application.

    Raises:
        ConfigError: If settings are invalid or unsafe for their
            environment (only when ``runtime`` is None).
    """
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    if runtime is None:
        settings.check()
        runtime = build_runtime(settings)
    init_sentry(settings)

    app = FastAPI(
        title="TMPBOT backend",
        version=__version__,
        # Interactive docs only off production; the API is private.
        docs_url=None if settings.env == "production" else "/docs",
        redoc_url=None,
        openapi_url=None if settings.env == "production" else "/openapi.json",
    )
    app.state.settings = settings
    app.state.runtime = runtime
    app.add_middleware(AccessLogMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe for Render's health check.

        Returns:
            dict[str, str]: Status, version and environment name.
        """
        return {"status": "ok", "version": __version__, "env": settings.env}

    @app.post("/api/v1/chat", dependencies=[Depends(require_proxy_secret)])
    async def chat(
        req: ChatRequest,
        request: Request,
        authorization: Optional[str] = Header(default=None),
        visitor_cookie: Optional[str] = Header(default=None,
                                               alias=VISITOR_HEADER),
        client_ip_header: Optional[str] = Header(default=None,
                                                 alias=CLIENT_IP_HEADER),
        turnstile_token: Optional[str] = Header(default=None,
                                                alias=TURNSTILE_HEADER),
    ) -> StreamingResponse:
        """Answer one chat turn as a stream of SSE records.

        Args:
            req: The validated chat request.
            request: The raw request, for the request id and client.
            authorization: ``Bearer`` access token, if signed in.
            visitor_cookie: Visitor id from the proxy's cookie.
            client_ip_header: The visitor's IP, from the proxy.
            turnstile_token: Turnstile token, when the browser sent one.

        Returns:
            StreamingResponse: ``text/event-stream`` body of SSE records.
            A signed-out visitor's id comes back in the
            ``X-TMPBOT-Visitor-Id`` header for the proxy's cookie.
        """
        message_id = wire.new_message_id()
        request_id = request.scope.get("state", {}).get("request_id")
        demo = req.demo
        if demo is not None and not settings.demo_allowed:
            log.info("demo switch ignored in production",
                     extra={"demo": demo})
            demo = None
        if demo is not None:
            return _sse(stream_canned_reply(
                req.language, message_id, demo=demo,
                delay_s=settings.canned_token_delay_s), message_id)

        identity = resolve_identity(authorization, runtime.verifier)
        if identity.signed_in:
            # Signed in on a browser that chatted signed out: carry its
            # conversations over before this turn looks for its own.
            await _claim(runtime, visitor_cookie, identity)
        client_ip = _client_ip(client_ip_header, request)
        admission = await admit(
            settings=settings, identity=identity,
            visitors=runtime.visitors, limiter=runtime.limiter,
            turnstile=runtime.turnstile, cookie_id=visitor_cookie,
            client_ip=client_ip, turnstile_token=turnstile_token,
        )
        headers = {}
        if admission.visitor is not None:
            headers[VISITOR_HEADER] = admission.visitor.visitor_id
        log.info(
            "chat turn",
            extra={
                "request_id": request_id,
                "message_id": message_id,
                "language": req.language,
                "history_len": len(req.messages),
                "plan": identity.plan,
                "refusal": admission.refusal,
            },
        )
        if admission.refusal is not None:
            return _sse(_refusal_stream(admission, message_id,
                                        req.language),
                        message_id, headers)

        turn = Turn(
            language=req.language,
            history=[(m.role, m.content) for m in req.messages],
            identity=identity,
            message_id=message_id,
            audience=req.audience if req.audience in AUDIENCES else None,
            conversation_id=parse_uuid(req.conversation_id),
            visitor_id=(admission.visitor.visitor_id
                        if admission.visitor else None),
            on_failure=_refund(runtime, admission),
        )
        return _sse(stream_turn(turn, runtime.deps), message_id, headers)

    @app.post("/api/v1/claim", dependencies=[Depends(require_proxy_secret)])
    async def claim(
        authorization: Optional[str] = Header(default=None),
        visitor_cookie: Optional[str] = Header(default=None,
                                               alias=VISITOR_HEADER),
    ) -> JSONResponse:
        """Attach the visitor's signed-out conversations to the account.

        The Next.js sign-in routes call this right after a sign-in, so
        the conversations are the account's at once (the next chat turn
        would also do it).

        Args:
            authorization: ``Bearer`` access token.
            visitor_cookie: Visitor id from the proxy's cookie.

        Returns:
            JSONResponse: ``{"claimed": n}``.

        Raises:
            HTTPException: 401 without a valid access token.
        """
        identity = resolve_identity(authorization, runtime.verifier)
        if not identity.signed_in:
            raise HTTPException(status_code=401, detail="sign_in_required")
        moved = await _claim(runtime, visitor_cookie, identity)
        return JSONResponse({"claimed": moved},
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/v1/downloads/{publication_id}",
              dependencies=[Depends(require_proxy_secret)])
    async def download(
        publication_id: str,
        authorization: Optional[str] = Header(default=None),
    ) -> JSONResponse:
        """Issue a short-lived download link to a subscriber.

        The subscription is checked now, not when the card was drawn
        (spec 3b).

        Args:
            publication_id: The card's publication id.
            authorization: ``Bearer`` access token.

        Returns:
            JSONResponse: ``{"url": ...}``.

        Raises:
            HTTPException: 403 without TMP Access, 404 for an id that is
                not published, 503 while downloads are not set up.
        """
        identity: Identity = resolve_identity(authorization,
                                              runtime.verifier)
        if not identity.subscriber:
            raise HTTPException(status_code=403, detail="tmp_access_required")
        if runtime.catalog.publication(publication_id) is None:
            raise HTTPException(status_code=404, detail="not_published")
        url = await runtime.downloads.signed_url(publication_id)
        if url is None:
            raise HTTPException(status_code=503,
                                detail="downloads_not_configured")
        return JSONResponse({"url": url},
                            headers={"Cache-Control": "no-store"})

    return app


def _sse(records: AsyncIterator[str], message_id: str,
         headers: Optional[dict[str, str]] = None) -> StreamingResponse:
    """Wrap SSE records in a streaming response.

    Args:
        records: Encoded records.
        message_id: Assistant turn id, for logs.
        headers: Extra response headers.

    Returns:
        StreamingResponse: The response.
    """
    return StreamingResponse(
        _guarded_stream(records, message_id),
        media_type="text/event-stream; charset=utf-8",
        headers={**SSE_HEADERS, **(headers or {})},
    )


def _client_ip(header: Optional[str], request: Request) -> str:
    """The visitor's IP: the proxy's header, else the TCP peer.

    Args:
        header: ``X-TMPBOT-Client-IP`` value.
        request: The request.

    Returns:
        str: A valid IP address as text, or "" when none is usable.
    """
    candidates = [header or ""]
    if request.client is not None:
        candidates.append(request.client.host)
    for raw in candidates:
        try:
            return str(ipaddress.ip_address(raw.strip()))
        except ValueError:
            continue
    return ""


async def _claim(runtime: Runtime, visitor_cookie: Optional[str],
                 identity: Identity) -> int:
    """Move a visitor's conversations to the signed-in user.

    A failure is logged and never blocks the turn: the conversation
    simply starts afresh under the account.

    Args:
        runtime: The runtime (its conversation store).
        visitor_cookie: The visitor id the proxy forwarded, if any.
        identity: The signed-in caller.

    Returns:
        int: Conversations moved (0 on failure or without a visitor).
    """
    visitor_id = parse_uuid(visitor_cookie)
    if visitor_id is None or identity.user_id is None:
        return 0
    try:
        moved = await asyncio.to_thread(runtime.deps.conversations.claim,
                                        visitor_id, identity.user_id)
    except Exception:  # noqa: BLE001 - never block the turn
        log.exception("could not attach visitor conversations")
        return 0
    if moved:
        log.info("attached visitor conversations", extra={"moved": moved})
    return moved


def _refund(runtime: Runtime, admission: Admission):
    """Build the failure hook that gives a free question back.

    Args:
        runtime: The runtime.
        admission: The admission outcome.

    Returns:
        Optional[Callable[[], None]]: The hook, or None when no question
        was counted.
    """
    if not admission.counted_question or admission.visitor is None:
        return None
    visitor_id = admission.visitor.visitor_id

    def refund() -> None:
        runtime.visitors.refund_question(visitor_id)

    return refund


async def _refusal_stream(admission: Admission, message_id: str,
                          language: str) -> AsyncIterator[str]:
    """The short stream for a turn the checks refused.

    Args:
        admission: The refusal.
        message_id: Assistant turn id.
        language: Reply locale.

    Yields:
        str: ``meta``, then the free-limit card and ``done``, or an
        ``error``.
    """
    yield wire.event_meta(message_id, language)
    await asyncio.sleep(0)
    refusal = admission.refusal
    if refusal == "free_limit":
        yield wire.event_gate(wire.GATE_FREE_LIMIT)
        yield wire.event_done(wire.STOP_END_TURN)
    elif refusal == "fair_use":
        yield wire.event_error(wire.ERR_RATE_LIMITED,
                               "Daily fair-use cap reached.",
                               wire.DETAIL_FAIR_USE_DAILY)
    elif refusal == "burst":
        yield wire.event_error(wire.ERR_RATE_LIMITED,
                               "Too many questions in a minute.",
                               wire.DETAIL_BURST)
    elif refusal == "turnstile":
        yield wire.event_error(wire.ERR_INVALID_REQUEST,
                               "Turnstile check needed.",
                               wire.DETAIL_TURNSTILE_REQUIRED)
    else:
        yield wire.event_error(wire.ERR_UPSTREAM_UNAVAILABLE,
                               "Chat is unavailable.")


async def _guarded_stream(
    records: AsyncIterator[str], message_id: str
) -> AsyncIterator[bytes]:
    """Encode records to UTF-8 and end with ``error`` if the source fails.

    The frontend treats only ``done`` and ``error`` as terminal, and a
    stream that just stops leaves the chat bubble hanging (a NINOBOT
    lesson). So any exception after the first byte becomes an
    ``internal_error`` event instead of a dropped connection.

    Args:
        records: Encoded SSE records from the answer generator.
        message_id: Assistant turn id, for the log line.

    Yields:
        bytes: UTF-8 encoded SSE records.
    """
    terminal_sent = False
    try:
        async for record in records:
            if record.startswith(
                tuple(f"event: {e}\n" for e in wire.TERMINAL_EVENTS)
            ):
                terminal_sent = True
            yield record.encode("utf-8")
    except Exception:  # noqa: BLE001 - the stream must end cleanly
        log.exception("chat stream failed", extra={"message_id": message_id})
        if not terminal_sent:
            yield wire.event_error(
                wire.ERR_INTERNAL, "The answer stream failed."
            ).encode("utf-8")
