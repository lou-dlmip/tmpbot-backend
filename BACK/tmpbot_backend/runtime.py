# Builds the chat's collaborators from settings.
#
# Real services where configured, local fakes otherwise (rule: until an
# account exists, code reads the name and falls back to a fake behind the
# same interface). Settings.check() has already refused combinations
# production must not run with. Tests build a Runtime by hand instead.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Assemble the chat runtime."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

from .catalog import Catalog, load_catalog
from .config import Settings
from .db import Database
from .embedding import (
    FakeEmbedder,
    OpenAIQueryEmbedder,
    load_embedding_config,
)
from .identity import (
    JwksVerifier,
    NoTokenVerifier,
    TokenVerifier,
)
from .llm import LiteLLMClient, LocalFakeLLM
from .orchestrator import ChatDeps
from .prompt import load_layers
from .retrieval import MemoryRetriever, PgRetriever
from .routing import load_policy
from .stores import (
    MemoryConversationStore,
    MemoryRateLimiter,
    MemoryUsageLog,
    MemoryVisitorStore,
    PgConversationStore,
    PgRateLimiter,
    PgUsageLog,
    PgVisitorStore,
    RateLimiter,
    VisitorStore,
)
from .turnstile import (
    CloudflareTurnstile,
    DisabledTurnstile,
    TurnstileVerifier,
)

log = logging.getLogger("tmpbot.runtime")


class DownloadIssuer(Protocol):
    """Issues short-lived download links for publications (Phase 7)."""

    async def signed_url(self, publication_id: str) -> Optional[str]:
        """Return a signed URL, or None when downloads are not set up.

        Args:
            publication_id: A published publication id.

        Returns:
            Optional[str]: The URL.
        """
        ...


class NoDownloads:
    """Downloads are not configured yet (no Storage bucket, M7.3)."""

    async def signed_url(self, publication_id: str) -> Optional[str]:
        """Return None: nothing to issue.

        Args:
            publication_id: Ignored.

        Returns:
            Optional[str]: Always None.
        """
        return None


@dataclass(frozen=True)
class Runtime:
    """Everything the chat routes use.

    Attributes:
        deps: The orchestrator's dependencies.
        visitors: Free-question store.
        limiter: Rate limiter.
        turnstile: Turnstile verifier.
        verifier: Access-token verifier.
        downloads: Download link issuer.
    """

    deps: ChatDeps
    visitors: VisitorStore
    limiter: RateLimiter
    turnstile: TurnstileVerifier
    verifier: TokenVerifier
    downloads: DownloadIssuer

    @property
    def catalog(self) -> Catalog:
        """The published-content catalog."""
        return self.deps.catalog


def build_runtime(settings: Settings) -> Runtime:
    """Build the runtime from settings.

    Args:
        settings: Checked settings.

    Returns:
        Runtime: The runtime.
    """
    policy = load_policy(settings.routing_file)
    layers = load_layers(settings.prompt_dir)
    catalog = load_catalog(settings.cops_dir)

    if settings.llm_provider == "litellm":
        llm = LiteLLMClient()
    else:
        log.warning("using the local fake writer; answers are test text")
        llm = LocalFakeLLM(delay_s=settings.canned_token_delay_s)

    if settings.database_url:
        db = Database(settings.database_url)
        config = load_embedding_config(settings.routing_file)
        if settings.embedder == "openai":
            embedder = OpenAIQueryEmbedder(
                settings.openai_api_key or "", config["model"],
                config["dimensions"])
        else:
            embedder = FakeEmbedder(dimensions=config["dimensions"])
        retriever = PgRetriever(db, embedder)
        visitors: VisitorStore = PgVisitorStore(db)
        limiter: RateLimiter = PgRateLimiter(db)
        usage = PgUsageLog(db)
        conversations = PgConversationStore(db)
    else:
        log.warning("no DATABASE_URL: in-memory stores and COPS retrieval")
        retriever = MemoryRetriever.from_cops(settings.cops_dir)
        visitors = MemoryVisitorStore()
        limiter = MemoryRateLimiter()
        usage = MemoryUsageLog()
        conversations = MemoryConversationStore()

    turnstile: TurnstileVerifier = (
        CloudflareTurnstile(settings.turnstile_secret)
        if settings.turnstile_secret else DisabledTurnstile()
    )
    # With accounts switched off nobody is signed in, whatever token
    # arrives: every caller is a signed-out visitor (launch phase L2).
    if settings.accounts_enabled and settings.supabase_url:
        verifier: TokenVerifier = JwksVerifier(settings.supabase_url,
                                               settings.jwt_audience)
    else:
        if settings.accounts_enabled:
            log.warning("accounts are on but SUPABASE_URL is unset; "
                        "every caller is treated as signed out")
        verifier = NoTokenVerifier()
    deps = ChatDeps(policy=policy, layers=layers, catalog=catalog,
                    llm=llm, retriever=retriever, usage=usage,
                    conversations=conversations)
    return Runtime(deps=deps, visitors=visitors, limiter=limiter,
                   turnstile=turnstile, verifier=verifier,
                   downloads=NoDownloads())
