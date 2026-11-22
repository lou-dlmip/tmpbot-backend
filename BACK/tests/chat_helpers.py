# Helpers shared by the backend tests: test keys, a token signer and an
# in-memory runtime builder. Kept out of conftest.py so test modules can
# import them by a unique module name (three test folders each have a
# conftest.py).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Test helpers for the chat backend."""

from __future__ import annotations

import dataclasses
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import jwt

from tmpbot_backend.catalog import load_catalog
from tmpbot_backend.config import REPO_ROOT, Settings
from tmpbot_backend.identity import StaticKeyVerifier
from tmpbot_backend.llm import ScriptedLLM
from tmpbot_backend.orchestrator import ChatDeps
from tmpbot_backend.prompt import load_layers
from tmpbot_backend.retrieval import MemoryRetriever
from tmpbot_backend.routing import load_policy
from tmpbot_backend.runtime import NoDownloads, Runtime
from tmpbot_backend.stores import (
    MemoryConversationStore,
    MemoryRateLimiter,
    MemoryUsageLog,
    MemoryVisitorStore,
)
from tmpbot_backend.turnstile import DisabledTurnstile

TEST_SECRET = "test-shared-secret-0123456789"
TEST_JWT_KEY = "test-jwt-signing-key-0123456789abcdef"
COPS = REPO_ROOT / "COPS"


def make_token(plan: Optional[str] = None,
               user_id: Optional[str] = None,
               key: str = TEST_JWT_KEY, expires_in: int = 3600) -> str:
    """Sign a Supabase-style access token for tests.

    Args:
        plan: ``access`` for a subscriber; None for a free account.
        user_id: The ``sub`` claim; a fresh UUID when None.
        key: Signing key.
        expires_in: Seconds until expiry (negative for expired).

    Returns:
        str: The JWT.
    """
    claims: dict[str, Any] = {
        "sub": user_id or str(uuid.uuid4()),
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + expires_in,
    }
    if plan is not None:
        claims["plan"] = plan
    return jwt.encode(claims, key, algorithm="HS256")


def build_runtime(settings: Settings, llm: Any = None,
                  cops_dir: Path = COPS, **overrides: Any) -> Runtime:
    """An in-memory runtime over the real policy, prompts and content.

    Args:
        settings: Settings (routing and prompt paths are read).
        llm: Writer; a fresh ScriptedLLM when None.
        cops_dir: Content folder for the catalog and retriever.
        **overrides: Runtime or ChatDeps fields to replace.

    Returns:
        Runtime: The runtime.
    """
    dep_fields = {f.name for f in dataclasses.fields(ChatDeps)}
    deps = ChatDeps(
        policy=load_policy(settings.routing_file),
        layers=load_layers(settings.prompt_dir),
        catalog=load_catalog(cops_dir),
        llm=llm if llm is not None else ScriptedLLM(),
        retriever=MemoryRetriever.from_cops(cops_dir),
        usage=MemoryUsageLog(),
        conversations=MemoryConversationStore(),
    )
    deps = dataclasses.replace(
        deps, **{k: v for k, v in overrides.items() if k in dep_fields})
    runtime = Runtime(
        deps=deps,
        visitors=MemoryVisitorStore(),
        limiter=MemoryRateLimiter(),
        turnstile=DisabledTurnstile(),
        verifier=StaticKeyVerifier(TEST_JWT_KEY),
        downloads=NoDownloads(),
    )
    return dataclasses.replace(
        runtime,
        **{k: v for k, v in overrides.items() if k not in dep_fields})


def db_user(url: str) -> str:
    """Create a user in the test database's auth.users shim.

    Args:
        url: Database URL (superuser).

    Returns:
        str: The new user's id.
    """
    import psycopg

    with psycopg.connect(url, autocommit=True) as conn:
        return str(conn.execute(
            "INSERT INTO auth.users DEFAULT VALUES RETURNING id"
        ).fetchone()[0])
