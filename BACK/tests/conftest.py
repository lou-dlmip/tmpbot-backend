# Shared pytest fixtures for the TMPBOT backend tests.
#
# The chat app is built with an injected Runtime: the real routing
# policy, prompt layers and published-content catalog from the repo, a
# scripted fake writer (no provider is ever called), in-memory stores,
# and a token verifier keyed with a test secret.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Fixtures: test settings, runtime, tokens and a FastAPI test client."""

from __future__ import annotations

from typing import Callable, Optional

import pytest
from chat_helpers import TEST_SECRET, build_runtime
from db_support import (  # noqa: F401 - fixtures for the DB tests
    as_role,
    fresh_db,
    migrated_db,
    pg_admin_url,
)
from fastapi.testclient import TestClient

from tmpbot_backend.config import Settings
from tmpbot_backend.llm import ScriptedLLM
from tmpbot_backend.main import create_app
from tmpbot_backend.runtime import Runtime


@pytest.fixture
def settings() -> Settings:
    """Local settings with a known secret and no token delay.

    Returns:
        Settings: Settings for a local test app.
    """
    return Settings(
        env="local",
        shared_secret=TEST_SECRET,
        log_level="INFO",
        sentry_dsn=None,
        canned_token_delay_s=0.0,
        ip_hash_salt="test-salt-0123456789",
    )


@pytest.fixture
def llm() -> ScriptedLLM:
    """A scripted writer with no replies queued (default answers).

    Returns:
        ScriptedLLM: The fake.
    """
    return ScriptedLLM()


@pytest.fixture
def runtime(settings: Settings, llm: ScriptedLLM) -> Runtime:
    """The in-memory runtime used by ``client``.

    Args:
        settings: Settings fixture.
        llm: Writer fixture.

    Returns:
        Runtime: The runtime.
    """
    return build_runtime(settings, llm)


@pytest.fixture
def client(settings: Settings, runtime: Runtime) -> TestClient:
    """A test client for an app built from ``settings`` and ``runtime``.

    Args:
        settings: The settings fixture.
        runtime: The runtime fixture.

    Returns:
        TestClient: Client bound to a fresh app.
    """
    return TestClient(create_app(settings, runtime))


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers carrying the correct proxy secret.

    Returns:
        dict[str, str]: The header mapping.
    """
    return {"X-TMPBOT-Proxy-Secret": TEST_SECRET}


@pytest.fixture
def make_body() -> Callable[..., dict]:
    """Factory for a minimal valid chat request body.

    Returns:
        Callable[..., dict]: ``make_body(language="en", demo=None,
        question=...)``.
    """

    def _make(language: str = "en", demo: Optional[str] = None,
              question: str = "What does TMP do?") -> dict:
        body: dict = {
            "language": language,
            "messages": [{"role": "user", "content": question}],
        }
        if demo is not None:
            body["demo"] = demo
        return body

    return _make
