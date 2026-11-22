# Backend settings read from environment variables.
#
# Every name read here is documented in INFR/env_inventory.md. Settings
# are read once into a frozen dataclass, so tests can build one directly
# and hand it to create_app() without touching os.environ.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Runtime configuration for the TMPBOT backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

ENV_NAMES = ("local", "staging", "production")

DEFAULT_LOG_LEVEL = "INFO"

# Pause between canned tokens. Long enough that a person watching curl
# sees the text arrive word by word; tests set it to zero.
DEFAULT_CANNED_TOKEN_DELAY_S = 0.04

# Repository root: BACK/tmpbot_backend/config.py -> parents[2]. Render
# runs from the repo root (INFR/render.yaml has no root directory), so
# BOTS/ and COPS/ sit beside BACK/ in production too.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Signed-out free questions before the free-limit card (S14).
DEFAULT_FREE_QUESTION_LIMIT = 10
# The launch default when accounts are switched off
# (TMPBOT_ACCOUNTS_ENABLED unset or false): nobody can sign in for more,
# so signed-out visitors get more questions (launch phase L2).
DEFAULT_FREE_QUESTION_LIMIT_NO_ACCOUNTS = 20
# Daily questions per free account (S18). Adjusted from real usage.
DEFAULT_FAIR_USE_DAILY_CAP = 100
# Short-burst limits, questions per minute. A person types far slower;
# the per-IP figure is higher because offices share one address.
DEFAULT_RATE_PER_MINUTE = 6
DEFAULT_RATE_PER_IP_PER_MINUTE = 30
# How far back a visitor without a cookie is matched to an earlier
# visitor from the same hashed IP, so clearing cookies does not reset
# the free count (spec 3b).
DEFAULT_VISITOR_IP_WINDOW_DAYS = 7

LLM_PROVIDERS = ("litellm", "fake")
EMBEDDERS = ("openai", "fake")


class ConfigError(ValueError):
    """Raised when an environment variable holds an unusable value."""


@dataclass(frozen=True)
class Settings:
    """Backend settings.

    Attributes:
        env: ``local``, ``staging`` or ``production`` (``TMPBOT_ENV``).
        shared_secret: Secret the Next.js proxy sends with every chat
            request (``TMPBOT_SHARED_SECRET``). Empty means the chat
            endpoint refuses everything.
        log_level: Python logging level name (``TMPBOT_LOG_LEVEL``).
        sentry_dsn: Sentry DSN (``SENTRY_DSN_BACKEND``); ``None`` keeps
            Sentry off.
        canned_token_delay_s: Seconds between tokens of the local fake
            writer. Not read from the environment; tests set it to zero.
        database_url: Postgres URL (``DATABASE_URL``). ``None`` uses
            in-memory stores, allowed only when ``env`` is ``local``.
        llm_provider: ``litellm`` (real providers) or ``fake`` (the
            local scripted writer) (``TMPBOT_LLM_PROVIDER``).
        embedder: ``openai`` or ``fake`` (``TMPBOT_EMBEDDER``). Must
            match the embedder the corpus was indexed with.
        openai_api_key: Embeddings key (``OPENAI_API_KEY``).
        ip_hash_salt: Salt for hashing visitor IPs
            (``TMPBOT_IP_HASH_SALT``).
        accounts_enabled: Whether free accounts are switched on
            (``TMPBOT_ACCOUNTS_ENABLED``). Off: every caller is treated as
            a signed-out visitor, whatever token arrives.
        free_question_limit: Signed-out questions before the free-limit
            card (``TMPBOT_FREE_QUESTION_LIMIT``; 10 with accounts on,
            20 with accounts off, when unset).
        fair_use_daily_cap: Daily questions per free account
            (``TMPBOT_FAIR_USE_DAILY_CAP``).
        rate_per_minute: Questions per minute per visitor or account
            (``TMPBOT_RATE_LIMIT_PER_MINUTE``).
        rate_per_ip_per_minute: Questions per minute per IP address
            (``TMPBOT_RATE_LIMIT_PER_IP_PER_MINUTE``).
        visitor_ip_window_days: Days within which a cookieless visitor
            is matched to an earlier one by hashed IP
            (``TMPBOT_VISITOR_IP_WINDOW_DAYS``).
        turnstile_secret: Cloudflare Turnstile secret
            (``TURNSTILE_SECRET_KEY``). Unset turns the check off, which
            only ``local`` and ``staging`` allow.
        supabase_url: Supabase project URL (``SUPABASE_URL``); the JWKS
            for access-token checks lives under it.
        jwt_audience: Expected access-token audience
            (``SUPABASE_JWT_AUDIENCE``).
        routing_file: The writer routing policy.
        prompt_dir: The prompt layers.
        cops_dir: Published content, for checking card ids.
    """

    env: str = "local"
    shared_secret: str = ""
    log_level: str = DEFAULT_LOG_LEVEL
    sentry_dsn: Optional[str] = None
    canned_token_delay_s: float = DEFAULT_CANNED_TOKEN_DELAY_S
    database_url: Optional[str] = None
    llm_provider: str = "fake"
    embedder: str = "fake"
    openai_api_key: Optional[str] = None
    ip_hash_salt: str = ""
    accounts_enabled: bool = False
    free_question_limit: int = DEFAULT_FREE_QUESTION_LIMIT
    fair_use_daily_cap: int = DEFAULT_FAIR_USE_DAILY_CAP
    rate_per_minute: int = DEFAULT_RATE_PER_MINUTE
    rate_per_ip_per_minute: int = DEFAULT_RATE_PER_IP_PER_MINUTE
    visitor_ip_window_days: int = DEFAULT_VISITOR_IP_WINDOW_DAYS
    turnstile_secret: Optional[str] = None
    supabase_url: Optional[str] = None
    jwt_audience: str = "authenticated"
    routing_file: Path = REPO_ROOT / "BOTS" / "POLC" / "llm_routing.yaml"
    prompt_dir: Path = REPO_ROOT / "BOTS" / "PRMT"
    cops_dir: Path = REPO_ROOT / "COPS"

    @property
    def demo_allowed(self) -> bool:
        """Whether the request's ``demo`` field may force a card event.

        Demo switches let the frontend draw every card without a model.
        Production ignores them.
        """
        return self.env != "production"

    def check(self) -> None:
        """Refuse settings that would run the public chat unprotected.

        Production must have a database (the free count, limits and
        usage log live there), the real writer and embedder, an IP salt
        and Turnstile. Staging and local may run with fakes so they can
        work before any vendor account exists. ``local`` alone may run
        without a database.

        Raises:
            ConfigError: If a required setting is missing or unusable.
        """
        if self.llm_provider not in LLM_PROVIDERS:
            raise ConfigError(
                f"TMPBOT_LLM_PROVIDER must be one of {LLM_PROVIDERS}; "
                f"got {self.llm_provider!r}"
            )
        if self.embedder not in EMBEDDERS:
            raise ConfigError(
                f"TMPBOT_EMBEDDER must be one of {EMBEDDERS}; "
                f"got {self.embedder!r}"
            )
        for name in ("free_question_limit", "fair_use_daily_cap",
                     "rate_per_minute", "rate_per_ip_per_minute",
                     "visitor_ip_window_days"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1")
        if self.embedder == "openai" and not self.openai_api_key:
            raise ConfigError("TMPBOT_EMBEDDER=openai needs OPENAI_API_KEY")
        if self.env != "local" and not self.database_url:
            raise ConfigError(
                f"DATABASE_URL is required when TMPBOT_ENV={self.env}"
            )
        if self.env == "production":
            missing = []
            if self.llm_provider != "litellm":
                missing.append("TMPBOT_LLM_PROVIDER=litellm")
            if self.embedder != "openai":
                missing.append("TMPBOT_EMBEDDER=openai")
            if len(self.ip_hash_salt) < 16:
                missing.append("TMPBOT_IP_HASH_SALT (16+ characters)")
            if not self.turnstile_secret:
                missing.append("TURNSTILE_SECRET_KEY")
            if self.accounts_enabled and not self.supabase_url:
                missing.append("SUPABASE_URL (accounts are on)")
            if missing:
                raise ConfigError(
                    "production needs: " + ", ".join(missing)
                )


def _int(source: Mapping[str, str], name: str, default: int) -> int:
    """Read a positive integer variable.

    Args:
        source: Environment mapping.
        name: Variable name.
        default: Value when unset or blank.

    Returns:
        int: The value.

    Raises:
        ConfigError: If the value is not a whole number.
    """
    raw = (source.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number; got {raw!r}") \
            from exc


TRUE_WORDS = ("1", "true", "yes", "on")
FALSE_WORDS = ("", "0", "false", "no", "off")


def _bool(source: Mapping[str, str], name: str) -> bool:
    """Read an on/off switch; unset means off.

    The same words are accepted as by the frontend's
    ``FRNT/lib/accounts.ts``, so one value means the same on both sides.

    Args:
        source: Environment mapping.
        name: Variable name.

    Returns:
        bool: True for 1, true, yes or on (any case).

    Raises:
        ConfigError: For any other value, so a typo never silently
            leaves a switch in the wrong position.
    """
    raw = (source.get(name) or "").strip().lower()
    if raw in TRUE_WORDS:
        return True
    if raw in FALSE_WORDS:
        return False
    raise ConfigError(f"{name} must be true or false; got {raw!r}")


def load_settings(environ: Optional[Mapping[str, str]] = None) -> Settings:
    """Read settings from the environment.

    Args:
        environ: Mapping to read from; defaults to ``os.environ``.

    Returns:
        Settings: The parsed settings.

    Raises:
        ConfigError: If a variable holds a value outside its allowed set.
    """
    source = os.environ if environ is None else environ

    env = (source.get("TMPBOT_ENV") or "local").strip().lower()
    if env not in ENV_NAMES:
        raise ConfigError(
            f"TMPBOT_ENV must be one of {', '.join(ENV_NAMES)}; got {env!r}"
        )

    log_level = (
        (source.get("TMPBOT_LOG_LEVEL") or DEFAULT_LOG_LEVEL).strip().upper()
    )
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ConfigError(f"TMPBOT_LOG_LEVEL is not a level: {log_level!r}")

    def text(name: str) -> Optional[str]:
        return (source.get(name) or "").strip() or None

    accounts_enabled = _bool(source, "TMPBOT_ACCOUNTS_ENABLED")
    default_provider = "litellm" if env == "production" else "fake"
    default_embedder = "openai" if env == "production" else "fake"

    return Settings(
        env=env,
        shared_secret=(source.get("TMPBOT_SHARED_SECRET") or "").strip(),
        log_level=log_level,
        sentry_dsn=text("SENTRY_DSN_BACKEND"),
        database_url=text("DATABASE_URL"),
        llm_provider=(text("TMPBOT_LLM_PROVIDER")
                      or default_provider).lower(),
        embedder=(text("TMPBOT_EMBEDDER") or default_embedder).lower(),
        openai_api_key=text("OPENAI_API_KEY"),
        ip_hash_salt=text("TMPBOT_IP_HASH_SALT") or "",
        accounts_enabled=accounts_enabled,
        free_question_limit=_int(
            source, "TMPBOT_FREE_QUESTION_LIMIT",
            DEFAULT_FREE_QUESTION_LIMIT if accounts_enabled
            else DEFAULT_FREE_QUESTION_LIMIT_NO_ACCOUNTS),
        fair_use_daily_cap=_int(source, "TMPBOT_FAIR_USE_DAILY_CAP",
                                DEFAULT_FAIR_USE_DAILY_CAP),
        rate_per_minute=_int(source, "TMPBOT_RATE_LIMIT_PER_MINUTE",
                             DEFAULT_RATE_PER_MINUTE),
        rate_per_ip_per_minute=_int(
            source, "TMPBOT_RATE_LIMIT_PER_IP_PER_MINUTE",
            DEFAULT_RATE_PER_IP_PER_MINUTE),
        visitor_ip_window_days=_int(source, "TMPBOT_VISITOR_IP_WINDOW_DAYS",
                                    DEFAULT_VISITOR_IP_WINDOW_DAYS),
        turnstile_secret=text("TURNSTILE_SECRET_KEY"),
        supabase_url=(text("SUPABASE_URL") or "").rstrip("/") or None,
        jwt_audience=text("SUPABASE_JWT_AUDIENCE") or "authenticated",
    )
