# Tests for settings loading (config.py), the JSON log formatter
# (logging_setup.py), Sentry start-up (observability.py), the locale list
# (locales.py) and the canned token splitter (canned.py).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Configuration, logging, Sentry, locale and splitter tests."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest

from tmpbot_backend import canned, locales
from tmpbot_backend.config import ConfigError, Settings, load_settings
from tmpbot_backend.logging_setup import JsonFormatter, configure_logging
from tmpbot_backend.observability import init_sentry

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- config


def test_load_settings_defaults():
    """An empty environment gives safe local defaults: accounts off, so
    20 free questions (launch phase L2)."""
    s = load_settings({})
    assert s == Settings(env="local", shared_secret="", log_level="INFO",
                         sentry_dsn=None, accounts_enabled=False,
                         free_question_limit=20)
    assert s.demo_allowed


def test_load_settings_reads_values():
    """Values are read, trimmed and normalised."""
    s = load_settings({
        "TMPBOT_ENV": " Production ",
        "TMPBOT_SHARED_SECRET": " s3cret ",
        "TMPBOT_LOG_LEVEL": "debug",
        "SENTRY_DSN_BACKEND": "https://key@o0.ingest.sentry.io/1",
    })
    assert s.env == "production"
    assert s.shared_secret == "s3cret"
    assert s.log_level == "DEBUG"
    assert s.sentry_dsn == "https://key@o0.ingest.sentry.io/1"
    assert not s.demo_allowed


def test_blank_sentry_dsn_means_off():
    """An empty or blank DSN leaves Sentry off."""
    assert load_settings({"SENTRY_DSN_BACKEND": "  "}).sentry_dsn is None


@pytest.mark.parametrize("env", [{"TMPBOT_ENV": "prod"},
                                 {"TMPBOT_LOG_LEVEL": "LOUD"}])
def test_load_settings_rejects_bad_values(env):
    """Typos in the environment fail loudly at start-up."""
    with pytest.raises(ConfigError):
        load_settings(env)


def test_backend_env_names_are_documented():
    """Every name config.py reads is in INFR/env_inventory.md and
    .env.example."""
    source = (REPO_ROOT / "BACK" / "tmpbot_backend" / "config.py").read_text(
        encoding="utf-8")
    names = {a or b for a, b in re.findall(
        r'(?:source\.get|text)\("([A-Z_]+)"\)|_int\(\s*source,\s*"([A-Z_]+)"',
        source)}
    assert names == {
        "TMPBOT_ENV", "TMPBOT_SHARED_SECRET", "TMPBOT_LOG_LEVEL",
        "SENTRY_DSN_BACKEND", "DATABASE_URL", "TMPBOT_LLM_PROVIDER",
        "TMPBOT_EMBEDDER", "OPENAI_API_KEY", "TMPBOT_IP_HASH_SALT",
        "TMPBOT_FREE_QUESTION_LIMIT", "TMPBOT_FAIR_USE_DAILY_CAP",
        "TMPBOT_RATE_LIMIT_PER_MINUTE",
        "TMPBOT_RATE_LIMIT_PER_IP_PER_MINUTE",
        "TMPBOT_VISITOR_IP_WINDOW_DAYS", "TURNSTILE_SECRET_KEY",
        "SUPABASE_URL", "SUPABASE_JWT_AUDIENCE",
    }
    inventory = (REPO_ROOT / "INFR" / "env_inventory.md").read_text(
        encoding="utf-8")
    example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in names:
        assert f"`{name}`" in inventory, name
        assert f"\n{name}=" in example, name


# --------------------------------------------------------------- logging


def _record(msg="hello %s", args=("world",), **extra):
    """Build a LogRecord with optional extra fields.

    Args:
        msg: Format string.
        args: Format arguments.
        **extra: Attributes added as ``extra=`` would.

    Returns:
        logging.LogRecord: The record.
    """
    rec = logging.LogRecord("tmpbot.test", logging.INFO, __file__, 1, msg,
                            args, None)
    for key, value in extra.items():
        setattr(rec, key, value)
    return rec


def test_json_formatter_basic_fields():
    """One JSON object with a UTC timestamp, level, logger and message."""
    line = JsonFormatter().format(_record())
    data = json.loads(line)
    assert data["msg"] == "hello world"
    assert data["level"] == "INFO"
    assert data["logger"] == "tmpbot.test"
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$", data["ts"])
    assert "\n" not in line


def test_json_formatter_includes_extra_and_odd_types():
    """extra= fields appear; values JSON cannot encode become strings."""
    data = json.loads(JsonFormatter().format(
        _record(request_id="abc", path=Path("/x"), status=200)))
    assert data["request_id"] == "abc"
    assert data["path"] == "/x"
    assert data["status"] == 200


def test_json_formatter_exception():
    """A logged exception lands in the exc field."""
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys
        rec = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed",
                                None, sys.exc_info())
    data = json.loads(JsonFormatter().format(rec))
    assert "RuntimeError: boom" in data["exc"]


def test_json_formatter_drops_uvicorn_colour_copy():
    """uvicorn's color_message (ANSI codes) never reaches the log."""
    data = json.loads(JsonFormatter().format(
        _record(color_message="\x1b[36mhello\x1b[0m")))
    assert "color_message" not in data


def test_configure_logging_silences_uvicorn_access(capsys):
    """The app logs requests itself; uvicorn's access line is dropped,
    while uvicorn's server messages come out as JSON."""
    configure_logging("INFO")
    logging.getLogger("uvicorn.access").info("GET / 200")
    logging.getLogger("uvicorn.error").info("server started")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    assert len(lines) == 1
    assert json.loads(lines[0])["msg"] == "server started"


def test_json_formatter_keeps_japanese():
    """Japanese log text stays readable."""
    line = JsonFormatter().format(_record("質問 %s", ("受信",)))
    assert "質問 受信" in line


# ---------------------------------------------------------------- sentry


def test_sentry_off_without_dsn():
    """No DSN: init is never called."""
    calls = []
    assert init_sentry(Settings(), init_fn=calls.append) is False
    assert calls == []


def test_sentry_on_with_dsn():
    """With a DSN: init runs once, with no personal data and the env."""
    calls = []
    settings = Settings(env="staging", sentry_dsn="https://k@o0/1")
    assert init_sentry(settings,
                       init_fn=lambda **kw: calls.append(kw)) is True
    assert len(calls) == 1
    assert calls[0]["dsn"] == "https://k@o0/1"
    assert calls[0]["environment"] == "staging"
    assert calls[0]["send_default_pii"] is False


def test_sentry_failure_does_not_block_start():
    """A broken SDK or DSN is logged, not raised."""
    def broken(**_):
        raise RuntimeError("bad dsn")

    assert init_sentry(Settings(sentry_dsn="x"), init_fn=broken) is False


def test_sentry_real_sdk_accepts_our_arguments():
    """The pinned sentry-sdk accepts the keyword arguments we pass.

    Uses a syntactically valid DSN that points nowhere; with tracing at
    zero and no events captured, nothing is sent.
    """
    import sentry_sdk

    try:
        assert init_sentry(
            Settings(sentry_dsn="https://public@o0.ingest.invalid/1")
        ) is True
    finally:
        sentry_sdk.init(dsn=None)


# --------------------------------------------------------------- locales


def test_seven_locales():
    """The seven site languages, Japanese fallback (CLAUDE.md, D9)."""
    assert set(locales.LOCALES) == {"en", "ja", "es", "pt", "fr", "de", "it"}
    assert len(locales.LOCALES) == 7
    assert locales.FALLBACK_LOCALE == "ja"
    assert re.match(locales.LOCALE_PATTERN, "it")
    assert not re.match(locales.LOCALE_PATTERN, "zh")


def test_canned_texts_cover_every_locale():
    """Every locale has a canned reply and teaser."""
    assert set(canned.CANNED_REPLY) == set(locales.LOCALES)
    assert set(canned.CANNED_TEASER) == set(locales.LOCALES)


# -------------------------------------------------------------- splitter


@pytest.mark.parametrize("lang", locales.LOCALES)
def test_split_tokens_round_trips(lang):
    """Joining the pieces gives the text back exactly."""
    text = canned.CANNED_REPLY[lang]
    pieces = canned.split_tokens(text, lang)
    assert "".join(pieces) == text
    assert len(pieces) > 5


def test_split_tokens_words_keep_trailing_space():
    """Spaced languages split after each word's trailing space."""
    assert canned.split_tokens("one two  three", "en") == [
        "one ", "two  ", "three"]


def test_split_tokens_japanese_in_small_pieces():
    """Japanese streams two characters at a time."""
    assert canned.split_tokens("気候変動リスク", "ja") == [
        "気候", "変動", "リス", "ク"]
