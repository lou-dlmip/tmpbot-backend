# Hold the backend's locale list equal to the frontend's authoritative one.
#
# FRNT/i18n/routing.ts owns the list (CLAUDE.md, spec 7a). The backend
# validates `language` against BACK/tmpbot_backend/locales.py. If the two
# drift, the frontend could send a locale the backend rejects, or the
# reverse. NINOBOT needed a migration when a hard-coded list missed
# Italian; this test stops that class of bug at review time. (PLAN M2.7)
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Locale parity between FRNT/i18n/routing.ts and tmpbot_backend."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tmpbot_backend.locales import FALLBACK_LOCALE, LOCALES

REPO_ROOT = Path(__file__).resolve().parents[2]
ROUTING_TS = REPO_ROOT / "FRNT" / "i18n" / "routing.ts"


def _ts_string_list(source: str, name: str) -> list[str]:
    """Return the string items of `export const NAME = [...]` in TS source.

    Args:
        source: TypeScript source text.
        name: The constant's name.

    Returns:
        The quoted items, in order.

    Raises:
        AssertionError: If the constant is not found.
    """
    match = re.search(
        rf"export const {name}\s*=\s*\[([^\]]*)\]", source, re.DOTALL
    )
    assert match, f"{name} not found in {ROUTING_TS}"
    return re.findall(r"\"([^\"]+)\"", match.group(1))


def _ts_string_const(source: str, name: str) -> str:
    """Return the value of `export const NAME: T = "value"` in TS source.

    Args:
        source: TypeScript source text.
        name: The constant's name.

    Returns:
        The quoted value.

    Raises:
        AssertionError: If the constant is not found.
    """
    match = re.search(
        rf"export const {name}(?::\s*\w+)?\s*=\s*\"([^\"]+)\"", source
    )
    assert match, f"{name} not found in {ROUTING_TS}"
    return match.group(1)


@pytest.fixture(scope="module")
def routing_source() -> str:
    """Read FRNT/i18n/routing.ts once.

    Returns:
        The file's text.
    """
    return ROUTING_TS.read_text(encoding="utf-8")


def test_locales_match_frontend(routing_source: str) -> None:
    """The backend accepts exactly the frontend's locales, in order."""
    assert tuple(_ts_string_list(routing_source, "LOCALES")) == LOCALES


def test_fallback_matches_frontend(routing_source: str) -> None:
    """Both sides fall back to the same locale (ja, D9)."""
    assert _ts_string_const(routing_source, "FALLBACK_LOCALE") == (
        FALLBACK_LOCALE
    )
    assert FALLBACK_LOCALE == "ja"


def test_seven_locales(routing_source: str) -> None:
    """There are seven, with Italian included."""
    locales = _ts_string_list(routing_source, "LOCALES")
    assert len(locales) == 7
    assert "it" in locales
