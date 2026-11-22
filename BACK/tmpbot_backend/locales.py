# The seven languages the backend accepts.
#
# FRNT/i18n/routing.ts is the authoritative list (CLAUDE.md, spec 7a).
# BACK/tests/test_locales_parity.py holds this tuple equal to it.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Locale list for request validation."""

from __future__ import annotations

# Order follows the project facts in CLAUDE.md, not the splash order.
LOCALES: tuple[str, ...] = ("en", "ja", "es", "pt", "fr", "de", "it")

# Unmatched browser languages fall back to Japanese (D9).
FALLBACK_LOCALE = "ja"

# Regex for pydantic ``Field(pattern=...)``; built from LOCALES so the two
# cannot drift.
LOCALE_PATTERN = "^(" + "|".join(LOCALES) + ")$"
