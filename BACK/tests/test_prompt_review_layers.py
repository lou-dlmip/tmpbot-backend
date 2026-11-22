# The 2026-11-22 prompt review: the shipped prompt layers carry Lou's
# notes (what TMP does, tone, 当社, audience tailoring, the 150-word
# allowance, contact sparingly, off-topic, dates, fees) and they reach
# the system message (tmpbot_backend.prompt).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Prompt layers after the 2026-11-22 review."""

from __future__ import annotations

from pathlib import Path

import pytest

from tmpbot_backend.prompt import load_layers

PRMT = Path(__file__).resolve().parents[2] / "BOTS" / "PRMT"


@pytest.fixture(scope="module")
def layers():
    """Load the real prompt layers from BOTS/PRMT."""
    return load_layers(PRMT)


def test_voice_has_what_tmp_does_and_tone(layers):
    assert "since 2009 in more than 70 countries" in layers.voice
    assert "never \"TMP provides\"" in layers.voice
    assert "about 200 characters" in layers.voice
    assert "best available intelligence" in layers.voice
    assert "Tone: neutral and informative" in layers.voice
    assert "a little dry" not in layers.voice


def test_voice_uses_formal_japanese_self_reference(layers):
    assert "formal 当社 (not 私たち)" in layers.voice


def test_voice_tailors_to_every_audience(layers):
    for label in ("Investor", "Corporate", "Government", "Funder",
                  "Non-Profit"):
        assert f"- {label}:" in layers.voice
    assert "never the facts" in layers.voice


def test_length_allows_150_words_for_detail(layers):
    assert "up to 150 words" in layers.voice
    assert "about 90 words" in layers.voice
    assert "at most 150 words" in layers.length_mistral


def test_rules_cover_review_notes(layers):
    rules = layers.rules
    assert "Dates and status:" in rules
    assert '"2023–present"' in rules
    assert "Getting in touch:" in rules
    assert "at most once in a conversation" in rules
    assert "Off-topic questions:" in rules
    assert "Fees:" in rules
    assert "Do not invent a figure" in rules
    # The old Facts rule ended every gap with "suggest getting in touch".
    assert "suggest getting in touch" not in rules


def test_layer_headers_are_stripped(layers):
    for text in (layers.voice, layers.rules, layers.length_mistral):
        assert "<!--" not in text
        assert "Copyright" not in text
