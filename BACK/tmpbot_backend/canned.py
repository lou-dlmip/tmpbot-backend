# The canned chat reply used until the real chat lands in Phase 5.
#
# Streams a fixed placeholder answer in the requested language, piece by
# piece, through the real wire format, so the frontend proxy and chat UI
# can be built and checked end to end before any model account exists.
# The texts say plainly that they are test replies; they make no claims
# about TMP. They are developer-facing and never reach the public site.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Canned streamed reply and demo card switches."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator, Optional

from . import wire

log = logging.getLogger("tmpbot.canned")

CANNED_REPLY: dict[str, str] = {
    "en": (
        "This is a test reply from the TMPBOT backend. The real chat "
        "arrives in a later phase. For now every question gets this "
        "same answer, sent word by word."
    ),
    "ja": (
        "これはTMPBOTバックエンドからのテスト応答です。本来のチャットは"
        "後のフェーズで導入されます。現在はどの質問にもこの同じ回答を、"
        "少しずつ送信しています。"
    ),
    "es": (
        "Esta es una respuesta de prueba del backend de TMPBOT. El chat "
        "real llegará en una fase posterior. Por ahora, cada pregunta "
        "recibe esta misma respuesta, enviada palabra por palabra."
    ),
    "pt": (
        "Esta é uma resposta de teste do backend do TMPBOT. O chat real "
        "chega numa fase posterior. Por enquanto, todas as perguntas "
        "recebem esta mesma resposta, enviada palavra por palavra."
    ),
    "fr": (
        "Ceci est une réponse de test du backend TMPBOT. Le vrai chat "
        "arrivera dans une phase ultérieure. Pour l'instant, chaque "
        "question reçoit cette même réponse, envoyée mot par mot."
    ),
    "de": (
        "Dies ist eine Testantwort des TMPBOT-Backends. Der echte Chat "
        "folgt in einer späteren Phase. Bis dahin erhält jede Frage "
        "diese gleiche Antwort, Wort für Wort gesendet."
    ),
    "it": (
        "Questa è una risposta di prova del backend di TMPBOT. La vera "
        "chat arriverà in una fase successiva. Per ora ogni domanda "
        "riceve questa stessa risposta, inviata parola per parola."
    ),
}

# One-sentence teaser for the ``gate`` demo (the paywall card follows).
CANNED_TEASER: dict[str, str] = {
    "en": "This is a test teaser before the TMP Access card.",
    "ja": "これはTMP Accessカードの前に表示されるテスト用の導入文です。",
    "es": "Este es un adelanto de prueba antes de la tarjeta TMP Access.",
    "pt": "Este é um resumo de teste antes do cartão TMP Access.",
    "fr": "Ceci est un aperçu de test avant la carte TMP Access.",
    "de": "Dies ist ein Test-Anriss vor der TMP-Access-Karte.",
    "it": "Questa è un'anteprima di prova prima della scheda TMP Access.",
}

DEMO_MODES = ("cases", "gate", "limit", "download")

# Real case-study slugs from the 2026-11-13 scrape of asktmp.com
# (DOCS/DSGN/webflow_export_2026-11/case_studies/). A test checks that
# each still exists there; Phase 5 replaces this with ids from COPS/CASE/.
DEMO_CASE_IDS: tuple[str, ...] = (
    "dryad",
    "transition-minerals",
    "landscope-and-riverscope",
)

# No publication exists yet, so the download demo uses an id that says
# what it is. Demo switches never run in production (Settings.demo_allowed).
DEMO_PUBLICATION = {
    "publication_id": "demo-publication",
    "title": "Sample publication (demo only)",
    "year": 2026,
}

# Languages written without spaces between words are streamed a few
# characters at a time instead of word by word.
_NO_SPACE_LOCALES = frozenset({"ja"})
_CJK_PIECE_CHARS = 2
_WORD_RE = re.compile(r"\S+\s*")


def split_tokens(text: str, language: str) -> list[str]:
    """Split text into the pieces sent as ``token`` events.

    Joining the pieces gives back the original text exactly, minus any
    leading whitespace.

    Args:
        text: The text to stream.
        language: Locale of the text.

    Returns:
        list[str]: Pieces in order.
    """
    if language in _NO_SPACE_LOCALES:
        return [
            text[i:i + _CJK_PIECE_CHARS]
            for i in range(0, len(text), _CJK_PIECE_CHARS)
        ]
    return _WORD_RE.findall(text)


async def stream_canned_reply(
    language: str,
    message_id: str,
    demo: Optional[str] = None,
    delay_s: float = 0.0,
) -> AsyncIterator[str]:
    """Yield the SSE records for one canned assistant turn.

    Order: ``meta``, ``token`` events, then any card event, then
    ``done``. With ``demo``:

    - ``cases``: the canned reply, then a ``cases`` event.
    - ``gate``: a one-sentence teaser, then ``gate`` (subscriber_only).
    - ``limit``: no text, only ``gate`` (free_limit).
    - ``download``: the canned reply, then a ``download`` event.

    Args:
        language: Locale of the answer; must be a key of CANNED_REPLY.
        message_id: Assistant turn id for ``meta``.
        demo: One of DEMO_MODES, or None for the plain reply. The caller
            decides whether demo switches are allowed.
        delay_s: Pause between tokens, in seconds.

    Yields:
        str: Encoded SSE records.

    Raises:
        KeyError: If ``language`` has no canned text.
        ValueError: If ``demo`` is not None and not in DEMO_MODES.
    """
    if demo is not None and demo not in DEMO_MODES:
        raise ValueError(f"unknown demo mode: {demo!r}")

    yield wire.event_meta(message_id, language)

    if demo == "limit":
        text = ""
    elif demo == "gate":
        text = CANNED_TEASER[language]
    else:
        text = CANNED_REPLY[language]

    pieces = split_tokens(text, language)
    for piece in pieces:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        yield wire.event_token(piece)

    if demo == "cases":
        cases = wire.event_cases(DEMO_CASE_IDS)
        if cases is not None:
            yield cases
    elif demo == "gate":
        yield wire.event_gate(wire.GATE_SUBSCRIBER_ONLY)
    elif demo == "limit":
        yield wire.event_gate(wire.GATE_FREE_LIMIT)
    elif demo == "download":
        yield wire.event_download(**DEMO_PUBLICATION)

    # No model ran, so usage is zero: the cost log (Phase 5) must never
    # count canned turns as model spend.
    yield wire.event_done(wire.STOP_END_TURN, input_tokens=0, output_tokens=0)
