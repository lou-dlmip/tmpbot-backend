# The three card tools and their checks (spec 3b).
#
# The model asks for cards by calling tools; the backend decides what is
# drawn. Every id is checked against published content (catalog.py) and
# a made-up id is dropped, never rendered. Cards are collected during the
# turn and sent after the last token, in a fixed order: cases, gate,
# downloads (DOCS/STND/streaming_wire_format.md).
#
# Plan rules:
#   - show_access_prompt is offered to non-subscribers only. A subscriber
#     never sees the paywall card, whatever the model does.
#   - offer_download is offered to subscribers only. If a non-subscriber's
#     model calls it anyway, the paywall card is shown instead: the
#     download card would only lead to a refusal.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Tool schemas, dispatch and the cards they produce."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from . import wire
from .catalog import Catalog
from .identity import Identity

log = logging.getLogger("tmpbot.tools")

SHOW_CASE_STUDIES = "show_case_studies"
SHOW_ACCESS_PROMPT = "show_access_prompt"
OFFER_DOWNLOAD = "offer_download"

_CASES_SCHEMA = {
    "type": "function",
    "function": {
        "name": SHOW_CASE_STUDIES,
        "description": (
            "Show one to three case-study cards under your answer. Use "
            "only ids of references whose doc_type is case_study."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": wire.MAX_CASE_CARDS,
                    "description": "Case-study ids, most relevant first.",
                },
            },
            "required": ["ids"],
        },
    },
}

_GATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": SHOW_ACCESS_PROMPT,
        "description": (
            "Show the TMP Access card after your one or two sentence "
            "teaser, for a subscriber-only question."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_DOWNLOAD_SCHEMA = {
    "type": "function",
    "function": {
        "name": OFFER_DOWNLOAD,
        "description": (
            "Offer a publication as a download card under your answer. "
            "Use only ids of references whose doc_type is publication."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "publication_id": {"type": "string"},
            },
            "required": ["publication_id"],
        },
    },
}


def tools_for(identity: Identity) -> list[dict[str, Any]]:
    """The tool schemas offered to the writer for this caller.

    Args:
        identity: The caller.

    Returns:
        list[dict[str, Any]]: OpenAI-style function schemas.
    """
    if identity.subscriber:
        return [_CASES_SCHEMA, _DOWNLOAD_SCHEMA]
    return [_CASES_SCHEMA, _GATE_SCHEMA]


@dataclass(frozen=True)
class ToolOutcome:
    """The result of one tool call.

    Attributes:
        ok: Whether the call did what it asked.
        detail: Short reason when not ok (wire ``tool_result.detail``).
        for_model: Text returned to the writer as the tool result.
    """

    ok: bool
    detail: str = ""
    for_model: str = "ok"


@dataclass
class CardCollector:
    """Collects the cards one turn will show.

    Args:
        catalog: Published ids.
        identity: The caller.
        language: Reply locale, for download titles.
    """

    catalog: Catalog
    identity: Identity
    language: str
    case_ids: list[str] = field(default_factory=list)
    gate: bool = False
    downloads: list[dict[str, Any]] = field(default_factory=list)

    def apply(self, name: str, args: Any) -> ToolOutcome:
        """Check and apply one tool call.

        Args:
            name: Tool name the model used.
            args: Parsed arguments (anything; checked here).

        Returns:
            ToolOutcome: What happened.
        """
        if not isinstance(args, dict):
            args = {}
        if name == SHOW_CASE_STUDIES:
            return self._cases(args.get("ids"))
        if name == SHOW_ACCESS_PROMPT:
            return self._gate()
        if name == OFFER_DOWNLOAD:
            return self._download(args.get("publication_id"))
        log.warning("unknown tool called", extra={"tool": name})
        return ToolOutcome(False, "unknown tool", "error: unknown tool")

    def _cases(self, ids: Any) -> ToolOutcome:
        """Keep published case-study ids, up to three in total.

        Args:
            ids: The ids the model sent.

        Returns:
            ToolOutcome: ok when at least one id is kept.
        """
        if not isinstance(ids, list):
            return ToolOutcome(False, "ids must be a list",
                               "error: ids must be a list")
        kept, dropped = [], []
        for raw in ids:
            if self.catalog.is_case_study(raw):
                if raw not in self.case_ids and raw not in kept:
                    kept.append(raw)
            else:
                dropped.append(raw)
        if dropped:
            log.warning("dropping unpublished case ids",
                        extra={"dropped": [repr(d) for d in dropped]})
        room = wire.MAX_CASE_CARDS - len(self.case_ids)
        self.case_ids.extend(kept[:max(room, 0)])
        if not kept:
            return ToolOutcome(False, "no published case-study ids",
                               "error: none of those ids is published")
        return ToolOutcome(True, for_model="shown")

    def _gate(self) -> ToolOutcome:
        """Show the paywall card, for non-subscribers only.

        Returns:
            ToolOutcome: Not ok for a subscriber.
        """
        if self.identity.subscriber:
            log.warning("show_access_prompt for a subscriber; ignored")
            return ToolOutcome(False, "visitor is a subscriber",
                               "error: the visitor is a subscriber")
        self.gate = True
        return ToolOutcome(True, for_model="shown")

    def _download(self, publication_id: Any) -> ToolOutcome:
        """Offer a published publication to a subscriber.

        Args:
            publication_id: The id the model sent.

        Returns:
            ToolOutcome: What happened.
        """
        pub = self.catalog.publication(publication_id)
        if pub is None:
            log.warning("dropping unpublished publication id",
                        extra={"publication_id": repr(publication_id)})
            return ToolOutcome(False, "unknown publication id",
                               "error: that publication is not published")
        if not self.identity.subscriber:
            # The card would only lead to a refusal; show the paywall.
            self.gate = True
            return ToolOutcome(True, "paywall shown instead",
                               "shown the TMP Access card instead")
        if not 1900 <= pub.year <= 2100:
            log.error("publication has no valid year; card dropped",
                      extra={"publication_id": pub.publication_id})
            return ToolOutcome(False, "publication has no year",
                               "error: publication unavailable")
        if all(d["publication_id"] != pub.publication_id
               for d in self.downloads):
            self.downloads.append({
                "publication_id": pub.publication_id,
                "title": pub.title_in(self.language),
                "year": pub.year,
            })
        return ToolOutcome(True, for_model="shown")

    def events(self) -> list[str]:
        """Encode the collected cards, in wire order.

        Returns:
            list[str]: SSE records: ``cases``, ``gate``, ``download``.
        """
        out: list[str] = []
        cases = wire.event_cases(self.case_ids)
        if cases is not None:
            out.append(cases)
        if self.gate and not self.identity.subscriber:
            out.append(wire.event_gate(wire.GATE_SUBSCRIBER_ONLY))
        for d in self.downloads:
            out.append(wire.event_download(**d))
        return out

    def stored(self) -> dict[str, Any]:
        """The cards as stored with the assistant message.

        Returns:
            dict[str, Any]: Only the keys that have cards.
        """
        data: dict[str, Any] = {}
        if self.case_ids:
            data["cases"] = list(self.case_ids)
        if self.gate and not self.identity.subscriber:
            data["gate"] = wire.GATE_SUBSCRIBER_ONLY
        if self.downloads:
            data["downloads"] = list(self.downloads)
        return data
