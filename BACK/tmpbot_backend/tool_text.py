# Recover tool calls that a writer typed into its answer text.
#
# Mistral Large 3 (the French writer) intermittently writes a tool call's
# JSON arguments as ordinary answer text instead of a structured tool
# call. Seen in the 2026-11-20 trial: the answer ended with
# `{"ids": ["el-nino-risk-analysis"]}` as visible words, and no card was
# drawn. The API reported finish_reason "stop", with no tool_calls delta.
#
# TextToolFilter sits between the stream and the visitor. It holds back
# everything from the first "{" (French prose does not use braces), and
# at the end of the call either turns the held JSON into the tool call it
# was meant to be, drops it, or, when it was just prose, releases it.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Hold back and recover tool-call JSON typed into answer text."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

log = logging.getLogger("tmpbot.tool_text")

# Provider ids for Mistral tool calls are exactly nine letters or digits.
RECOVERED_ID_PREFIX = "txtcall"


@dataclass(frozen=True)
class FilterResult:
    """What is left once the stream has ended.

    Attributes:
        release: Held text that was ordinary prose after all, plus any
            prose that followed the JSON. Show it.
        tool_call: A recovered tool call (``id``, ``name``,
            ``arguments``), or None.
    """

    release: str = ""
    tool_call: Optional[dict[str, Any]] = None


def _looks_like_tool_json(held: str) -> bool:
    """Say whether held text starts like a JSON object, not prose.

    Args:
        held: Text from the first ``{`` onward.

    Returns:
        bool: True for ``{"`` or ``{}`` (with optional spaces).
    """
    rest = held[1:].lstrip()
    return rest[:1] in ('"', "}")


def _match_tool(args: dict[str, Any],
                tools: Sequence[dict[str, Any]]) -> Optional[str]:
    """Find the offered tool a JSON object is the arguments of.

    Args:
        args: The parsed JSON object.
        tools: OpenAI-style function schemas offered to the writer.

    Returns:
        Optional[str]: The tool name, or None when no offered tool takes
        exactly these keys (empty ``{}`` matches nothing: it is ambiguous).
    """
    if not args:
        return None
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {}) or {}
        props = set((params.get("properties") or {}).keys())
        required = set(params.get("required") or [])
        keys = set(args)
        if props and keys <= props and required <= keys:
            return str(fn.get("name"))
    return None


class TextToolFilter:
    """Hold back possible tool-call JSON from streamed answer text.

    Args:
        tools: The tool schemas offered to the writer.
    """

    def __init__(self, tools: Sequence[dict[str, Any]]) -> None:
        """Create the filter.

        Args:
            tools: The tool schemas offered to the writer.
        """
        self._tools = list(tools)
        self._held = ""

    def feed(self, text: str) -> str:
        """Take one piece of streamed text.

        Args:
            text: The piece.

        Returns:
            str: The part that is safe to show now (may be empty).
        """
        if self._held:
            self._held += text
            return ""
        brace = text.find("{")
        if brace < 0:
            return text
        self._held = text[brace:]
        return text[:brace]

    def finish(self, has_tool_calls: bool) -> FilterResult:
        """Decide what the held text was, once the stream has ended.

        Args:
            has_tool_calls: Whether the writer also made a structured
                tool call (then typed JSON is a duplicate and is dropped).

        Returns:
            FilterResult: Text to release and a recovered tool call.
        """
        held, self._held = self._held, ""
        if not held:
            return FilterResult()
        if not _looks_like_tool_json(held):
            return FilterResult(release=held)
        try:
            args, end = json.JSONDecoder().raw_decode(held)
        except json.JSONDecodeError:
            log.warning("dropping unparseable tool JSON typed into the "
                        "answer", extra={"held": held[:200]})
            return FilterResult()
        rest = held[end:].strip()
        if not isinstance(args, dict):
            return FilterResult(release=rest)
        name = None if has_tool_calls else _match_tool(args, self._tools)
        if name is None:
            log.warning("dropping tool JSON typed into the answer",
                        extra={"held": held[:200],
                               "had_tool_calls": has_tool_calls})
            return FilterResult(release=rest)
        log.warning("recovered a tool call typed into the answer",
                    extra={"tool": name})
        return FilterResult(release=rest, tool_call={
            "id": f"{RECOVERED_ID_PREFIX}00",
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False),
        })
