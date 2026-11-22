# The LLM adapter: one small interface in front of the writer model.
#
# The orchestrator talks only to LLMClient.stream(). Three clients sit
# behind it:
#
#   - LiteLLMClient: the real one (ADR-0002). LiteLLM turns Anthropic and
#     Mistral streams into the OpenAI delta shape; this module normalises
#     those chunks to StreamDelta (adapted from NINOBOT origin/main
#     orchestrator._iter_stream/_extract_delta). Imported lazily, and
#     only ever constructed when TMPBOT_LLM_PROVIDER=litellm.
#   - ScriptedLLM: the test double. It plays back scripted replies (text
#     and tool calls) and records every request exactly as the
#     orchestrator sent it, so tests can assert on the prompt itself.
#   - LocalFakeLLM: a stand-in for local work without keys. It answers
#     from the reference material in the prompt with a plainly labelled
#     test reply and real tool calls, so the cards can be seen end to
#     end. Production refuses it (Settings.check).
#
# No test ever calls a real provider: TMPBOT has no keys yet.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Writer-model clients behind one streaming interface."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Protocol, Sequence

from .adaptive_thinking import is_synthetic_tool_delta
from .cost import TurnUsage

log = logging.getLogger("tmpbot.llm")


class LLMError(Exception):
    """A writer call failed.

    Attributes:
        code: Wire-format error code, ``upstream_timeout`` or
            ``upstream_unavailable``.
    """

    def __init__(self, code: str, message: str) -> None:
        """Create the error.

        Args:
            code: Wire-format error code.
            message: Detail for logs.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolCallDelta:
    """A piece of a streamed tool call.

    Attributes:
        index: Which tool call this piece belongs to.
        call_id: Provider id of the call, on its first piece.
        name: Tool name, on its first piece.
        arguments: A fragment of the JSON arguments.
    """

    index: int
    call_id: Optional[str]
    name: str
    arguments: str


@dataclass(frozen=True)
class StreamDelta:
    """One normalised stream chunk.

    Attributes:
        text: Answer text in this chunk.
        tool_calls: Tool-call pieces in this chunk.
        finish_reason: Provider finish reason on the last chunk.
        usage: Token usage, usually on the final chunk only.
        reasoning: Chain-of-thought text in this chunk. Never shown to
            the visitor; kept so the orchestrator can see a writer that
            spends its token cap on thinking.
    """

    text: str = ""
    tool_calls: tuple[ToolCallDelta, ...] = ()
    finish_reason: Optional[str] = None
    usage: Optional[TurnUsage] = None
    reasoning: str = ""


class LLMClient(Protocol):
    """What the orchestrator needs from a writer."""

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Stream one writer call.

        Args:
            model: LiteLLM model id.
            messages: Chat messages, system first.
            tools: OpenAI-style function schemas.
            timeout_s: Seconds before the call is abandoned.
            **kwargs: Model-dependent arguments (max_tokens and so on).

        Returns:
            AsyncIterator[StreamDelta]: The normalised chunks.
        """
        ...


# ----------------------------------------------------------------------
# Chunk normalisation (LiteLLM's OpenAI-shaped stream)
# ----------------------------------------------------------------------


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Read a key or attribute, whichever the object has.

    Args:
        obj: A dict or an object.
        name: Key or attribute name.
        default: Value when missing.

    Returns:
        Any: The value or ``default``.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage(raw: Any) -> Optional[TurnUsage]:
    """Convert a LiteLLM usage object to TurnUsage.

    Args:
        raw: LiteLLM ``Usage`` model or dict, or None.

    Returns:
        Optional[TurnUsage]: The counts, or None when absent.
    """
    if not raw:
        return None

    def count(name: str) -> int:
        return int(_attr(raw, name, 0) or 0)

    return TurnUsage(
        input_tokens=count("prompt_tokens"),
        output_tokens=count("completion_tokens"),
        cache_read_tokens=count("cache_read_input_tokens"),
        cache_write_tokens=count("cache_creation_input_tokens"),
    )


# Content-part types that carry the writer's chain of thought, not the
# answer. Mistral Large 4 sends these inside list-shaped ``content``.
_REASONING_PART_TYPES = frozenset({"thinking", "reasoning"})


def _split_content(content: Any) -> tuple[str, str]:
    """Split a delta's ``content`` into answer text and reasoning text.

    OpenAI-style providers send ``content`` as a string. Mistral's
    reasoning models send a list of typed parts instead:
    ``[{"type": "thinking", "thinking": [{"type": "text", "text": ...}]}]``
    for the chain of thought and ``[{"type": "text", "text": ...}]`` for
    the answer. Only the answer may reach the visitor.

    Args:
        content: A string, a list of parts, or None.

    Returns:
        tuple[str, str]: ``(answer_text, reasoning_text)``.
    """
    if not content:
        return "", ""
    if isinstance(content, str):
        return content, ""
    if not isinstance(content, (list, tuple)):
        log.warning("unexpected delta content type",
                    extra={"type": type(content).__name__})
        return "", ""
    text: list[str] = []
    reasoning: list[str] = []
    for part in content:
        if isinstance(part, str):
            text.append(part)
            continue
        kind = _attr(part, "type", "text")
        if kind in _REASONING_PART_TYPES:
            inner = _attr(part, "thinking", None)
            if inner is None:
                inner = _attr(part, "text", "")
            reasoning.append(_split_content(inner)[0])
        elif kind == "text":
            text.append(str(_attr(part, "text", "") or ""))
        # Other part types (references, images) carry no answer text.
    return "".join(text), "".join(reasoning)


def extract_delta(chunk: Any) -> StreamDelta:
    """Normalise one LiteLLM stream chunk.

    Args:
        chunk: A ``ModelResponse`` chunk or an equivalent dict.

    Returns:
        StreamDelta: Text, tool-call pieces, finish reason and usage.
    """
    usage = _usage(_attr(chunk, "usage", None))
    choices = _attr(chunk, "choices", None) or []
    if not choices:
        return StreamDelta(usage=usage)
    choice = choices[0]
    delta = _attr(choice, "delta", None) or {}
    pieces: list[ToolCallDelta] = []
    for tc in _attr(delta, "tool_calls", None) or []:
        fn = _attr(tc, "function", None) or {}
        index = _attr(tc, "index", 0)
        call_id = _attr(tc, "id", None)
        name = _attr(fn, "name", None)
        arguments = _attr(fn, "arguments", None)
        if is_synthetic_tool_delta(index, call_id, name, arguments):
            continue
        pieces.append(ToolCallDelta(
            index=int(index or 0),
            call_id=call_id,
            name=name or "",
            arguments=arguments or "",
        ))
    text, reasoning = _split_content(_attr(delta, "content", None))
    # Newer LiteLLM moves the chain of thought to ``reasoning_content``.
    reasoning += str(_attr(delta, "reasoning_content", "") or "")
    return StreamDelta(
        text=text,
        tool_calls=tuple(pieces),
        finish_reason=_attr(choice, "finish_reason", None),
        usage=usage,
        reasoning=reasoning,
    )


# ----------------------------------------------------------------------
# LiteLLM (real providers)
# ----------------------------------------------------------------------


class LiteLLMClient:
    """Writer calls through LiteLLM's ``acompletion``.

    API keys come from the environment LiteLLM reads
    (``ANTHROPIC_API_KEY``, ``MISTRAL_API_KEY``).
    """

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Stream one writer call through LiteLLM.

        Args:
            model: LiteLLM model id.
            messages: Chat messages, system first.
            tools: Function schemas; omitted from the call when empty.
            timeout_s: Seconds before the call is abandoned.
            **kwargs: Model-dependent arguments.

        Yields:
            StreamDelta: Normalised chunks.

        Raises:
            LLMError: When the provider times out or fails.
        """
        import litellm  # heavy import, only on the real path

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=tools or None,
                stream=True,
                # Mistral sends no usage on streams unless asked
                # (NINOBOT M9.40); Anthropic ignores the flag.
                stream_options={"include_usage": True},
                timeout=timeout_s,
                **kwargs,
            )
            async for chunk in response:
                yield extract_delta(chunk)
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider errors vary
            code = ("upstream_timeout" if "timeout" in type(exc).__name__
                    .lower() else "upstream_unavailable")
            raise LLMError(code, f"{type(exc).__name__}: {exc}") from exc


# ----------------------------------------------------------------------
# Scripted fake (tests)
# ----------------------------------------------------------------------


@dataclass
class ScriptedReply:
    """One scripted writer call.

    Attributes:
        text: Answer text, streamed a word at a time.
        tool_calls: ``(name, arguments)`` pairs, sent after the text.
            ``arguments`` may be a dict or a raw (possibly broken) JSON
            string.
        finish_reason: Finish reason on the last chunk; defaults to
            ``tool_calls`` when there are tool calls, else ``stop``.
        usage: Usage reported on the final chunk.
        error: Raise this before any chunk, as a failed call would.
        fail_after_text: Raise an upstream error after the text.
    """

    text: str = ""
    tool_calls: list[tuple[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: TurnUsage = field(
        default_factory=lambda: TurnUsage(input_tokens=100,
                                          output_tokens=20)
    )
    error: Optional[LLMError] = None
    fail_after_text: bool = False


@dataclass
class RecordedCall:
    """A request the orchestrator sent to the writer.

    Attributes:
        model: Model id.
        messages: A deep copy of the messages, system prompt included.
        tools: The tool schemas offered.
        kwargs: Every other argument.
    """

    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    kwargs: dict[str, Any]

    def prompt_text(self) -> str:
        """Return every piece of text in the request, joined.

        Returns:
            str: System blocks, history and tool schemas as one string,
            for "does the prompt contain X" assertions.
        """
        return json.dumps(
            {"messages": self.messages, "tools": self.tools},
            ensure_ascii=False,
        )


class ScriptedLLM:
    """Plays back scripted replies and records each request.

    Args:
        replies: Replies in call order. When they run out, further calls
            get a plain short answer.
        delay_s: Pause between chunks.
    """

    def __init__(self, replies: Sequence[ScriptedReply] = (),
                 delay_s: float = 0.0) -> None:
        """Create the fake.

        Args:
            replies: Replies in call order.
            delay_s: Pause between chunks.
        """
        self.replies = list(replies)
        self.delay_s = delay_s
        self.calls: list[RecordedCall] = []

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Record the request and stream the next scripted reply.

        Args:
            model: Model id.
            messages: Chat messages.
            tools: Tool schemas.
            timeout_s: Recorded with the other arguments.
            **kwargs: Recorded.

        Yields:
            StreamDelta: The scripted chunks.

        Raises:
            LLMError: When the reply is scripted to fail.
        """
        self.calls.append(RecordedCall(
            model=model,
            messages=copy.deepcopy(messages),
            tools=copy.deepcopy(tools),
            kwargs={"timeout_s": timeout_s, **copy.deepcopy(kwargs)},
        ))
        reply = (self.replies.pop(0) if self.replies
                 else ScriptedReply(text="Scripted default answer."))
        if reply.error is not None:
            raise reply.error
        for piece in re.findall(r"\S+\s*|\s+", reply.text):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield StreamDelta(text=piece)
        if reply.fail_after_text:
            raise LLMError("upstream_unavailable", "scripted mid-stream")
        for k, (name, args) in enumerate(reply.tool_calls):
            raw = args if isinstance(args, str) else json.dumps(args)
            yield StreamDelta(tool_calls=(
                ToolCallDelta(k, f"call_{k}", name, raw),
            ))
        finish = reply.finish_reason or (
            "tool_calls" if reply.tool_calls else "stop"
        )
        yield StreamDelta(finish_reason=finish, usage=reply.usage)


# ----------------------------------------------------------------------
# Local stand-in (no keys)
# ----------------------------------------------------------------------

_REF_RE = re.compile(
    r'<reference id="([^"]+)" doc_type="([^"]+)" access="([^"]+)" '
    r'title="([^"]*)"'
)
_GATED_HINT_RE = re.compile(
    r"(download|report|publication|pdf|dataset|rank|forecast|"
    r"asset[- ]level|ダウンロード|レポート|報告書)",
    re.IGNORECASE,
)


class LocalFakeLLM:
    """Answers from the prompt's reference material without a model.

    The reply says plainly that it is a local test answer. It names the
    best reference titles, shows up to two case cards, and shows the
    paywall card when the prompt carries the gating layer and the
    question looks gated (the prototype's GATE_RX). Used only when
    ``TMPBOT_LLM_PROVIDER=fake``; never in production.

    Args:
        delay_s: Pause between chunks, so streaming is visible.
    """

    def __init__(self, delay_s: float = 0.0) -> None:
        """Create the stand-in.

        Args:
            delay_s: Pause between chunks.
        """
        self.delay_s = delay_s

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        **kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        """Produce a labelled test answer with real tool calls.

        Args:
            model: Ignored.
            messages: Chat messages; the system prompt is read.
            tools: Tool schemas offered.
            timeout_s: Ignored.
            **kwargs: Ignored.

        Yields:
            StreamDelta: Chunks of the test answer.
        """
        system = json.dumps(messages[0].get("content"), ensure_ascii=False)
        system = system.replace('\\"', '"')
        refs = _REF_RE.findall(system)
        question = str(messages[-1].get("content", ""))
        offered = {t["function"]["name"] for t in tools}
        if messages[-1].get("role") == "tool":
            # Follow-up after a tool call with no text yet.
            text, calls = "Here is what the local test writer found.", []
        else:
            gated = ("show_access_prompt" in offered
                     and _GATED_HINT_RE.search(question))
            titles = [r[3] for r in refs if r[2] == "public"][:2]
            if gated:
                text = ("This is a local test answer, not a model. That "
                        "question is part of TMP Access.")
                calls = [("show_access_prompt", {})]
            else:
                found = ("; ".join(titles) if titles
                         else "nothing in the corpus")
                text = ("This is a local test answer, not a model. The "
                        f"closest material is: {found}.")
                cases = [r[0] for r in refs
                         if r[1] == "case_study" and r[2] == "public"][:2]
                calls = ([("show_case_studies", {"ids": cases})]
                         if cases and "show_case_studies" in offered
                         else [])
        for piece in re.findall(r"\S+\s*", text):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield StreamDelta(text=piece)
        for k, (name, args) in enumerate(calls):
            yield StreamDelta(tool_calls=(
                ToolCallDelta(k, f"local_{k}", name, json.dumps(args)),
            ))
        yield StreamDelta(
            finish_reason="tool_calls" if calls else "stop",
            usage=TurnUsage(),
        )
