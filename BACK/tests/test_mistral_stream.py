# Regression tests for the French (Mistral) writer stream, 2026-11-20.
#
# The first real-key trial found two failures. Mistral's -latest alias had
# moved to Mistral Large 4, a reasoning model that streams its chain of
# thought as list-shaped `content` chunks:
#
#   {"delta": {"content": [{"type": "thinking",
#                           "thinking": [{"type": "text", "text": "..."}],
#                           "closed": true}]}}
#
# 1. The thinking spent the 400-token cap (usage showed 399 reasoning
#    tokens of 400) so the visitor saw almost no answer, with
#    stop = max_tokens.
# 2. LiteLLM 1.53.1 crashes on a list `content` ("can only concatenate
#    str (not list) to str"), so the call failed before any text and the
#    orchestrator fell back to Claude.
#
# The fix pins French to the dated, non-reasoning mistral-large-2512 and
# makes the stream handling safe against list content, reasoning text and
# an answer that never starts.
#
# A third shape showed up on the pinned model: Mistral Large 3 sometimes
# types a tool call's JSON arguments into the answer text instead of
# sending a tool call. tool_text.py recovers it.
#
# These tests replay the failure shapes with scripted chunks; no
# provider is called.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Mistral stream shapes: thinking parts, tool-call order, token cap."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, AsyncIterator, Sequence

import pytest
from chat_helpers import TEST_SECRET, build_runtime
from fastapi.testclient import TestClient

from tmpbot_backend import wire
from tmpbot_backend.cost import TurnUsage
from tmpbot_backend.identity import ANONYMOUS, Identity
from tmpbot_backend.llm import (
    LiteLLMClient,
    LLMError,
    StreamDelta,
    ToolCallDelta,
    extract_delta,
)
from tmpbot_backend.main import create_app
from tmpbot_backend.routing import load_policy
from tmpbot_backend.tool_text import RECOVERED_ID_PREFIX, TextToolFilter
from tmpbot_backend.tools import tools_for

HEADERS = {"X-TMPBOT-Proxy-Secret": TEST_SECRET,
           "X-TMPBOT-Client-IP": "192.0.2.1"}
FR_QUESTION = "Que faites-vous sur le foncier ?"


# ------------------------------------------------------------- helpers


def _chunk(content: Any = None, *, tool_calls: Any = None,
           finish: Any = None, usage: Any = None,
           **delta_extra: Any) -> dict[str, Any]:
    """Build one OpenAI-shaped stream chunk as LiteLLM yields it.

    Args:
        content: The delta ``content`` (string, list of parts or None).
        tool_calls: Raw tool-call pieces.
        finish: Finish reason.
        usage: Usage dict.
        **delta_extra: Other delta fields (``reasoning_content``).

    Returns:
        dict[str, Any]: The chunk.
    """
    delta: dict[str, Any] = dict(delta_extra)
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    chunk: dict[str, Any] = {"choices": [{"index": 0, "delta": delta,
                                          "finish_reason": finish}]}
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _thinking(text: str, closed: bool = True) -> list[dict[str, Any]]:
    """A Mistral thinking part, exactly as the API streamed it.

    Args:
        text: Chain-of-thought text.
        closed: Whether Mistral marked the block closed.

    Returns:
        list[dict[str, Any]]: The ``content`` list.
    """
    part: dict[str, Any] = {
        "type": "thinking",
        "thinking": [{"type": "text", "text": text}],
    }
    if closed:
        part["closed"] = True
    return [part]


def _text_part(text: str) -> list[dict[str, Any]]:
    """A Mistral answer part.

    Args:
        text: Answer text.

    Returns:
        list[dict[str, Any]]: The ``content`` list.
    """
    return [{"type": "text", "text": text}]


class ChunkLLM:
    """A writer that replays hand-built StreamDelta lists, one per call.

    Args:
        calls: For each writer call, the deltas to stream.
    """

    def __init__(self, calls: Sequence[Sequence[StreamDelta]]) -> None:
        """Create the writer.

        Args:
            calls: For each writer call, the deltas to stream.
        """
        self.calls = [list(c) for c in calls]
        self.models: list[str] = []

    async def stream(self, *, model: str, messages: Any, tools: Any,
                     timeout_s: float,
                     **kwargs: Any) -> AsyncIterator[StreamDelta]:
        """Replay the next call's deltas.

        Args:
            model: Recorded.
            messages: Ignored.
            tools: Ignored.
            timeout_s: Ignored.
            **kwargs: Ignored.

        Yields:
            StreamDelta: The scripted chunks.
        """
        self.models.append(model)
        for delta in self.calls.pop(0):
            yield delta


def _ask(settings: Any, llm: Any,
         question: str = FR_QUESTION,
         language: str = "fr") -> list[tuple[str, Any]]:
    """Send one question through the real app with a scripted writer.

    Args:
        settings: Settings fixture.
        llm: The writer.
        question: The question.
        language: Request language.

    Returns:
        list[tuple[str, Any]]: Parsed SSE events.
    """
    runtime = build_runtime(settings, llm)
    client = TestClient(create_app(settings, runtime))
    resp = client.post("/api/v1/chat", json={
        "language": language,
        "messages": [{"role": "user", "content": question}]},
        headers=HEADERS)
    assert resp.status_code == 200, resp.text
    return wire.parse_sse(resp.text)


def _answer(events: Sequence[tuple[str, Any]]) -> str:
    """Join the token events.

    Args:
        events: Parsed events.

    Returns:
        str: The visible answer.
    """
    return "".join(p["content"] for n, p in events if n == "token")


# ------------------------------------------- extract_delta, list content


def test_thinking_parts_are_reasoning_not_answer():
    """The Large 4 failure shape: a thinking part is never visible text."""
    delta = extract_delta(_chunk(_thinking("L'utilisateur demande ce")))
    assert delta.text == ""
    assert delta.reasoning == "L'utilisateur demande ce"


def test_text_parts_become_answer_text():
    """A list-shaped answer chunk is joined into plain text."""
    delta = extract_delta(_chunk(_text_part("Nous aidons")))
    assert delta.text == "Nous aidons" and delta.reasoning == ""


def test_mixed_thinking_and_text_parts_in_one_chunk():
    """Thinking and answer in one chunk are split; the first answer word
    is kept (the 'first word missing' symptom)."""
    delta = extract_delta(_chunk(
        _thinking("Je dois répondre.") + _text_part("Chez")
        + [{"type": "reference", "reference_ids": [1]}]))
    assert delta.text == "Chez"
    assert delta.reasoning == "Je dois répondre."


def test_plain_string_content_is_unchanged():
    """The Large 3 shape (and Claude's): string content passes through."""
    assert extract_delta(_chunk("Bonjour ")).text == "Bonjour "
    assert extract_delta(_chunk("")).text == ""
    assert extract_delta(_chunk(None)).text == ""


def test_newer_litellm_reasoning_content_is_reasoning():
    """LiteLLM 1.110 puts thinking in reasoning_content, content null."""
    delta = extract_delta(_chunk(None, reasoning_content="Je réfléchis"))
    assert delta.text == "" and delta.reasoning == "Je réfléchis"


def test_unknown_content_type_is_dropped_not_raised():
    """A surprise content type must not become text or crash."""
    assert extract_delta(_chunk(42)).text == ""


def test_text_and_tool_call_in_one_chunk_keep_both():
    """Mistral can send the last words and the tool call together."""
    delta = extract_delta(_chunk(
        "fin de phrase.",
        tool_calls=[{"index": 0, "id": "abc123xyz",
                     "function": {"name": "show_case_studies",
                                  "arguments": '{"ids": ["dryad"]}'}}],
        finish="tool_calls"))
    assert delta.text == "fin de phrase."
    assert delta.tool_calls[0].name == "show_case_studies"
    assert delta.finish_reason == "tool_calls"


# ----------------------------------- LiteLLMClient over a stubbed stream


def _stub_litellm(monkeypatch: pytest.MonkeyPatch,
                  chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Replace ``litellm.acompletion`` with a scripted stream.

    Args:
        monkeypatch: The pytest fixture.
        chunks: The chunks to stream.

    Returns:
        dict[str, Any]: Filled with the keyword arguments of the call.
    """
    seen: dict[str, Any] = {}

    async def acompletion(**kwargs: Any) -> AsyncIterator[Any]:
        seen.update(kwargs)

        async def gen() -> AsyncIterator[Any]:
            for c in chunks:
                yield c
        return gen()

    monkeypatch.setitem(sys.modules, "litellm",
                        types.SimpleNamespace(acompletion=acompletion))
    return seen


def _collect(client: LiteLLMClient, model: str) -> list[StreamDelta]:
    """Run one LiteLLMClient call to the end.

    Args:
        client: The client.
        model: Model id.

    Returns:
        list[StreamDelta]: Every delta.
    """
    async def go() -> list[StreamDelta]:
        return [d async for d in client.stream(
            model=model, messages=[], tools=[], timeout_s=5,
            max_tokens=400)]
    return asyncio.run(go())


def test_litellm_client_survives_a_reasoning_stream(monkeypatch):
    """Thinking chunks, then answer chunks: the client yields the answer
    text only and flags the reasoning."""
    chunks = [
        _chunk("", role="assistant"),
        _chunk(_thinking("Je dois répondre")),
        _chunk(_thinking(" en français.", closed=False)),
        _chunk(_text_part("Nous ")),
        _chunk(_text_part("accompagnons.")),
        _chunk("", finish="stop",
               usage={"prompt_tokens": 240, "completion_tokens": 60}),
    ]
    seen = _stub_litellm(monkeypatch, chunks)
    deltas = _collect(LiteLLMClient(), "mistral/mistral-large-2512")
    assert "".join(d.text for d in deltas) == "Nous accompagnons."
    assert "".join(d.reasoning for d in deltas) == \
        "Je dois répondre en français."
    assert seen["stream"] is True and seen["max_tokens"] == 400


def test_litellm_client_wraps_provider_errors(monkeypatch):
    """The 1.53.1 list-content crash arrives as an LLMError, so the
    orchestrator's fallback path can handle it."""
    async def acompletion(**kwargs: Any) -> Any:
        raise TypeError('can only concatenate str (not "list") to str')

    monkeypatch.setitem(sys.modules, "litellm",
                        types.SimpleNamespace(acompletion=acompletion))
    with pytest.raises(LLMError) as err:
        _collect(LiteLLMClient(), "mistral/mistral-large-2512")
    assert err.value.code == "upstream_unavailable"


# ------------------------------------------ orchestrator, French turns


def _tc(index: int, call_id: Any, name: str, args: str) -> ToolCallDelta:
    """Build a tool-call piece.

    Args:
        index: Call index.
        call_id: Provider id or None.
        name: Tool name or "".
        args: Argument fragment.

    Returns:
        ToolCallDelta: The piece.
    """
    return ToolCallDelta(index, call_id, name, args)


def test_full_french_answer_with_reasoning_and_cards(settings):
    """Thinking first, then the whole answer, then a tool call split
    across chunks: all text shown from the first word, cards drawn,
    stop is end_turn, and the thinking never reaches the visitor."""
    words = ["Nous ", "aidons ", "les ", "acteurs ", "fonciers ",
             "à ", "mesurer ", "leurs ", "risques ", "climatiques."]
    deltas = [StreamDelta(reasoning="Je dois répondre en 2 phrases.")]
    deltas += [StreamDelta(text=w) for w in words]
    deltas += [
        StreamDelta(tool_calls=(_tc(0, "abc123xyz", "show_case_studies",
                                    '{"ids": ["dry'),)),
        StreamDelta(tool_calls=(_tc(0, None, "", 'ad", "angel"]}'),)),
        StreamDelta(finish_reason="tool_calls",
                    usage=TurnUsage(input_tokens=2800, output_tokens=120)),
    ]
    events = _ask(settings, ChunkLLM([deltas]))
    assert _answer(events) == "".join(words)
    assert _answer(events).startswith("Nous ")
    assert "Je dois" not in _answer(events)
    assert ("cases", {"ids": ["dryad", "angel"]}) in events
    done = events[-1]
    assert done[0] == "done" and done[1]["stop_reason"] == "end_turn"
    assert done[1]["usage"]["output_tokens"] == 120


def test_first_word_arrives_with_the_tool_call_delta(settings):
    """The first content piece and a tool-call piece share a chunk: the
    first word is still shown (no 'starts mid-sentence')."""
    events = _ask(settings, ChunkLLM([[
        StreamDelta(text="Nous ", tool_calls=(
            _tc(0, "abc123xyz", "show_case_studies",
                '{"ids": ["dryad"]}'),)),
        StreamDelta(text="travaillons sur le foncier."),
        StreamDelta(finish_reason="tool_calls"),
    ]]))
    assert _answer(events) == "Nous travaillons sur le foncier."
    assert ("cases", {"ids": ["dryad"]}) in events


def test_tool_call_before_text_gets_a_follow_up_answer(settings):
    """A writer that calls the tool first, then writes: the follow-up
    call carries the answer from its first word."""
    llm = ChunkLLM([
        [StreamDelta(tool_calls=(_tc(0, "abc123xyz", "show_case_studies",
                                     '{"ids": ["dryad"]}'),)),
         StreamDelta(finish_reason="tool_calls")],
        [StreamDelta(text="Voici "), StreamDelta(text="un exemple."),
         StreamDelta(finish_reason="stop")],
    ])
    events = _ask(settings, llm)
    assert _answer(events) == "Voici un exemple."
    assert ("cases", {"ids": ["dryad"]}) in events
    assert events[-1][1]["stop_reason"] == "end_turn"


def test_token_cap_eaten_by_thinking_falls_back(settings):
    """The Large 4 failure: only reasoning, then finish_reason length.
    The visitor must not get a blank answer; the fallback writer runs."""
    llm = ChunkLLM([
        [StreamDelta(reasoning="x" * 800),
         StreamDelta(reasoning="y" * 800, finish_reason="length",
                     usage=TurnUsage(input_tokens=241, output_tokens=400))],
        [StreamDelta(text="Bonjour, voici la réponse."),
         StreamDelta(finish_reason="stop")],
    ])
    runtime = build_runtime(settings, llm)
    client = TestClient(create_app(settings, runtime))
    resp = client.post("/api/v1/chat", json={
        "language": "fr",
        "messages": [{"role": "user", "content": FR_QUESTION}]},
        headers=HEADERS)
    events = wire.parse_sse(resp.text)
    assert llm.models == ["mistral/mistral-large-2512",
                          "anthropic/claude-haiku-5-5"]
    assert _answer(events) == "Bonjour, voici la réponse."
    assert runtime.deps.usage.rows[-1].fallback_used


def test_token_cap_with_no_answer_and_no_fallback_is_an_error(settings):
    """Both writers return nothing but thinking: one error event."""
    empty = [StreamDelta(reasoning="r" * 500, finish_reason="length")]
    events = _ask(settings, ChunkLLM([empty, empty]))
    assert events[-1][0] == "error"
    assert not any(n == "done" for n, _ in events)
    assert _answer(events) == ""


def test_a_long_answer_cut_by_the_cap_still_reports_max_tokens(settings):
    """Visible text that hits the cap is a real truncation, not a
    failure: the text stays and the stop reason says max_tokens."""
    events = _ask(settings, ChunkLLM([[
        StreamDelta(text="Une réponse coupée"),
        StreamDelta(finish_reason="length"),
    ]]))
    assert _answer(events) == "Une réponse coupée"
    assert events[-1][1]["stop_reason"] == "max_tokens"


# --------------------------------------------------- routing is pinned


def test_french_writer_is_a_pinned_non_reasoning_model():
    """Routing never uses a moving alias: -latest moved to a reasoning
    model once and broke French on launch week."""
    from tmpbot_backend.config import REPO_ROOT
    policy = load_policy(REPO_ROOT / "BOTS" / "POLC" / "llm_routing.yaml")
    routed = set(policy.locale_models.values()) | {policy.fallback_model}
    assert policy.model_for("fr") == "mistral/mistral-large-2512"
    assert not [m for m in routed if m.endswith("-latest")]
    for model in routed:
        assert policy.price_for(model) is not None
    # 400 tokens is enough for ~90 French words plus one tool call.
    assert policy.max_tokens == 400


# --------------------------- tool-call JSON typed into the answer text
# Seen live on 2026-11-20 with Mistral Large 3: the answer ended with
# `{"ids": ["el-nino-risk-analysis"]}` as visible words, finish "stop",
# no tool_calls delta, no card.

LEAK_PIECES = ['{"ids', '": ["el', '-nino-risk', '-analysis",',
               ' "transition-min', 'erals"]}']


def _leak_deltas(prose: str, pieces: Sequence[str]) -> list[StreamDelta]:
    """Prose streamed word by word, then typed JSON, finish 'stop'.

    Args:
        prose: The answer.
        pieces: The typed JSON pieces.

    Returns:
        list[StreamDelta]: The stream.
    """
    out = [StreamDelta(text=w + " ") for w in prose.split()]
    out += [StreamDelta(text=p) for p in pieces]
    out.append(StreamDelta(finish_reason="stop",
                           usage=TurnUsage(input_tokens=3096,
                                           output_tokens=167)))
    return out


def test_typed_tool_json_becomes_a_card_and_stays_out_of_the_text(
        settings):
    """The visitor sees the prose only; the card is drawn from the JSON."""
    prose = "TMP a analysé El Niño sur des actifs critiques."
    events = _ask(settings, ChunkLLM([_leak_deltas(prose, LEAK_PIECES)]),
                  question="Que dit TMP sur El Niño ?")
    assert _answer(events).strip() == prose
    assert "{" not in _answer(events) and "ids" not in _answer(events)
    assert ("tool_call", {"name": "show_case_studies", "args": {
        "ids": ["el-nino-risk-analysis", "transition-minerals"]}}) \
        in events
    assert [p for n, p in events if n == "cases"] == [
        {"ids": ["el-nino-risk-analysis", "transition-minerals"]}]
    assert events[-1][0] == "done"
    assert events[-1][1]["stop_reason"] == "end_turn"


def test_typed_json_in_a_single_chunk_is_recovered(settings):
    """Same, when text and JSON arrive in one piece."""
    events = _ask(settings, ChunkLLM([[
        StreamDelta(text='Voici. {"ids": ["dryad"]}'),
        StreamDelta(finish_reason="stop"),
    ]]))
    assert _answer(events) == "Voici. "
    assert ("cases", {"ids": ["dryad"]}) in events


def test_typed_json_beside_a_real_tool_call_is_dropped(settings):
    """A duplicate (typed and structured) draws one card set only."""
    events = _ask(settings, ChunkLLM([[
        StreamDelta(text="Voici. "),
        StreamDelta(text='{"ids": ["angel"]}'),
        StreamDelta(tool_calls=(_tc(0, "abc123xyz", "show_case_studies",
                                    '{"ids": ["dryad"]}'),)),
        StreamDelta(finish_reason="tool_calls"),
    ]]))
    assert _answer(events) == "Voici. "
    assert [p for n, p in events if n == "cases"] == [{"ids": ["dryad"]}]


def test_truncated_or_empty_typed_json_is_hidden_not_shown(settings):
    """Broken or ambiguous JSON never reaches the visitor."""
    for pieces in (['{"ids": ["dry'], ["{}"], ['{"unknown": 1}']):
        events = _ask(settings, ChunkLLM([_leak_deltas("Bonjour.",
                                                       pieces)]))
        assert _answer(events).strip() == "Bonjour."
        assert "cases" not in [n for n, _ in events]
        assert events[-1][0] == "done"


def test_braces_in_prose_are_shown_in_the_end(settings):
    """A brace that is not tool JSON is released, in order."""
    events = _ask(settings, ChunkLLM([[
        StreamDelta(text="La clé {pays} reste. "),
        StreamDelta(text="Fin."),
        StreamDelta(finish_reason="stop"),
    ]]))
    assert _answer(events) == "La clé {pays} reste. Fin."


def test_typed_json_only_still_gets_a_written_answer(settings):
    """A writer that only types the tool call gets the follow-up call,
    like a real tool call before text."""
    llm = ChunkLLM([
        [StreamDelta(text='{"ids": ["dryad"]}'),
         StreamDelta(finish_reason="stop")],
        [StreamDelta(text="Voici un exemple."),
         StreamDelta(finish_reason="stop")],
    ])
    events = _ask(settings, llm)
    assert _answer(events) == "Voici un exemple."
    assert ("cases", {"ids": ["dryad"]}) in events


def test_recovered_tool_call_id_fits_mistral():
    """Mistral rejects tool-call ids that are not 9 letters or digits."""
    filt = TextToolFilter(tools_for(ANONYMOUS))
    assert filt.feed('ok {"ids": ["dryad"]}') == "ok "
    result = filt.finish(False)
    assert result.tool_call is not None
    call_id = result.tool_call["id"]
    assert call_id.startswith(RECOVERED_ID_PREFIX)
    assert len(call_id) == 9 and call_id.isalnum()


def test_download_json_is_matched_for_subscribers():
    """{"publication_id": ...} maps to offer_download when offered."""
    filt = TextToolFilter(tools_for(Identity(plan="access", user_id="u")))
    filt.feed('{"publication_id": "p1"}')
    result = filt.finish(False)
    assert result.tool_call and result.tool_call["name"] == "offer_download"
