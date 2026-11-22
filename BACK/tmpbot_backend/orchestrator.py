# One chat turn: retrieval, prompt, writer stream, tool loop, cards.
#
# Adapted from NINOBOT origin/main orchestrator.py (its tool loop, the
# pre-stream fallback to a second writer, usage accounting across
# iterations), cut down to TMPBOT's contract (spec 3b):
#
#   meta -> token* (with tool_call/tool_result pairs) -> cards -> done
#                                                     \-> error
#
# TMPBOT's tools only draw cards, so they need no follow-up call once the
# writer has written its answer: the turn ends after the first call that
# produced text. A follow-up runs only when the writer called a tool
# before writing anything, so it still gets to write the answer.
#
# Cards are held back until the text is finished and every id has been
# checked against published content (tools.py).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Stream one chat turn."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional, Sequence

from . import wire
from .adaptive_thinking import writer_call_kwargs
from .catalog import Catalog
from .cost import TurnUsage, turn_cost_usd
from .fees import FeeSchedule
from .identity import Identity
from .llm import LLMClient, LLMError
from .prompt import PromptLayers, supports_prompt_cache, system_message
from .retrieval import RetrievalBundle, Retriever, merge_bundles
from .routing import RoutingPolicy
from .stores import (
    ConversationStore,
    Owner,
    StoredTurn,
    UsageLog,
    UsageRow,
)
from .tool_text import TextToolFilter
from .tools import CardCollector, tools_for

log = logging.getLogger("tmpbot.orchestrator")

MAX_TOOL_ITERATIONS = 3
# Earlier user questions searched again for carry-over material.
CARRY_OVER_QUESTIONS = 2


@dataclass(frozen=True)
class ChatDeps:
    """Everything a turn needs, injected so tests can swap any part.

    Attributes:
        policy: Writer routing.
        layers: Prompt layers.
        catalog: Published ids.
        llm: Writer client.
        retriever: Reference-material search.
        usage: Usage log.
        conversations: Conversation store.
        fees: Fee lines with daily exchange rates; None uses the
            fallback rates in BOTS/POLC/fees.yaml.
    """

    policy: RoutingPolicy
    layers: PromptLayers
    catalog: Catalog
    llm: LLMClient
    retriever: Retriever
    usage: UsageLog
    conversations: ConversationStore
    fees: Optional[FeeSchedule] = None


@dataclass(frozen=True)
class Turn:
    """One validated chat turn.

    Attributes:
        language: Reply locale.
        history: ``(role, content)`` pairs, the last from the user.
        identity: The caller.
        message_id: Wire id of the assistant turn.
        audience: "Asking as" key, or None.
        conversation_id: Conversation the client says it is in.
        visitor_id: Signed-out visitor, if any.
        on_failure: Called when the turn fails before any text (gives a
            counted free question back).
    """

    language: str
    history: Sequence[tuple[str, str]]
    identity: Identity
    message_id: str
    audience: Optional[str] = None
    conversation_id: Optional[str] = None
    visitor_id: Optional[str] = None
    on_failure: Optional[Callable[[], None]] = None


def retrieval_query(history: Sequence[tuple[str, str]]) -> str:
    """The text to search with: the last question, plus the one before.

    A follow-up like "And land tenure?" carries little on its own; the
    previous question adds the subject.

    Args:
        history: The conversation.

    Returns:
        str: Search text.
    """
    questions = [c for r, c in history if r == "user"]
    return " ".join(questions[-2:])


def carry_over_queries(history: Sequence[tuple[str, str]]) -> list[str]:
    """Earlier user questions to search again, most recent first.

    The writer sees its earlier answers in the history but not the
    material they came from. Searching the previous questions again
    puts that material back in front of it, so it does not retract
    accurate answers (see ``merge_bundles``).

    Args:
        history: The conversation; the last message is the current
            question.

    Returns:
        list[str]: Up to ``CARRY_OVER_QUESTIONS`` questions, without
        blanks or repeats, and without the current question.
    """
    questions = [c for r, c in history if r == "user"]
    current = questions[-1].strip() if questions else ""
    out: list[str] = []
    for q in reversed(questions[:-1]):
        text = q.strip()
        if text and text != current and text not in out:
            out.append(text)
        if len(out) == CARRY_OVER_QUESTIONS:
            break
    return out


async def retrieve_for_turn(retriever: Retriever,
                            history: Sequence[tuple[str, str]],
                            language: str,
                            identity: Identity,
                            audience: Optional[str] = None
                            ) -> RetrievalBundle:
    """Retrieve for the current question and carry earlier material over.

    A fault in the current question's search raises, as before. A fault
    in a carry-over search is logged and ignored: the turn still has
    its own material. Every search uses the turn's audience, so earlier
    questions' material follows the same audience rules.

    Args:
        retriever: The search.
        history: The conversation.
        language: Reply locale.
        identity: The caller.
        audience: The "Asking as" key, or None.

    Returns:
        RetrievalBundle: The merged material, current question first.

    Raises:
        Exception: Whatever the retriever raises for the current query.
    """
    queries = carry_over_queries(history)
    results = await asyncio.gather(
        retriever.retrieve(retrieval_query(history), language, identity,
                           audience=audience),
        *(retriever.retrieve(q, language, identity, audience=audience)
          for q in queries),
        return_exceptions=True,
    )
    primary = results[0]
    if isinstance(primary, BaseException):
        raise primary
    carried = []
    for result in results[1:]:
        if isinstance(result, BaseException):
            log.warning("carry-over retrieval failed; skipped",
                        extra={"error": repr(result)})
        else:
            carried.append(result)
    return merge_bundles(primary, carried)


def writer_messages(system: dict[str, Any],
                    history: Sequence[tuple[str, str]],
                    keep: int) -> list[dict[str, Any]]:
    """The messages for the writer: system, then recent history.

    Args:
        system: The system message.
        history: The conversation.
        keep: Most history messages to send.

    Returns:
        list[dict[str, Any]]: Messages; history starts on a user turn.
    """
    recent = list(history)[-keep:]
    while recent and recent[0][0] != "user":
        recent.pop(0)
    return [system] + [{"role": r, "content": c} for r, c in recent]


@dataclass
class _Call:
    """What one writer call produced."""

    text: str
    tool_calls: list[dict[str, Any]]
    finish_reason: Optional[str]
    usage: TurnUsage
    reasoning_chars: int = 0


async def stream_turn(turn: Turn, deps: ChatDeps) -> AsyncIterator[str]:
    """Stream one chat turn as SSE records.

    Args:
        turn: The validated turn.
        deps: Dependencies.

    Yields:
        str: Encoded SSE records, ending with ``done`` or ``error``.
    """
    yield wire.event_meta(turn.message_id, turn.language)
    policy = deps.policy
    model = policy.model_for(turn.language)
    usage = TurnUsage()
    answer = ""
    fallback_used = False
    stop_reason = wire.STOP_END_TURN
    cards = CardCollector(deps.catalog, turn.identity, turn.language)
    if deps.fees is not None:
        # Background fetch when the day's rates are due; never waits.
        deps.fees.maybe_refresh()

    try:
        bundle = await retrieve_for_turn(
            deps.retriever, turn.history, turn.language, turn.identity,
            audience=turn.audience,
        )
    except Exception:  # noqa: BLE001 - any retrieval fault ends the turn
        log.exception("retrieval failed", extra={"message_id":
                                                 turn.message_id})
        await asyncio.to_thread(_fail, turn)
        await _record_usage(deps, turn, model, usage, wire.ERR_INTERNAL,
                            False)
        yield wire.event_error(wire.ERR_INTERNAL, "Retrieval failed.")
        return

    tools = tools_for(turn.identity)
    messages = _build_messages(deps, turn, model, bundle)

    try:
        for iteration in range(MAX_TOOL_ITERATIONS):
            call = _Call("", [], None, TurnUsage())
            try:
                async for record in _run_call(deps, model, messages,
                                              tools, call):
                    yield record
            except LLMError as first_exc:
                if (iteration == 0 and not answer and not call.text
                        and not fallback_used
                        and model != policy.fallback_model):
                    # Log the cause: without it a provider-side change
                    # (a model alias that moved, a new stream shape)
                    # looks like a plain outage.
                    log.warning("writer failed before any text; falling "
                                "back", extra={"model": model,
                                               "code": first_exc.code,
                                               "detail": str(first_exc)})
                    model = policy.fallback_model
                    fallback_used = True
                    messages = _build_messages(deps, turn, model, bundle)
                    call = _Call("", [], None, TurnUsage())
                    async for record in _run_call(deps, model, messages,
                                                  tools, call):
                        yield record
                else:
                    usage = usage.plus(call.usage)
                    answer += call.text
                    raise
            usage = usage.plus(call.usage)
            answer += call.text

            for tc in call.tool_calls:
                args = _parse_args(tc["arguments"])
                yield wire.event_tool_call(tc["name"], args)
                outcome = cards.apply(tc["name"], args)
                yield wire.event_tool_result(
                    tc["name"], outcome.ok, outcome.detail or None)
                tc["result"] = outcome.for_model

            if not call.tool_calls:
                stop_reason = _stop_reason(call.finish_reason, answer)
                break
            if answer.strip():
                # Display tools need no follow-up once the answer exists.
                stop_reason = wire.STOP_END_TURN
                break
            messages = messages + _tool_messages(call)
    except LLMError as exc:
        log.warning("writer failed", extra={"code": exc.code,
                                            "model": model,
                                            "detail": str(exc)})
        if not answer.strip():
            await asyncio.to_thread(_fail, turn)
        await _record_usage(deps, turn, model, usage, exc.code,
                            fallback_used)
        yield wire.event_error(exc.code, "The writer model failed.")
        return

    for record in cards.events():
        yield record

    conversation_id = await _save(deps, turn, answer, cards)
    await _record_usage(deps, turn, model, usage, stop_reason,
                        fallback_used, conversation_id)
    yield wire.event_done(stop_reason, usage.input_tokens,
                          usage.output_tokens, conversation_id)


def _build_messages(deps: ChatDeps, turn: Turn, model: str,
                    bundle: RetrievalBundle) -> list[dict[str, Any]]:
    """System message plus trimmed history for a model.

    Args:
        deps: Dependencies.
        turn: The turn.
        model: Writer model (decides caching and length layers).
        bundle: Retrieved material.

    Returns:
        list[dict[str, Any]]: Writer messages.
    """
    system = system_message(
        deps.layers, turn.identity, model, turn.language, turn.audience,
        bundle, deps.policy.mistral_length_shaping,
        deps.fees.ranges_text() if deps.fees is not None else None,
    )
    return writer_messages(system, turn.history,
                           deps.policy.history_messages)


async def _run_call(deps: ChatDeps, model: str,
                    messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]],
                    call: _Call) -> AsyncIterator[str]:
    """Run one writer call, yielding token records and filling ``call``.

    Args:
        deps: Dependencies.
        model: Writer model.
        messages: Writer messages.
        tools: Tool schemas.
        call: Filled with text, tool calls, finish reason and usage.

    Yields:
        str: ``token`` records.

    Raises:
        LLMError: When the writer fails.
    """
    policy = deps.policy
    kwargs = writer_call_kwargs(
        model, max_tokens=policy.max_tokens,
        temperature=policy.temperature, effort=policy.thinking_effort,
    )
    buckets: dict[int, dict[str, Any]] = {}
    # Mistral sometimes types a tool call's JSON into the answer; hold
    # it back so the visitor never sees it (tool_text.py).
    leak_filter = TextToolFilter(tools)
    async for delta in deps.llm.stream(model=model, messages=messages,
                                       tools=tools,
                                       timeout_s=policy.timeout_s,
                                       **kwargs):
        if delta.reasoning:
            call.reasoning_chars += len(delta.reasoning)
        shown = leak_filter.feed(delta.text) if delta.text else ""
        if shown:
            call.text += shown
            yield wire.event_token(shown)
        for piece in delta.tool_calls:
            bucket = buckets.setdefault(
                piece.index, {"id": None, "name": "", "arguments": ""})
            if piece.call_id:
                bucket["id"] = piece.call_id
            if piece.name:
                bucket["name"] = piece.name
            bucket["arguments"] += piece.arguments
        if delta.finish_reason:
            call.finish_reason = delta.finish_reason
        if delta.usage is not None:
            # Providers repeat cumulative usage; keep the largest.
            u = delta.usage
            call.usage = TurnUsage(
                max(call.usage.input_tokens, u.input_tokens),
                max(call.usage.output_tokens, u.output_tokens),
                max(call.usage.cache_read_tokens, u.cache_read_tokens),
                max(call.usage.cache_write_tokens, u.cache_write_tokens),
            )
    call.tool_calls = [
        {"id": b["id"] or f"call_{i}", "name": b["name"],
         "arguments": b["arguments"]}
        for i, b in sorted(buckets.items()) if b["name"]
    ]
    leaked = leak_filter.finish(bool(call.tool_calls))
    if leaked.release:
        call.text += leaked.release
        yield wire.event_token(leaked.release)
    if leaked.tool_call is not None:
        call.tool_calls.append(leaked.tool_call)
    if call.reasoning_chars:
        log.warning("writer streamed reasoning; it counts against "
                    "max_tokens", extra={
                        "model": model,
                        "reasoning_chars": call.reasoning_chars,
                        "visible_chars": len(call.text),
                        "finish_reason": call.finish_reason})
    if (call.finish_reason == "length" and not call.text.strip()
            and not call.tool_calls):
        # The cap ran out before the first visible word (the writer
        # spent it all on thinking). An empty answer must not reach the
        # visitor as a normal turn; fail so the fallback writer runs.
        raise LLMError("upstream_unavailable",
                       "max_tokens reached before any visible text "
                       f"({call.reasoning_chars} reasoning chars)")


def _parse_args(raw: str) -> Any:
    """Parse tool arguments, tolerating broken JSON.

    Args:
        raw: The JSON text the writer streamed.

    Returns:
        Any: The parsed value, or ``{}`` when unparseable.
    """
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        log.warning("unparseable tool arguments", extra={"raw": raw[:200]})
        return {}


def _tool_messages(call: _Call) -> list[dict[str, Any]]:
    """The assistant tool-call message and its tool results.

    Args:
        call: The call that made tool calls.

    Returns:
        list[dict[str, Any]]: Messages for the follow-up call.
    """
    out: list[dict[str, Any]] = [{
        "role": "assistant",
        "content": call.text or None,
        "tool_calls": [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"],
                          "arguments": tc["arguments"] or "{}"}}
            for tc in call.tool_calls
        ],
    }]
    for tc in call.tool_calls:
        out.append({"role": "tool", "tool_call_id": tc["id"],
                    "content": tc.get("result", "ok")})
    return out


def _stop_reason(finish_reason: Optional[str], answer: str) -> str:
    """Map the provider's finish reason to the wire stop reason.

    Args:
        finish_reason: Provider finish reason.
        answer: The answer so far.

    Returns:
        str: One of ``wire.STOP_REASONS``.
    """
    if finish_reason == "length":
        return wire.STOP_MAX_TOKENS
    if finish_reason in ("refusal", "content_filter"):
        log.warning("writer refused")
        return wire.STOP_REFUSAL
    if not answer.strip():
        return wire.STOP_ABSTENTION
    return wire.STOP_END_TURN


def _fail(turn: Turn) -> None:
    """Run the turn's failure hook, never raising.

    Args:
        turn: The turn.
    """
    if turn.on_failure is None:
        return
    try:
        turn.on_failure()
    except Exception:  # noqa: BLE001 - a refund must not break the stream
        log.exception("failure hook failed")


async def _save(deps: ChatDeps, turn: Turn, answer: str,
                cards: CardCollector) -> Optional[str]:
    """Store the turn; failures are logged, not raised.

    Args:
        deps: Dependencies.
        turn: The turn.
        answer: Answer text.
        cards: The cards shown.

    Returns:
        Optional[str]: The conversation id, or None if storing failed.
    """
    owner = Owner(user_id=turn.identity.user_id,
                  visitor_id=None if turn.identity.signed_in
                  else turn.visitor_id)
    if owner.user_id is None and owner.visitor_id is None:
        return None
    stored = StoredTurn(
        language=turn.language, audience=turn.audience,
        question=turn.history[-1][1], answer=answer.strip(),
        wire_message_id=turn.message_id, events=cards.stored() or None,
    )
    try:
        return await asyncio.to_thread(
            deps.conversations.save_turn, owner, turn.conversation_id,
            stored,
        )
    except Exception:  # noqa: BLE001 - storage must not lose the answer
        log.exception("storing the conversation failed")
        return None


async def _record_usage(deps: ChatDeps, turn: Turn, model: str,
                        usage: TurnUsage, stop: str, fallback_used: bool,
                        conversation_id: Optional[str] = None) -> None:
    """Write the usage row; failures are logged, not raised.

    Args:
        deps: Dependencies.
        turn: The turn.
        model: Model that answered (or failed).
        usage: Token counts.
        stop: Stop reason or error code.
        fallback_used: Whether the fallback writer ran.
        conversation_id: Stored conversation, if any.
    """
    row = UsageRow(
        model=model, language=turn.language, plan=turn.identity.plan,
        message_id=turn.message_id, conversation_id=conversation_id,
        visitor_id=turn.visitor_id, user_id=turn.identity.user_id,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cost_usd=turn_cost_usd(deps.policy, model, usage),
        prompt_cache_enabled=supports_prompt_cache(model),
        stop_reason=stop, fallback_used=fallback_used,
    )
    try:
        await asyncio.to_thread(deps.usage.record, row)
    except Exception:  # noqa: BLE001 - logging spend must not break chat
        log.exception("usage_log write failed",
                      extra={"message_id": turn.message_id})
    log.info("chat turn cost", extra={
        "message_id": turn.message_id, "model": model,
        "plan": turn.identity.plan, "cost_usd": row.cost_usd,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens, "stop": stop,
    })
