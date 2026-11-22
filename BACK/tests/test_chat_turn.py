# One chat turn through the real app with the scripted writer and the
# in-memory stores: SSE event order with cards, id checks, the tool
# loop, the fallback writer, routing per locale, prompt caching and the
# usage log (tmpbot_backend.orchestrator, tools, prompt).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Chat-turn behaviour with a scripted writer."""

from __future__ import annotations

import pytest
from chat_helpers import TEST_SECRET, build_runtime, make_token
from fastapi.testclient import TestClient

from tmpbot_backend import wire
from tmpbot_backend.llm import LLMError, ScriptedLLM, ScriptedReply
from tmpbot_backend.main import create_app

HEADERS = {"X-TMPBOT-Proxy-Secret": TEST_SECRET,
           "X-TMPBOT-Client-IP": "192.0.2.1"}


def _run(settings, replies, *, language="en", question="Tell me about "
         "your deforestation work", token=None, history=(), **body_extra):
    """Ask one question and return events, the writer and the runtime.

    Args:
        settings: Settings fixture.
        replies: Scripted replies.
        language: Request language.
        question: The question.
        token: Access token.
        history: Earlier messages.
        **body_extra: Extra request fields.

    Returns:
        tuple[list, ScriptedLLM, Runtime]: Events, writer, runtime.
    """
    llm = ScriptedLLM(replies)
    runtime = build_runtime(settings, llm)
    client = TestClient(create_app(settings, runtime))
    headers = dict(HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {"language": language,
            "messages": [*history, {"role": "user", "content": question}],
            **body_extra}
    resp = client.post("/api/v1/chat", json=body, headers=headers)
    assert resp.status_code == 200, resp.text
    return wire.parse_sse(resp.text), llm, runtime


def _names(events):
    """Event names in order.

    Args:
        events: Parsed events.

    Returns:
        list[str]: Names.
    """
    return [n for n, _ in events]


# ------------------------------------------------------------ event order


def test_event_order_with_cards(settings):
    """meta, tokens, tool pairs, then cases, gate, done: cards after the
    last token, one terminal event."""
    reply = ScriptedReply(
        text="Forests are where we started. The Dryad work shows how.",
        tool_calls=[("show_case_studies", {"ids": ["dryad"]}),
                    ("show_access_prompt", {})],
    )
    events, _, _ = _run(settings, [reply])
    names = _names(events)
    assert names[0] == "meta" and names.count("meta") == 1
    last_token = max(i for i, n in enumerate(names) if n == "token")
    assert names[last_token + 1:] == [
        "tool_call", "tool_result", "tool_call", "tool_result",
        "cases", "gate", "done"]
    assert events[-3][1] == {"ids": ["dryad"]}
    assert events[-2][1] == {"reason": "subscriber_only"}
    assert sum(n in wire.TERMINAL_EVENTS for n in names) == 1


def test_tool_result_events_report_outcome(settings):
    """tool_call carries the model's arguments; tool_result says ok."""
    reply = ScriptedReply(text="Here.", tool_calls=[
        ("show_case_studies", {"ids": ["dryad", "invented-case"]})])
    events, _, _ = _run(settings, [reply])
    assert ("tool_call", {"name": "show_case_studies",
                          "args": {"ids": ["dryad", "invented-case"]}}) \
        in events
    assert ("tool_result", {"name": "show_case_studies", "ok": True}) \
        in events


# ---------------------------------------------------------------- ids


def test_made_up_case_ids_are_dropped(settings):
    """Only published ids become cards; order kept, at most three."""
    reply = ScriptedReply(text="Some of our work.", tool_calls=[(
        "show_case_studies",
        {"ids": ["made-up-project", "dryad", "https://evil.example/x",
                 "angel", "dryad", "transition-minerals", "bonsucro"]},
    )])
    events, _, _ = _run(settings, [reply])
    cases = [p for n, p in events if n == "cases"]
    assert cases == [{"ids": ["dryad", "angel", "transition-minerals"]}]


def test_all_made_up_ids_send_no_cases_event(settings):
    """When nothing is published, nothing is drawn."""
    reply = ScriptedReply(text="Hmm.", tool_calls=[
        ("show_case_studies", {"ids": ["ghost-one", "ghost-two"]})])
    events, _, _ = _run(settings, [reply])
    assert "cases" not in _names(events)
    assert ("tool_result", {"name": "show_case_studies", "ok": False,
                            "detail": "no published case-study ids"}) \
        in events


@pytest.mark.parametrize("args", [
    '{"ids": "dryad"}', '{"ids": [1, null]}', "not json at all", "{}",
])
def test_malformed_tool_arguments_draw_nothing(settings, args):
    """Broken or wrongly typed arguments never become a card."""
    reply = ScriptedReply(text="Answer.",
                          tool_calls=[("show_case_studies", args)])
    events, _, _ = _run(settings, [reply])
    assert "cases" not in _names(events)
    assert events[-1][0] == "done"


def test_unknown_tool_and_unpublished_download_draw_nothing(settings):
    """An invented tool name or publication id is refused."""
    reply = ScriptedReply(text="Answer.", tool_calls=[
        ("delete_everything", {}),
        ("offer_download", {"publication_id": "no-such-report"})])
    events, _, _ = _run(settings, [reply],
                        token=make_token(plan="access"))
    assert not {"cases", "gate", "download"} & set(_names(events))


def test_subscriber_never_sees_the_paywall(settings):
    """show_access_prompt is not offered to subscribers, and ignored if
    called anyway."""
    reply = ScriptedReply(text="Full answer.",
                          tool_calls=[("show_access_prompt", {})])
    events, llm, _ = _run(settings, [reply],
                          token=make_token(plan="access"))
    assert "gate" not in _names(events)
    offered = [t["function"]["name"] for t in llm.calls[0].tools]
    assert "show_access_prompt" not in offered


def test_non_subscriber_tools(settings):
    """Non-subscribers are offered cases and the paywall, no download."""
    _, llm, _ = _run(settings, [ScriptedReply(text="Hi.")])
    offered = [t["function"]["name"] for t in llm.calls[0].tools]
    assert offered == ["show_case_studies", "show_access_prompt"]


# ------------------------------------------------------------ tool loop


def test_tool_call_before_text_gets_a_follow_up(settings):
    """A writer that calls a tool first is called again to write."""
    replies = [
        ScriptedReply(tool_calls=[("show_case_studies",
                                   {"ids": ["dryad"]})]),
        ScriptedReply(text="The Dryad project is a good start."),
    ]
    events, llm, _ = _run(settings, replies)
    assert len(llm.calls) == 2
    follow = llm.calls[1].messages
    assert follow[-2]["role"] == "assistant"
    assert follow[-2]["tool_calls"][0]["function"]["name"] == \
        "show_case_studies"
    assert follow[-1] == {"role": "tool", "tool_call_id": "call_0",
                          "content": "shown"}
    text = "".join(p["content"] for n, p in events if n == "token")
    assert text == "The Dryad project is a good start."
    assert ("cases", {"ids": ["dryad"]}) in events
    usage = events[-1][1]["usage"]
    assert usage == {"input_tokens": 200, "output_tokens": 40}


def test_text_plus_tools_needs_no_second_call(settings):
    """Display tools after a written answer end the turn."""
    reply = ScriptedReply(text="Done.", tool_calls=[
        ("show_case_studies", {"ids": ["angel"]})])
    _, llm, _ = _run(settings, [reply])
    assert len(llm.calls) == 1


def test_tool_loop_is_bounded(settings):
    """A writer that only ever calls tools stops after three calls."""
    loop = ScriptedReply(tool_calls=[("show_access_prompt", {})])
    events, llm, _ = _run(settings, [loop] * 5)
    assert len(llm.calls) == 3
    assert events[-1][0] == "done"


# ------------------------------------------------------- stop and failure


def test_max_tokens_stop_reason(settings):
    """A length stop is reported as max_tokens."""
    events, _, _ = _run(settings, [ScriptedReply(text="Cut off",
                                                 finish_reason="length")])
    assert events[-1][1]["stop_reason"] == "max_tokens"


def test_no_text_is_abstention(settings):
    """A writer that says nothing ends with abstention."""
    events, _, _ = _run(settings, [ScriptedReply(text="")])
    assert events[-1][1]["stop_reason"] == "abstention"


def test_writer_failure_falls_back_once(settings):
    """French on Mistral fails before text: the fallback answers."""
    replies = [ScriptedReply(error=LLMError("upstream_unavailable", "x")),
               ScriptedReply(text="Bonjour.")]
    events, llm, runtime = _run(settings, replies, language="fr")
    assert [c.model for c in llm.calls] == [
        "mistral/mistral-large-2512", "anthropic/claude-haiku-5-5"]
    assert events[-1][0] == "done"
    row = runtime.deps.usage.rows[-1]
    assert row.fallback_used and row.model == "anthropic/claude-haiku-5-5"


def test_both_writers_fail_gives_error_event(settings):
    """Writer and fallback both down: one error event, no done."""
    down = LLMError("upstream_timeout", "slow")
    events, _, runtime = _run(settings, [ScriptedReply(error=down),
                                         ScriptedReply(error=down)],
                              language="fr")
    assert _names(events) == ["meta", "error"]
    assert events[-1][1]["code"] == "upstream_timeout"
    assert runtime.deps.usage.rows[-1].stop_reason == "upstream_timeout"


def test_failure_after_text_ends_with_error(settings):
    """A mid-stream failure keeps the text sent and ends with error."""
    events, llm, _ = _run(settings, [ScriptedReply(
        text="Half an", fail_after_text=True)])
    assert _names(events)[-1] == "error"
    assert len(llm.calls) == 1, "no fallback once text was shown"


def test_failure_refunds_the_free_question(settings):
    """The signed-out visitor keeps the question a failed turn used."""
    down = LLMError("upstream_unavailable", "x")
    _, _, runtime = _run(settings, [ScriptedReply(error=down),
                                    ScriptedReply(error=down)])
    visitor = next(iter(runtime.visitors.visitors.values()))
    assert visitor.questions_used == 0


# ------------------------------------------------------ routing and prompt


def test_routing_per_locale(settings):
    """French writes on Mistral; the other six on the default."""
    for lang, model in (("fr", "mistral/mistral-large-2512"),
                        ("it", "anthropic/claude-haiku-5-5"),
                        ("ja", "anthropic/claude-haiku-5-5")):
        _, llm, _ = _run(settings, [ScriptedReply(text="x")],
                         language=lang)
        assert llm.calls[0].model == model


def test_anthropic_prompt_is_cached_and_mistral_is_shaped(settings):
    """Claude gets cache_control on the stable block; Mistral gets one
    string with the length layer and no thinking arguments."""
    _, llm, _ = _run(settings, [ScriptedReply(text="x")], language="en")
    system = llm.calls[0].messages[0]["content"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system[1]
    assert "REFERENCE MATERIAL" in system[1]["text"]
    assert "REFERENCE MATERIAL" not in system[0]["text"]
    assert llm.calls[0].kwargs["output_config"] == {"effort": "low"}
    assert "temperature" not in llm.calls[0].kwargs

    _, llm, _ = _run(settings, [ScriptedReply(text="x")], language="fr")
    system = llm.calls[0].messages[0]["content"]
    assert isinstance(system, str)
    assert "Hard limit: one paragraph, at most 4 sentences" in system
    assert "Always reply in French" in system
    assert llm.calls[0].kwargs["max_tokens"] == 400
    assert "output_config" not in llm.calls[0].kwargs


def test_audience_goes_into_the_prompt(settings):
    """The "Asking as" choice reaches the prompt; "Just curious" adds
    nothing."""
    _, llm, _ = _run(settings, [ScriptedReply(text="x")],
                     audience="investor")
    assert "The visitor is asking as: Investor." in llm.calls[0]\
        .prompt_text()
    _, llm, _ = _run(settings, [ScriptedReply(text="x")],
                     audience="just_curious")
    assert "asking as" not in llm.calls[0].prompt_text()


def test_gating_layer_by_plan(settings):
    """Non-subscribers get the gating layer; subscribers the access
    layer."""
    _, llm, _ = _run(settings, [ScriptedReply(text="x")])
    text = llm.calls[0].prompt_text()
    assert "The visitor is NOT a subscriber" in text
    assert "is a TMP Access subscriber" not in text
    _, llm, _ = _run(settings, [ScriptedReply(text="x")],
                     token=make_token(plan="access"))
    text = llm.calls[0].prompt_text()
    assert "The visitor is a TMP Access subscriber" in text
    assert "NOT a subscriber" not in text


def test_retrieval_material_reaches_the_prompt(settings):
    """Facts come from retrieval: a question about Dryad brings its
    text."""
    _, llm, _ = _run(settings, [ScriptedReply(text="x")],
                     question="What was the Dryad project in Cameroon?")
    system = llm.calls[0].messages[0]["content"][1]["text"]
    assert '<reference id="dryad" doc_type="case_study"' in system
    assert "Cameroon" in system


def test_history_is_trimmed(settings):
    """Only the last twelve history messages reach the writer, starting
    on a user turn."""
    history = []
    for k in range(20):
        history += [{"role": "user", "content": f"q{k}"},
                    {"role": "assistant", "content": f"a{k}"}]
    _, llm, _ = _run(settings, [ScriptedReply(text="x")], history=history)
    sent = llm.calls[0].messages[1:]
    assert len(sent) <= 12
    assert sent[0]["role"] == "user" and sent[-1]["content"].startswith(
        "Tell me")


def test_usage_row_has_no_text(settings):
    """The usage log row carries counts, not the conversation."""
    _, _, runtime = _run(settings, [ScriptedReply(text="Secret answer")])
    row = runtime.deps.usage.rows[-1]
    assert row.plan == "anonymous" and row.stop_reason == "end_turn"
    assert row.input_tokens == 100 and row.prompt_cache_enabled
    assert "Secret answer" not in repr(row)


def test_stored_conversation_keeps_cards(settings):
    """The assistant message is stored with the cards it showed."""
    reply = ScriptedReply(text="See Dryad.", tool_calls=[
        ("show_case_studies", {"ids": ["dryad"]})])
    events, _, runtime = _run(settings, [reply])
    cid = events[-1][1]["conversation_id"]
    stored = runtime.deps.conversations.messages[cid]
    assert stored[1]["events"] == {"cases": ["dryad"]}
