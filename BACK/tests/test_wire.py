# Tests for the SSE encoders in tmpbot_backend.wire, including the three
# TMPBOT card events (cases, gate, download).
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Wire-format encoder tests."""

from __future__ import annotations

import pytest

from tmpbot_backend import wire


def test_encode_event_shape():
    """One event line, one data line, blank line terminator."""
    out = wire.encode_event("token", {"content": "hi"})
    assert out == 'event: token\ndata: {"content":"hi"}\n\n'


def test_encode_event_keeps_non_ascii():
    """Japanese and accents stay readable on the wire."""
    out = wire.encode_event("token", {"content": "気候 é"})
    assert "気候 é" in out


def test_encode_event_newline_stays_on_one_data_line():
    """A newline inside the text is JSON-escaped, so SSE framing holds."""
    out = wire.encode_event("token", {"content": "a\n\nb"})
    assert out.count("\n") == 3
    assert wire.parse_sse(out) == [("token", {"content": "a\n\nb"})]


def test_encode_event_rejects_unknown_name():
    """Only the documented events can be sent."""
    with pytest.raises(ValueError):
        wire.encode_event("cards", {})


def test_new_message_id_format_and_uniqueness():
    """Ids match the documented pattern and do not repeat."""
    ids = {wire.new_message_id() for _ in range(200)}
    assert len(ids) == 200
    assert all(wire.MESSAGE_ID_RE.match(i) for i in ids)


def test_event_meta():
    """meta carries message_id and language."""
    out = wire.event_meta("TMP-0123456789ab", "ja")
    assert wire.parse_sse(out) == [
        ("meta", {"message_id": "TMP-0123456789ab", "language": "ja"})
    ]


def test_event_meta_rejects_bad_id():
    """A malformed message id is a programming error."""
    with pytest.raises(ValueError):
        wire.event_meta("NIN-C5", "en")


def test_event_tool_call_and_result():
    """tool_call echoes args; tool_result adds detail only when given."""
    call = wire.parse_sse(wire.event_tool_call("show_case_studies",
                                               {"ids": ["dryad"]}))
    assert call == [("tool_call", {"name": "show_case_studies",
                                   "args": {"ids": ["dryad"]}})]
    ok = wire.parse_sse(wire.event_tool_result("show_case_studies", True))
    assert ok == [("tool_result", {"name": "show_case_studies", "ok": True})]
    miss = wire.parse_sse(
        wire.event_tool_result("offer_download", False, "not published")
    )
    assert miss[0][1]["detail"] == "not published"


def test_event_cases_keeps_order_and_dedupes():
    """Display order is kept; repeats are dropped."""
    out = wire.event_cases(["dryad", "angel", "dryad"])
    assert wire.parse_sse(out) == [("cases", {"ids": ["dryad", "angel"]})]


def test_event_cases_drops_malformed_ids():
    """Ids that are not slugs never reach the frontend."""
    out = wire.event_cases(["dryad", "../etc", "Bad Id", "", None, 7])
    assert wire.parse_sse(out) == [("cases", {"ids": ["dryad"]})]


def test_event_cases_caps_at_three():
    """The design shows at most three case cards."""
    out = wire.event_cases(["a", "b", "c", "d"])
    assert wire.parse_sse(out)[0][1]["ids"] == ["a", "b", "c"]


def test_event_cases_none_when_nothing_valid():
    """No valid id means no event at all."""
    assert wire.event_cases([]) is None
    assert wire.event_cases(["NOT A SLUG"]) is None


@pytest.mark.parametrize("reason", wire.GATE_REASONS)
def test_event_gate_reasons(reason):
    """Both gate reasons encode."""
    assert wire.parse_sse(wire.event_gate(reason)) == [
        ("gate", {"reason": reason})
    ]


def test_event_gate_rejects_unknown_reason():
    """An unknown gate reason is refused."""
    with pytest.raises(ValueError):
        wire.event_gate("paywall")


def test_event_download():
    """download carries id, title and year, and no file URL."""
    out = wire.event_download("tenure-risk-2025", "Tenure risk", 2025)
    payload = wire.parse_sse(out)[0][1]
    assert payload == {"publication_id": "tenure-risk-2025",
                       "title": "Tenure risk", "year": 2025}
    assert "url" not in out


@pytest.mark.parametrize(
    "args",
    [("Bad Id", "T", 2025), ("ok-id", "  ", 2025), ("ok-id", "T", 1800)],
)
def test_event_download_rejects_bad_input(args):
    """Bad ids, empty titles and impossible years are refused."""
    with pytest.raises(ValueError):
        wire.event_download(*args)


def test_event_done():
    """done carries stop reason and usage."""
    out = wire.event_done("max_tokens", 120, 40)
    assert wire.parse_sse(out) == [
        ("done", {"stop_reason": "max_tokens",
                  "usage": {"input_tokens": 120, "output_tokens": 40}})
    ]


def test_event_done_rejects_unknown_reason():
    """Only documented stop reasons."""
    with pytest.raises(ValueError):
        wire.event_done("finished")


def test_event_error():
    """error carries code and message; unknown codes are refused."""
    out = wire.event_error("rate_limited", "slow down")
    assert wire.parse_sse(out) == [
        ("error", {"code": "rate_limited", "message": "slow down"})
    ]
    with pytest.raises(ValueError):
        wire.event_error("oops", "x")


def test_parse_sse_rejects_malformed():
    """The decoder refuses records it did not write."""
    with pytest.raises(ValueError):
        wire.parse_sse("event: token\n\n")
    with pytest.raises(ValueError):
        wire.parse_sse("event: token\ndata: {not json}\n\n")
