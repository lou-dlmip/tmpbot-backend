# Fee ranges with daily exchange rates (tmpbot_backend.fees): rounding,
# rendering in yen | dollars | euros, the rate response check, the daily
# refresh with its fallback, and the chat turn passing the lines on.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Fee ranges and exchange rates."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from tmpbot_backend import fees as fees_mod
from tmpbot_backend.fees import (
    DEFAULT_FEES_FILE,
    Fee,
    FeeSchedule,
    FeesError,
    Rates,
    format_jpy,
    load_fees,
    parse_rates,
    render_fee_ranges,
    round_sig,
)

# The ECB rates of 2026-11-20, as in fees.yaml.
RATES = Rates(date="2026-11-20", usd_per_jpy=0.0063034,
              eur_per_jpy=0.0057356)


def _run(coro):
    return asyncio.run(coro)


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(usd: float = 0.0070, eur: float = 0.0060,
        date: str = "2026-11-23"):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.frankfurter.dev"
        return httpx.Response(200, json={
            "amount": 1.0, "base": "JPY", "date": date,
            "rates": {"USD": usd, "EUR": eur}})
    return handler


@pytest.mark.parametrize("value, expected", [
    (18910.2, 19000), (37820.4, 38000), (47275.5, 47000),
    (126068.0, 130000), (9455.1, 9500), (8603.4, 8600),
    (344136.0, 340000), (0, 0),
])
def test_round_sig(value, expected):
    assert round_sig(value) == expected


@pytest.mark.parametrize("amount, expected", [
    (3_000_000, "¥3 million"), (7_500_000, "¥7.5 million"),
    (1_500_000, "¥1.5 million"), (20_000_000, "¥20 million"),
])
def test_format_jpy(amount, expected):
    assert format_jpy(amount) == expected


def test_real_fees_file_renders_all_three_currencies():
    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    assert len(fees) == 5
    assert fallback == RATES
    text = render_fee_ranges(fees, fallback)
    lines = text.splitlines()
    assert len(lines) == 5
    assert lines[0] == (
        "- Risk briefing or screening: about ¥3 million to ¥6 million | "
        "about $19,000 to $38,000 | about €17,000 to €34,000.")
    assert lines[1] == (
        "- Bespoke analysis project: usually not less than about "
        "¥7.5 million, up to about ¥20 million | usually not less than "
        "about $47,000, up to about $130,000 | usually not less than "
        "about €43,000, up to about €110,000.")
    assert "about ¥30 million to ¥60 million a year" in lines[2]
    assert "about $190,000 to $380,000 a year" in lines[2]
    assert "about €8,600 to €17,000 a month" in lines[3]
    assert lines[4].endswith(
        "such projects have never cost less than ¥7.5 million | "
        "$47,000 | €43,000.")


def test_new_rates_change_the_figures():
    fees, _ = load_fees(DEFAULT_FEES_FILE)
    text = render_fee_ranges(fees, Rates("2026-12-01", 0.0070, 0.0060))
    assert "about $21,000 to $42,000" in text
    assert "about €18,000 to €36,000" in text
    # Yen never moves.
    assert "about ¥3 million to ¥6 million" in text


def test_bad_fees_file_is_refused(tmp_path: Path):
    path = tmp_path / "fees.yaml"
    path.write_text("fallback_rates: {date: x, usd_per_jpy: 0.0063, "
                    "eur_per_jpy: 0.0057}\nfees:\n  - label: A\n"
                    "    kind: range\n    low_jpy: 5\n    high_jpy: 3\n",
                    encoding="utf-8")
    with pytest.raises(FeesError):
        load_fees(path)
    path.write_text("fallback_rates: {date: x, usd_per_jpy: 158.6, "
                    "eur_per_jpy: 0.0057}\nfees:\n  - label: A\n"
                    "    kind: floor\n    low_jpy: 5\n", encoding="utf-8")
    with pytest.raises(FeesError, match="USD per JPY"):
        load_fees(path)


@pytest.mark.parametrize("payload", [
    {"base": "USD", "date": "d", "rates": {"USD": 0.006, "EUR": 0.005}},
    {"base": "JPY", "date": "d", "rates": {"USD": 0.006}},
    # Inverted: yen per dollar instead of dollars per yen.
    {"base": "JPY", "date": "d", "rates": {"USD": 158.6, "EUR": 0.005}},
    {"base": "JPY", "date": "d", "rates": {"USD": "x", "EUR": 0.005}},
    [],
])
def test_parse_rates_rejects_bad_responses(payload):
    with pytest.raises(FeesError):
        parse_rates(payload)


def test_refresh_stores_new_rates_and_rerenders():
    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    schedule = FeeSchedule(fees, fallback, client=_client(_ok()),
                           clock=FakeClock())
    before = schedule.ranges_text()
    assert "$19,000" in before
    assert _run(schedule.refresh()) is True
    assert schedule.rates.date == "2026-11-23"
    assert "about $21,000 to $42,000" in schedule.ranges_text()


def test_failed_refresh_keeps_the_old_rates():
    def down(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    clock = FakeClock()
    schedule = FeeSchedule(fees, fallback, client=_client(down),
                           clock=clock)
    assert _run(schedule.refresh()) is False
    assert schedule.rates == fallback
    assert "$19,000" in schedule.ranges_text()


def test_out_of_band_rate_is_ignored():
    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    schedule = FeeSchedule(fees, fallback,
                           client=_client(_ok(usd=158.6)),
                           clock=FakeClock())
    assert _run(schedule.refresh()) is False
    assert schedule.rates == fallback


def test_maybe_refresh_fetches_once_a_day():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _ok()(request)

    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    clock = FakeClock()
    schedule = FeeSchedule(fees, fallback, client=_client(handler),
                           clock=clock)

    async def turns(n: int) -> None:
        for _ in range(n):
            schedule.maybe_refresh()
            await asyncio.sleep(0)
            if schedule._task is not None:
                await schedule._task

    _run(turns(3))
    assert len(calls) == 1
    clock.now += fees_mod.REFRESH_AFTER_S - 1
    _run(turns(2))
    assert len(calls) == 1
    clock.now += 2
    _run(turns(2))
    assert len(calls) == 2


def test_failed_fetch_retries_after_an_hour():
    calls = []

    def down(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    clock = FakeClock()
    schedule = FeeSchedule(fees, fallback, client=_client(down),
                           clock=clock)

    async def turn() -> None:
        schedule.maybe_refresh()
        if schedule._task is not None:
            await schedule._task

    _run(turn())
    _run(turn())
    assert len(calls) == 1
    clock.now += fees_mod.RETRY_AFTER_S + 1
    _run(turn())
    assert len(calls) == 2


def test_no_client_never_fetches():
    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    schedule = FeeSchedule(fees, fallback, client=None, clock=FakeClock())

    async def turn() -> None:
        schedule.maybe_refresh()

    _run(turn())
    assert schedule._task is None
    assert _run(schedule.refresh()) is False


def test_floor_without_unit_renders_only_the_figure():
    fee = Fee(label="X", kind="floor", low_jpy=1_000_000, high_jpy=None,
              unit="", note="never less than")
    text = render_fee_ranges([fee], RATES)
    assert text == "- X: never less than ¥1 million | $6,300 | €5,700."
