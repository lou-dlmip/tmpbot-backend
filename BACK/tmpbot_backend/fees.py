# Typical project fees for the chat, in yen, US dollars and euros.
#
# Lou's ranges live in yen in BOTS/POLC/fees.yaml. The bot quotes yen in
# Japanese, euros in French, German and Italian, and US dollars in the
# other languages (BOTS/PRMT/rules.md, "Fees"). Language models convert
# currencies badly, so the backend does it: it fetches the European
# Central Bank reference rate once a day and renders the fee lines that
# replace FEE_RANGES_TOKEN in the rules layer.
#
# The fetch never blocks a chat turn. A turn that finds the rates older
# than a day starts a background refresh and uses the rates it has; the
# fallback rates in fees.yaml cover the time before the first fetch and
# any outage. A rate outside a sane band is rejected and logged.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Fee ranges with daily exchange rates."""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import yaml

log = logging.getLogger("tmpbot.fees")

# Frankfurter serves the ECB reference rates (free, no key). The old
# api.frankfurter.app host redirects here.
RATES_URL = "https://api.frankfurter.dev/v1/latest?from=JPY&to=USD,EUR"

# BACK/tmpbot_backend/fees.py -> parents[2] is the repository root.
DEFAULT_FEES_FILE = (Path(__file__).resolve().parents[2] / "BOTS" / "POLC"
                     / "fees.yaml")

# Placeholder in BOTS/PRMT/rules.md that the rendered fee lines replace.
FEE_RANGES_TOKEN = "[[FEE_RANGES]]"

REFRESH_AFTER_S = 24 * 3600
# After a failed fetch, wait this long before trying again.
RETRY_AFTER_S = 3600
FETCH_TIMEOUT_S = 10.0

# Sane bands for one yen. The yen has traded between about 75 and 162
# to the dollar since 2010; these bands allow far more than that but
# catch an inverted or garbled rate.
USD_PER_JPY_BAND = (0.002, 0.03)
EUR_PER_JPY_BAND = (0.002, 0.03)

KINDS = ("range", "floor_range", "floor")


class FeesError(ValueError):
    """Raised when fees.yaml is missing or malformed."""


@dataclass(frozen=True)
class Rates:
    """Yen exchange rates.

    Attributes:
        date: The ECB reference date, ``YYYY-MM-DD``.
        usd_per_jpy: US dollars for one yen.
        eur_per_jpy: Euros for one yen.
    """

    date: str
    usd_per_jpy: float
    eur_per_jpy: float


@dataclass(frozen=True)
class Fee:
    """One fee line, in yen.

    Attributes:
        label: What the work is.
        kind: ``range``, ``floor_range`` or ``floor``.
        low_jpy: Low end, or the floor.
        high_jpy: High end; None for ``floor``.
        unit: Text after the figures ("a month"), or "".
        note: Text before the figures, for ``floor``.
    """

    label: str
    kind: str
    low_jpy: int
    high_jpy: Optional[int]
    unit: str
    note: str


def _check_rates(usd: float, eur: float) -> None:
    """Reject rates outside the sane bands.

    Args:
        usd: US dollars per yen.
        eur: Euros per yen.

    Raises:
        FeesError: If either rate is out of band.
    """
    if not USD_PER_JPY_BAND[0] <= usd <= USD_PER_JPY_BAND[1]:
        raise FeesError(f"USD per JPY {usd} is outside {USD_PER_JPY_BAND}")
    if not EUR_PER_JPY_BAND[0] <= eur <= EUR_PER_JPY_BAND[1]:
        raise FeesError(f"EUR per JPY {eur} is outside {EUR_PER_JPY_BAND}")


def _parse_fee(raw: dict[str, Any]) -> Fee:
    """Validate one entry of ``fees``.

    Args:
        raw: The YAML mapping.

    Returns:
        Fee: The fee line.

    Raises:
        FeesError: If a field is missing or invalid.
    """
    try:
        label = str(raw["label"]).strip()
        kind = str(raw["kind"])
        low = int(raw["low_jpy"])
        high = raw.get("high_jpy")
        high = None if high is None else int(high)
    except (KeyError, TypeError, ValueError) as exc:
        raise FeesError(f"bad fee entry {raw!r}: {exc}") from exc
    if not label or kind not in KINDS or low <= 0:
        raise FeesError(f"bad fee entry {raw!r}")
    if kind != "floor" and (high is None or high <= low):
        raise FeesError(f"fee {label!r} needs high_jpy above low_jpy")
    return Fee(label=label, kind=kind, low_jpy=low, high_jpy=high,
               unit=str(raw.get("unit") or "").strip(),
               note=str(raw.get("note") or "").strip())


def load_fees(path: Path) -> tuple[list[Fee], Rates]:
    """Read ``fees.yaml``.

    Args:
        path: ``BOTS/POLC/fees.yaml``.

    Returns:
        tuple[list[Fee], Rates]: The fee lines and the fallback rates.

    Raises:
        FeesError: If the file is missing or malformed.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FeesError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or not data.get("fees"):
        raise FeesError(f"{path} has no fees")
    fees = [_parse_fee(entry) for entry in data["fees"]]
    try:
        raw = data["fallback_rates"]
        rates = Rates(date=str(raw["date"]),
                      usd_per_jpy=float(raw["usd_per_jpy"]),
                      eur_per_jpy=float(raw["eur_per_jpy"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise FeesError(f"{path}: bad fallback_rates: {exc}") from exc
    _check_rates(rates.usd_per_jpy, rates.eur_per_jpy)
    return fees, rates


def round_sig(value: float, digits: int = 2) -> int:
    """Round to significant figures.

    Args:
        value: A positive amount.
        digits: Significant figures to keep.

    Returns:
        int: The rounded amount (18910 -> 19000).
    """
    if value <= 0:
        return 0
    step = 10 ** (math.floor(math.log10(value)) - digits + 1)
    return int(round(value / step) * step)


def format_jpy(amount: int) -> str:
    """Yen in millions: 7500000 -> "¥7.5 million".

    Args:
        amount: Yen.

    Returns:
        str: The figure.
    """
    millions = f"{amount / 1_000_000:.2f}".rstrip("0").rstrip(".")
    return f"¥{millions} million"


def _money(symbol: str, amount: float) -> str:
    """A converted figure: 18910.2 -> "$19,000".

    Args:
        symbol: "$" or "€".
        amount: The converted amount.

    Returns:
        str: The rounded figure.
    """
    return f"{symbol}{round_sig(amount):,}"


def _fee_text(fee: Fee, fmt: Callable[[int], str]) -> str:
    """One currency's wording for a fee line.

    Args:
        fee: The fee line.
        fmt: Formats a yen amount in the currency.

    Returns:
        str: For example "about $19,000 to $38,000 a month".
    """
    unit = f" {fee.unit}" if fee.unit else ""
    if fee.kind == "floor" or fee.high_jpy is None:
        return f"{fmt(fee.low_jpy)}{unit}"
    if fee.kind == "floor_range":
        return (f"usually not less than about {fmt(fee.low_jpy)}, "
                f"up to about {fmt(fee.high_jpy)}{unit}")
    return f"about {fmt(fee.low_jpy)} to {fmt(fee.high_jpy)}{unit}"


def render_fee_ranges(fees: list[Fee], rates: Rates) -> str:
    """The fee lines for the rules layer, yen | dollars | euros.

    Args:
        fees: Fee lines in yen.
        rates: Exchange rates.

    Returns:
        str: One line per fee, each starting "- ".
    """
    formats = (
        format_jpy,
        lambda jpy: _money("$", jpy * rates.usd_per_jpy),
        lambda jpy: _money("€", jpy * rates.eur_per_jpy),
    )
    lines = []
    for fee in fees:
        columns = " | ".join(_fee_text(fee, fmt) for fmt in formats)
        prefix = f"{fee.note} " if fee.note else ""
        lines.append(f"- {fee.label}: {prefix}{columns}.")
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def default_ranges_text() -> str:
    """Fee lines at the fallback rates of the default ``fees.yaml``.

    For prompts built without a FeeSchedule (tests, tools).

    Returns:
        str: The rendered fee lines.
    """
    fees, fallback = load_fees(DEFAULT_FEES_FILE)
    return render_fee_ranges(fees, fallback)


def parse_rates(payload: Any) -> Rates:
    """Read a Frankfurter ``latest`` response.

    Args:
        payload: The decoded JSON.

    Returns:
        Rates: The rates.

    Raises:
        FeesError: If the shape or the values are wrong.
    """
    try:
        if payload["base"] != "JPY":
            raise FeesError(f"rate base is {payload['base']}, not JPY")
        usd = float(payload["rates"]["USD"])
        eur = float(payload["rates"]["EUR"])
        date = str(payload["date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FeesError(f"bad rate response: {exc}") from exc
    _check_rates(usd, eur)
    return Rates(date=date, usd_per_jpy=usd, eur_per_jpy=eur)


class FeeSchedule:
    """Fee lines plus exchange rates that refresh once a day.

    Args:
        fees: Fee lines in yen.
        fallback: Rates to use until a fetch succeeds.
        client: An ``httpx.AsyncClient``; tests pass a mock transport.
            None means no fetching at all (tests and local runs that
            should not call out).
        clock: Monotonic clock in seconds; tests pass a fake.
    """

    def __init__(self, fees: list[Fee], fallback: Rates,
                 client: Optional[httpx.AsyncClient] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.fees = fees
        self.rates = fallback
        self._client = client
        self._clock = clock
        # Due at once: the first turn starts a fetch.
        self._next_fetch = clock()
        self._task: Optional[asyncio.Task[None]] = None
        self._text = render_fee_ranges(fees, fallback)

    @classmethod
    def from_file(cls, path: Path, fetch: bool = True) -> "FeeSchedule":
        """Build from ``fees.yaml``.

        Args:
            path: ``BOTS/POLC/fees.yaml``.
            fetch: Whether to fetch daily rates.

        Returns:
            FeeSchedule: The schedule.

        Raises:
            FeesError: If the file is missing or malformed.
        """
        fees, fallback = load_fees(path)
        client = (httpx.AsyncClient(timeout=FETCH_TIMEOUT_S,
                                    follow_redirects=True)
                  if fetch else None)
        return cls(fees, fallback, client=client)

    def ranges_text(self) -> str:
        """The rendered fee lines for the current rates."""
        return self._text

    def maybe_refresh(self) -> None:
        """Start a background fetch if the rates are due.

        Never waits and never raises; call it from a running event loop.
        """
        if self._client is None or self._clock() < self._next_fetch:
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self.refresh())

    async def refresh(self) -> bool:
        """Fetch today's rates now.

        Returns:
            bool: True if new rates were stored.
        """
        if self._client is None:
            return False
        try:
            response = await self._client.get(RATES_URL)
            response.raise_for_status()
            rates = parse_rates(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            # FeesError is a ValueError.
            self._next_fetch = self._clock() + RETRY_AFTER_S
            log.warning("exchange-rate fetch failed; keeping rates of %s: "
                        "%s", self.rates.date, exc)
            return False
        self.rates = rates
        self._text = render_fee_ranges(self.fees, rates)
        self._next_fetch = self._clock() + REFRESH_AFTER_S
        log.info("exchange rates updated",
                 extra={"rate_date": rates.date,
                        "usd_per_jpy": rates.usd_per_jpy,
                        "eur_per_jpy": rates.eur_per_jpy})
        return True
