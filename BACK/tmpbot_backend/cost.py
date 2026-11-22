# Per-turn LLM cost for the usage log.
#
# Adapted from NINOBOT origin/main BACK/ninobot_backend/cost.py. The price
# table moved out of code into BOTS/POLC/llm_routing.yaml (`prices:`), so
# a model change and its price change together. Cache tokens are carried
# separately because LiteLLM's streaming path leaves them out of
# prompt_tokens (NINOBOT M17.31); folding them in here keeps the cost
# from being under-reported.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Compute the USD cost of a chat turn."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .routing import RoutingPolicy

log = logging.getLogger("tmpbot.cost")

# Anthropic prompt-cache multipliers against the input rate (5-minute
# TTL). Zero cache tokens on other providers, so they are unaffected.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


@dataclass(frozen=True)
class TurnUsage:
    """Token counts for one turn, summed over its writer calls.

    Attributes:
        input_tokens: Fresh input tokens.
        output_tokens: Output tokens (thinking included).
        cache_read_tokens: Prompt-cache reads.
        cache_write_tokens: Prompt-cache writes.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def plus(self, other: "TurnUsage") -> "TurnUsage":
        """Return the sum of two usages.

        Args:
            other: Usage to add.

        Returns:
            TurnUsage: The sum.
        """
        return TurnUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


def turn_cost_usd(policy: RoutingPolicy, model: str,
                  usage: TurnUsage) -> float:
    """Price a turn in USD, rounded to six decimal places.

    Args:
        policy: Routing policy holding the price table.
        model: LiteLLM model id the turn ran on.
        usage: The turn's token counts.

    Returns:
        float: Cost in USD; 0.0 (with a warning) for an unpriced model.
    """
    price = policy.price_for(model)
    if price is None:
        log.warning("no price for model; logging cost 0",
                    extra={"model": model})
        return 0.0
    per = 1_000_000
    cost = (
        max(usage.input_tokens, 0) * price.input / per
        + max(usage.output_tokens, 0) * price.output / per
        + max(usage.cache_write_tokens, 0) * price.input
        * CACHE_WRITE_MULTIPLIER / per
        + max(usage.cache_read_tokens, 0) * price.input
        * CACHE_READ_MULTIPLIER / per
    )
    return round(cost, 6)
