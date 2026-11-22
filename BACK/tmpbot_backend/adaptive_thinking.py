# Writer-call settings for adaptive-thinking Claude models.
#
# Lifted from NINOBOT origin/main BACK/ninobot_backend/adaptive_thinking.py
# (ADR-0071 there). Claude Haiku 5.5 thinks before it answers, which
# changes three things about a writer call:
#
# 1. Effort is set per request with output_config={"effort": ...};
#    NINOBOT's 2026-10-15 trial found "low" as accurate as higher levels
#    and much faster, so "low" is the default.
# 2. Thinking counts against max_tokens, so the cap gets headroom.
# 3. Sampling parameters (temperature, top_p, top_k) are rejected with a
#    400, so they are dropped.
#
# It also filters LiteLLM 1.53's synthetic end-of-block tool delta
# (no id, no name, arguments "{}"), which otherwise becomes a phantom
# tool call or corrupts the previous call's arguments.
#
# Which models count as adaptive is matched on a fragment of the model
# id, the one place a model family is named in code; the model choice
# itself stays in BOTS/POLC/llm_routing.yaml.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Per-model keyword arguments for writer calls."""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("tmpbot.adaptive_thinking")

# Model-id fragments of writers that think adaptively and reject
# sampling parameters.
ADAPTIVE_THINKING_MODELS: tuple[str, ...] = ("claude-haiku-5-5",)

DEFAULT_EFFORT = "low"
VALID_EFFORTS: frozenset[str] = frozenset(
    {"off", "low", "medium", "high", "xhigh", "max"}
)

# Tokens added to max_tokens when thinking is on. NINOBOT: 6,000 was
# enough for every low- and medium-effort turn in its trial.
THINKING_HEADROOM_TOKENS = 6_000


def is_adaptive_thinking_model(model: str) -> bool:
    """Return True for writers that take effort and reject sampling.

    Args:
        model: LiteLLM model id, with or without the provider prefix.

    Returns:
        bool: True for adaptive-thinking models.
    """
    return any(tag in model for tag in ADAPTIVE_THINKING_MODELS)


def writer_call_kwargs(
    model: str,
    *,
    max_tokens: int,
    temperature: Optional[float] = None,
    effort: Optional[str] = None,
) -> dict[str, Any]:
    """Build the model-dependent keyword arguments of one writer call.

    Args:
        model: The writer's LiteLLM model id for this call.
        max_tokens: The visible-answer cap (thinking headroom is added).
        temperature: Sampling temperature; None sends none.
        effort: Thinking effort; None means ``DEFAULT_EFFORT``.

    Returns:
        dict[str, Any]: ``max_tokens`` plus either ``temperature`` or
        ``output_config`` / ``thinking``.

    Raises:
        ValueError: If ``effort`` is not one of ``VALID_EFFORTS``.
    """
    if not is_adaptive_thinking_model(model):
        kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    level = effort or DEFAULT_EFFORT
    if level not in VALID_EFFORTS:
        raise ValueError(f"unknown thinking effort {level!r}")
    if temperature is not None:
        log.warning("dropping temperature for an adaptive-thinking model",
                    extra={"model": model, "temperature": temperature})
    if level == "off":
        return {"max_tokens": max_tokens, "thinking": {"type": "disabled"}}
    return {
        "max_tokens": max_tokens + THINKING_HEADROOM_TOKENS,
        "output_config": {"effort": level},
    }


def is_synthetic_tool_delta(
    index: Any, call_id: Any, name: Any, arguments: Any
) -> bool:
    """Return True for LiteLLM 1.53's end-of-block filler tool delta.

    Args:
        index: The delta's ``index``.
        call_id: The delta's ``id``.
        name: The delta's function name.
        arguments: The delta's function arguments.

    Returns:
        bool: True when the delta has no id and no name and its
        arguments are exactly ``"{}"``, or its index is negative.
    """
    if isinstance(index, int) and index < 0:
        return True
    return not call_id and not name and arguments == "{}"
