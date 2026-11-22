# Writer routing: which model writes the answer, per locale.
#
# Reads BOTS/POLC/llm_routing.yaml (ADR-0002). Adapted from NINOBOT's
# model_routing.py (origin/main) with the multi-bot overrides removed:
# TMPBOT has one bot, so resolution is locale -> "*" default. Model ids
# never appear in code; the YAML names them, and the per-model prices
# for the usage log sit beside them.
#
# Author: Lou Munden
# Copyright 2026 合同会社 TMP Climate and Lou Munden

"""Load and query the writer routing policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .locales import LOCALES

log = logging.getLogger("tmpbot.routing")


class RoutingError(ValueError):
    """Raised when the routing policy file is missing or malformed."""


@dataclass(frozen=True)
class Price:
    """List price of a model in USD per million tokens.

    Attributes:
        input: Fresh input tokens.
        output: Output tokens.
    """

    input: float
    output: float


@dataclass(frozen=True)
class RoutingPolicy:
    """The parsed writer routing policy.

    Attributes:
        locale_models: Locale code (or ``"*"``) to LiteLLM model id.
        fallback_model: Model used once when the writer fails before
            any text.
        temperature: Sampling temperature, or None for the provider
            default.
        thinking_effort: Effort for adaptive-thinking writers.
        max_tokens: Visible-answer token cap.
        timeout_s: Seconds before a writer call is abandoned.
        history_messages: Conversation messages sent to the writer.
        mistral_length_shaping: Whether Mistral writers get the length
            block.
        prices: Model id to list price.
    """

    locale_models: Mapping[str, str]
    fallback_model: str
    temperature: Optional[float] = None
    thinking_effort: Optional[str] = None
    max_tokens: int = 400
    timeout_s: float = 60.0
    history_messages: int = 12
    mistral_length_shaping: bool = True
    prices: Mapping[str, Price] = field(default_factory=dict)

    def model_for(self, locale: str) -> str:
        """Return the writer model for a locale.

        Args:
            locale: One of the seven locale codes.

        Returns:
            str: The LiteLLM model id.

        Raises:
            RoutingError: If neither the locale nor ``"*"`` is mapped.
        """
        model = self.locale_models.get(locale) or self.locale_models.get(
            "*"
        )
        if not model:
            raise RoutingError(f"no writer model for locale {locale!r}")
        return model

    def price_for(self, model: str) -> Optional[Price]:
        """Return the list price of a model, accepting a bare id.

        Args:
            model: LiteLLM model id, with or without the provider prefix.

        Returns:
            Optional[Price]: The price, or None when the table lacks it.
        """
        if model in self.prices:
            return self.prices[model]
        for key, price in self.prices.items():
            if key.split("/", 1)[-1] == model.split("/", 1)[-1]:
                return price
        return None


def _number(raw: Any, name: str) -> float:
    """Coerce a YAML value to a non-negative number.

    Args:
        raw: The value.
        name: Where it came from, for the error message.

    Returns:
        float: The number.

    Raises:
        RoutingError: If the value is not a non-negative number.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RoutingError(f"{name} must be a number; got {raw!r}")
    if raw < 0:
        raise RoutingError(f"{name} must not be negative")
    return float(raw)


def parse_policy(data: Mapping[str, Any]) -> RoutingPolicy:
    """Build a policy from the YAML document.

    Args:
        data: The parsed YAML.

    Returns:
        RoutingPolicy: The policy.

    Raises:
        RoutingError: If required keys are missing or a value is invalid.
    """
    defaults = data.get("defaults")
    if not isinstance(defaults, Mapping):
        raise RoutingError("llm_routing.yaml needs a 'defaults' mapping")
    locales = defaults.get("locales")
    if not isinstance(locales, Mapping) or "*" not in locales:
        raise RoutingError("defaults.locales needs a '*' entry")
    unknown = set(locales) - set(LOCALES) - {"*"}
    if unknown:
        raise RoutingError(f"unknown locales in routing: {sorted(unknown)}")
    fallback = defaults.get("fallback") or locales["*"]

    temperature = defaults.get("temperature")
    if temperature is not None:
        temperature = _number(temperature, "temperature")

    prices: dict[str, Price] = {}
    for model, row in (data.get("prices") or {}).items():
        if not isinstance(row, Mapping):
            raise RoutingError(f"price for {model} must be a mapping")
        prices[str(model)] = Price(
            input=_number(row.get("input"), f"{model}.input"),
            output=_number(row.get("output"), f"{model}.output"),
        )

    policy = RoutingPolicy(
        locale_models={str(k): str(v) for k, v in locales.items()},
        fallback_model=str(fallback),
        temperature=temperature,
        thinking_effort=defaults.get("thinking_effort"),
        max_tokens=int(_number(defaults.get("max_tokens", 400),
                               "max_tokens")),
        timeout_s=_number(defaults.get("timeout_s", 60), "timeout_s"),
        history_messages=int(_number(defaults.get("history_messages", 12),
                                     "history_messages")),
        mistral_length_shaping=bool(
            defaults.get("mistral_length_shaping", True)
        ),
        prices=prices,
    )
    if policy.max_tokens < 1 or policy.history_messages < 1:
        raise RoutingError("max_tokens and history_messages must be >= 1")
    for model in set(policy.locale_models.values()) | {fallback}:
        if policy.price_for(model) is None:
            log.warning("no price for routed model; usage cost logs $0",
                        extra={"model": model})
    return policy


def load_policy(path: Path) -> RoutingPolicy:
    """Read the routing policy file.

    Args:
        path: Path to ``llm_routing.yaml``.

    Returns:
        RoutingPolicy: The policy.

    Raises:
        RoutingError: If the file cannot be read or parsed.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RoutingError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise RoutingError(f"{path} is not a YAML mapping")
    return parse_policy(data)
