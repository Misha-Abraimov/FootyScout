"""Explicit, versioned token pricing with no built-in guessed prices."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PRICING_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "runtime_metadata" / "ai_pricing.json"
)


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    effective_date: date | None = None
    input_usd_per_million_tokens: float = Field(ge=0)
    cached_input_usd_per_million_tokens: float | None = Field(default=None, ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    units: Literal["usd_per_million_tokens"] = "usd_per_million_tokens"
    source: str | None = None


class PricingRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str | None = None
    entries: tuple[ModelPricing, ...] = ()

    def find(self, provider: str, model: str) -> ModelPricing | None:
        provider_key = provider.strip().casefold()
        model_key = model.strip().casefold()
        return next(
            (
                entry
                for entry in self.entries
                if entry.provider.strip().casefold() == provider_key
                and entry.model.strip().casefold() == model_key
            ),
            None,
        )


EMPTY_PRICING_REGISTRY = PricingRegistry()


def load_pricing_registry(path: str | Path) -> PricingRegistry:
    """Load one explicit pricing snapshot; malformed files fail at startup/configuration."""
    return PricingRegistry.model_validate_json(
        Path(path).resolve().read_text(encoding="utf-8")
    )


def load_default_pricing_registry() -> PricingRegistry:
    """Load the packaged pricing snapshot independently of process working directory."""
    return load_pricing_registry(DEFAULT_PRICING_REGISTRY_PATH)


def estimate_cost_usd(
    *,
    input_tokens: int | None,
    cached_input_tokens: int | None = None,
    output_tokens: int | None,
    pricing: ModelPricing | None,
) -> float | None:
    if input_tokens is None or output_tokens is None or pricing is None:
        return None
    cached_tokens = cached_input_tokens or 0
    if cached_tokens < 0 or cached_tokens > input_tokens:
        return None
    cached_price = (
        pricing.cached_input_usd_per_million_tokens
        if pricing.cached_input_usd_per_million_tokens is not None
        else pricing.input_usd_per_million_tokens
    )
    estimate = (
        (input_tokens - cached_tokens) * pricing.input_usd_per_million_tokens
        + cached_tokens * cached_price
        + output_tokens * pricing.output_usd_per_million_tokens
    ) / 1_000_000
    return round(estimate, 12)
