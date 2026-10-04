"""Lazy public facade for AI Scout observability.

Importing a concrete observability submodule must never initialize workflow or
domain modules. Public compatibility exports are resolved only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AIScoutRunTrace": ("app.ai.observability.models", "AIScoutRunTrace"),
    "AIScoutTraceSummary": ("app.ai.observability.aggregation", "AIScoutTraceSummary"),
    "DEFAULT_PRICING_REGISTRY_PATH": (
        "app.ai.observability.pricing",
        "DEFAULT_PRICING_REGISTRY_PATH",
    ),
    "EMPTY_PRICING_REGISTRY": (
        "app.ai.observability.pricing",
        "EMPTY_PRICING_REGISTRY",
    ),
    "EntityResolutionTrace": (
        "app.ai.observability.models",
        "EntityResolutionTrace",
    ),
    "ErrorTrace": ("app.ai.observability.models", "ErrorTrace"),
    "InMemoryTraceSink": ("app.ai.observability.sinks", "InMemoryTraceSink"),
    "JsonlTraceSink": ("app.ai.observability.sinks", "JsonlTraceSink"),
    "ModelPricing": ("app.ai.observability.pricing", "ModelPricing"),
    "NullTraceSink": ("app.ai.observability.sinks", "NullTraceSink"),
    "PlannedToolTrace": ("app.ai.observability.models", "PlannedToolTrace"),
    "PricingRegistry": ("app.ai.observability.pricing", "PricingRegistry"),
    "TRACE_SCHEMA_VERSION": ("app.ai.observability.models", "TRACE_SCHEMA_VERSION"),
    "ToolExecutionTrace": ("app.ai.observability.models", "ToolExecutionTrace"),
    "TraceSink": ("app.ai.observability.sinks", "TraceSink"),
    "ValidationFindingTrace": (
        "app.ai.observability.models",
        "ValidationFindingTrace",
    ),
    "build_run_trace": ("app.ai.observability.tracing", "build_run_trace"),
    "estimate_cost_usd": ("app.ai.observability.pricing", "estimate_cost_usd"),
    "load_jsonl_traces": ("app.ai.observability.aggregation", "load_jsonl_traces"),
    "load_default_pricing_registry": (
        "app.ai.observability.pricing",
        "load_default_pricing_registry",
    ),
    "load_pricing_registry": (
        "app.ai.observability.pricing",
        "load_pricing_registry",
    ),
    "record_trace_safely": (
        "app.ai.observability.sinks",
        "record_trace_safely",
    ),
    "summarize_traces": ("app.ai.observability.aggregation", "summarize_traces"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve compatibility exports without eager cross-layer imports."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
