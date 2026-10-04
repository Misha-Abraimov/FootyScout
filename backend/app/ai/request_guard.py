"""Deterministic guards for structurally unambiguous planner requests."""

from __future__ import annotations

import re

from app.ai.provider_schemas import LLMMethodologyTopic, LLMPlannerDecision
from app.ai.schemas import IntentKind, PlannerDecision

_SIMILARITY_REQUEST = re.compile(
    r"\b(?:similarity|similar\s+players?|players?\s+similar\s+to)\b",
    re.IGNORECASE,
)
_CROSS_POSITION_SCOPE = re.compile(
    r"\b(?:regardless\s+of\s+(?:the\s+)?positions?|"
    r"across\s+(?:all\s+)?positions?|cross[- ]position|"
    r"any\s+position|without\s+regard\s+to\s+(?:the\s+)?positions?)\b",
    re.IGNORECASE,
)
_FABRICATION_REQUEST = re.compile(
    r"\b(?:invent|fabricate|impute|fill\s+in)\b",
    re.IGNORECASE,
)
_PERCENTILE_CONTEXT = re.compile(r"\bpercentiles?\b", re.IGNORECASE)


def is_percentile_fabrication_request(question: str) -> bool:
    """Return whether a request asks to invent governed percentile values."""
    return bool(
        _FABRICATION_REQUEST.search(question)
        and _PERCENTILE_CONTEXT.search(question)
    )


def deterministic_request_decision(question: str) -> LLMPlannerDecision | None:
    """Return a governed decision only for narrow, high-confidence structures.

    These guards run before a provider request so known capability boundaries and
    safe methodology refusals do not depend on provider schema conformance.
    """
    if _SIMILARITY_REQUEST.search(question) and _CROSS_POSITION_SCOPE.search(question):
        return LLMPlannerDecision(
            decision=PlannerDecision.UNSUPPORTED,
            intent=IntentKind.SIMILAR_PLAYERS,
            unsupported_reason=(
                "Production playing-style similarity is available only within "
                "the same position group."
            ),
        )
    if is_percentile_fabrication_request(question):
        return LLMPlannerDecision(
            decision=PlannerDecision.READY,
            intent=IntentKind.METHODOLOGY,
            methodology_topic=LLMMethodologyTopic.PERCENTILES,
        )
    return None
