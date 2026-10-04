"""Narrow deterministic correction for explicit pure-methodology questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ai.methodology import KnowledgeTopic
from app.ai.provider_schemas import LLMMethodologyTopic, LLMPlannerDecision
from app.ai.schemas import IntentKind, PlannerDecision

_TOPIC_ALIASES: dict[KnowledgeTopic, tuple[str, ...]] = {
    KnowledgeTopic.XPASS: ("xpass", "expected pass completion", "pass difficulty"),
    KnowledgeTopic.XG: ("xg", "expected goals"),
    KnowledgeTopic.POSSESSION_VALUE: (
        "possession value",
        "possession state",
        "possession state model",
    ),
    KnowledgeTopic.ATTACKING_IMPACT: ("attacking impact",),
    KnowledgeTopic.PLAYER_PROFILES: ("player profile", "player profiles"),
    KnowledgeTopic.PERCENTILES: ("percentile", "percentiles"),
    KnowledgeTopic.ARCHETYPES: ("archetype", "archetypes", "player archetypes"),
    KnowledgeTopic.SIMILARITY: (
        "player similarity",
        "style similarity",
        "similarity score",
    ),
    KnowledgeTopic.TEAM_INTELLIGENCE: ("team intelligence",),
    KnowledgeTopic.ROLE_FIT: ("role fit",),
    KnowledgeTopic.ROLE_RECOMMENDATIONS: (
        "scouting recommendation",
        "scouting recommendations",
        "role recommendation",
        "role recommendations",
    ),
}

_QUESTION_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("what does ", " mean and what are its limitations"),
    ("what does ", " mean"),
    ("what are the limitations of ", ""),
    ("what is ", ""),
    ("explain ", ""),
    ("how does ", " work"),
    ("how is ", " calculated"),
    ("how is ", " interpreted"),
    ("how are ", " interpreted"),
    ("how should ", " be interpreted"),
)
_ARCHETYPE_QUALITY_HIERARCHY = re.compile(
    r"\b(?:which|what)\s+archetype\b[^?.!]{0,80}"
    r"\b(?:objectively\s+)?(?:best|worst)\b|"
    r"\brank\s+(?:the\s+)?archetypes?\b[^?.!]{0,80}"
    r"\b(?:best|worst)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MethodologyGuardResult:
    decision: LLMPlannerDecision
    topic: KnowledgeTopic | None
    applied: bool


def apply_pure_methodology_guard(
    question: str,
    decision: LLMPlannerDecision,
) -> MethodologyGuardResult:
    """Correct provider routing only when the whole question is a known definition form."""
    topic = detect_pure_methodology_topic(question)
    if topic is None:
        return MethodologyGuardResult(decision=decision, topic=None, applied=False)

    corrected = LLMPlannerDecision(
        decision=PlannerDecision.READY,
        intent=IntentKind.METHODOLOGY,
        methodology_topic=LLMMethodologyTopic(topic.value),
    )
    return MethodologyGuardResult(
        decision=corrected,
        topic=topic,
        applied=corrected != decision,
    )


def detect_pure_methodology_topic(question: str) -> KnowledgeTopic | None:
    """Return a topic only for a complete, bounded methodology-question template."""
    normalized = _normalize(question)
    if is_archetype_quality_hierarchy_request(normalized):
        return KnowledgeTopic.ARCHETYPES
    if _is_possession_value_model_comparison(normalized):
        return KnowledgeTopic.POSSESSION_VALUE
    for prefix, suffix in _QUESTION_TEMPLATES:
        if not normalized.startswith(prefix):
            continue
        if suffix and not normalized.endswith(suffix):
            continue
        end = len(normalized) - len(suffix) if suffix else len(normalized)
        subject = normalized[len(prefix) : end].strip()
        matches = [topic for topic, aliases in _TOPIC_ALIASES.items() if subject in aliases]
        if len(matches) == 1:
            return matches[0]
    return None


def is_archetype_quality_hierarchy_request(question: str) -> bool:
    """Return whether a request asks for an unsupported archetype quality ranking."""
    return bool(_ARCHETYPE_QUALITY_HIERARCHY.search(question))


def _is_possession_value_model_comparison(normalized: str) -> bool:
    """Recognize the governed production-vs-experimental possession-model contrast."""
    if re.search(r"\bxg\b|\bexpected goals?\b", normalized):
        return False
    return bool(
        "xgboost" in normalized
        and "transformer" in normalized
        and re.search(r"\b(?:production|model|used|chosen|selected)\b", normalized)
    )


def _normalize(value: str) -> str:
    value = value.casefold().replace("-", " ")
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())
