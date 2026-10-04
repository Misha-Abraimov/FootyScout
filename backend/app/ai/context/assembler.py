"""Rule-based methodology selection and context sufficiency evaluation."""

from __future__ import annotations

from app.ai.content_hygiene import sanitize_user_facing_text
from app.ai.context.schemas import ContextStatus, SelectedContext
from app.ai.grounding import EvidenceCategory, EvidenceRecord
from app.ai.methodology import KnowledgeTopic
from app.ai.schemas import (
    IntentKind,
    MethodologyIntent,
    MethodologyTopic,
    PlayerDossierSection,
    PlayerProfileIntent,
    ScoutIntent,
    ToolExecutionStatus,
    ToolName,
)

_INTENT_TOPICS: dict[IntentKind, tuple[KnowledgeTopic, ...]] = {
    IntentKind.PLAYER_SEARCH: (),
    IntentKind.PLAYER_PROFILE: (KnowledgeTopic.PLAYER_PROFILES,),
    IntentKind.PLAYER_COMPARISON: (
        KnowledgeTopic.PLAYER_PROFILES,
        KnowledgeTopic.PERCENTILES,
    ),
    IntentKind.SIMILAR_PLAYERS: (KnowledgeTopic.SIMILARITY,),
    IntentKind.LEADERBOARD: (KnowledgeTopic.PERCENTILES,),
    IntentKind.TEAM_ANALYSIS: (KnowledgeTopic.TEAM_INTELLIGENCE,),
    IntentKind.ROLE_FIT: (KnowledgeTopic.ROLE_FIT,),
    IntentKind.ROLE_RECOMMENDATIONS: (
        KnowledgeTopic.ROLE_FIT,
        KnowledgeTopic.ROLE_RECOMMENDATIONS,
    ),
    IntentKind.METHODOLOGY: (),
}

_METHODOLOGY_TOPIC_MAP = {
    topic.value: KnowledgeTopic(topic.value) for topic in MethodologyTopic
}


def select_methodology_topics(intent: ScoutIntent) -> tuple[KnowledgeTopic, ...]:
    """Choose only the methodology required by the validated intent."""
    topics = list(_INTENT_TOPICS[intent.kind])
    if isinstance(intent, MethodologyIntent):
        topics.append(_METHODOLOGY_TOPIC_MAP[intent.topic.value])
    elif isinstance(intent, PlayerProfileIntent):
        section_topics = {
            PlayerDossierSection.PASSING: KnowledgeTopic.XPASS,
            PlayerDossierSection.SHOOTING: KnowledgeTopic.XG,
            PlayerDossierSection.ATTACKING_IMPACT: KnowledgeTopic.ATTACKING_IMPACT,
            PlayerDossierSection.INTELLIGENCE: KnowledgeTopic.PLAYER_PROFILES,
        }
        topics.extend(section_topics[section] for section in intent.sections)
    return tuple(dict.fromkeys(topics))


def assemble_context(
    *,
    question: str,
    intent: ScoutIntent,
    records: tuple[EvidenceRecord, ...],
    limitations: tuple[str, ...] = (),
) -> SelectedContext:
    analytics = tuple(
        record
        for record in records
        if record.evidence_category is EvidenceCategory.ANALYTICS
        and record.execution_status is ToolExecutionStatus.SUCCESS
    )
    methodology = tuple(
        record
        for record in records
        if record.evidence_category is EvidenceCategory.METHODOLOGY
        and record.execution_status is ToolExecutionStatus.SUCCESS
    )
    web = tuple(
        record
        for record in records
        if record.evidence_category is EvidenceCategory.WEB
        and record.execution_status is ToolExecutionStatus.SUCCESS
    )
    warnings = tuple(
        dict.fromkeys(
            sanitized
            for warning in [
                *limitations,
                *(
                    warning
                    for record in (*analytics, *methodology, *web)
                    for warning in record.warnings
                ),
            ]
            if (sanitized := sanitize_user_facing_text(warning))
        )
    )
    context = SelectedContext(
        question=question,
        intent=intent.kind.value,
        analytics_evidence=analytics,
        methodology_evidence=methodology,
        web_evidence=web,
        limitations=warnings,
    )
    return context.model_copy(update={"status": evaluate_context_sufficiency(context)})


def evaluate_context_sufficiency(context: SelectedContext) -> ContextStatus:
    if context.analytics_evidence or context.methodology_evidence or context.web_evidence:
        return ContextStatus.SUFFICIENT
    return ContextStatus.INSUFFICIENT_EVIDENCE


def methodology_topics_already_present(
    records: tuple[EvidenceRecord, ...],
) -> frozenset[str]:
    return frozenset(
        record.methodology_topic
        for record in records
        if record.tool_name is ToolName.GET_METHODOLOGY and record.methodology_topic is not None
    )
