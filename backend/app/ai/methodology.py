"""Deterministic, curated FootyScout methodology retrieval without embeddings."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from app.ai.content_hygiene import sanitize_user_facing_text
from app.ai.grounding import EvidenceCategory, EvidenceRecord
from app.ai.schemas import (
    MethodologyInput,
    MethodologyTopic,
    ProductionStatus,
    SourceCategory,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.tools.methodology import get_methodology


class KnowledgeTopic(str, Enum):
    XPASS = "xpass"
    XG = "xg"
    POSSESSION_VALUE = "possession_value"
    ATTACKING_IMPACT = "attacking_impact"
    PLAYER_PROFILES = "player_profiles"
    PERCENTILES = "percentiles"
    ARCHETYPES = "archetypes"
    SIMILARITY = "similarity"
    TEAM_INTELLIGENCE = "team_intelligence"
    ROLE_FIT = "role_fit"
    ROLE_RECOMMENDATIONS = "role_recommendations"


class MethodologyDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: KnowledgeTopic
    section: str
    source_id: str
    source_path: str
    content: str
    production_status: ProductionStatus
    limitations: tuple[str, ...] = ()

    @field_validator("section", "content")
    @classmethod
    def sanitize_visible_text(cls, value: str) -> str:
        return sanitize_user_facing_text(value)

    @field_validator("limitations")
    @classmethod
    def sanitize_visible_limitations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sanitize_user_facing_text(item) for item in value)


class MethodologyService(Protocol):
    def get(self, topic: KnowledgeTopic) -> MethodologyDocument:
        """Return one allowlisted methodology document."""


_STATIC_DOCUMENTS: dict[KnowledgeTopic, MethodologyDocument] = {
    KnowledgeTopic.PLAYER_PROFILES: MethodologyDocument(
        topic=KnowledgeTopic.PLAYER_PROFILES,
        section="Unified player intelligence profiles",
        source_id="player_profiles:analytics_readme",
        source_path="backend/analytics/README.md",
        content=(
            "Player profiles combine governed passing, shooting, attacking-impact, and "
            "playing-style outputs. Eligibility and missing metrics are preserved rather "
            "than imputed into ratings."
        ),
        production_status=ProductionStatus.PRODUCTION,
        limitations=("A profile describes observed event data, not future performance.",),
    ),
    KnowledgeTopic.TEAM_INTELLIGENCE: MethodologyDocument(
        topic=KnowledgeTopic.TEAM_INTELLIGENCE,
        section="Team intelligence and positional-role profiles",
        source_id="team_intelligence:analytics_readme",
        source_path="backend/analytics/README.md",
        content=(
            "Team intelligence summarizes observed team event data. Positional-role "
            "profiles pool the observed actions of supported outfield position groups."
        ),
        production_status=ProductionStatus.PRODUCTION,
        limitations=(
            "Team intelligence describes observed event data, not coaching intent.",
            "Current production coverage is limited to qualified teams and outfield roles.",
        ),
    ),
    KnowledgeTopic.ROLE_RECOMMENDATIONS: MethodologyDocument(
        topic=KnowledgeTopic.ROLE_RECOMMENDATIONS,
        section="Scouting recommendations",
        source_id="role_recommendations:analytics_readme",
        source_path="backend/analytics/README.md",
        content=(
            "Scouting recommendations order eligible external same-position players by "
            "their persisted Role Fit distance to a supported observed team role."
        ),
        production_status=ProductionStatus.PRODUCTION,
        limitations=(
            "Recommendations are style matches, not predictions of transfer success.",
            "Lower raw Role Fit distance means closer observed style resemblance.",
        ),
    ),
}


class CuratedMethodologyService:
    """Resolve a small governed corpus by exact topic rather than vector similarity."""

    def get(self, topic: KnowledgeTopic) -> MethodologyDocument:
        if topic in _STATIC_DOCUMENTS:
            return _STATIC_DOCUMENTS[topic]

        response = get_methodology(MethodologyInput(topic=MethodologyTopic(topic.value)))
        primary = response.sources[0]
        return MethodologyDocument(
            topic=topic,
            section=primary.section,
            source_id=primary.source_id,
            source_path=primary.path,
            content=response.summary,
            production_status=response.production_status,
            limitations=tuple(response.limitations),
        )


def methodology_evidence(
    *,
    service: MethodologyService,
    topics: tuple[KnowledgeTopic, ...],
    run_id: str,
    start_order: int,
) -> tuple[EvidenceRecord, ...]:
    records: list[EvidenceRecord] = []
    for offset, topic in enumerate(topics):
        document = service.get(topic)
        records.append(
            EvidenceRecord(
                evidence_id=f"{run_id}:evidence-{start_order + offset}",
                evidence_category=EvidenceCategory.METHODOLOGY,
                tool_name=ToolName.GET_METHODOLOGY,
                producer="curated_methodology_service",
                normalized_arguments={"topic": topic.value},
                execution_status=ToolExecutionStatus.SUCCESS,
                result=document.model_dump(mode="json"),
                source_category=SourceCategory.CURATED_DOCUMENTATION,
                warnings=document.limitations,
                production_status=document.production_status,
                execution_order=start_order + offset,
                methodology_topic=topic.value,
                methodology_sources=(document.source_id,),
                provenance={
                    "source_id": document.source_id,
                    "source_path": document.source_path,
                    "section": document.section,
                    "retrieval": "exact_topic",
                },
            )
        )
    return tuple(records)
