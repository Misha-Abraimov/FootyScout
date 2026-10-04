"""Provider-compatible DTOs converted into FootyScout's strict internal ScoutPlan."""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Literal

from pydantic import Field, model_validator

from app.ai.policy import MAX_TOOL_CALLS
from app.ai.schemas import (
    AiModel,
    EntityRef,
    IntentKind,
    LeaderboardIntent,
    MethodologyIntent,
    MethodologyTopic,
    PlannedLeaderboardCall,
    PlannedMethodologyCall,
    PlannedPlayerDossierCall,
    PlannedRoleFitCall,
    PlannedRoleRecommendationsCall,
    PlannedSearchPlayersCall,
    PlannedSimilarPlayersCall,
    PlannedTeamIntelligenceCall,
    PlannerDecision,
    PlayerDossierSection,
    PlayerProfileIntent,
    PlayerSearchIntent,
    RoleFitIntent,
    RoleRecommendationsIntent,
    ScoutPlan,
    SimilarPlayersIntent,
    TeamAnalysisIntent,
    ToolName,
)
from app.schemas import LeaderboardMetric, PlayerSortField, PositionGroup, SortOrder

logger = logging.getLogger(__name__)

_SIMILARITY_ADVISORY_NEGATIONS = (
    "cannot guarantee",
    "can't guarantee",
    "does not guarantee",
    "doesn't guarantee",
    "do not guarantee",
    "not guarantee",
    "no guarantee",
    "never guarantee",
)
_SIMILARITY_ADVISORY_SUBJECT = re.compile(
    r"\b(?:similar(?:ity| player)?|style|quality|performance|future|as good)\b",
    re.IGNORECASE,
)


def _is_similarity_non_guarantee_advisory(value: object) -> bool:
    """Recognize only an explicit similarity non-guarantee safety caveat."""
    if not isinstance(value, str):
        return False
    text = " ".join(value.split())
    if not text or len(text) > 500:
        return False
    lowered = text.casefold().replace("’", "'")
    return (
        any(negation in lowered for negation in _SIMILARITY_ADVISORY_NEGATIONS)
        and _SIMILARITY_ADVISORY_SUBJECT.search(lowered) is not None
    )


def normalize_ready_similarity_advisory(value: Any) -> Any:
    """Move a misplaced READY similarity safety advisory into ``limitation``.

    This intentionally accepts only an otherwise answerable similarity request
    with a player reference. Genuine unsupported/clarification decisions and
    arbitrary explanations remain subject to the strict provider schema.
    """
    if not isinstance(value, dict):
        return value
    if value.get("decision") != PlannerDecision.READY:
        return value
    if value.get("intent") != IntentKind.SIMILAR_PLAYERS:
        return value
    has_player_reference = bool(
        str(value.get("primary_player_name", "")).strip()
        or isinstance(value.get("primary_player_id"), int)
        and value["primary_player_id"] > 0
    )
    if not has_player_reference:
        return value

    advisory_fields = tuple(
        field
        for field in ("unsupported_reason", "clarification_message")
        if isinstance(value.get(field), str) and value[field].strip()
    )
    if not advisory_fields or not all(
        _is_similarity_non_guarantee_advisory(value[field]) for field in advisory_fields
    ):
        return value

    normalized = dict(value)
    advisory_texts = [str(normalized[field]).strip() for field in advisory_fields]
    existing_limitation = str(normalized.get("limitation", "")).strip()
    limitation_parts = ([existing_limitation] if existing_limitation else []) + advisory_texts
    limitation = " ".join(dict.fromkeys(limitation_parts))
    if len(limitation) > 500:
        return value

    normalized["limitation"] = limitation
    normalized["unsupported_reason"] = ""
    normalized["clarification_message"] = ""
    logger.warning(
        "Recovered safe planner advisory before validation: "
        "decision=ready intent=similar_players player_reference=present "
        "advisory_fields=%s advisory_kind=non_guarantee_safety_limitation",
        ",".join(advisory_fields),
    )
    return normalized


class LLMPositionGroup(str, Enum):
    NONE = "none"
    GK = PositionGroup.GK.value
    DEF = PositionGroup.DEF.value
    MID = PositionGroup.MID.value
    FWD = PositionGroup.FWD.value


class LLMLeaderboardMetric(str, Enum):
    NONE = "none"
    COMPLETION_ABOVE_EXPECTED_PP = LeaderboardMetric.COMPLETION_ABOVE_EXPECTED_PP.value
    PRESSURE_ABOVE_EXPECTED_PP = LeaderboardMetric.PRESSURE_ABOVE_EXPECTED_PP.value
    PROGRESSIVE_ABOVE_EXPECTED_PP = LeaderboardMetric.PROGRESSIVE_ABOVE_EXPECTED_PP.value
    LONG_PASS_ABOVE_EXPECTED_PP = LeaderboardMetric.LONG_PASS_ABOVE_EXPECTED_PP.value
    FINAL_THIRD_ENTRIES_PER_100_PASSES = (
        LeaderboardMetric.FINAL_THIRD_ENTRIES_PER_100_PASSES.value
    )
    EXPECTED_COMPLETION_RATE = LeaderboardMetric.EXPECTED_COMPLETION_RATE.value
    PROGRESSIVE_PASS_RATE = LeaderboardMetric.PROGRESSIVE_PASS_RATE.value
    ATTACKING_VALUE_PER_100_ACTIONS = LeaderboardMetric.ATTACKING_VALUE_PER_100_ACTIONS.value
    PASS_VALUE_PER_100_PASSES = LeaderboardMetric.PASS_VALUE_PER_100_PASSES.value
    CARRY_VALUE_PER_100_CARRIES = LeaderboardMetric.CARRY_VALUE_PER_100_CARRIES.value
    PROGRESSIVE_VALUE_PER_100_ACTIONS = (
        LeaderboardMetric.PROGRESSIVE_VALUE_PER_100_ACTIONS.value
    )
    PRESSURE_VALUE_PER_100_ACTIONS = LeaderboardMetric.PRESSURE_VALUE_PER_100_ACTIONS.value


class LLMPlayerSortField(str, Enum):
    NONE = "none"
    PLAYER_NAME = PlayerSortField.PLAYER_NAME.value
    PASS_ATTEMPTS = PlayerSortField.PASS_ATTEMPTS.value
    ACTUAL_COMPLETION_RATE = PlayerSortField.ACTUAL_COMPLETION_RATE.value
    EXPECTED_COMPLETION_RATE = PlayerSortField.EXPECTED_COMPLETION_RATE.value
    COMPLETION_ABOVE_EXPECTED_PP = PlayerSortField.COMPLETION_ABOVE_EXPECTED_PP.value
    PROGRESSIVE_PASS_RATE = PlayerSortField.PROGRESSIVE_PASS_RATE.value
    PRESSURE_ABOVE_EXPECTED_PP = PlayerSortField.PRESSURE_ABOVE_EXPECTED_PP.value
    FINAL_THIRD_ENTRIES_PER_100_PASSES = (
        PlayerSortField.FINAL_THIRD_ENTRIES_PER_100_PASSES.value
    )


class LLMMethodologyTopic(str, Enum):
    NONE = "none"
    XPASS = MethodologyTopic.XPASS.value
    XG = MethodologyTopic.XG.value
    POSSESSION_VALUE = MethodologyTopic.POSSESSION_VALUE.value
    ATTACKING_IMPACT = MethodologyTopic.ATTACKING_IMPACT.value
    PLAYER_PROFILES = MethodologyTopic.PLAYER_PROFILES.value
    PERCENTILES = MethodologyTopic.PERCENTILES.value
    ARCHETYPES = MethodologyTopic.ARCHETYPES.value
    SIMILARITY = MethodologyTopic.SIMILARITY.value
    TEAM_INTELLIGENCE = MethodologyTopic.TEAM_INTELLIGENCE.value
    ROLE_FIT = MethodologyTopic.ROLE_FIT.value
    ROLE_RECOMMENDATIONS = MethodologyTopic.ROLE_RECOMMENDATIONS.value


class LLMPlannerDecision(AiModel):
    """Flat provider contract: understand the request, but never choose tools."""

    decision: PlannerDecision
    intent: IntentKind
    requested_analyses: list[IntentKind] = Field(default_factory=list, max_length=6)
    primary_player_name: str = Field(default="", max_length=200)
    primary_player_id: int = Field(default=0, ge=0)
    secondary_player_name: str = Field(default="", max_length=200)
    secondary_player_id: int = Field(default=0, ge=0)
    team_name: str = Field(default="", max_length=200)
    team_id: int = Field(default=0, ge=0)
    search_query: str = Field(default="", max_length=200)
    position_group: LLMPositionGroup = LLMPositionGroup.NONE
    leaderboard_metric: LLMLeaderboardMetric = LLMLeaderboardMetric.NONE
    player_sort_by: LLMPlayerSortField = LLMPlayerSortField.NONE
    sort_order: SortOrder = SortOrder.ASC
    methodology_topic: LLMMethodologyTopic = LLMMethodologyTopic.NONE
    requested_sections: list[PlayerDossierSection] = Field(default_factory=list, max_length=4)
    min_pass_attempts: int = Field(default=0, ge=0)
    limit: int = Field(default=0, ge=0)
    clarification_message: str = Field(default="", max_length=500)
    unsupported_reason: str = Field(default="", max_length=500)
    limitation: str = Field(default="", max_length=500)

    @model_validator(mode="before")
    @classmethod
    def preserve_ready_advisory_as_limitation(cls, value: Any) -> Any:
        return normalize_ready_similarity_advisory(value)

    @model_validator(mode="after")
    def require_decision_explanation(self) -> LLMPlannerDecision:
        if self.decision is PlannerDecision.CLARIFICATION_REQUIRED:
            if not self.clarification_message.strip():
                raise ValueError("Clarification decisions require a message.")
        elif self.clarification_message:
            raise ValueError("clarification_message is only valid for clarification decisions.")
        if self.decision is PlannerDecision.UNSUPPORTED:
            if not self.unsupported_reason.strip():
                raise ValueError("Unsupported decisions require a reason.")
        elif self.unsupported_reason:
            raise ValueError("unsupported_reason is only valid for unsupported decisions.")
        return self


class LLMPlayerComparisonIntent(AiModel):
    """Provider-safe comparison intent without a fixed tuple schema."""

    kind: Literal[IntentKind.PLAYER_COMPARISON] = IntentKind.PLAYER_COMPARISON
    player_a: EntityRef
    player_b: EntityRef


LLMScoutIntent = (
    PlayerSearchIntent
    | PlayerProfileIntent
    | LLMPlayerComparisonIntent
    | SimilarPlayersIntent
    | LeaderboardIntent
    | TeamAnalysisIntent
    | RoleFitIntent
    | RoleRecommendationsIntent
    | MethodologyIntent
)


class LLMPlannedComparePlayersInput(AiModel):
    """Provider-safe comparison arguments without JSON Schema prefixItems."""

    player_a: EntityRef
    player_b: EntityRef


class LLMPlannedComparePlayersCall(AiModel):
    name: Literal[ToolName.COMPARE_PLAYERS]
    arguments: LLMPlannedComparePlayersInput


LLMPlannedToolCall = (
    PlannedSearchPlayersCall
    | PlannedPlayerDossierCall
    | LLMPlannedComparePlayersCall
    | PlannedSimilarPlayersCall
    | PlannedLeaderboardCall
    | PlannedTeamIntelligenceCall
    | PlannedRoleFitCall
    | PlannedRoleRecommendationsCall
    | PlannedMethodologyCall
)


class LLMScoutPlan(AiModel):
    """Provider-facing plan using only the Structured Outputs JSON Schema subset."""

    decision: PlannerDecision
    intent: LLMScoutIntent
    calls: list[LLMPlannedToolCall] = Field(
        default_factory=list,
        max_length=MAX_TOOL_CALLS,
    )
    clarification_message: str | None = Field(default=None, max_length=500)
    unsupported_reason: str | None = Field(default=None, max_length=500)
    limitations: list[str] = Field(default_factory=list, max_length=10)

    def to_scout_plan(self) -> ScoutPlan:
        """Convert provider output through the authoritative internal validation."""
        intent = self.intent.model_dump(mode="json")
        if isinstance(self.intent, LLMPlayerComparisonIntent):
            intent = {
                "kind": self.intent.kind.value,
                "players": [
                    self.intent.player_a.model_dump(mode="json"),
                    self.intent.player_b.model_dump(mode="json"),
                ],
            }

        calls: list[dict[str, object]] = []
        for call in self.calls:
            if isinstance(call, LLMPlannedComparePlayersCall):
                calls.append(
                    {
                        "name": call.name.value,
                        "arguments": {
                            "players": [
                                call.arguments.player_a.model_dump(mode="json"),
                                call.arguments.player_b.model_dump(mode="json"),
                            ]
                        },
                    }
                )
            else:
                calls.append(call.model_dump(mode="json"))

        return ScoutPlan.model_validate(
            {
                "decision": self.decision.value,
                "intent": intent,
                "calls": calls,
                "clarification_message": self.clarification_message,
                "unsupported_reason": self.unsupported_reason,
                "limitations": self.limitations,
            }
        )
