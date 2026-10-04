"""Strict Phase 1 schemas for deterministic AI Scout intents and tools."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.policy import (
    LEADERBOARD_MAX_RESULTS,
    MAX_ENTITY_STRING_LENGTH,
    MAX_METHODOLOGY_QUESTION_LENGTH,
    MAX_TOOL_CALLS,
    PLAYER_SEARCH_MAX_RESULTS,
    RECOMMENDATIONS_MAX_RESULTS,
    SIMILAR_PLAYERS_MAX_RESULTS,
)
from app.schemas import (
    ActionValueModelInfoResponse,
    AttackingProfileResponse,
    ComparisonResponse,
    LeaderboardMetric,
    LeaderboardResponse,
    ModelInfoResponse,
    PlayerIdentity,
    PlayerIntelligenceResponse,
    PlayerListResponse,
    PlayerProfileResponse,
    PlayerRoleFitResponse,
    PlayerSortField,
    PositionGroup,
    ScoutingRecommendationsResponse,
    ShootingProfileResponse,
    SimilarPlayersResponse,
    SortOrder,
    TeamIntelligenceResponse,
    TeamRoleResponse,
    XGModelInfoResponse,
)

SUBJECT_CURRENT_TEAM_REFERENCE = "__subject_current_team__"
MISSING_PLAYER_REFERENCE = "__missing_player__"


class AiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class IntentKind(str, Enum):
    PLAYER_SEARCH = "player_search"
    PLAYER_PROFILE = "player_profile"
    PLAYER_COMPARISON = "player_comparison"
    SIMILAR_PLAYERS = "similar_players"
    LEADERBOARD = "leaderboard"
    TEAM_ANALYSIS = "team_analysis"
    ROLE_FIT = "role_fit"
    ROLE_RECOMMENDATIONS = "role_recommendations"
    METHODOLOGY = "methodology"


class PlayerDossierSection(str, Enum):
    PASSING = "passing"
    SHOOTING = "shooting"
    ATTACKING_IMPACT = "attacking_impact"
    INTELLIGENCE = "intelligence"


class MethodologyTopic(str, Enum):
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


class ToolName(str, Enum):
    SEARCH_PLAYERS = "search_players"
    GET_PLAYER_DOSSIER = "get_player_dossier"
    COMPARE_PLAYERS = "compare_players"
    GET_SIMILAR_PLAYERS = "get_similar_players"
    GET_LEADERBOARD = "get_leaderboard"
    GET_TEAM_INTELLIGENCE = "get_team_intelligence"
    GET_ROLE_FIT = "get_role_fit"
    GET_ROLE_RECOMMENDATIONS = "get_role_recommendations"
    GET_METHODOLOGY = "get_methodology"
    SEARCH_WEB = "search_web"


class SourceCategory(str, Enum):
    POSTGRESQL = "postgresql"
    RUNTIME_METADATA = "runtime_metadata"
    CURATED_DOCUMENTATION = "curated_documentation"
    MIXED = "mixed"
    EXTERNAL_WEB = "external_web"


class ProductionStatus(str, Enum):
    PRODUCTION = "production"
    EXPERIMENTAL = "experimental"
    NOT_APPLICABLE = "not_applicable"


class ToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_FOUND = "not_found"
    ERROR = "error"


class ToolErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    SERVICE_ERROR = "service_error"


class EntityResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


class PlannerDecision(str, Enum):
    READY = "ready"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED = "unsupported"


class PlanPreparationStatus(str, Enum):
    READY = "ready"
    CLARIFICATION_REQUIRED = "clarification_required"
    NOT_FOUND = "not_found"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


class PlanPreparationReasonCode(str, Enum):
    """Governed reason for a deterministic preparation halt."""

    CANDIDATE_POSITION_INCOMPATIBLE = "candidate_position_incompatible"
    UNSUPPORTED_TEAM_CAPABILITY = "unsupported_team_capability"


class EntityRef(AiModel):
    player_id: int | None = Field(default=None, gt=0)
    player_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_STRING_LENGTH,
    )

    @model_validator(mode="after")
    def require_exactly_one_reference(self) -> EntityRef:
        if (self.player_id is None) == (self.player_name is None):
            raise ValueError("Provide exactly one player_id or player_name.")
        return self


class TeamRef(AiModel):
    team_id: int | None = Field(default=None, gt=0)
    team_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_STRING_LENGTH,
    )

    @model_validator(mode="after")
    def require_exactly_one_reference(self) -> TeamRef:
        if (self.team_id is None) == (self.team_name is None):
            raise ValueError("Provide exactly one team_id or team_name.")
        return self


class SearchPlayersInput(AiModel):
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_STRING_LENGTH,
    )
    team: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_STRING_LENGTH,
    )
    position_group: PositionGroup | None = None
    min_pass_attempts: int = Field(default=0, ge=0)
    sort_by: PlayerSortField = PlayerSortField.PLAYER_NAME
    sort_order: SortOrder = SortOrder.ASC
    limit: int = Field(default=10, ge=1, le=PLAYER_SEARCH_MAX_RESULTS)


class PlayerDossierInput(AiModel):
    player_id: int = Field(gt=0)
    sections: list[PlayerDossierSection] = Field(
        default_factory=lambda: [PlayerDossierSection.INTELLIGENCE],
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def require_unique_sections(self) -> PlayerDossierInput:
        if len(self.sections) != len(set(self.sections)):
            raise ValueError("Dossier sections must be unique.")
        return self


class ComparePlayersInput(AiModel):
    player_ids: tuple[int, int]

    @model_validator(mode="after")
    def require_distinct_positive_ids(self) -> ComparePlayersInput:
        if any(player_id <= 0 for player_id in self.player_ids):
            raise ValueError("Player IDs must be positive.")
        if self.player_ids[0] == self.player_ids[1]:
            raise ValueError("Comparison requires two distinct player IDs.")
        return self


class SimilarPlayersInput(AiModel):
    player_id: int = Field(gt=0)
    limit: int = Field(default=6, ge=1, le=SIMILAR_PLAYERS_MAX_RESULTS)


class LeaderboardInput(AiModel):
    metric: LeaderboardMetric = LeaderboardMetric.COMPLETION_ABOVE_EXPECTED_PP
    position_group: PositionGroup | None = None
    team: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_ENTITY_STRING_LENGTH,
    )
    limit: int = Field(default=10, ge=1, le=LEADERBOARD_MAX_RESULTS)


class TeamIntelligenceInput(AiModel):
    team_id: int = Field(gt=0)
    position_group: PositionGroup | None = None

    @model_validator(mode="after")
    def reject_goalkeeper_role(self) -> TeamIntelligenceInput:
        if self.position_group is PositionGroup.GK:
            raise ValueError("Team role intelligence supports DEF, MID, and FWD only.")
        return self


class RoleFitInput(AiModel):
    player_id: int = Field(gt=0)
    target_team_id: int = Field(gt=0)


class RoleRecommendationsInput(AiModel):
    team_id: int = Field(gt=0)
    position_group: PositionGroup
    limit: int = Field(default=10, ge=1, le=RECOMMENDATIONS_MAX_RESULTS)

    @model_validator(mode="after")
    def reject_goalkeeper_role(self) -> RoleRecommendationsInput:
        if self.position_group is PositionGroup.GK:
            raise ValueError("Role recommendations support DEF, MID, and FWD only.")
        return self


class MethodologyInput(AiModel):
    topic: MethodologyTopic


class PlayerSearchIntent(SearchPlayersInput):
    kind: Literal[IntentKind.PLAYER_SEARCH] = IntentKind.PLAYER_SEARCH


class PlayerProfileIntent(AiModel):
    kind: Literal[IntentKind.PLAYER_PROFILE] = IntentKind.PLAYER_PROFILE
    player: EntityRef
    sections: list[PlayerDossierSection] = Field(
        default_factory=lambda: [PlayerDossierSection.INTELLIGENCE],
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def require_unique_sections(self) -> PlayerProfileIntent:
        if len(self.sections) != len(set(self.sections)):
            raise ValueError("Dossier sections must be unique.")
        return self


class PlayerComparisonIntent(AiModel):
    kind: Literal[IntentKind.PLAYER_COMPARISON] = IntentKind.PLAYER_COMPARISON
    players: tuple[EntityRef, EntityRef]

    @model_validator(mode="after")
    def require_distinct_references(self) -> PlayerComparisonIntent:
        if self.players[0] == self.players[1]:
            raise ValueError("Comparison requires two distinct player references.")
        return self


class SimilarPlayersIntent(AiModel):
    kind: Literal[IntentKind.SIMILAR_PLAYERS] = IntentKind.SIMILAR_PLAYERS
    player: EntityRef
    candidate_position_group: PositionGroup | None = None
    limit: int = Field(default=6, ge=1, le=SIMILAR_PLAYERS_MAX_RESULTS)


class LeaderboardIntent(LeaderboardInput):
    kind: Literal[IntentKind.LEADERBOARD] = IntentKind.LEADERBOARD


class TeamAnalysisIntent(AiModel):
    kind: Literal[IntentKind.TEAM_ANALYSIS] = IntentKind.TEAM_ANALYSIS
    team: TeamRef
    position_group: PositionGroup | None = None


class RoleFitIntent(AiModel):
    kind: Literal[IntentKind.ROLE_FIT] = IntentKind.ROLE_FIT
    player: EntityRef
    target_team: TeamRef


class RoleRecommendationsIntent(AiModel):
    kind: Literal[IntentKind.ROLE_RECOMMENDATIONS] = IntentKind.ROLE_RECOMMENDATIONS
    team: TeamRef
    position_group: PositionGroup
    limit: int = Field(default=10, ge=1, le=RECOMMENDATIONS_MAX_RESULTS)


class MethodologyIntent(AiModel):
    kind: Literal[IntentKind.METHODOLOGY] = IntentKind.METHODOLOGY
    topic: MethodologyTopic
    question: str | None = Field(
        default=None,
        min_length=1,
        max_length=MAX_METHODOLOGY_QUESTION_LENGTH,
    )


ScoutIntent = Annotated[
    PlayerSearchIntent
    | PlayerProfileIntent
    | PlayerComparisonIntent
    | SimilarPlayersIntent
    | LeaderboardIntent
    | TeamAnalysisIntent
    | RoleFitIntent
    | RoleRecommendationsIntent
    | MethodologyIntent,
    Field(discriminator="kind"),
]


class PlayerResolution(AiModel):
    status: EntityResolutionStatus
    query: str | int
    player_id: int | None = None
    player: PlayerIdentity | None = None
    candidates: list[PlayerIdentity] = Field(default_factory=list)


class TeamResolutionCandidate(AiModel):
    team_id: int
    team_name: str
    analytics_supported: bool


class TeamResolution(AiModel):
    status: EntityResolutionStatus
    query: str | int
    team_id: int | None = None
    team: TeamResolutionCandidate | None = None
    candidates: list[TeamResolutionCandidate] = Field(default_factory=list)
    analytics_supported: bool = False


class PlannedPlayerDossierInput(AiModel):
    player: EntityRef
    sections: list[PlayerDossierSection] = Field(
        default_factory=lambda: [PlayerDossierSection.INTELLIGENCE],
        min_length=1,
        max_length=4,
    )

    @model_validator(mode="after")
    def require_unique_sections(self) -> PlannedPlayerDossierInput:
        if len(self.sections) != len(set(self.sections)):
            raise ValueError("Dossier sections must be unique.")
        return self


class PlannedComparePlayersInput(AiModel):
    players: tuple[EntityRef, EntityRef]

    @model_validator(mode="after")
    def require_distinct_references(self) -> PlannedComparePlayersInput:
        if self.players[0] == self.players[1]:
            raise ValueError("Comparison requires two distinct player references.")
        return self


class PlannedSimilarPlayersInput(AiModel):
    player: EntityRef
    candidate_position_group: PositionGroup | None = None
    limit: int = Field(default=6, ge=1, le=SIMILAR_PLAYERS_MAX_RESULTS)


class PlannedTeamIntelligenceInput(AiModel):
    team: TeamRef
    position_group: PositionGroup | None = None


class PlannedRoleFitInput(AiModel):
    player: EntityRef
    target_team: TeamRef


class PlannedRoleRecommendationsInput(AiModel):
    team: TeamRef
    position_group: PositionGroup
    limit: int = Field(default=10, ge=1, le=RECOMMENDATIONS_MAX_RESULTS)


class PlannedSearchPlayersCall(AiModel):
    name: Literal[ToolName.SEARCH_PLAYERS]
    arguments: SearchPlayersInput


class PlannedPlayerDossierCall(AiModel):
    name: Literal[ToolName.GET_PLAYER_DOSSIER]
    arguments: PlannedPlayerDossierInput


class PlannedComparePlayersCall(AiModel):
    name: Literal[ToolName.COMPARE_PLAYERS]
    arguments: PlannedComparePlayersInput


class PlannedSimilarPlayersCall(AiModel):
    name: Literal[ToolName.GET_SIMILAR_PLAYERS]
    arguments: PlannedSimilarPlayersInput


class PlannedLeaderboardCall(AiModel):
    name: Literal[ToolName.GET_LEADERBOARD]
    arguments: LeaderboardInput


class PlannedTeamIntelligenceCall(AiModel):
    name: Literal[ToolName.GET_TEAM_INTELLIGENCE]
    arguments: PlannedTeamIntelligenceInput


class PlannedRoleFitCall(AiModel):
    name: Literal[ToolName.GET_ROLE_FIT]
    arguments: PlannedRoleFitInput


class PlannedRoleRecommendationsCall(AiModel):
    name: Literal[ToolName.GET_ROLE_RECOMMENDATIONS]
    arguments: PlannedRoleRecommendationsInput


class PlannedMethodologyCall(AiModel):
    name: Literal[ToolName.GET_METHODOLOGY]
    arguments: MethodologyInput


PlannedToolCall = Annotated[
    PlannedSearchPlayersCall
    | PlannedPlayerDossierCall
    | PlannedComparePlayersCall
    | PlannedSimilarPlayersCall
    | PlannedLeaderboardCall
    | PlannedTeamIntelligenceCall
    | PlannedRoleFitCall
    | PlannedRoleRecommendationsCall
    | PlannedMethodologyCall,
    Field(discriminator="name"),
]


class ScoutPlan(AiModel):
    decision: PlannerDecision
    intent: ScoutIntent
    calls: list[PlannedToolCall] = Field(default_factory=list, max_length=MAX_TOOL_CALLS)
    clarification_message: str | None = Field(default=None, max_length=500)
    unsupported_reason: str | None = Field(default=None, max_length=500)
    limitations: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def require_decision_details(self) -> ScoutPlan:
        if self.decision is PlannerDecision.CLARIFICATION_REQUIRED:
            if not self.clarification_message:
                raise ValueError("Clarification decisions require a message.")
        elif self.clarification_message is not None:
            raise ValueError("clarification_message is only valid when clarification is required.")
        if self.decision is PlannerDecision.UNSUPPORTED:
            if not self.unsupported_reason:
                raise ValueError("Unsupported decisions require a reason.")
        elif self.unsupported_reason is not None:
            raise ValueError("unsupported_reason is only valid for unsupported decisions.")
        return self


class ResolvedEntity(AiModel):
    entity_type: Literal["player", "team"]
    query: str | int
    stable_id: int
    display_name: str


class NormalizedToolCall(AiModel):
    call_id: str = Field(pattern=r"^call-[1-6]$")
    name: ToolName
    arguments: dict[str, Any]


class NormalizedPlan(AiModel):
    intent: ScoutIntent
    calls: list[NormalizedToolCall] = Field(max_length=MAX_TOOL_CALLS)
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PlanPreparationResult(AiModel):
    status: PlanPreparationStatus
    reason_code: PlanPreparationReasonCode | None = None
    normalized_plan: NormalizedPlan | None = None
    player_resolutions: list[PlayerResolution] = Field(default_factory=list)
    team_resolutions: list[TeamResolution] = Field(default_factory=list)
    clarification_message: str | None = None
    unsupported_reason: str | None = None
    error_message: str | None = None
    limitations: list[str] = Field(default_factory=list)


class PlayerDossierResponse(AiModel):
    player: PlayerIdentity
    requested_sections: list[PlayerDossierSection]
    passing: PlayerProfileResponse | None = None
    shooting: ShootingProfileResponse | None = None
    attacking_impact: AttackingProfileResponse | None = None
    intelligence: PlayerIntelligenceResponse | None = None
    unavailable_sections: dict[PlayerDossierSection, str] = Field(default_factory=dict)


class TeamIntelligenceToolResponse(AiModel):
    intelligence: TeamIntelligenceResponse | None = None
    role: TeamRoleResponse | None = None


class MethodologySource(AiModel):
    source_id: str
    path: str
    section: str
    status: ProductionStatus


class MethodologyResponse(AiModel):
    topic: MethodologyTopic
    production_status: ProductionStatus
    current_production_model: str | None = None
    experimental_models: list[str] = Field(default_factory=list)
    summary: str
    structured_metadata: (
        ModelInfoResponse
        | XGModelInfoResponse
        | ActionValueModelInfoResponse
        | dict[str, Any]
        | None
    ) = None
    sources: list[MethodologySource]
    limitations: list[str] = Field(default_factory=list)


class ToolError(AiModel):
    code: ToolErrorCode
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


class ToolExecutionResult(AiModel):
    tool_name: ToolName
    arguments: dict[str, Any]
    status: ToolExecutionStatus
    result: dict[str, Any] | None = None
    source_category: SourceCategory
    entity_ids: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    internal_notes: list[str] = Field(default_factory=list)
    production_status: ProductionStatus = ProductionStatus.PRODUCTION
    error: ToolError | None = None


ToolOutput = (
    PlayerListResponse
    | PlayerDossierResponse
    | ComparisonResponse
    | SimilarPlayersResponse
    | LeaderboardResponse
    | TeamIntelligenceToolResponse
    | PlayerRoleFitResponse
    | ScoutingRecommendationsResponse
    | MethodologyResponse
)
