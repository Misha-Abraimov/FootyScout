"""Fixed registry and evidence wrapper for deterministic AI Scout tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from fastapi import HTTPException, status
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.ai.policy import (
    ROLE_FIT_LIMITATIONS,
    SIMILARITY_LIMITATIONS,
    TEAM_INTELLIGENCE_LIMITATIONS,
)
from app.ai.schemas import (
    AiModel,
    ComparePlayersInput,
    LeaderboardInput,
    MethodologyInput,
    MethodologyResponse,
    PlayerDossierInput,
    PlayerDossierResponse,
    ProductionStatus,
    RoleFitInput,
    RoleRecommendationsInput,
    SearchPlayersInput,
    SimilarPlayersInput,
    SourceCategory,
    TeamIntelligenceInput,
    TeamIntelligenceToolResponse,
    ToolError,
    ToolErrorCode,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.tools import methodology as methodology_tools
from app.ai.tools import players as player_tools
from app.ai.tools import teams as team_tools

ToolExecutor = Callable[[Session, Any], BaseModel]


@dataclass(frozen=True)
class ToolDefinition:
    name: ToolName
    input_model: type[AiModel]
    executor: ToolExecutor
    description: str
    limitations: tuple[str, ...]
    source_category: SourceCategory
    internal_notes: tuple[str, ...] = ()


TOOL_REGISTRY: dict[ToolName, ToolDefinition] = {
    ToolName.SEARCH_PLAYERS: ToolDefinition(
        name=ToolName.SEARCH_PLAYERS,
        input_model=SearchPlayersInput,
        executor=player_tools.search_players,
        description=(
            "Searches the loaded player cohort with existing validated filters. "
            "Use to resolve names or build a simple bounded shortlist."
        ),
        limitations=(),
        source_category=SourceCategory.POSTGRESQL,
        internal_notes=(
            (
                "Player identity uses deterministic normalized matching; ambiguous names "
                "may require clarification."
            ),
        ),
    ),
    ToolName.GET_PLAYER_DOSSIER: ToolDefinition(
        name=ToolName.GET_PLAYER_DOSSIER,
        input_model=PlayerDossierInput,
        executor=player_tools.get_player_dossier,
        description=(
            "Returns requested passing, shooting, attacking-impact, and intelligence "
            "sections for one stable player ID."
        ),
        limitations=("A profile describes observed event data, not future performance.",),
        source_category=SourceCategory.POSTGRESQL,
        internal_notes=("The dossier does not return raw event rows.",),
    ),
    ToolName.COMPARE_PLAYERS: ToolDefinition(
        name=ToolName.COMPARE_PLAYERS,
        input_model=ComparePlayersInput,
        executor=player_tools.compare_players,
        description=(
            "Compares exactly two players with the existing passing, attacking, and "
            "intelligence response contract."
        ),
        limitations=(
            "The comparison does not include every shooting or Role Fit field.",
            "Cross-position percentiles use different same-position peer groups.",
        ),
        source_category=SourceCategory.POSTGRESQL,
    ),
    ToolName.GET_SIMILAR_PLAYERS: ToolDefinition(
        name=ToolName.GET_SIMILAR_PLAYERS,
        input_model=SimilarPlayersInput,
        executor=player_tools.get_similar_players,
        description=(
            "Returns persisted same-position playing-style similarity. Do not use to "
            "determine player quality, transfer success, or team-role fit."
        ),
        limitations=SIMILARITY_LIMITATIONS,
        source_category=SourceCategory.POSTGRESQL,
    ),
    ToolName.GET_LEADERBOARD: ToolDefinition(
        name=ToolName.GET_LEADERBOARD,
        input_model=LeaderboardInput,
        executor=player_tools.get_leaderboard,
        description=(
            "Ranks eligible players for one governed leaderboard metric using the "
            "existing reliability thresholds."
        ),
        limitations=("A single-metric rank is not an overall player-quality rating.",),
        source_category=SourceCategory.POSTGRESQL,
    ),
    ToolName.GET_TEAM_INTELLIGENCE: ToolDefinition(
        name=ToolName.GET_TEAM_INTELLIGENCE,
        input_model=TeamIntelligenceInput,
        executor=team_tools.get_team_intelligence,
        description=(
            "Returns observed team intelligence or one supported outfield role. Do not "
            "use to infer coaching intent."
        ),
        limitations=TEAM_INTELLIGENCE_LIMITATIONS,
        source_category=SourceCategory.POSTGRESQL,
    ),
    ToolName.GET_ROLE_FIT: ToolDefinition(
        name=ToolName.GET_ROLE_FIT,
        input_model=RoleFitInput,
        executor=team_tools.get_role_fit,
        description=(
            "Returns resemblance to a supported team's observed positional role. It "
            "does not predict future performance or transfer success."
        ),
        limitations=ROLE_FIT_LIMITATIONS,
        source_category=SourceCategory.POSTGRESQL,
    ),
    ToolName.GET_ROLE_RECOMMENDATIONS: ToolDefinition(
        name=ToolName.GET_ROLE_RECOMMENDATIONS,
        input_model=RoleRecommendationsInput,
        executor=team_tools.get_role_recommendations,
        description=(
            "Returns external same-position candidates ordered by persisted Role Fit. "
            "Do not use as a prediction of transfer success."
        ),
        limitations=ROLE_FIT_LIMITATIONS,
        source_category=SourceCategory.POSTGRESQL,
    ),
    ToolName.GET_METHODOLOGY: ToolDefinition(
        name=ToolName.GET_METHODOLOGY,
        input_model=MethodologyInput,
        executor=lambda _session, request: methodology_tools.get_methodology(request),
        description=(
            "Returns allowlisted production metadata and stable methodology source "
            "references. It does not calculate player analytics."
        ),
        limitations=(),
        source_category=SourceCategory.MIXED,
        internal_notes=("Phase 1 uses no semantic retrieval or generated explanation.",),
    ),
}


def get_tool_definition(name: ToolName | str) -> ToolDefinition:
    try:
        normalized = ToolName(name)
    except ValueError as exc:
        raise ValueError(f"Unknown AI Scout tool: {name!r}.") from exc
    return TOOL_REGISTRY[normalized]


def execute_tool(
    session: Session,
    name: ToolName | str,
    arguments: Mapping[str, Any],
) -> ToolExecutionResult:
    definition = get_tool_definition(name)
    try:
        validated = definition.input_model.model_validate(dict(arguments))
    except ValidationError as exc:
        return ToolExecutionResult(
            tool_name=definition.name,
            arguments=dict(arguments),
            status=ToolExecutionStatus.INVALID_ARGUMENTS,
            source_category=definition.source_category,
            warnings=list(definition.limitations),
            internal_notes=list(definition.internal_notes),
            error=ToolError(
                code=ToolErrorCode.VALIDATION_ERROR,
                message="Tool arguments failed validation.",
                details=exc.errors(include_url=False),
            ),
        )

    normalized_arguments = validated.model_dump(mode="json")
    try:
        output = definition.executor(session, validated)
    except HTTPException as exc:
        execution_status = (
            ToolExecutionStatus.NOT_FOUND
            if exc.status_code == status.HTTP_404_NOT_FOUND
            else ToolExecutionStatus.ERROR
        )
        error_code = (
            ToolErrorCode.NOT_FOUND
            if exc.status_code == status.HTTP_404_NOT_FOUND
            else ToolErrorCode.SERVICE_ERROR
        )
        return ToolExecutionResult(
            tool_name=definition.name,
            arguments=normalized_arguments,
            status=execution_status,
            source_category=definition.source_category,
            warnings=list(definition.limitations),
            internal_notes=list(definition.internal_notes),
            error=ToolError(code=error_code, message=str(exc.detail)),
        )
    except RuntimeError:
        return ToolExecutionResult(
            tool_name=definition.name,
            arguments=normalized_arguments,
            status=ToolExecutionStatus.ERROR,
            source_category=definition.source_category,
            warnings=list(definition.limitations),
            internal_notes=list(definition.internal_notes),
            error=ToolError(
                code=ToolErrorCode.SERVICE_ERROR,
                message="The governed methodology source is unavailable.",
            ),
        )

    result = output.model_dump(mode="json")
    return ToolExecutionResult(
        tool_name=definition.name,
        arguments=normalized_arguments,
        status=ToolExecutionStatus.SUCCESS,
        result=result,
        source_category=_source_category(definition, output),
        entity_ids=_entity_ids(definition.name, validated, output),
        warnings=_warnings(definition, output),
        internal_notes=list(definition.internal_notes),
        production_status=_production_status(output),
    )


def _source_category(
    definition: ToolDefinition,
    output: BaseModel,
) -> SourceCategory:
    if not isinstance(output, MethodologyResponse):
        return definition.source_category
    categories = {
        SourceCategory.RUNTIME_METADATA
        if source.path.endswith(".json")
        else SourceCategory.CURATED_DOCUMENTATION
        for source in output.sources
    }
    return categories.pop() if len(categories) == 1 else SourceCategory.MIXED


def _production_status(output: BaseModel) -> ProductionStatus:
    if isinstance(output, MethodologyResponse):
        return output.production_status
    return ProductionStatus.PRODUCTION


def _entity_ids(
    name: ToolName,
    request: AiModel,
    output: BaseModel,
) -> list[int]:
    if isinstance(request, PlayerDossierInput):
        return [request.player_id]
    if isinstance(request, ComparePlayersInput):
        return list(request.player_ids)
    if isinstance(request, SimilarPlayersInput):
        response = cast(Any, output)
        return [request.player_id, *[item.similar_player_id for item in response.items]]
    if isinstance(request, RoleFitInput):
        return [request.player_id, request.target_team_id]
    if isinstance(request, RoleRecommendationsInput):
        response = cast(Any, output)
        return [request.team_id, *[item.player.player_id for item in response.items]]
    if isinstance(request, TeamIntelligenceInput):
        return [request.team_id]
    if isinstance(request, SearchPlayersInput):
        response = cast(Any, output)
        return [item.player_id for item in response.items]
    if isinstance(request, LeaderboardInput):
        response = cast(Any, output)
        return [item.player_id for item in response.items]
    return []


def _warnings(definition: ToolDefinition, output: BaseModel) -> list[str]:
    warnings = list(definition.limitations)
    if isinstance(output, PlayerDossierResponse):
        warnings.extend(output.unavailable_sections.values())
    elif isinstance(output, TeamIntelligenceToolResponse):
        if output.role is not None:
            warnings.append(output.role.support_message)
        elif output.intelligence is not None:
            warnings.extend(role.support_message for role in output.intelligence.roles)
    elif isinstance(output, MethodologyResponse):
        warnings.extend(output.limitations)
    else:
        unavailable_reason = getattr(output, "unavailable_reason", None)
        if unavailable_reason:
            warnings.append(unavailable_reason)
        sample_message = getattr(output, "sample_support_message", None)
        if sample_message:
            warnings.append(sample_message)
        role_message = getattr(output, "role_support_message", None)
        if role_message:
            warnings.append(role_message)
    return list(dict.fromkeys(warnings))
