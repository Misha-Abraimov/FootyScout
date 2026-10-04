"""Deterministic plan preparation, execution, and evidence tests."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.orm import Session

from app.ai import executor as executor_module
from app.ai.executor import PlanExecutionStatus, execute_plan
from app.ai.planner import PlannerResult, PlannerStatus
from app.ai.prompts import PLANNER_PROMPT_VERSION
from app.ai.provider import PlannerUsage
from app.ai.run import AIScoutRunStatus, run_ai_scout
from app.ai.schemas import (
    NormalizedPlan,
    NormalizedToolCall,
    ProductionStatus,
    ScoutIntent,
    ScoutPlan,
    SourceCategory,
    ToolError,
    ToolErrorCode,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolName,
)


class FakePlanner:
    provider = "fake"
    model = "fake-planner"

    def __init__(self, plan: ScoutPlan) -> None:
        self._plan = plan

    def plan(self, question: str) -> PlannerResult:
        del question
        return PlannerResult(
            status=PlannerStatus.SUCCESS,
            provider=self.provider,
            model=self.model,
            prompt_version=PLANNER_PROMPT_VERSION,
            plan=self._plan,
            usage=PlannerUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )


def _plan(payload: dict[str, object]) -> ScoutPlan:
    return ScoutPlan.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "expected_tools"),
    [
        (
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice", "limit": 5},
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Alice", "limit": 5},
                    }
                ],
            },
            [ToolName.SEARCH_PLAYERS],
        ),
        (
            {
                "decision": "ready",
                "intent": {"kind": "similar_players", "player": {"player_id": 1}},
                "calls": [
                    {
                        "name": "get_similar_players",
                        "arguments": {"player": {"player_id": 1}, "limit": 3},
                    }
                ],
            },
            [ToolName.GET_SIMILAR_PLAYERS],
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_comparison",
                    "players": [{"player_id": 1}, {"player_id": 2}],
                },
                "calls": [
                    {
                        "name": "compare_players",
                        "arguments": {"players": [{"player_id": 1}, {"player_id": 2}]},
                    }
                ],
            },
            [ToolName.COMPARE_PLAYERS],
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "role_fit",
                    "player": {"player_id": 4},
                    "target_team": {"team_id": 904},
                },
                "calls": [
                    {
                        "name": "get_role_fit",
                        "arguments": {
                            "player": {"player_id": 4},
                            "target_team": {"team_id": 904},
                        },
                    }
                ],
            },
            [ToolName.GET_ROLE_FIT],
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "role_recommendations",
                    "team": {"team_name": "Leverkusen"},
                    "position_group": "MID",
                },
                "calls": [
                    {
                        "name": "get_role_recommendations",
                        "arguments": {
                            "team": {"team_name": "Leverkusen"},
                            "position_group": "MID",
                            "limit": 5,
                        },
                    }
                ],
            },
            [ToolName.GET_ROLE_RECOMMENDATIONS],
        ),
        (
            {
                "decision": "ready",
                "intent": {"kind": "methodology", "topic": "possession_value"},
                "calls": [
                    {
                        "name": "get_methodology",
                        "arguments": {"topic": "possession_value"},
                    }
                ],
            },
            [ToolName.GET_METHODOLOGY],
        ),
        (
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_profile",
                    "player": {"player_name": "Bob"},
                    "sections": ["passing"],
                },
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Bob", "limit": 10},
                    },
                    {
                        "name": "get_player_dossier",
                        "arguments": {
                            "player": {"player_name": "Bob"},
                            "sections": ["passing"],
                        },
                    },
                ],
            },
            [ToolName.SEARCH_PLAYERS, ToolName.GET_PLAYER_DOSSIER],
        ),
    ],
)
def test_valid_plans_execute_without_provider_access(
    ai_session: Session,
    payload: dict[str, object],
    expected_tools: list[ToolName],
) -> None:
    result = run_ai_scout(
        ai_session,
        question="provider-free test",
        planner=FakePlanner(_plan(payload)),
    )
    assert result.status is AIScoutRunStatus.COMPLETED
    assert result.normalized_plan is not None
    assert [call.name for call in result.normalized_plan.calls] == expected_tools
    assert [record.tool_name for record in result.evidence] == expected_tools
    assert [record.execution_order for record in result.evidence] == list(
        range(1, len(expected_tools) + 1)
    )
    assert result.usage is not None and result.usage.total_tokens == 2


def test_ambiguous_and_missing_players_stop_before_execution(ai_session: Session) -> None:
    ambiguous = _plan(
        {
            "decision": "ready",
            "intent": {"kind": "similar_players", "player": {"player_name": "Alice"}},
            "calls": [
                {
                    "name": "get_similar_players",
                    "arguments": {"player": {"player_name": "Alice"}},
                }
            ],
        }
    )
    missing = _plan(
        {
            "decision": "ready",
            "intent": {
                "kind": "player_profile",
                "player": {"player_name": "Nobody Here"},
            },
            "calls": [
                {
                    "name": "get_player_dossier",
                    "arguments": {"player": {"player_name": "Nobody Here"}},
                }
            ],
        }
    )
    ambiguous_result = run_ai_scout(
        ai_session,
        question="Who is like Alice?",
        planner=FakePlanner(ambiguous),
    )
    missing_result = run_ai_scout(
        ai_session,
        question="Show Nobody Here",
        planner=FakePlanner(missing),
    )
    assert ambiguous_result.status is AIScoutRunStatus.CLARIFICATION_REQUIRED
    assert ambiguous_result.clarification_required is True
    assert len(ambiguous_result.player_resolutions[0].candidates) == 2
    assert "Alice Playmaker — Alpha FC, Center Midfield" in (
        ambiguous_result.clarification_message or ""
    )
    assert "Alice Runner — Delta Town, Right Midfield" in (
        ambiguous_result.clarification_message or ""
    )
    assert ambiguous_result.evidence == ()
    assert missing_result.status is AIScoutRunStatus.NOT_FOUND
    assert missing_result.evidence == ()


def test_loaded_team_without_production_profile_is_insufficient(ai_session: Session) -> None:
    plan = _plan(
        {
            "decision": "ready",
            "intent": {"kind": "team_analysis", "team": {"team_name": "Alpha FC"}},
            "calls": [
                {
                    "name": "get_team_intelligence",
                    "arguments": {"team": {"team_name": "Alpha FC"}},
                }
            ],
        }
    )
    result = run_ai_scout(
        ai_session,
        question="Analyze Alpha FC",
        planner=FakePlanner(plan),
    )
    assert result.status is AIScoutRunStatus.NOT_FOUND
    assert result.team_resolutions[0].analytics_supported is False
    assert "exists in the loaded cohort" in (result.error_message or "")


@pytest.mark.parametrize("request_kind", ["role_fit", "role_recommendations"])
def test_goalkeeper_role_requests_are_stopped_by_policy(
    ai_session: Session,
    request_kind: str,
) -> None:
    if request_kind == "role_fit":
        payload = {
            "decision": "ready",
            "intent": {
                "kind": "role_fit",
                "player": {"player_id": 6},
                "target_team": {"team_id": 904},
            },
            "calls": [
                {
                    "name": "get_role_fit",
                    "arguments": {
                        "player": {"player_id": 6},
                        "target_team": {"team_id": 904},
                    },
                }
            ],
        }
    else:
        payload = {
            "decision": "ready",
            "intent": {
                "kind": "role_recommendations",
                "team": {"team_id": 904},
                "position_group": "GK",
            },
            "calls": [
                {
                    "name": "get_role_recommendations",
                    "arguments": {"team": {"team_id": 904}, "position_group": "GK"},
                }
            ],
        }
    result = run_ai_scout(
        ai_session,
        question="goalkeeper request",
        planner=FakePlanner(_plan(payload)),
    )
    assert result.status is AIScoutRunStatus.UNSUPPORTED
    assert result.evidence == ()


def test_comparison_that_resolves_to_same_player_requires_clarification(
    ai_session: Session,
) -> None:
    result = run_ai_scout(
        ai_session,
        question="Compare Alice Playmaker with player 1",
        planner=FakePlanner(
                _plan(
                    {
                        "decision": "ready",
                        "intent": {
                            "kind": "player_comparison",
                            "players": [
                                {"player_name": "Alice Playmaker"},
                                {"player_id": 1},
                            ],
                        },
                        "calls": [
                            {
                                "name": "compare_players",
                                "arguments": {
                                    "players": [
                                        {"player_name": "Alice Playmaker"},
                                        {"player_id": 1},
                                    ]
                                },
                            }
                        ],
                    }
                )
        ),
    )
    assert result.status is AIScoutRunStatus.CLARIFICATION_REQUIRED
    assert result.clarification_required is True


def _normalized_plan(*names: ToolName) -> NormalizedPlan:
    intent_payload: dict[str, object] = (
        {
            "kind": "player_profile",
            "player": {"player_name": "Alice"},
            "sections": ["passing"],
        }
        if ToolName.SEARCH_PLAYERS in names
        else {"kind": "methodology", "topic": "xpass"}
    )
    intent = TypeAdapter(ScoutIntent).validate_python(intent_payload)
    return NormalizedPlan(
        intent=intent,
        calls=[
            NormalizedToolCall(
                call_id=f"call-{index}",
                name=name,
                arguments=(
                    {"topic": "xpass"}
                    if name is ToolName.GET_METHODOLOGY
                    else {"query": "Alice", "limit": 5}
                ),
            )
            for index, name in enumerate(names, start=1)
        ],
    )


def test_executor_stops_on_tool_error_and_records_it(
    ai_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_tool(*_args: object, **_kwargs: object) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=ToolName.GET_METHODOLOGY,
            arguments={"topic": "xpass"},
            status=ToolExecutionStatus.ERROR,
            source_category=SourceCategory.RUNTIME_METADATA,
            error=ToolError(code=ToolErrorCode.SERVICE_ERROR, message="safe failure"),
        )

    monkeypatch.setattr(executor_module, "execute_tool", fail_tool)
    execution = execute_plan(
        ai_session,
        _normalized_plan(ToolName.GET_METHODOLOGY),
        run_id="run-test",
    )
    assert execution.status is PlanExecutionStatus.STOPPED_ON_ERROR
    assert execution.stopped_on_call_id == "call-1"
    assert execution.evidence.records[0].execution_status is ToolExecutionStatus.ERROR


def test_evidence_order_status_and_ledger_immutability(
    ai_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = iter([ProductionStatus.PRODUCTION, ProductionStatus.EXPERIMENTAL])

    def succeed(
        _session: Session,
        name: ToolName,
        arguments: dict[str, object],
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=name,
            arguments=arguments,
            status=ToolExecutionStatus.SUCCESS,
            result={"ok": True},
            source_category=SourceCategory.POSTGRESQL,
            entity_ids=[1],
            warnings=["traceable warning"],
            production_status=next(statuses),
        )

    monkeypatch.setattr(executor_module, "execute_tool", succeed)
    execution = execute_plan(
        ai_session,
        _normalized_plan(ToolName.SEARCH_PLAYERS, ToolName.GET_METHODOLOGY),
        run_id="run-status",
    )
    assert execution.status is PlanExecutionStatus.COMPLETED
    assert [row.execution_order for row in execution.evidence.records] == [1, 2]
    assert [row.production_status for row in execution.evidence.records] == [
        ProductionStatus.PRODUCTION,
        ProductionStatus.EXPERIMENTAL,
    ]
    assert execution.evidence.records[0].stable_entity_ids == (1,)
    with pytest.raises(ValidationError):
        execution.evidence.records[0].execution_order = 2


def test_exact_duplicate_calls_are_removed_deterministically(ai_session: Session) -> None:
    call = {"name": "search_players", "arguments": {"query": "Alice", "limit": 5}}
    result = run_ai_scout(
        ai_session,
        question="Find Alice once",
        planner=FakePlanner(
                _plan(
                    {
                        "decision": "ready",
                        "intent": {"kind": "player_search", "query": "Alice"},
                        "calls": [call, call],
                    }
                )
        ),
    )
    assert result.status is AIScoutRunStatus.COMPLETED
    assert result.normalized_plan is not None
    assert len(result.normalized_plan.calls) == 1
    assert any("duplicate" in limitation for limitation in result.limitations)
