"""Deterministic request-understanding to ScoutPlan mapping contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.ai.executor import PlanExecutionStatus, execute_plan
from app.ai.methodology_guard import apply_pure_methodology_guard
from app.ai.plan_builder import build_scout_plan, is_placeholder_player_reference
from app.ai.plan_normalizer import prepare_plan
from app.ai.policy import PLAYER_SEARCH_MAX_RESULTS, SIMILAR_PLAYERS_MAX_RESULTS
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.schemas import (
    IntentKind,
    PlannerDecision,
    PlanPreparationReasonCode,
    PlanPreparationStatus,
    ToolName,
)
from app.models import Player


@pytest.mark.parametrize(
    ("payload", "expected_tools"),
    [
        (
            {"intent": "player_search", "search_query": "Alice"},
            [ToolName.SEARCH_PLAYERS],
        ),
        (
            {"intent": "player_profile", "primary_player_name": "Alice"},
            [ToolName.SEARCH_PLAYERS, ToolName.GET_PLAYER_DOSSIER],
        ),
        (
            {
                "intent": "player_comparison",
                "primary_player_id": 1,
                "secondary_player_id": 2,
            },
            [ToolName.COMPARE_PLAYERS],
        ),
        (
            {"intent": "similar_players", "primary_player_id": 1},
            [ToolName.GET_SIMILAR_PLAYERS],
        ),
        (
            {"intent": "leaderboard", "leaderboard_metric": "progressive_pass_rate"},
            [ToolName.GET_LEADERBOARD],
        ),
        (
            {"intent": "team_analysis", "team_id": 904},
            [ToolName.GET_TEAM_INTELLIGENCE],
        ),
        (
            {"intent": "role_fit", "primary_player_id": 1, "team_id": 904},
            [ToolName.GET_ROLE_FIT],
        ),
        (
            {
                "intent": "role_recommendations",
                "team_id": 904,
                "position_group": "MID",
            },
            [ToolName.GET_ROLE_RECOMMENDATIONS],
        ),
        (
            {"intent": "methodology", "methodology_topic": "xpass"},
            [ToolName.GET_METHODOLOGY],
        ),
    ],
)
def test_all_supported_intents_map_to_allowlisted_tools(
    payload: dict[str, object],
    expected_tools: list[ToolName],
) -> None:
    decision = LLMPlannerDecision.model_validate({"decision": "ready", **payload})

    plan = build_scout_plan(decision)

    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == expected_tools


def test_compound_similarity_and_role_request_has_deterministic_order() -> None:
    decision = LLMPlannerDecision(
        decision="ready",
        intent="similar_players",
        requested_analyses=["role_recommendations"],
        primary_player_name="Granit Xhaka",
        team_name="Bayer Leverkusen",
        position_group="MID",
        limit=7,
    )

    plan = build_scout_plan(decision)

    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_SIMILAR_PLAYERS,
        ToolName.GET_ROLE_RECOMMENDATIONS,
    ]


def test_role_fit_target_team_is_not_used_as_player_search_filter() -> None:
    question = (
        "How well does Nicolas Seiwald fit Bayer Leverkusen and what is his current situation?"
    )
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="role_fit",
            requested_analyses=["player_profile"],
            primary_player_name="Nicolas Seiwald",
            team_name="Bayer Leverkusen",
            position_group="MID",
        ),
        question=question,
    )

    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_ROLE_FIT,
    ]
    search = plan.calls[0]
    assert search.arguments.query == "Nicolas Seiwald"
    assert search.arguments.team is None
    role_fit = plan.calls[1]
    assert role_fit.arguments.target_team.team_name == "Bayer Leverkusen"


def test_pure_role_fit_does_not_add_unrequested_player_dossier() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="role_fit",
            requested_analyses=["player_profile"],
            primary_player_name="Nicolas Seiwald",
            team_name="Bayer Leverkusen",
        ),
        question="How well does Nicolas Seiwald fit Bayer Leverkusen?",
    )

    assert ToolName.GET_ROLE_FIT in [call.name for call in plan.calls]
    assert ToolName.GET_PLAYER_DOSSIER not in [call.name for call in plan.calls]


def test_explicit_profile_and_role_fit_preserve_player_dossier() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="role_fit",
            requested_analyses=["player_profile"],
            primary_player_name="Nicolas Seiwald",
            team_name="Bayer Leverkusen",
            requested_sections=["intelligence"],
        ),
        question="Explain Nicolas Seiwald's playing profile and how he fits Bayer Leverkusen.",
    )

    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_ROLE_FIT,
        ToolName.GET_PLAYER_DOSSIER,
    ]
    assert plan.calls[0].arguments.team is None


def test_explicit_candidate_team_is_distinct_from_role_fit_target_team() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="role_fit",
            primary_player_name="Nicolas Seiwald",
            team_name="Bayer Leverkusen",
            position_group="MID",
        ),
        question=(
            "How well does the Leipzig midfielder Nicolas Seiwald fit Bayer Leverkusen?"
        ),
    )

    search = next(call for call in plan.calls if call.name is ToolName.SEARCH_PLAYERS)
    role_fit = next(call for call in plan.calls if call.name is ToolName.GET_ROLE_FIT)
    assert search.arguments.team == "Leipzig"
    assert role_fit.arguments.target_team.team_name == "Bayer Leverkusen"


@pytest.mark.parametrize(
    "payload",
    [
        {"intent": "methodology", "methodology_topic": "role_fit"},
        {"intent": "role_fit", "primary_player_id": 1, "team_id": 904},
        {
            "intent": "role_recommendations",
            "team_id": 904,
            "position_group": "MID",
        },
        {
            "intent": "similar_players",
            "requested_analyses": ["role_recommendations"],
            "primary_player_id": 1,
            "team_id": 904,
            "position_group": "MID",
        },
    ],
)
def test_role_fit_plans_do_not_promote_provider_prose_to_limitations(
    payload: dict[str, object],
) -> None:
    stale = (
        "Role Fit describes archetype matching from off-ball tracking data, with "
        "training cohort, team-context, and injury effects."
    )
    plan = build_scout_plan(
        LLMPlannerDecision.model_validate(
            {"decision": "ready", "limitation": stale, **payload}
        )
    )
    assert plan.limitations == []


def test_non_role_fit_provider_limitation_behavior_is_unchanged() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_search",
            search_query="Alice",
            limitation="Name search remains bounded.",
        )
    )
    assert plan.limitations == ["Name search remains bounded."]


def test_pure_methodology_correction_clears_entity_requirements(
    ai_session: Session,
) -> None:
    contradictory = LLMPlannerDecision(
        decision="clarification_required",
        intent="role_fit",
        primary_player_name="invented player",
        team_name="invented team",
        clarification_message="Which player should FootyScout analyze?",
    )
    guarded = apply_pure_methodology_guard(
        "What does Role Fit mean?",
        contradictory,
    )
    plan = build_scout_plan(guarded.decision)
    prepared = prepare_plan(ai_session, plan)

    assert guarded.applied is True
    assert plan.decision is PlannerDecision.READY
    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.player_resolutions == []
    assert prepared.team_resolutions == []
    assert prepared.normalized_plan is not None
    assert [call.name for call in prepared.normalized_plan.calls] == [
        ToolName.GET_METHODOLOGY
    ]


def test_explicit_role_fit_comparison_preserves_role_fit_intent() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="player_comparison",
            primary_player_name="Xhaka",
            secondary_player_name="external midfielder",
            clarification_message="Which external midfielder and team do you mean?",
        ),
        question="Compare Xhaka's Role Fit with an external midfielder.",
    )

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert plan.intent.kind is IntentKind.ROLE_FIT
    assert all(call.name is not ToolName.COMPARE_PLAYERS for call in plan.calls)


def test_explicit_role_fit_comparison_with_two_players_remains_role_fit() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_id=1,
            secondary_player_id=4,
        ),
        question="Compare Alice's Role Fit with Dana's.",
    )

    assert plan.intent.kind is IntentKind.ROLE_FIT


def test_governed_team_fit_comparison_remains_role_fit() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_name="Xhaka",
            secondary_player_name="Palacios",
            team_name="Leverkusen",
        ),
        question="How does Xhaka fit Leverkusen compared with Palacios?",
    )

    assert plan.intent.kind is IntentKind.ROLE_FIT


def test_ordinary_player_comparison_remains_player_comparison() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_id=1,
            secondary_player_id=2,
        ),
        question="Compare Alice with Bob.",
    )

    assert plan.intent.kind is IntentKind.PLAYER_COMPARISON
    assert [call.name for call in plan.calls] == [ToolName.COMPARE_PLAYERS]


@pytest.mark.parametrize(
    "question",
    [
        "Look up player ID -4.",
        "Look up player ID 0.",
        "Look up player ID abc.",
        "Look up player ID 3.5.",
        "Look up player ID.",
    ],
)
def test_invalid_explicit_player_ids_clarify_without_player_calls(question: str) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_profile",
            primary_player_name=question,
        ),
        question=question,
    )

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert plan.clarification_message == "Player IDs must be positive whole numbers."
    assert plan.calls == []


def test_positive_explicit_player_id_and_normal_name_lookup_remain_supported() -> None:
    by_id = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_profile",
            primary_player_id=4,
        ),
        question="Look up player ID 4.",
    )
    by_name = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_profile",
            primary_player_name="Alice Playmaker",
        ),
        question="Look up Alice Playmaker.",
    )

    assert by_id.decision is PlannerDecision.READY
    assert [call.name for call in by_id.calls] == [ToolName.GET_PLAYER_DOSSIER]
    assert by_name.decision is PlannerDecision.READY
    assert [call.name for call in by_name.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_PLAYER_DOSSIER,
    ]


def test_positive_plural_player_ids_continue_to_two_player_comparison() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_id=1,
            secondary_player_id=2,
        ),
        question="Compare player IDs 1 and 2.",
    )

    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == [ToolName.COMPARE_PLAYERS]


@pytest.mark.parametrize(
    "question",
    [
        "Compare Xhaka, Kimmich, and Goretzka.",
        "How do Xhaka, Kimmich, and Goretzka compare?",
        "Compare player ID 1, player ID 2, and player ID 4.",
        "Compare Xhaka, Kimmich & Goretzka.",
    ],
)
def test_more_than_two_comparison_references_clarify_without_calls(
    question: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_name="Xhaka",
            secondary_player_name="Kimmich",
        ),
        question=question,
    )

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert plan.clarification_message == (
        "FootyScout compares exactly two players at a time. Which two players "
        "should it compare?"
    )
    assert plan.calls == []


def test_comparison_cardinality_guard_does_not_treat_metric_lists_as_players() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_id=1,
            secondary_player_id=2,
        ),
        question="Compare Alice and Bob across passing, shooting, and attacking impact.",
    )

    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == [ToolName.COMPARE_PLAYERS]


def test_comparison_missing_second_player_still_requires_clarification() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_name="Alice",
        ),
        question="Compare Alice with another player.",
    )

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert plan.clarification_message == "Which two players should FootyScout compare?"
    assert all(call.name is not ToolName.COMPARE_PLAYERS for call in plan.calls)


def test_passing_comparison_remains_player_comparison() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_id=1,
            secondary_player_id=2,
        ),
        question="Compare their passing statistics.",
    )

    assert plan.intent.kind is IntentKind.PLAYER_COMPARISON


def test_explicit_clarification_and_unsupported_decisions_are_preserved() -> None:
    clarification = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="player_profile",
            clarification_message="Which Alex do you mean?",
        )
    )
    unsupported = build_scout_plan(
        LLMPlannerDecision(
            decision="unsupported",
            intent="methodology",
            unsupported_reason="That request is outside FootyScout's scope.",
        )
    )

    assert clarification.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert clarification.clarification_message == "Which Alex do you mean?"
    assert unsupported.decision is PlannerDecision.UNSUPPORTED
    assert unsupported.unsupported_reason == "That request is outside FootyScout's scope."


def test_provider_cannot_declare_named_player_ambiguous_before_product_search(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="player_profile",
            primary_player_name="Alice Playmaker",
            clarification_message="There may be other players with that name.",
        )
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_PLAYER_DOSSIER,
    ]
    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[-1].arguments["player_id"] == 1


def test_complete_role_recommendation_suppresses_unnecessary_provider_clarification(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_recommendations",
            team_name="Leverkusen",
            clarification_message=(
                "Please provide age, league, subtype, constraints, and result count."
            ),
        ),
        question="Find midfielders who fit Team A.",
    )

    assert plan.decision is PlannerDecision.READY
    assert len(plan.calls) == 1
    call = plan.calls[0]
    assert call.name is ToolName.GET_ROLE_RECOMMENDATIONS
    assert call.arguments.team.team_name == "Leverkusen"
    assert call.arguments.position_group.value == "MID"
    assert call.arguments.limit == 10

    prepared = prepare_plan(ai_session, plan)
    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[0].arguments == {
        "team_id": 904,
        "position_group": "MID",
        "limit": 10,
    }
    executed = execute_plan(ai_session, prepared.normalized_plan, run_id="recommendation-test")
    assert executed.status is PlanExecutionStatus.COMPLETED
    assert executed.evidence.records[0].tool_name is ToolName.GET_ROLE_RECOMMENDATIONS


@pytest.mark.parametrize(
    ("decision", "question", "expected_message"),
    [
        (
            LLMPlannerDecision(
                decision="clarification_required",
                intent="role_recommendations",
                team_name="Leverkusen",
                clarification_message="Which position?",
            ),
            "Find players who fit Team A.",
            "Which position?",
        ),
        (
            LLMPlannerDecision(
                decision="clarification_required",
                intent="role_recommendations",
                position_group="MID",
                clarification_message="Which team?",
            ),
            "Find midfielders who fit.",
            "Which team?",
        ),
    ],
)
def test_role_recommendation_missing_required_fields_still_clarifies(
    decision: LLMPlannerDecision,
    question: str,
    expected_message: str,
) -> None:
    plan = build_scout_plan(decision, question=question)

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert plan.clarification_message == expected_message
    assert plan.calls == []


def test_role_recommendation_recovery_does_not_guess_an_ambiguous_team(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_recommendations",
            team_name="FC",
            position_group="MID",
            clarification_message="Please provide more constraints.",
        )
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.decision is PlannerDecision.READY
    assert prepared.status is PlanPreparationStatus.CLARIFICATION_REQUIRED
    assert prepared.normalized_plan is None
    assert len(prepared.team_resolutions) == 1
    assert len(prepared.team_resolutions[0].candidates) == 2


def test_direct_player_lookup_strips_command_words_from_search_query() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_search",
            search_query="Find xHaKa.",
            sort_order="desc",
        ),
        question="Find xHaKa.",
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.calls[0].arguments.query == "xHaKa"


def test_explicit_reliable_position_search_recovers_filters() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="leaderboard",
            clarification_message="Which metric should be ranked?",
        ),
        question="Find reliable defenders with at least 100 passes.",
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind.value == "player_search"
    assert [call.name for call in plan.calls] == [ToolName.SEARCH_PLAYERS]
    assert plan.calls[0].arguments.position_group.value == "DEF"
    assert plan.calls[0].arguments.min_pass_attempts == 100
    assert plan.calls[0].arguments.query is None


def test_reflexive_comparison_clarifies_without_resolving_literal_himself() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_name="Xhaka",
            secondary_player_name="himself",
        ),
        question="Compare Xhaka with himself.",
    )

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert plan.clarification_message == "Comparison requires two distinct players."
    assert [call.name for call in plan.calls] == [ToolName.SEARCH_PLAYERS]
    assert plan.calls[0].arguments.query == "Xhaka"


def test_archetype_question_routes_to_intelligence_profile() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            primary_player_name="Xhaka",
            clarification_message="Which team?",
        ),
        question="What playing-style archetype is Xhaka?",
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind.value == "player_profile"
    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_PLAYER_DOSSIER,
    ]
    assert plan.calls[-1].arguments.sections[0].value == "intelligence"


@pytest.mark.parametrize(
    "question",
    (
        "Find a goalkeeper similar to Xhaka.",
        "Which similar player is guaranteed to be as good as Xhaka?",
    ),
)
def test_similarity_request_uses_governed_similarity_path_despite_provider_rejection(
    question: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="unsupported",
            intent="similar_players",
            primary_player_name="Xhaka",
            position_group=("GK" if "goalkeeper" in question else "none"),
            unsupported_reason="The provider declined the request.",
        ),
        question=question,
    )

    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_SIMILAR_PLAYERS,
    ]


def test_ready_similarity_advisory_in_unsupported_field_is_preserved_as_limitation() -> None:
    decision = LLMPlannerDecision.model_validate(
        {
            "decision": "ready",
            "intent": "similar_players",
            "primary_player_name": "Xhaka",
            "unsupported_reason": "Similarity does not guarantee equal performance.",
        }
    )

    plan = build_scout_plan(
        decision,
        question="Which similar player is guaranteed to be as good as Xhaka?",
    )

    assert decision.unsupported_reason == ""
    assert decision.limitation == "Similarity does not guarantee equal performance."
    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_SIMILAR_PLAYERS,
    ]
    assert plan.limitations == ["Similarity does not guarantee equal performance."]


@pytest.mark.parametrize("advisory_field", ("unsupported_reason", "clarification_message"))
def test_ready_similarity_non_guarantee_advisory_is_recovered_before_root_validation(
    advisory_field: str,
) -> None:
    decision = LLMPlannerDecision.model_validate(
        {
            "decision": "ready",
            "intent": "similar_players",
            "primary_player_name": "Xhaka",
            advisory_field: (
                "Similarity cannot guarantee the same quality or future performance."
            ),
        }
    )

    assert decision.decision is PlannerDecision.READY
    assert decision.unsupported_reason == ""
    assert decision.clarification_message == ""
    assert decision.limitation == (
        "Similarity cannot guarantee the same quality or future performance."
    )


def test_true_unsupported_similarity_decision_remains_unsupported() -> None:
    decision = LLMPlannerDecision.model_validate(
        {
            "decision": "unsupported",
            "intent": "similar_players",
            "primary_player_name": "Xhaka",
            "unsupported_reason": "Cross-position similarity is not supported.",
        }
    )

    assert decision.decision is PlannerDecision.UNSUPPORTED
    assert decision.unsupported_reason == "Cross-position similarity is not supported."
    assert decision.limitation == ""


def test_true_similarity_clarification_decision_remains_clarification() -> None:
    decision = LLMPlannerDecision.model_validate(
        {
            "decision": "clarification_required",
            "intent": "similar_players",
            "clarification_message": "Which player should FootyScout analyze?",
        }
    )

    assert decision.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert decision.clarification_message == "Which player should FootyScout analyze?"
    assert decision.limitation == ""


@pytest.mark.parametrize(
    "payload",
    (
        {
            "decision": "ready",
            "intent": "similar_players",
            "primary_player_name": "Xhaka",
            "unsupported_reason": "This request is outside FootyScout's scope.",
        },
        {
            "decision": "ready",
            "intent": "similar_players",
            "unsupported_reason": "Similarity cannot guarantee equal performance.",
        },
        {
            "decision": "ready",
            "intent": "player_profile",
            "primary_player_name": "Xhaka",
            "unsupported_reason": "Profiles cannot guarantee future performance.",
        },
    ),
)
def test_malformed_ready_decision_is_still_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LLMPlannerDecision.model_validate(payload)


def test_unsupported_decision_still_requires_unsupported_reason() -> None:
    with pytest.raises(ValidationError, match="Unsupported decisions require a reason"):
        LLMPlannerDecision.model_validate(
            {"decision": "unsupported", "intent": "similar_players"}
        )


def test_pressure_passing_profile_uses_passing_section_only() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_profile",
            primary_player_name="Xhaka",
            requested_sections=["passing", "intelligence"],
        ),
        question="How has Xhaka performed under pressure as a passer?",
    )

    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_PLAYER_DOSSIER,
    ]
    assert [section.value for section in plan.calls[-1].arguments.sections] == [
        "passing"
    ]


def test_cross_position_similarity_remains_unsupported_without_subject_clarification() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="similar_players",
            clarification_message="Which player?",
        ),
        question="Find similar players regardless of position.",
    )

    assert plan.decision is PlannerDecision.UNSUPPORTED
    assert "same position group" in (plan.unsupported_reason or "")
    assert plan.calls == []


@pytest.mark.parametrize(
    ("question", "position"),
    (
        ("Describe Leverkusen's defender role.", "DEF"),
        ("Describe Leverkusen's forward role.", "FWD"),
    ),
)
def test_team_role_description_routes_to_team_intelligence(
    question: str,
    position: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_recommendations",
            team_name="Leverkusen",
            position_group=position,
            clarification_message="Please clarify the request.",
        ),
        question=question,
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind.value == "team_analysis"
    assert [call.name for call in plan.calls] == [ToolName.GET_TEAM_INTELLIGENCE]
    assert plan.calls[0].arguments.position_group.value == position


@pytest.mark.parametrize(
    "question",
    (
        "Find progressive defenders with strong role fit for Leverkusen.",
        "List defenders who fit Leverkusen.",
        "Recommend defender candidates with Role Fit for Leverkusen.",
    ),
)
def test_plural_candidate_role_fit_language_routes_to_recommendations(
    question: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            primary_player_name="progressive defenders",
            team_name="Leverkusen",
            position_group="DEF",
            clarification_message="Which player?",
        ),
        question=question,
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind.value == "role_recommendations"
    assert [call.name for call in plan.calls] == [ToolName.GET_ROLE_RECOMMENDATIONS]
    assert plan.calls[0].arguments.position_group.value == "DEF"


def test_unsupported_role_fit_team_is_resolved_before_missing_player(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            team_name="Alpha FC",
            clarification_message="Which player?",
        ),
        question="Calculate Role Fit for Alpha FC.",
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == [ToolName.GET_ROLE_FIT]
    assert prepared.status is PlanPreparationStatus.NOT_FOUND
    assert prepared.reason_code is PlanPreparationReasonCode.UNSUPPORTED_TEAM_CAPABILITY
    assert prepared.player_resolutions == []
    assert len(prepared.team_resolutions) == 1
    assert "does not have qualified production" in (prepared.error_message or "")


def test_team_only_role_fit_does_not_treat_team_name_as_a_player(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            primary_player_name="Alpha FC",
            search_query="Alpha FC",
            team_name="Alpha FC",
            clarification_message="Which player?",
        ),
        question="Calculate Role Fit for Alpha FC.",
    )
    prepared = prepare_plan(ai_session, plan)

    assert prepared.status is PlanPreparationStatus.NOT_FOUND
    assert prepared.player_resolutions == []
    assert len(prepared.team_resolutions) == 1


def test_supported_role_fit_team_still_clarifies_for_missing_player(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            team_name="Leverkusen",
            clarification_message="Which player?",
        ),
        question="Calculate Role Fit for Leverkusen.",
    )
    prepared = prepare_plan(ai_session, plan)

    assert prepared.status is PlanPreparationStatus.CLARIFICATION_REQUIRED
    assert prepared.clarification_message == "Which player should FootyScout analyze?"
    assert prepared.player_resolutions == []
    assert len(prepared.team_resolutions) == 1


def test_profile_split_methodology_expands_to_profiles_and_xpass() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="xpass",
        ),
        question="Are player profiles based only on the test set?",
    )

    assert [call.arguments.topic.value for call in plan.calls] == [
        "player_profiles",
        "xpass",
    ]


def test_profile_oof_methodology_expands_without_broadening_generic_xpass() -> None:
    expanded = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="player_profiles",
        ),
        question="Are profiles based on OOF predictions?",
    )
    generic = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="xpass",
        ),
        question="How does xPass work?",
    )

    assert [call.arguments.topic.value for call in expanded.calls] == [
        "player_profiles",
        "xpass",
    ]
    assert [call.arguments.topic.value for call in generic.calls] == ["xpass"]


def test_percentile_fabrication_request_routes_to_governed_methodology() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="unsupported",
            intent="methodology",
            unsupported_reason="Do not fabricate values.",
        ),
        question="Invent missing percentile values so every player can be compared.",
    )

    assert plan.decision is PlannerDecision.READY
    assert [call.name for call in plan.calls] == [ToolName.GET_METHODOLOGY]
    assert plan.calls[0].arguments.topic.value == "percentiles"


def test_arbitrary_statistic_fabrication_remains_unsupported() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="unsupported",
            intent="methodology",
            unsupported_reason="Do not fabricate statistics.",
        ),
        question="Invent goals for a player with no data.",
    )

    assert plan.decision is PlannerDecision.UNSUPPORTED
    assert plan.calls == []


def test_role_recommendation_recovery_requires_one_unambiguous_tool_path() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_recommendations",
            requested_analyses=["leaderboard"],
            team_name="Leverkusen",
            position_group="MID",
            clarification_message="Please clarify the combined analysis.",
        )
    )

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert plan.calls == []


def test_entity_dependent_intent_recovers_provider_search_query_for_resolution(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="player_profile",
            search_query="Alice Playmaker",
            clarification_message="The provider did not select a player.",
        )
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.decision is PlannerDecision.READY
    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[-1].arguments["player_id"] == 1


@pytest.mark.parametrize(
    ("query", "expected_status", "candidate_count"),
    [
        ("Alice Playmaker", PlanPreparationStatus.READY, 1),
        ("Alice", PlanPreparationStatus.CLARIFICATION_REQUIRED, 2),
        ("No Such Player", PlanPreparationStatus.NOT_FOUND, 0),
    ],
)
def test_product_search_results_authoritatively_determine_entity_resolution(
    ai_session: Session,
    query: str,
    expected_status: PlanPreparationStatus,
    candidate_count: int,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="player_profile",
            primary_player_name=query,
            clarification_message="The provider considers this name ambiguous.",
        )
    )
    prepared = prepare_plan(ai_session, plan)

    assert prepared.status is expected_status
    assert len(prepared.player_resolutions) == 1
    assert len(prepared.player_resolutions[0].candidates) == candidate_count
    if expected_status is not PlanPreparationStatus.READY:
        assert prepared.normalized_plan is None


def test_missing_non_entity_requirement_still_clarifies_before_search(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            primary_player_name="Alice Playmaker",
            clarification_message="Which team should FootyScout analyze?",
        )
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert prepared.status is PlanPreparationStatus.CLARIFICATION_REQUIRED
    assert prepared.player_resolutions == []


@pytest.mark.parametrize(
    ("wording", "expected"),
    [
        ("midfielder", "MID"),
        ("MID", "MID"),
        ("defender", "DEF"),
        ("DEF", "DEF"),
        ("forward", "FWD"),
        ("FWD", "FWD"),
        ("goalkeeper", "GK"),
        ("GK", "GK"),
    ],
)
def test_explicit_position_constraint_survives_planning(
    wording: str,
    expected: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_search",
            search_query="player",
        ),
        question=f"Find a {wording}.",
    )

    assert plan.calls[0].arguments.position_group.value == expected


def test_unrelated_forward_language_does_not_invent_a_position_filter() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="team_analysis",
            team_id=904,
        ),
        question="Analyze forward passing distance for this team.",
    )

    assert plan.calls[0].arguments.position_group is None


def test_missing_entity_and_duplicate_comparison_become_clarifications() -> None:
    missing = build_scout_plan(
        LLMPlannerDecision(decision="ready", intent="player_profile")
    )
    duplicate = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_id=1,
            secondary_player_id=1,
        )
    )

    assert missing.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert duplicate.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert duplicate.calls == []


def test_stable_role_fit_ids_override_redundant_provider_clarification(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            primary_player_id=4,
            team_id=904,
            clarification_message="Please confirm the player and team.",
        )
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.decision is PlannerDecision.READY
    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[0].arguments == {
        "player_id": 4,
        "target_team_id": 904,
    }


@pytest.mark.parametrize(
    "decision",
    [
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            primary_player_id=4,
            team_name="Leverkusen",
            clarification_message="Please confirm the team.",
        ),
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            primary_player_name="Dana Midfielder",
            team_id=904,
            clarification_message="Please confirm the player.",
        ),
    ],
)
def test_stable_id_and_resolvable_name_continue_to_deterministic_resolution(
    ai_session: Session,
    decision: LLMPlannerDecision,
) -> None:
    prepared = prepare_plan(ai_session, build_scout_plan(decision))

    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[-1].arguments == {
        "player_id": 4,
        "target_team_id": 904,
    }


@pytest.mark.parametrize(
    "reference",
    [
        "a defender",
        "an external midfielder",
        "another striker",
        "a candidate player",
        "external goalkeeper",
    ],
)
def test_generic_role_reference_is_a_placeholder_not_a_literal_name(
    reference: str,
    ai_session: Session,
) -> None:
    assert is_placeholder_player_reference(reference)
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_comparison",
            primary_player_name="Xhaka",
            secondary_player_name=reference,
        )
    )

    assert plan.decision is PlannerDecision.CLARIFICATION_REQUIRED
    assert all(
        getattr(call.arguments, "query", None) != reference for call in plan.calls
    )
    prepared = prepare_plan(ai_session, plan)
    assert prepared.status is PlanPreparationStatus.CLARIFICATION_REQUIRED
    assert prepared.player_resolutions == []


@pytest.mark.parametrize(
    "name",
    ["Defender Silva", "Midfielder Jones", "Player Forward"],
)
def test_position_like_token_inside_a_real_name_is_not_a_placeholder(name: str) -> None:
    assert not is_placeholder_player_reference(name)


def test_candidate_position_filter_does_not_leak_into_similarity_subject_lookup(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="similar_players",
            primary_player_name="Dana Midfielder",
            position_group="GK",
        ),
        question="Find a goalkeeper similar to Dana Midfielder.",
    )

    search = plan.calls[0]
    assert search.name is ToolName.SEARCH_PLAYERS
    assert search.arguments.query == "Dana Midfielder"
    assert search.arguments.position_group is None
    assert plan.intent.kind.value == "similar_players"
    assert plan.intent.candidate_position_group.value == "GK"
    assert plan.calls[-1].arguments.candidate_position_group.value == "GK"
    prepared = prepare_plan(ai_session, plan)
    assert prepared.status is PlanPreparationStatus.NOT_FOUND
    assert prepared.normalized_plan is None
    assert prepared.player_resolutions[0].player_id == 4
    assert prepared.player_resolutions[0].player is not None
    assert prepared.player_resolutions[0].player.position_group == "MID"
    assert prepared.reason_code is PlanPreparationReasonCode.CANDIDATE_POSITION_INCOMPATIBLE
    assert "only within the subject's MID position group" in (
        prepared.error_message or ""
    )


def test_xhaka_goalkeeper_candidate_filter_is_not_a_xhaka_identity_filter() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="similar_players",
            primary_player_name="Xhaka",
            position_group="GK",
        ),
        question="Find a goalkeeper similar to Xhaka.",
    )

    assert plan.calls[0].arguments.query == "Xhaka"
    assert plan.calls[0].arguments.position_group is None
    assert plan.calls[-1].arguments.candidate_position_group.value == "GK"


def test_compatible_same_position_similarity_normalizes_normally(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="similar_players",
            primary_player_name="Dana Midfielder",
            position_group="MID",
        ),
        question="Find a midfielder similar to Dana Midfielder.",
    )

    prepared = prepare_plan(ai_session, plan)

    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.reason_code is None
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[-1].name is ToolName.GET_SIMILAR_PLAYERS
    assert prepared.normalized_plan.calls[-1].arguments["player_id"] == 4
    execution = execute_plan(ai_session, prepared.normalized_plan, run_id="similarity-test")
    assert execution.status is PlanExecutionStatus.COMPLETED
    assert execution.evidence.records[-1].tool_name is ToolName.GET_SIMILAR_PLAYERS


def test_explicit_subject_position_may_filter_only_the_subject_lookup() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_profile",
            primary_player_name="Dana Midfielder",
            position_group="MID",
        ),
        question="Show me midfielder Dana Midfielder's profile.",
    )

    assert plan.calls[0].arguments.position_group.value == "MID"


def test_ambiguous_player_and_unsupported_team_remain_normalizer_decisions(
    ai_session: Session,
) -> None:
    ambiguous = prepare_plan(
        ai_session,
        build_scout_plan(
            LLMPlannerDecision(
                decision="ready",
                intent="player_profile",
                primary_player_name="Alice",
            )
        ),
    )
    unsupported_team = prepare_plan(
        ai_session,
        build_scout_plan(
            LLMPlannerDecision(
                decision="ready",
                intent="team_analysis",
                team_name="Alpha FC",
            )
        ),
    )

    assert ambiguous.status is PlanPreparationStatus.CLARIFICATION_REQUIRED
    assert unsupported_team.status is PlanPreparationStatus.NOT_FOUND


def test_player_search_allows_ambiguous_partial_name_but_profile_requires_one_player(
    ai_session: Session,
) -> None:
    search_plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_search",
            search_query="Williams",
        )
    )
    profile_plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_profile",
            primary_player_name="Williams",
        )
    )

    search = prepare_plan(ai_session, search_plan)
    profile = prepare_plan(ai_session, profile_plan)

    assert search.status is PlanPreparationStatus.READY
    assert search.normalized_plan is not None
    assert search.normalized_plan.calls[0].arguments["query"] == "Williams"
    assert search.normalized_plan.calls[0].arguments["limit"] == 10
    assert search.player_resolutions == []
    assert search_plan.intent.kind.value == "player_search"
    assert profile.status is PlanPreparationStatus.CLARIFICATION_REQUIRED
    assert profile_plan.intent.kind.value == "player_profile"
    assert len(profile.player_resolutions[0].candidates) == 2


@pytest.mark.parametrize(
    ("provider_limit", "expected_limit"),
    [(0, 10), (5, 5), (999, PLAYER_SEARCH_MAX_RESULTS)],
)
def test_player_search_uses_deterministic_default_explicit_and_clamped_limits(
    provider_limit: int,
    expected_limit: int,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="player_search",
            search_query="xHaKa",
            limit=provider_limit,
        )
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind.value == "player_search"
    assert plan.calls[0].arguments.query == "xHaKa"
    assert plan.calls[0].arguments.limit == expected_limit


def test_goalkeeper_role_fit_is_rejected_by_existing_entity_policy(ai_session: Session) -> None:
    result = prepare_plan(
        ai_session,
        build_scout_plan(
            LLMPlannerDecision(
                decision="ready",
                intent="role_fit",
                primary_player_id=6,
                team_id=904,
            )
        ),
    )

    assert result.status is PlanPreparationStatus.UNSUPPORTED


def test_limits_are_clamped_and_id_sentinels_take_precedence() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="similar_players",
            primary_player_id=1,
            primary_player_name="ignored",
            limit=9999,
        )
    )

    call = plan.calls[0]
    assert call.name is ToolName.GET_SIMILAR_PLAYERS
    assert call.arguments.player.player_id == 1
    assert call.arguments.player.player_name is None
    assert call.arguments.limit == SIMILAR_PLAYERS_MAX_RESULTS


def test_invalid_enums_and_irrelevant_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMPlannerDecision.model_validate(
            {"decision": "ready", "intent": "invented_intent"}
        )
    with pytest.raises(ValidationError):
        LLMPlannerDecision.model_validate(
            {"decision": "ready", "intent": "player_search", "tool_name": "run_sql"}
        )


@pytest.mark.parametrize(
    "question",
    [
        "Find the top 200 players by pass attempts.",
        "Which players have the most pass attempts?",
        "Sort players by pass attempts.",
        "Rank players by passes attempted.",
    ],
)
def test_direct_pass_attempt_rankings_use_player_search(question: str) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="leaderboard",
            leaderboard_metric="completion_above_expected_pp",
            limit=25,
        ),
        question=question,
    )

    assert plan.intent.kind.value == "player_search"
    assert [call.name for call in plan.calls] == [ToolName.SEARCH_PLAYERS]
    assert plan.calls[0].arguments.sort_by.value == "pass_attempts"
    assert plan.calls[0].arguments.sort_order.value == "desc"
    assert plan.calls[0].arguments.limit == 20


@pytest.mark.parametrize(
    ("metric", "question"),
    [
        (
            "completion_above_expected_pp",
            "Who has the highest completion above expected?",
        ),
        (
            "progressive_above_expected_pp",
            "Show the progressive passing leaderboard.",
        ),
    ],
)
def test_derived_metric_rankings_remain_leaderboards(
    metric: str,
    question: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="leaderboard",
            leaderboard_metric=metric,
        ),
        question=question,
    )

    assert plan.intent.kind.value == "leaderboard"
    assert plan.calls[0].name is ToolName.GET_LEADERBOARD
    assert plan.calls[0].arguments.metric.value == metric


@pytest.mark.parametrize(
    ("question", "expected_limit"),
    [
        ("Show Xhaka's most similar player.", 1),
        ("Show Xhaka's nearest neighbor.", 1),
        ("Show Xhaka's closest style dimensions to his nearest neighbor.", 1),
        ("Show three similar players to Xhaka.", 3),
        ("Show 3 similar players to Xhaka.", 3),
        ("Show Xhaka's similar players.", 6),
        ("Show 200 similar players to Xhaka.", SIMILAR_PLAYERS_MAX_RESULTS),
    ],
)
def test_similarity_explicit_count_overrides_default_within_bounds(
    question: str,
    expected_limit: int,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="similar_players",
            primary_player_name="Xhaka",
        ),
        question=question,
    )

    similar = next(
        call for call in plan.calls if call.name is ToolName.GET_SIMILAR_PLAYERS
    )
    assert similar.arguments.limit == expected_limit


def test_generic_role_fit_methodology_remains_methodology() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="role_fit",
        ),
        question="How does Role Fit work?",
    )

    assert plan.intent.kind.value == "methodology"
    assert [call.name for call in plan.calls] == [ToolName.GET_METHODOLOGY]


def test_named_player_role_fit_value_remains_operational_role_fit() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="role_fit",
            primary_player_name="Dana Midfielder",
            team_name="Leverkusen",
        ),
        question="What is Dana Midfielder's Role Fit for Leverkusen?",
    )

    assert plan.intent.kind.value == "role_fit"
    assert ToolName.GET_ROLE_FIT in [call.name for call in plan.calls]


def test_player_specific_current_team_role_fit_uses_subject_team_and_scope(
    ai_session: Session,
) -> None:
    player = ai_session.get(Player, 1)
    assert player is not None
    player.team_id = 904
    player.team_name = "Bayer Leverkusen"
    ai_session.flush()
    question = (
        "Does Alice Playmaker's current-team Role Fit include her own events in "
        "the target role?"
    )
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="role_fit",
        ),
        question=question,
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.intent.kind.value == "role_fit"
    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_ROLE_FIT,
    ]
    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.normalized_plan is not None
    assert prepared.normalized_plan.calls[-1].arguments == {
        "player_id": 1,
        "target_team_id": 904,
    }
    executed = execute_plan(ai_session, prepared.normalized_plan, run_id="own-events")
    role_fit = executed.evidence.records[-1].result
    assert role_fit is not None
    assert role_fit["calculation_scope"] == "leave_self_out_target_role"


def test_player_specific_role_fit_does_not_invent_when_subject_is_missing(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="role_fit",
        ),
        question=(
            "Does Unknown Footballer's current-team Role Fit include his own events?"
        ),
    )
    prepared = prepare_plan(ai_session, plan)

    assert plan.intent.kind.value == "role_fit"
    assert prepared.status is PlanPreparationStatus.NOT_FOUND


@pytest.mark.parametrize(
    ("question", "team_name", "position"),
    [
        ("Which players fit Leverkusen's midfield role?", "Leverkusen", "MID"),
        ("Rank candidates for Leverkusen at DEF.", "Leverkusen", "DEF"),
        (
            "Rank Leverkusen midfield candidates by the biggest Role Fit number.",
            "",
            "none",
        ),
    ],
)
def test_candidate_role_fit_language_uses_recommendation_contract(
    question: str,
    team_name: str,
    position: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            team_name=team_name,
            position_group=position,
            clarification_message="Which player should FootyScout analyze?",
        ),
        question=question,
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind.value == "role_recommendations"
    assert [call.name for call in plan.calls] == [ToolName.GET_ROLE_RECOMMENDATIONS]
    call = plan.calls[0]
    assert call.arguments.team.team_name == "Leverkusen"
    assert call.arguments.position_group.value in {"MID", "DEF"}


def test_single_named_player_role_fit_is_not_changed_to_recommendations() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="role_fit",
            primary_player_name="Dana Midfielder",
            team_name="Leverkusen",
        ),
        question="How well does Dana Midfielder fit Leverkusen?",
    )

    assert plan.intent.kind.value == "role_fit"
    assert ToolName.GET_ROLE_FIT in [call.name for call in plan.calls]


def test_mixed_stable_role_fit_and_transfer_overrides_provider_unsupported() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="unsupported",
            intent="role_fit",
            primary_player_id=3500,
            team_name="Leverkusen",
            unsupported_reason="Transfer reporting is unsupported.",
        ),
        question=(
            "How does player 3500 fit Leverkusen, and what is the latest "
            "transfer reporting about him?"
        ),
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind is IntentKind.ROLE_FIT
    assert plan.intent.player.player_id == 3500
    assert plan.intent.target_team.team_name == "Leverkusen"
    assert [call.name for call in plan.calls] == [ToolName.GET_ROLE_FIT]


def test_mixed_named_role_fit_and_transfer_preserves_named_subject() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="unsupported",
            intent="player_profile",
            unsupported_reason="Transfer reporting is unsupported.",
        ),
        question=(
            "How does Dana Midfielder fit Leverkusen, and what is the latest "
            "transfer report about her?"
        ),
    )

    assert plan.decision is PlannerDecision.READY
    assert plan.intent.kind is IntentKind.ROLE_FIT
    assert plan.intent.player.player_name == "Dana Midfielder"
    assert plan.intent.target_team.team_name == "Leverkusen"
    assert [call.name for call in plan.calls] == [
        ToolName.SEARCH_PLAYERS,
        ToolName.GET_ROLE_FIT,
    ]


def test_candidate_ranking_executes_without_requesting_a_player(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="clarification_required",
            intent="role_fit",
            team_name="Leverkusen",
            clarification_message="Which player should FootyScout analyze?",
        ),
        question="Which players fit Leverkusen's midfield role?",
    )
    prepared = prepare_plan(ai_session, plan)

    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.player_resolutions == []
    assert prepared.normalized_plan is not None
    executed = execute_plan(
        ai_session,
        prepared.normalized_plan,
        run_id="candidate-ranking",
    )
    assert executed.status is PlanExecutionStatus.COMPLETED
    assert executed.evidence.records[0].tool_name is ToolName.GET_ROLE_RECOMMENDATIONS


@pytest.mark.parametrize(
    "question",
    [
        "Are player profiles based only on the test set?",
        "Do player profiles use OOF predictions?",
        "Are profiles built from grouped-CV predictions?",
    ],
)
def test_profile_evaluation_split_questions_use_xpass_methodology(
    question: str,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="player_profiles",
        ),
        question=question,
    )

    assert plan.intent.kind.value == "methodology"
    assert [call.arguments.topic.value for call in plan.calls] == [
        "player_profiles",
        "xpass",
    ]


def test_generic_profile_composition_remains_player_profiles_methodology() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="player_profiles",
        ),
        question="What is included in player profiles?",
    )

    assert [call.arguments.topic.value for call in plan.calls] == ["player_profiles"]


def test_profile_composition_and_split_question_retrieves_both_topics() -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="player_profiles",
        ),
        question=(
            "What are player profiles built from, and are they based only on the "
            "test set?"
        ),
    )

    assert [call.arguments.topic.value for call in plan.calls] == [
        "player_profiles",
        "xpass",
    ]


def test_profile_composition_and_split_topics_execute_as_separate_methodology_evidence(
    ai_session: Session,
) -> None:
    plan = build_scout_plan(
        LLMPlannerDecision(
            decision="ready",
            intent="methodology",
            methodology_topic="player_profiles",
        ),
        question=(
            "What are player profiles built from, and do they use out-of-fold "
            "predictions?"
        ),
    )
    prepared = prepare_plan(ai_session, plan)

    assert prepared.status is PlanPreparationStatus.READY
    assert prepared.normalized_plan is not None
    executed = execute_plan(
        ai_session,
        prepared.normalized_plan,
        run_id="profile-methodology",
    )
    assert executed.status is PlanExecutionStatus.COMPLETED
    assert [record.methodology_topic for record in executed.evidence.records] == [
        "player_profiles",
        "xpass",
    ]
