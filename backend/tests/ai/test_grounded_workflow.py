"""Offline contracts for the bounded grounded-response workflow."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.ai.context import ContextStatus, select_methodology_topics
from app.ai.grounding import EvidenceLedger
from app.ai.methodology import CuratedMethodologyService, KnowledgeTopic, MethodologyDocument
from app.ai.observability import InMemoryTraceSink
from app.ai.plan_builder import build_scout_plan
from app.ai.planner import PlannerResult, PlannerStatus
from app.ai.prompts import PLANNER_PROMPT_VERSION
from app.ai.provider import PlannerUsage
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.run import UsageMetadata
from app.ai.schemas import IntentKind, ProductionStatus, ScoutPlan, ToolName
from app.ai.synthesis import (
    GroundedAnswerStatus,
    LLMGroundedAnswer,
    SynthesisResult,
    validate_answer_citations,
)
from app.ai.workflow import AIScoutWorkflow, _validation_failure_message
from app.models import Player


class FakePlanner:
    provider = "fake-planner-provider"
    model = "fake-planner-model"

    def __init__(
        self,
        plan: ScoutPlan,
        *,
        provider_decision: LLMPlannerDecision | None = None,
        methodology_guard_applied: bool = False,
    ) -> None:
        self.plan_value = plan
        self.provider_decision = provider_decision
        self.methodology_guard_applied = methodology_guard_applied

    def plan(self, _question: str) -> PlannerResult:
        return PlannerResult(
            status=PlannerStatus.SUCCESS,
            provider=self.provider,
            model=self.model,
            prompt_version=PLANNER_PROMPT_VERSION,
            provider_decision=self.provider_decision,
            plan=self.plan_value,
            usage=PlannerUsage(input_tokens=11, output_tokens=7, total_tokens=18),
            methodology_guard_applied=self.methodology_guard_applied,
            provider_attempts=1,
        )


class FakeSynthesizer:
    provider = "fake-synthesis-provider"
    model = "fake-synthesis-model"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, context):
        self.calls += 1
        evidence_ids = context.evidence_ids
        citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
        return SynthesisResult(
            answer=LLMGroundedAnswer(
                answer_markdown=f"Grounded response {citations}",
                evidence_ids=evidence_ids,
                status=GroundedAnswerStatus.ANSWERED,
            ),
            provider=self.provider,
            model=self.model,
            usage=UsageMetadata(input_tokens=20, output_tokens=8, total_tokens=28),
        )


class MisclassifiedGroundedSynthesizer(FakeSynthesizer):
    """Simulate a provider returning grounded content with a stale answer status."""

    def synthesize(self, context):
        self.calls += 1
        evidence_ids = context.evidence_ids
        citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
        return SynthesisResult(
            answer=LLMGroundedAnswer(
                answer_markdown=f"Grounded player analysis {citations}",
                evidence_ids=evidence_ids,
                status=GroundedAnswerStatus.CLARIFICATION_REQUIRED,
            ),
            provider=self.provider,
            model=self.model,
        )


class FakeMethodologyService:
    def get(self, topic: KnowledgeTopic) -> MethodologyDocument:
        return MethodologyDocument(
            topic=topic,
            section="Test section",
            source_id=f"{topic.value}:test",
            source_path="docs/test.md",
            content=f"Governed {topic.value} methodology.",
            production_status=ProductionStatus.PRODUCTION,
            limitations=("Test limitation.",),
        )


def _plan(payload: dict[str, object]) -> ScoutPlan:
    return ScoutPlan.model_validate(payload)


def _workflow(
    session: Session,
    plan: ScoutPlan,
    synthesizer: FakeSynthesizer,
) -> AIScoutWorkflow:
    return AIScoutWorkflow(
        session=session,
        planner=FakePlanner(plan),
        synthesizer=synthesizer,
        methodology_service=FakeMethodologyService(),
    )


def test_graph_routes_analytics_only_and_records_trace(ai_session: Session) -> None:
    synthesizer = FakeSynthesizer()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice", "limit": 5},
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Alice", "limit": 5},
                    }
                ],
            }
        ),
        synthesizer,
    ).run("Find Alice")

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert result.context is not None
    assert len(result.context.analytics_evidence) == 1
    assert result.context.methodology_evidence == ()
    assert synthesizer.calls == 1
    assert result.diagnostics.provider == "fake-planner-provider"
    assert result.diagnostics.synthesis_model == "fake-synthesis-model"
    assert result.diagnostics.synthesis_prompt_version == "grounded-synthesis-v21"
    assert result.diagnostics.planner_usage.total_tokens == 18
    assert result.diagnostics.synthesis_usage.total_tokens == 28
    assert result.diagnostics.planner_provider_attempts == 1
    assert result.diagnostics.synthesis_provider_attempts == 1
    assert result.diagnostics.tools_executed == ("search_players",)
    assert result.diagnostics.methodology_sources_derived == ()
    assert result.diagnostics.pre_normalization_tool_names == ("search_players",)
    assert result.diagnostics.post_normalization_tool_names == ("search_players",)
    assert result.diagnostics.terminal_branch_reason == "answered"
    assert {
        "planner",
        "preparation",
        "analytics_execution",
        "methodology_retrieval",
        "context_assembly",
        "synthesis",
        "answer_validation",
    }.issubset(result.diagnostics.stage_timings_ms)


def test_governed_synthesis_records_no_synthesis_provider_call(
    ai_session: Session,
) -> None:
    synthesizer = FakeSynthesizer()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {"kind": "methodology", "topic": "percentiles"},
                "calls": [
                    {
                        "name": "get_methodology",
                        "arguments": {"topic": "percentiles"},
                    }
                ],
            }
        ),
        synthesizer,
    ).run("Can you impute missing percentile values?")

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert synthesizer.calls == 0
    assert result.diagnostics.planner_provider_attempts == 1
    assert result.diagnostics.synthesis_provider_attempts == 0
    assert result.diagnostics.synthesis_usage is None
    assert "synthesis" in result.diagnostics.stage_timings_ms


def test_planner_prose_never_becomes_a_public_context_limitation(
    ai_session: Session,
) -> None:
    planner_note = (
        "Will produce a passing-focused profile. Specify a season or additional "
        "sections if you want them included."
    )
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice", "limit": 5},
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Alice", "limit": 5},
                    }
                ],
                "limitations": [planner_note],
            }
        ),
        FakeSynthesizer(),
    ).run("Find Alice")

    assert result.context is not None
    assert planner_note not in result.context.limitations
    assert planner_note not in result.answer.limitations


def test_workflow_exposes_safe_methodology_routing_diagnostics(
    ai_session: Session,
) -> None:
    provider_decision = LLMPlannerDecision(
        decision="clarification_required",
        intent="role_fit",
        requested_analyses=["player_profile"],
        primary_player_name="provider player",
        team_name="provider team",
        clarification_message="Which player should FootyScout analyze?",
    )
    corrected_plan = _plan(
        {
            "decision": "ready",
            "intent": {"kind": "methodology", "topic": "role_fit"},
            "calls": [{"name": "get_methodology", "arguments": {"topic": "role_fit"}}],
        }
    )
    synthesizer = FakeSynthesizer()
    workflow = AIScoutWorkflow(
        session=ai_session,
        planner=FakePlanner(
            corrected_plan,
            provider_decision=provider_decision,
            methodology_guard_applied=True,
        ),
        synthesizer=synthesizer,
    )
    result = workflow.run("What does Role Fit mean?")

    diagnostics = result.diagnostics
    assert diagnostics.planner_primary_intent == "role_fit"
    assert diagnostics.planner_methodology_topic == "none"
    assert diagnostics.planner_player_reference == "provider player"
    assert diagnostics.planner_team_reference == "provider team"
    assert diagnostics.planner_requested_analyses == ("player_profile",)
    assert diagnostics.methodology_guard_applied is True
    assert diagnostics.pre_normalization_tool_names == ("get_methodology",)
    assert diagnostics.post_normalization_tool_names == ("get_methodology",)
    assert diagnostics.terminal_branch_reason == "answered"


def test_methodology_only_question_uses_exact_topic(ai_session: Session) -> None:
    synthesizer = FakeSynthesizer()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {"kind": "methodology", "topic": "role_fit"},
                "calls": [
                    {"name": "get_methodology", "arguments": {"topic": "role_fit"}}
                ],
            }
        ),
        synthesizer,
    ).run("What does Role Fit mean?")

    assert result.context is not None
    assert result.context.analytics_evidence == ()
    assert len(result.context.methodology_evidence) == 1
    assert result.context.methodology_evidence[0].methodology_topic == "role_fit"
    assert result.context.status is ContextStatus.SUFFICIENT
    assert result.answer.methodology_sources == ("role_fit:primary",)
    assert result.diagnostics.methodology_sources_derived == ("role_fit:primary",)
    methodology_record = result.context.methodology_evidence[0]
    assert methodology_record.result is not None
    summary = methodology_record.result["summary"]
    assert "supported team's observed positional-role style" in summary
    assert "lower raw distance means closer resemblance" in summary
    assert "archetype" not in summary.casefold()
    assert "off-ball" not in summary.casefold()
    assert any(
        "Phase 1 uses no semantic retrieval" in note
        for note in methodology_record.internal_notes
    )
    assert all("Phase 1" not in limitation for limitation in result.answer.limitations)
    assert result.answer.limitations == (
        "Role Fit supports eligible outfield positions only.",
        "Role Fit does not predict transfer success or future performance.",
        "Role Fit is not lineup selection or a tactical guarantee.",
    )
    assert result.context.limitations == result.answer.limitations
    assert len(result.answer.limitations) == len(set(result.answer.limitations))
    forbidden = ("archetype", "off-ball", "tracking data", "training cohort", "injury")
    assert all(
        term not in limitation.casefold()
        for limitation in result.answer.limitations
        for term in forbidden
    )


def test_analytics_plus_methodology_selects_only_relevant_topics(ai_session: Session) -> None:
    synthesizer = FakeSynthesizer()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_profile",
                    "player": {"player_id": 1},
                    "sections": ["passing"],
                },
                "calls": [
                    {
                        "name": "get_player_dossier",
                        "arguments": {
                            "player": {"player_id": 1},
                            "sections": ["passing"],
                        },
                    }
                ],
            }
        ),
        synthesizer,
    ).run("Explain Alice's passing")

    assert result.context is not None
    assert {row.methodology_topic for row in result.context.methodology_evidence} == {
        "player_profiles",
        "xpass",
    }
    assert "xg" not in {row.methodology_topic for row in result.context.methodology_evidence}
    assert all(
        row.producer == "curated_methodology_service"
        and row.provenance["retrieval"] == "exact_topic"
        and row.methodology_sources
        for row in result.context.methodology_evidence
    )


def test_clarification_and_unsupported_branches_skip_synthesis(ai_session: Session) -> None:
    clarification_synth = FakeSynthesizer()
    clarification = _workflow(
        ai_session,
        _plan(
            {
                "decision": "clarification_required",
                "intent": {"kind": "similar_players", "player": {"player_name": "Smith"}},
                "clarification_message": "Which Smith do you mean?",
            }
        ),
        clarification_synth,
    ).run("Find players like Smith")
    assert clarification.answer.status is GroundedAnswerStatus.CLARIFICATION_REQUIRED
    assert clarification.answer.answer_markdown == "Which Smith do you mean?"
    assert clarification.answer.evidence_ids == ()
    assert clarification.answer.methodology_sources == ()
    assert clarification.answer.web_sources == ()
    assert clarification_synth.calls == 0

    unsupported_synth = FakeSynthesizer()
    unsupported = _workflow(
        ai_session,
        _plan(
            {
                "decision": "unsupported",
                "intent": {"kind": "player_search", "query": "Alice"},
                "unsupported_reason": "Future transfer success is not supported.",
            }
        ),
        unsupported_synth,
    ).run("Will Alice definitely succeed after a transfer?")
    assert unsupported.answer.status is GroundedAnswerStatus.UNSUPPORTED
    assert unsupported_synth.calls == 0


def test_role_fit_clarification_preserves_raw_and_governed_intents(
    ai_session: Session,
) -> None:
    provider_decision = LLMPlannerDecision(
        decision="clarification_required",
        intent="player_comparison",
        primary_player_name="Xhaka",
        secondary_player_name="external midfielder",
        clarification_message="Which external midfielder should FootyScout compare?",
    )
    plan = build_scout_plan(
        provider_decision,
        question="Compare Xhaka's Role Fit with an external midfielder.",
    )
    synthesizer = FakeSynthesizer()
    trace_sink = InMemoryTraceSink()
    workflow = AIScoutWorkflow(
        session=ai_session,
        planner=FakePlanner(plan, provider_decision=provider_decision),
        synthesizer=synthesizer,
        methodology_service=FakeMethodologyService(),
        trace_sink=trace_sink,
    )

    result = workflow.run("Compare Xhaka's Role Fit with an external midfielder.")

    assert result.answer.status is GroundedAnswerStatus.CLARIFICATION_REQUIRED
    assert result.diagnostics.planner_primary_intent == IntentKind.PLAYER_COMPARISON.value
    assert result.diagnostics.normalized_intent == IntentKind.ROLE_FIT.value
    assert result.diagnostics.pre_normalization_tool_names == (
        ToolName.SEARCH_PLAYERS.value,
    )
    assert result.diagnostics.post_normalization_tool_names == ()
    assert result.diagnostics.tools_executed == ()
    assert synthesizer.calls == 0
    assert len(trace_sink.traces) == 1
    assert trace_sink.traces[0].primary_intent == IntentKind.PLAYER_COMPARISON.value
    assert trace_sink.traces[0].normalized_intent == IntentKind.ROLE_FIT.value


def test_validated_grounded_answer_overrides_provider_clarification_status(
    ai_session: Session,
) -> None:
    synthesizer = MisclassifiedGroundedSynthesizer()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_profile",
                    "player": {"player_id": 1},
                    "sections": ["passing", "intelligence"],
                },
                "calls": [
                    {
                        "name": "get_player_dossier",
                        "arguments": {
                            "player": {"player_id": 1},
                            "sections": ["passing", "intelligence"],
                        },
                    }
                ],
            }
        ),
        synthesizer,
    ).run("Give me Alice Playmaker's full profile")

    assert synthesizer.calls == 1
    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert result.answer.evidence_ids
    assert "Grounded player analysis" in result.answer.answer_markdown
    assert result.diagnostics.terminal_status is GroundedAnswerStatus.ANSWERED
    assert result.diagnostics.terminal_branch_reason == "answered"


def test_omitted_middle_name_resolves_and_runs_player_analytics(
    ai_session: Session,
) -> None:
    player = ai_session.get(Player, 1)
    assert player is not None
    player.player_name = "Karim-David Adeyemi"
    ai_session.flush()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_profile",
                    "player": {"player_name": "Karim Adeyemi"},
                    "sections": ["intelligence"],
                },
                "calls": [
                    {
                        "name": "get_player_dossier",
                        "arguments": {
                            "player": {"player_name": "Karim Adeyemi"},
                            "sections": ["intelligence"],
                        },
                    }
                ],
            }
        ),
        FakeSynthesizer(),
    ).run("Tell me about the style of Karim Adeyemi")

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert result.context is not None
    assert result.context.analytics_evidence
    dossier = result.context.analytics_evidence[0].result
    assert dossier is not None
    assert dossier["player"]["player_name"] == "Karim-David Adeyemi"
    assert result.diagnostics.terminal_branch_reason == "answered"


def test_empty_successful_plan_is_insufficient_and_skips_synthesis(ai_session: Session) -> None:
    synthesizer = FakeSynthesizer()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice"},
                "calls": [],
            }
        ),
        synthesizer,
    ).run("Find Alice")

    assert result.answer.status is GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.evidence == EvidenceLedger()
    assert synthesizer.calls == 0


def test_raw_event_export_request_is_insufficient_even_with_aggregate_profile(
    ai_session: Session,
) -> None:
    synthesizer = FakeSynthesizer()
    result = _workflow(
        ai_session,
        _plan(
            {
                "decision": "ready",
                "intent": {
                    "kind": "player_profile",
                    "player": {"player_name": "Alice Playmaker"},
                    "sections": ["intelligence"],
                },
                "calls": [
                    {
                        "name": "search_players",
                        "arguments": {"query": "Alice Playmaker", "limit": 10},
                    },
                    {
                        "name": "get_player_dossier",
                        "arguments": {
                            "player": {"player_name": "Alice Playmaker"},
                            "sections": ["intelligence"],
                        },
                    },
                ],
            }
        ),
        synthesizer,
    ).run("Give me every raw pass made by Alice Playmaker.")

    assert result.answer.status is GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.context is not None
    assert result.context.analytics_evidence
    assert synthesizer.calls == 0


def test_context_topic_selection_is_deterministic() -> None:
    intent = _plan(
        {
            "decision": "ready",
            "intent": {
                "kind": "role_recommendations",
                "team": {"team_id": 904},
                "position_group": "MID",
            },
            "calls": [],
        }
    ).intent
    assert select_methodology_topics(intent) == (
        KnowledgeTopic.ROLE_FIT,
        KnowledgeTopic.ROLE_RECOMMENDATIONS,
    )


def test_curated_methodology_covers_every_governed_topic() -> None:
    service = CuratedMethodologyService()
    for topic in KnowledgeTopic:
        document = service.get(topic)
        assert document.topic is topic
        assert document.source_id
        assert document.source_path
        assert document.section
        assert document.content


def test_methodology_control_characters_are_removed_before_context() -> None:
    document = MethodologyDocument(
        topic=KnowledgeTopic.ROLE_FIT,
        section="Role\x01 Fit",
        source_id="role_fit:test",
        source_path="docs/test.md",
        content="Observed\x01 positional-role style.",
        production_status=ProductionStatus.PRODUCTION,
        limitations=("Not a\x01 transfer prediction.",),
    )
    assert document.section == "Role Fit"
    assert document.content == "Observed positional-role style."
    assert document.limitations == ("Not a transfer prediction.",)


def test_nonexistent_evidence_and_methodology_citations_are_rejected() -> None:
    answer = LLMGroundedAnswer(
        answer_markdown="Unsupported [run:evidence-99]",
        evidence_ids=("run:evidence-99",),
        status=GroundedAnswerStatus.ANSWERED,
    )
    try:
        validate_answer_citations(answer, EvidenceLedger())
    except ValueError as exc:
        assert "unknown evidence IDs" in str(exc)
    else:  # pragma: no cover - protects the grounding invariant
        raise AssertionError("Unknown evidence IDs must be rejected.")


def test_validation_failure_messages_distinguish_authority_from_citation_errors() -> None:
    generic = _validation_failure_message("current_world_claim_requires_web")
    citation = _validation_failure_message("unknown_evidence_id")

    assert generic == "I couldn't produce a response that passed the evidence checks."
    assert citation == (
        "I could not validate the evidence citations in the generated answer."
    )
    assert "current_world_claim_requires_web" not in generic
    assert "evidence-" not in generic
