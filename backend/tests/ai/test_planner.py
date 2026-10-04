"""Provider-free structured planner and schema tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.ai import planner as planner_module
from app.ai.diagnostics import FailureStage
from app.ai.planner import PlannerStatus, ScoutPlanner
from app.ai.prompts import PLANNER_PROMPT_VERSION
from app.ai.prompts.planner_v1 import PLANNER_PROMPT as LEGACY_PLANNER_V2_PROMPT
from app.ai.prompts.planner_v2 import (
    PLANNER_PROMPT as PLANNER_V2_PROMPT,
)
from app.ai.prompts.planner_v2 import (
    PLANNER_PROMPT_VERSION as PLANNER_V2_PROMPT_VERSION,
)
from app.ai.provider import (
    OpenAIPlannerProvider,
    PlannerProviderError,
    PlannerUsage,
    ProviderPlanResponse,
)
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.schemas import IntentKind, ScoutPlan, ToolName


def _search_plan() -> ScoutPlan:
    return ScoutPlan.model_validate(
        {
            "decision": "ready",
            "intent": {"kind": "player_search", "query": "Alice", "limit": 10},
            "calls": [
                {
                    "name": "search_players",
                    "arguments": {"query": "Alice", "limit": 10},
                }
            ],
        }
    )


def _llm_search_decision() -> LLMPlannerDecision:
    return LLMPlannerDecision(
        decision="ready",
        intent="player_search",
        search_query="Alice",
    )


class FakeProvider:
    provider = "fake"
    model = "fake-structured-model"

    def __init__(self, response: ProviderPlanResponse | Exception) -> None:
        self.response = response
        self.question: str | None = None
        self.system_prompt: str | None = None

    def create_plan(self, *, question: str, system_prompt: str) -> ProviderPlanResponse:
        self.question = question
        self.system_prompt = system_prompt
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class FakeOpenAIRawResponse:
    def __init__(
        self,
        parsed: object,
        *,
        payload: dict[str, object] | None = None,
        request_id: str = "req_fake_123",
        status_code: int = 200,
    ) -> None:
        self._parsed = parsed
        self._payload = payload or {}
        self.request_id = request_id
        self.http_response = SimpleNamespace(
            status_code=status_code,
            json=lambda: self._payload,
        )

    def parse(self) -> object:
        if isinstance(self._parsed, Exception):
            raise self._parsed
        return self._parsed

    def json(self) -> dict[str, object]:
        return self._payload


class FakeOpenAIResponses:
    def __init__(self, raw_response: FakeOpenAIRawResponse) -> None:
        self.with_raw_response = self
        self.raw_response = raw_response
        self.captured: dict[str, object] = {}

    def parse(self, **kwargs: object) -> FakeOpenAIRawResponse:
        self.captured.update(kwargs)
        return self.raw_response


def test_planner_uses_versioned_prompt_and_fake_provider() -> None:
    provider = FakeProvider(
        ProviderPlanResponse(
            decision=_llm_search_decision(),
            usage=PlannerUsage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
    )
    result = ScoutPlanner(provider).plan("  Find Alice  ")
    assert result.status is PlannerStatus.SUCCESS
    assert result.prompt_version == PLANNER_PROMPT_VERSION
    assert result.plan == _search_plan()
    assert result.usage is not None and result.usage.total_tokens == 30
    assert result.provider_attempts == 1
    assert provider.question == "Find Alice"
    assert provider.system_prompt is not None
    assert "deterministic FootyScout code chooses tools" in provider.system_prompt
    assert "SQL" in provider.system_prompt


def test_planner_v3_preserves_v2_and_adds_narrow_semantic_rules() -> None:
    provider = FakeProvider(
        ProviderPlanResponse(decision=_llm_search_decision(), usage=PlannerUsage())
    )

    result = ScoutPlanner(provider).plan("Find players named Williams")

    assert PLANNER_V2_PROMPT_VERSION == "planner-v2"
    assert PLANNER_V2_PROMPT == LEGACY_PLANNER_V2_PROMPT
    assert "The primary intent is" not in PLANNER_V2_PROMPT
    assert result.prompt_version == PLANNER_PROMPT_VERSION == "planner-v3"
    assert provider.system_prompt is not None
    assert "final analytical goal" in provider.system_prompt
    assert "multiple matching players are\nnormal results" in provider.system_prompt
    assert "emit the existing 0\nsentinel in limit" in provider.system_prompt


@pytest.mark.parametrize(
    ("question", "topic"),
    [
        ("What does xPass mean?", "xpass"),
        ("Explain xG", "xg"),
        ("How does possession value work?", "possession_value"),
        ("What are the limitations of attacking impact?", "attacking_impact"),
        ("How are player profiles interpreted?", "player_profiles"),
        ("What is percentile?", "percentiles"),
        ("Explain archetypes", "archetypes"),
        ("How does player similarity work?", "similarity"),
        ("What does team intelligence mean?", "team_intelligence"),
        (
            "What does Role Fit mean and what are its limitations?",
            "role_fit",
        ),
        ("How are scouting recommendations interpreted?", "role_recommendations"),
    ],
)
def test_pure_methodology_guard_covers_canonical_topics(
    question: str,
    topic: str,
) -> None:
    contradictory = LLMPlannerDecision(
        decision="clarification_required",
        intent="role_fit",
        requested_analyses=["player_profile"],
        primary_player_name="invented player requirement",
        team_name="invented team requirement",
        clarification_message="Which player should FootyScout analyze?",
    )
    result = ScoutPlanner(
        FakeProvider(
            ProviderPlanResponse(decision=contradictory, usage=PlannerUsage())
        )
    ).plan(question)

    assert result.status is PlannerStatus.SUCCESS
    assert result.methodology_guard_applied is True
    assert result.provider_decision == contradictory
    assert result.plan is not None
    assert result.plan.decision.value == "ready"
    assert result.plan.intent.kind is IntentKind.METHODOLOGY
    assert result.plan.intent.topic.value == topic
    assert [call.name for call in result.plan.calls] == [ToolName.GET_METHODOLOGY]


def test_xgboost_transformer_production_question_routes_to_possession_value() -> None:
    contradictory = LLMPlannerDecision(
        decision="ready",
        intent="methodology",
        methodology_topic="xg",
    )
    result = ScoutPlanner(
        FakeProvider(ProviderPlanResponse(decision=contradictory, usage=PlannerUsage()))
    ).plan("Why is XGBoost used instead of the Transformer in production?")

    assert result.status is PlannerStatus.SUCCESS
    assert result.methodology_guard_applied is True
    assert result.plan is not None
    assert result.plan.intent.kind is IntentKind.METHODOLOGY
    assert result.plan.intent.topic.value == "possession_value"


def test_explicit_xg_model_comparison_remains_xg_methodology() -> None:
    xg = LLMPlannerDecision(
        decision="ready",
        intent="methodology",
        methodology_topic="xg",
    )
    result = ScoutPlanner(
        FakeProvider(ProviderPlanResponse(decision=xg, usage=PlannerUsage()))
    ).plan("Why does the xG model use XGBoost rather than a Transformer?")

    assert result.plan is not None
    assert result.plan.intent.topic.value == "xg"


def test_archetype_quality_hierarchy_routes_to_methodology_without_player_lookup() -> None:
    provider = FakeProvider(
        ProviderPlanResponse(decision=_llm_search_decision(), usage=PlannerUsage())
    )
    result = ScoutPlanner(provider).plan("Which archetype is objectively the best?")

    assert provider.question == "Which archetype is objectively the best?"
    assert result.status is PlannerStatus.SUCCESS
    assert result.methodology_guard_applied is True
    assert result.plan is not None
    assert result.plan.intent.kind is IntentKind.METHODOLOGY
    assert result.plan.intent.topic.value == "archetypes"
    assert [call.name for call in result.plan.calls] == [ToolName.GET_METHODOLOGY]


def test_named_player_archetype_query_remains_player_profile() -> None:
    decision = LLMPlannerDecision(
        decision="ready",
        intent="player_profile",
        primary_player_name="Xhaka",
    )
    result = ScoutPlanner(
        FakeProvider(ProviderPlanResponse(decision=decision, usage=PlannerUsage()))
    ).plan("What archetype is Xhaka?")

    assert result.plan is not None
    assert result.plan.intent.kind is IntentKind.PLAYER_PROFILE
    assert ToolName.GET_PLAYER_DOSSIER in {call.name for call in result.plan.calls}


@pytest.mark.parametrize(
    ("question", "decision"),
    [
        (
            "Why does Xhaka have this Role Fit distance and what does Role Fit mean?",
            LLMPlannerDecision(
                decision="ready",
                intent="role_fit",
                primary_player_name="Xhaka",
                team_name="Leverkusen",
            ),
        ),
        (
            "Which midfielder best fits Leverkusen and how is Role Fit calculated?",
            LLMPlannerDecision(
                decision="ready",
                intent="role_recommendations",
                team_name="Leverkusen",
                position_group="MID",
            ),
        ),
        (
            "Explain Wirtz's similarity score.",
            LLMPlannerDecision(
                decision="ready",
                intent="similar_players",
                primary_player_name="Wirtz",
            ),
        ),
    ],
)
def test_methodology_guard_does_not_override_mixed_analytics_questions(
    question: str,
    decision: LLMPlannerDecision,
) -> None:
    result = ScoutPlanner(
        FakeProvider(ProviderPlanResponse(decision=decision, usage=PlannerUsage()))
    ).plan(question)
    assert result.status is PlannerStatus.SUCCESS
    assert result.methodology_guard_applied is False
    assert result.plan is not None
    assert result.plan.intent.kind is decision.intent


def test_planner_returns_safe_errors_without_provider_details() -> None:
    empty = ScoutPlanner(FakeProvider(AssertionError("provider must not run"))).plan(" ")
    assert empty.status is PlannerStatus.ERROR
    assert empty.error_message == "A non-empty scouting question is required."

    failed = ScoutPlanner(FakeProvider(PlannerProviderError("secret provider detail"))).plan(
        "Find Alice"
    )
    assert failed.status is PlannerStatus.ERROR
    assert failed.error_message == "The planning provider request failed."
    assert "secret" not in failed.error_message


def test_provider_exception_is_recorded_and_secrets_are_redacted() -> None:
    result = ScoutPlanner(
        FakeProvider(
            RuntimeError(
                "OPENAI_API_KEY=super-secret-value Authorization: Bearer sk-live-secret123"
            )
        )
    ).plan("Find Alice")
    assert result.status is PlannerStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.failure_stage is FailureStage.PROVIDER
    assert result.diagnostic.error_type == "RuntimeError"
    serialized = result.diagnostic.model_dump_json()
    assert "super-secret-value" not in serialized
    assert "sk-live-secret123" not in serialized
    assert serialized.count("[REDACTED]") == 2


def test_validation_failure_is_classified_as_structured_parse() -> None:
    try:
        ScoutPlan.model_validate({"decision": "ready"})
    except ValidationError as exc:
        validation_error = exc
    result = ScoutPlanner(FakeProvider(validation_error)).plan("Find Alice")
    assert result.status is PlannerStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.failure_stage is FailureStage.STRUCTURED_PARSE
    assert result.diagnostic.error_type == "ValidationError"
    assert result.diagnostic.provider_response_received is False
    assert result.diagnostic.validation_issues
    assert all(issue.location for issue in result.diagnostic.validation_issues)
    assert all(issue.error_type for issue in result.diagnostic.validation_issues)
    assert "input" not in result.diagnostic.validation_issues[0].model_dump_json()


@pytest.mark.parametrize(
    "question",
    [
        "Find similar players regardless of position.",
        "Find similar players across positions.",
        "Run cross-position similarity.",
        "Find similar players at any position.",
    ],
)
def test_cross_position_similarity_is_guarded_before_provider_parse(
    question: str,
) -> None:
    provider = FakeProvider(AssertionError("provider must not run"))

    result = ScoutPlanner(provider).plan(question)

    assert provider.question is None
    assert result.status is PlannerStatus.SUCCESS
    assert result.response_received is False
    assert result.plan is not None
    assert result.plan.decision.value == "unsupported"
    assert result.plan.intent.kind is IntentKind.SIMILAR_PLAYERS
    assert result.plan.calls == []
    assert result.provider_attempts == 0
    assert "same position group" in (result.plan.unsupported_reason or "")


def test_percentile_fabrication_is_safe_methodology_before_provider_parse() -> None:
    provider = FakeProvider(AssertionError("provider must not run"))

    result = ScoutPlanner(provider).plan(
        "Invent missing percentile values so every player can be compared."
    )

    assert provider.question is None
    assert result.status is PlannerStatus.SUCCESS
    assert result.plan is not None
    assert result.plan.intent.kind is IntentKind.METHODOLOGY
    assert result.plan.intent.topic.value == "percentiles"
    assert [call.name for call in result.plan.calls] == [ToolName.GET_METHODOLOGY]
    assert result.provider_attempts == 0


def test_unrelated_fabricated_statistics_are_not_captured_by_percentile_guard() -> None:
    provider = FakeProvider(
        ProviderPlanResponse(
            decision=LLMPlannerDecision(
                decision="unsupported",
                intent="player_profile",
                unsupported_reason="Fabricated player statistics are unsupported.",
            ),
            usage=PlannerUsage(),
        )
    )

    result = ScoutPlanner(provider).plan("Invent a goals total for an unknown player.")

    assert provider.question == "Invent a goals total for an unknown player."
    assert result.status is PlannerStatus.SUCCESS
    assert result.plan is not None
    assert result.plan.decision.value == "unsupported"


def test_invalid_built_plan_is_recorded_as_planner_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_builder(
        _decision: LLMPlannerDecision, *, question: str | None = None
    ) -> ScoutPlan:
        del question
        return ScoutPlan.model_validate({"decision": "ready"})

    monkeypatch.setattr(planner_module, "build_scout_plan", invalid_builder)
    invalid_response = ProviderPlanResponse(
        decision=_llm_search_decision(),
        usage=PlannerUsage(input_tokens=6, output_tokens=4, total_tokens=10),
        request_id="req_validation_123",
    )
    result = ScoutPlanner(FakeProvider(invalid_response)).plan("Find Alice")
    assert result.status is PlannerStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.failure_stage is FailureStage.PLANNER_VALIDATION
    assert result.diagnostic.error_type == "ValidationError"
    assert result.diagnostic.provider_response_received is True
    assert result.diagnostic.request_id == "req_validation_123"
    assert result.usage is not None and result.usage.total_tokens == 10


def test_scout_plan_rejects_unknown_tools_invalid_arguments_and_extra_fields() -> None:
    base = _search_plan().model_dump(mode="json")
    with pytest.raises(ValidationError):
        ScoutPlan.model_validate(
            {**base, "calls": [{"name": "run_sql", "arguments": {"sql": "SELECT 1"}}]}
        )
    with pytest.raises(ValidationError):
        ScoutPlan.model_validate(
            {
                **base,
                "calls": [{"name": "search_players", "arguments": {"query": "Alice", "limit": 21}}],
            }
        )
    with pytest.raises(ValidationError):
        ScoutPlan.model_validate({**base, "unexpected": True})


def test_scout_plan_rejects_more_than_six_calls() -> None:
    call = {"name": "search_players", "arguments": {"query": "Alice", "limit": 10}}
    with pytest.raises(ValidationError):
        ScoutPlan.model_validate(
            {
                "decision": "ready",
                "intent": {"kind": "player_search", "query": "Alice"},
                "calls": [call] * 7,
            }
        )


def test_official_provider_uses_responses_parse_with_planner_decision() -> None:
    responses = FakeOpenAIResponses(
        FakeOpenAIRawResponse(
            SimpleNamespace(
                output_parsed=_llm_search_decision(),
                usage=SimpleNamespace(
                    input_tokens=4,
                    input_tokens_details=SimpleNamespace(cached_tokens=1),
                    output_tokens=3,
                    total_tokens=7,
                ),
            ),
        )
    )

    provider = OpenAIPlannerProvider.__new__(OpenAIPlannerProvider)
    provider._api_key = "test-openai-key"
    provider.model = "fake-openai-model"
    provider._client = SimpleNamespace(responses=responses)
    response = provider.create_plan(question="Find Alice", system_prompt="bounded prompt")
    assert response.decision == _llm_search_decision()
    assert response.usage.total_tokens == 7
    assert response.usage.cached_input_tokens == 1
    assert response.response_received is True
    assert responses.captured["model"] == "fake-openai-model"
    assert responses.captured["reasoning"] == {"effort": "low"}
    assert responses.captured["text_format"] is LLMPlannerDecision
    assert responses.captured["input"] == [
        {"role": "system", "content": "bounded prompt"},
        {"role": "user", "content": "Find Alice"},
    ]


def test_provider_logs_only_safe_shape_for_structured_parse_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_player_name = "Private Player Name"
    sensitive_message = "Which player? OPENAI_API_KEY=secret-value"
    try:
        LLMPlannerDecision.model_validate(
            {
                "decision": "ready",
                "intent": "similar_players",
                "primary_player_name": sensitive_player_name,
                "clarification_message": sensitive_message,
            }
        )
    except ValidationError as exc:
        validation_error = exc

    responses = FakeOpenAIResponses(
        FakeOpenAIRawResponse(
            validation_error,
            payload={
                "usage": {
                    "input_tokens": 13,
                    "input_tokens_details": {"cached_tokens": 5},
                    "output_tokens": 7,
                    "total_tokens": 20,
                }
            },
            request_id="req_parse_failure_123",
        )
    )

    provider = OpenAIPlannerProvider.__new__(OpenAIPlannerProvider)
    provider._api_key = "test-openai-key"
    provider.model = "fake-openai-model"
    provider._client = SimpleNamespace(responses=responses)

    with pytest.raises(PlannerProviderError) as caught:
        provider.create_plan(question="Find Alice", system_prompt="bounded prompt")

    diagnostic = caught.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.provider_response_received is True
    assert diagnostic.request_id == "req_parse_failure_123"
    assert diagnostic.http_status == 200
    assert diagnostic.input_tokens == 13
    assert diagnostic.cached_input_tokens == 5
    assert diagnostic.output_tokens == 7
    assert diagnostic.total_tokens == 20

    log_text = caplog.text
    assert "decision': 'ready'" in log_text
    assert "intent': 'similar_players'" in log_text
    assert "primary_player_reference': True" in log_text
    assert "clarification_message" in log_text
    assert sensitive_player_name not in log_text
    assert sensitive_message not in log_text
    assert "secret-value" not in log_text


def test_missing_parsed_output_preserves_safe_response_metadata() -> None:
    responses = FakeOpenAIResponses(
        FakeOpenAIRawResponse(
            SimpleNamespace(
                output_parsed=None,
                usage=SimpleNamespace(input_tokens=8, output_tokens=2, total_tokens=10),
                _request_id="req_safe_123",
            ),
        )
    )

    provider = OpenAIPlannerProvider.__new__(OpenAIPlannerProvider)
    provider._api_key = "test-openai-key"
    provider.model = "fake-openai-model"
    provider._client = SimpleNamespace(responses=responses)
    result = ScoutPlanner(provider).plan("Find Alice")
    assert result.status is PlannerStatus.REFUSED
    assert result.diagnostic is not None
    assert result.diagnostic.failure_stage is FailureStage.STRUCTURED_PARSE
    assert result.diagnostic.provider_response_received is True
    assert result.diagnostic.request_id == "req_safe_123"
    assert result.diagnostic.total_tokens == 10
    assert result.usage is not None and result.usage.total_tokens == 10


def test_structured_parse_failure_without_raw_usage_remains_unobserved() -> None:
    try:
        LLMPlannerDecision.model_validate({"decision": "ready"})
    except ValidationError as exc:
        validation_error = exc
    responses = FakeOpenAIResponses(
        FakeOpenAIRawResponse(
            validation_error,
            payload={"id": "response_without_usage"},
        )
    )
    provider = OpenAIPlannerProvider.__new__(OpenAIPlannerProvider)
    provider._api_key = "test-openai-key"
    provider.model = "fake-openai-model"
    provider._client = SimpleNamespace(responses=responses)

    result = ScoutPlanner(provider).plan("Find Alice")

    assert result.status is PlannerStatus.ERROR
    assert result.response_received is True
    assert result.provider_attempts == 1
    assert result.usage is None
    assert result.diagnostic is not None
    assert result.diagnostic.total_tokens is None
