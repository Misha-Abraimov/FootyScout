"""Offline Anthropic adapter and provider-selection contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from anthropic import transform_schema
from pydantic import ValidationError

from app.ai import provider as provider_module
from app.ai import provider_factory
from app.ai.diagnostics import FailureStage
from app.ai.planner import PlannerStatus, ScoutPlanner
from app.ai.provider import (
    AnthropicPlannerProvider,
    OpenAIPlannerProvider,
    PlannerProviderError,
)
from app.ai.provider_schemas import LLMPlannerDecision, LLMScoutPlan
from app.config import AIScoutProvider, Settings


def _provider_decision(*, profile: bool = False) -> LLMPlannerDecision:
    return LLMPlannerDecision(
        decision="ready",
        intent="player_profile" if profile else "player_search",
        primary_player_name="Alice" if profile else "",
        search_query="" if profile else "Alice",
        requested_sections=["passing"] if profile else [],
        limit=10,
    )


def _anthropic_provider(response: object) -> AnthropicPlannerProvider:
    class FakeMessages:
        def parse(self, **kwargs: object) -> object:
            self.kwargs = kwargs
            if isinstance(response, Exception):
                raise response
            return response

    provider = AnthropicPlannerProvider.__new__(AnthropicPlannerProvider)
    provider.model = "claude-test"
    provider.max_tokens = 4096
    provider._api_key = "test-anthropic-key"
    provider._client = SimpleNamespace(messages=FakeMessages())
    return provider


def _openai_provider(decision: LLMPlannerDecision) -> OpenAIPlannerProvider:
    response = SimpleNamespace(
        output_parsed=decision,
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        _request_id="req_openai_test",
    )

    class FakeRawResponse:
        def parse(self) -> object:
            return response

    class FakeResponses:
        def __init__(self) -> None:
            self.with_raw_response = self

        def parse(self, **_kwargs: object) -> object:
            return FakeRawResponse()

    provider = OpenAIPlannerProvider.__new__(OpenAIPlannerProvider)
    provider.model = "gpt-test"
    provider._api_key = "test-openai-key"
    provider._client = SimpleNamespace(responses=FakeResponses())
    return provider


def _schema_stats(schema: dict[str, object]) -> dict[str, int]:
    stats = {"any_of": 0, "nullable_unions": 0, "objects": 0, "max_depth": 0}

    def visit(value: object, depth: int = 0) -> None:
        stats["max_depth"] = max(stats["max_depth"], depth)
        if isinstance(value, dict):
            any_of = value.get("anyOf")
            if isinstance(any_of, list):
                stats["any_of"] += 1
                if any(
                    isinstance(branch, dict) and branch.get("type") == "null"
                    for branch in any_of
                ):
                    stats["nullable_unions"] += 1
            if value.get("type") == "object":
                stats["objects"] += 1
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                visit(child, depth + 1)

    visit(schema)
    stats["bytes"] = len(
        json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )
    stats["definitions"] = len(schema.get("$defs", {}))
    return stats


def test_flat_decision_schema_is_anthropic_transformable_and_bounded() -> None:
    old_schema = transform_schema(LLMScoutPlan.model_json_schema())
    new_schema = transform_schema(LLMPlannerDecision.model_json_schema())
    old = _schema_stats(old_schema)
    new = _schema_stats(new_schema)

    assert new_schema["type"] == "object"
    assert new_schema["additionalProperties"] is False
    assert new["bytes"] < 4_000
    assert new["bytes"] < old["bytes"] * 0.30
    assert new["definitions"] <= 8
    assert new["any_of"] == 0
    assert new["nullable_unions"] == 0
    assert new["objects"] == 1
    assert new["max_depth"] <= 4


def test_anthropic_messages_parse_returns_shared_decision_and_metadata() -> None:
    parsed = _provider_decision()
    response = SimpleNamespace(
        parsed_output=parsed,
        usage=SimpleNamespace(input_tokens=13, output_tokens=8),
        _request_id="req_anthropic_test",
    )
    provider = _anthropic_provider(response)

    result = provider.create_plan(question="Find Alice", system_prompt="bounded prompt")

    assert result.decision == parsed
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 8
    assert result.usage.total_tokens == 21
    assert result.request_id == "req_anthropic_test"
    request = provider._client.messages.kwargs
    assert request == {
        "model": "claude-test",
        "max_tokens": 4096,
        "system": "bounded prompt",
        "messages": [{"role": "user", "content": "Find Alice"}],
        "output_format": LLMPlannerDecision,
    }


def test_anthropic_and_openai_return_the_same_provider_neutral_decision() -> None:
    parsed = _provider_decision(profile=True)
    anthropic = _anthropic_provider(
        SimpleNamespace(
            parsed_output=parsed,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            _request_id="req_anthropic_multi",
        )
    )
    openai = _openai_provider(parsed)

    anthropic_result = anthropic.create_plan(question="Profile Alice", system_prompt="prompt")
    openai_result = openai.create_plan(question="Profile Alice", system_prompt="prompt")

    assert anthropic_result.decision == openai_result.decision == parsed


def test_anthropic_malformed_output_is_a_structured_parse_failure() -> None:
    with pytest.raises(ValidationError) as validation:
        LLMPlannerDecision.model_validate({"decision": "ready"})
    result = ScoutPlanner(_anthropic_provider(validation.value)).plan("Find Alice")

    assert result.status is PlannerStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.provider == "anthropic"
    assert result.diagnostic.failure_stage is FailureStage.STRUCTURED_PARSE
    assert result.diagnostic.provider_response_received is False


def test_anthropic_api_error_maps_safe_metadata_and_redacts_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAnthropicAPIError(Exception):
        request_id = "req_failed_anthropic"
        status_code = 429
        response = SimpleNamespace(
            status_code=429,
            headers={"request-id": "req_failed_anthropic"},
        )

    monkeypatch.setattr(provider_module, "AnthropicAPIError", FakeAnthropicAPIError)
    secret = "sk-ant-api03-never-serialize-this"
    provider = _anthropic_provider(
        FakeAnthropicAPIError(
            f"ANTHROPIC_API_KEY={secret} x-api-key: {secret} OPENAI_API_KEY=sk-proj-also-secret123"
        )
    )
    provider._api_key = secret

    with pytest.raises(PlannerProviderError) as failure:
        provider.create_plan(question="Find Alice", system_prompt="prompt")

    diagnostic = failure.value.diagnostic
    assert diagnostic is not None
    assert diagnostic.provider == "anthropic"
    assert diagnostic.failure_stage is FailureStage.PROVIDER
    assert diagnostic.error_type == "FakeAnthropicAPIError"
    assert diagnostic.http_status == 429
    assert diagnostic.request_id == "req_failed_anthropic"
    serialized = diagnostic.model_dump_json()
    assert secret not in serialized
    assert "sk-proj-also-secret123" not in serialized
    assert "[REDACTED]" in serialized


def test_anthropic_default_timeout_is_left_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_anthropic(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "Anthropic", fake_anthropic)
    provider = provider_factory.create_planner_provider(
        Settings(
            _env_file=None,
            ai_scout_provider="anthropic",
            anthropic_api_key="anthropic-test-key",
        )
    )

    assert isinstance(provider, AnthropicPlannerProvider)
    assert "timeout" not in captured
    assert captured["max_retries"] == 1


def test_anthropic_explicit_timeout_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_anthropic(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "Anthropic", fake_anthropic)
    provider_factory.create_planner_provider(
        Settings(
            _env_file=None,
            ai_scout_provider="anthropic",
            anthropic_api_key="anthropic-test-key",
            ai_scout_anthropic_timeout_seconds=45,
        )
    )

    assert captured["timeout"] == 45
    assert captured["max_retries"] == 1


def test_openai_keeps_explicit_configurable_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(provider_module, "OpenAI", fake_openai)
    provider = provider_factory.create_planner_provider(
        Settings(
            _env_file=None,
            ai_scout_provider="openai",
            openai_api_key="openai-test-key",
        )
    )

    assert isinstance(provider, OpenAIPlannerProvider)
    assert captured["timeout"] == 20.0
    assert captured["max_retries"] == 1
    assert provider.reasoning_effort == "low"


def test_openai_planner_reasoning_effort_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "OpenAI", lambda **_kwargs: SimpleNamespace())

    provider = provider_factory.create_planner_provider(
        Settings(
            _env_file=None,
            ai_scout_provider="openai",
            openai_api_key="openai-test-key",
            ai_scout_openai_planner_reasoning_effort="minimal",
        )
    )

    assert isinstance(provider, OpenAIPlannerProvider)
    assert provider.reasoning_effort == "minimal"


def test_anthropic_timeout_diagnostic_remains_safe_and_provider_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class APITimeoutError(Exception):
        request_id = None
        status_code = None
        response = None

    monkeypatch.setattr(provider_module, "AnthropicAPIError", APITimeoutError)
    result = ScoutPlanner(
        _anthropic_provider(APITimeoutError("Request timed out."))
    ).plan("Find Alice")

    assert result.status is PlannerStatus.ERROR
    assert result.diagnostic is not None
    assert result.diagnostic.provider == "anthropic"
    assert result.diagnostic.failure_stage is FailureStage.PROVIDER
    assert result.diagnostic.error_type == "APITimeoutError"
    assert result.diagnostic.provider_response_received is False
    assert result.diagnostic.http_status is None
    assert result.diagnostic.request_id is None


class _StubProvider:
    def __init__(self, *, model: str, provider: str, **_kwargs: object) -> None:
        self.model = model
        self.provider = provider

    def create_plan(self, *, question: str, system_prompt: str) -> object:
        raise AssertionError(f"Unexpected provider call: {question=} {system_prompt=}")


def test_anthropic_selection_never_constructs_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_openai(**_kwargs: object) -> object:
        raise AssertionError("OpenAI must not be constructed in Anthropic mode")

    monkeypatch.setattr(provider_factory, "OpenAIPlannerProvider", fail_openai)
    monkeypatch.setattr(
        provider_factory,
        "AnthropicPlannerProvider",
        lambda **kwargs: _StubProvider(provider="anthropic", **kwargs),
    )
    settings = Settings(
        _env_file=None,
        ai_scout_provider="anthropic",
        anthropic_api_key="anthropic-test-key",
        openai_api_key=None,
    )

    planner = provider_factory.create_planner(settings)

    assert planner.provider == "anthropic"
    assert planner.model == "claude-sonnet-5"


def test_openai_selection_never_constructs_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_anthropic(**_kwargs: object) -> object:
        raise AssertionError("Anthropic must not be constructed in OpenAI mode")

    monkeypatch.setattr(provider_factory, "AnthropicPlannerProvider", fail_anthropic)
    monkeypatch.setattr(
        provider_factory,
        "OpenAIPlannerProvider",
        lambda **kwargs: _StubProvider(provider="openai", **kwargs),
    )
    settings = Settings(
        _env_file=None,
        ai_scout_provider="openai",
        anthropic_api_key=None,
        openai_api_key="openai-test-key",
    )

    planner = provider_factory.create_planner(settings)

    assert planner.provider == "openai"
    assert planner.model == "gpt-5-mini"


@pytest.mark.parametrize(
    ("provider", "settings_payload", "expected_key"),
    [
        (
            AIScoutProvider.ANTHROPIC,
            {"anthropic_api_key": None, "openai_api_key": "unused"},
            "ANTHROPIC_API_KEY",
        ),
        (
            AIScoutProvider.OPENAI,
            {"anthropic_api_key": "unused", "openai_api_key": None},
            "OPENAI_API_KEY",
        ),
    ],
)
def test_only_selected_provider_key_is_required(
    provider: AIScoutProvider,
    settings_payload: dict[str, object],
    expected_key: str,
) -> None:
    settings = Settings(
        _env_file=None,
        ai_scout_provider=provider,
        **settings_payload,
    )
    with pytest.raises(ValueError, match=expected_key):
        provider_factory.create_planner_provider(settings)


def test_unsupported_provider_name_is_rejected_by_settings() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, ai_scout_provider="unsupported")
