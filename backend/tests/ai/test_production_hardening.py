"""Offline production observability, security, and reliability contracts."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from app.ai.diagnostics import FailureDiagnostic, FailureStage
from app.ai.executor import execute_plan
from app.ai.observability.pricing import (
    ModelPricing,
    PricingRegistry,
    estimate_cost_usd,
    load_pricing_registry,
)
from app.ai.observability.telemetry import (
    CorrelationIdMiddleware,
    configure_langsmith,
    current_correlation_id,
    langsmith_graph_config,
    set_test_tracer,
    start_span,
)
from app.ai.provider import (
    FallbackPlannerProvider,
    PlannerProviderError,
    PlannerUsage,
    ProviderPlanResponse,
)
from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.reliability import ProviderReliabilityPolicy
from app.ai.schemas import (
    NormalizedPlan,
    NormalizedToolCall,
    ScoutIntent,
    ToolName,
)
from app.ai.web import WebSearchRequest
from app.ai.web.provider import ExaWebSearchProvider, WebSearchProviderError
from app.schemas import AIScoutRequest


class _Span:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _SpanManager(AbstractContextManager[_Span]):
    def __init__(self, *, fail_exit: bool = False) -> None:
        self.span = _Span()
        self.fail_exit = fail_exit

    def __enter__(self) -> _Span:
        return self.span

    def __exit__(self, *_args: object) -> bool:
        if self.fail_exit:
            raise RuntimeError("exporter unavailable")
        return False


class _Tracer:
    def __init__(self, *, fail_exit: bool = False) -> None:
        self.managers: list[_SpanManager] = []
        self.fail_exit = fail_exit

    def start_as_current_span(self, _name: str) -> _SpanManager:
        manager = _SpanManager(fail_exit=self.fail_exit)
        self.managers.append(manager)
        return manager


def test_langsmith_configuration_is_optional_and_payload_metadata_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "LANGSMITH_API_KEY",
        "LANGSMITH_HIDE_INPUTS",
        "LANGSMITH_HIDE_OUTPUTS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert configure_langsmith(
        enabled=False,
        project="test",
        api_key=None,
        endpoint=None,
    ) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert configure_langsmith(
        enabled=True,
        project="footyscout-test",
        api_key="secret-value",
        endpoint=None,
    ) is True
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"

    config = langsmith_graph_config(
        correlation_id="run-safe",
        provider="openai",
        model="gpt-test",
        planner_prompt_version="planner-v3",
        synthesis_prompt_version="grounded-synthesis-v21",
        validation_mode="strict",
    )
    serialized = str(config)
    assert "secret-value" not in serialized
    assert "raw question" not in serialized


def test_otel_export_failure_is_fail_open() -> None:
    tracer = _Tracer(fail_exit=True)
    set_test_tracer(tracer)
    try:
        with start_span("ai_scout.test", {"ai.workflow": "ai_scout"}):
            pass
    finally:
        set_test_tracer(None)
    assert tracer.managers[0].span.attributes == {"ai.workflow": "ai_scout"}


def test_http_request_id_is_shared_with_request_context() -> None:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/correlation")
    def correlation() -> dict[str, str | None]:
        return {"correlation_id": current_correlation_id()}

    request_id = "11111111-1111-4111-8111-111111111111"
    response = TestClient(app).get("/correlation", headers={"X-Request-ID": request_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert response.json()["correlation_id"] == request_id

    generated = TestClient(app).get("/correlation", headers={"X-Request-ID": "unsafe"})
    assert generated.status_code == 200
    UUID(generated.headers["X-Request-ID"])


def test_versioned_pricing_is_loaded_and_unknown_cost_stays_null(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        '{"version":"test-v1","entries":[{"provider":"openai",'
        '"model":"gpt-test","effective_date":"2026-10-03",'
        '"input_usd_per_million_tokens":1.0,'
        '"cached_input_usd_per_million_tokens":0.1,'
        '"output_usd_per_million_tokens":2.0,'
        '"units":"usd_per_million_tokens"}]}',
        encoding="utf-8",
    )
    registry = load_pricing_registry(path)
    pricing = registry.find("openai", "gpt-test")
    assert registry.version == "test-v1"
    assert pricing is not None and pricing.cached_input_usd_per_million_tokens == 0.1
    assert estimate_cost_usd(input_tokens=100, output_tokens=50, pricing=pricing) == 0.0002
    assert estimate_cost_usd(
        input_tokens=100,
        cached_input_tokens=50,
        output_tokens=50,
        pricing=pricing,
    ) == 0.000155
    assert estimate_cost_usd(input_tokens=100, output_tokens=50, pricing=None) is None
    assert estimate_cost_usd(input_tokens=None, output_tokens=50, pricing=pricing) is None


def test_question_schema_rejects_controls_but_allows_normal_multiline_text() -> None:
    assert AIScoutRequest(question="Compare Xhaka\nwith another midfielder.").question
    with pytest.raises(ValidationError, match="unsupported control"):
        AIScoutRequest(question="ignore\x00validation")
    with pytest.raises(ValidationError, match="at most 2000"):
        AIScoutRequest(question="x" * 2001)


def test_executor_refuses_intent_unauthorized_tool_before_dispatch() -> None:
    intent = TypeAdapter(ScoutIntent).validate_python(
        {"kind": "methodology", "topic": "xpass"}
    )
    plan = NormalizedPlan(
        intent=intent,
        calls=[
            NormalizedToolCall(
                call_id="call-1",
                name=ToolName.SEARCH_PLAYERS,
                arguments={"query": "Xhaka", "limit": 5},
            )
        ],
    )
    with pytest.raises(ValueError, match="not authorized"):
        execute_plan(None, plan, run_id="security-test")  # type: ignore[arg-type]


class _PlannerProvider:
    def __init__(
        self,
        provider: str,
        model: str,
        result: ProviderPlanResponse | Exception,
    ) -> None:
        self.provider = provider
        self.model = model
        self.result = result
        self.calls = 0

    def create_plan(self, *, question: str, system_prompt: str) -> ProviderPlanResponse:
        del question, system_prompt
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _decision_response() -> ProviderPlanResponse:
    return ProviderPlanResponse(
        decision=LLMPlannerDecision(
            decision="ready",
            intent="player_search",
            search_query="Xhaka",
        ),
        usage=PlannerUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def test_fallback_runs_only_after_transient_provider_failure() -> None:
    timeout = PlannerProviderError(
        "timed out",
        diagnostic=FailureDiagnostic(
            failure_stage=FailureStage.PROVIDER,
            error_type="APITimeoutError",
            sanitized_message="timed out",
            provider="openai",
            provider_response_received=False,
        ),
    )
    primary = _PlannerProvider("openai", "primary", timeout)
    fallback = _PlannerProvider("anthropic", "fallback", _decision_response())
    result = FallbackPlannerProvider(primary, fallback).create_plan(
        question="Find Xhaka",
        system_prompt="safe",
    )
    assert primary.calls == fallback.calls == 1
    assert result.fallback_attempted is True
    assert result.fallback_provider == "anthropic"
    assert result.fallback_success is True
    assert result.provider_attempts == 2


def test_fallback_does_not_run_for_auth_or_structured_errors() -> None:
    auth = PlannerProviderError(
        "authentication failed",
        diagnostic=FailureDiagnostic(
            failure_stage=FailureStage.PROVIDER,
            error_type="AuthenticationError",
            sanitized_message="authentication failed",
            provider="openai",
            provider_response_received=True,
            http_status=401,
        ),
    )
    primary = _PlannerProvider("openai", "primary", auth)
    fallback = _PlannerProvider("anthropic", "fallback", _decision_response())
    with pytest.raises(PlannerProviderError):
        FallbackPlannerProvider(primary, fallback).create_plan(
            question="Find Xhaka",
            system_prompt="safe",
        )
    assert primary.calls == 1
    assert fallback.calls == 0


def test_exa_retries_timeout_once_and_records_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"results": []}

    def fake_post(*_args: object, **_kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("temporary timeout")
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = ExaWebSearchProvider(
        api_key="secret",
        timeout_seconds=1,
        max_retries=1,
    )
    provider._reliability = ProviderReliabilityPolicy(
        max_retries=1,
        base_backoff_seconds=0,
        max_backoff_seconds=0,
        jitter_seconds=0,
    )
    result = provider.search(WebSearchRequest(query="latest update", max_results=1))
    assert calls == 2
    assert result.provider_attempts == 2
    assert result.retry_count == 1


def test_exa_retries_429_but_not_authentication_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    request = httpx.Request("POST", "https://api.exa.ai/search")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"results": []}

    def rate_limited_then_ok(*_args: object, **_kwargs: object) -> FakeResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            response = httpx.Response(429, request=request, headers={"retry-after": "0"})
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", rate_limited_then_ok)
    provider = ExaWebSearchProvider(api_key="secret", timeout_seconds=1, max_retries=1)
    provider._reliability = ProviderReliabilityPolicy(
        max_retries=1,
        base_backoff_seconds=0,
        max_backoff_seconds=0,
        jitter_seconds=0,
    )
    result = provider.search(WebSearchRequest(query="latest update", max_results=1))
    assert result.rate_limit_events == 1
    assert result.provider_attempts == 2

    attempts = 0

    def authentication_failure(*_args: object, **_kwargs: object) -> FakeResponse:
        nonlocal attempts
        attempts += 1
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(httpx, "post", authentication_failure)
    with pytest.raises(WebSearchProviderError) as failure:
        provider.search(WebSearchRequest(query="latest update", max_results=1))
    assert attempts == 1
    assert failure.value.http_status == 401
    assert failure.value.retry_count == 0


def test_pricing_model_requires_explicit_supported_units() -> None:
    with pytest.raises(ValidationError):
        ModelPricing(
            provider="openai",
            model="gpt-test",
            input_usd_per_million_tokens=1,
            output_usd_per_million_tokens=2,
            units="usd_per_token",  # type: ignore[arg-type]
        )
    assert PricingRegistry().find("unknown", "model") is None
