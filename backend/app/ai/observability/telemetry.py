"""Optional LangSmith and OpenTelemetry integration with fail-open isolation."""

from __future__ import annotations

import logging
import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import UUID, uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

_CORRELATION_ID: ContextVar[str | None] = ContextVar(
    "ai_scout_correlation_id",
    default=None,
)
_OTEL_TRACER: Any = None


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach one safe request ID to HTTP, LangSmith, OTel, and AI run records."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        correlation_id = _valid_correlation_id(request.headers.get("x-request-id"))
        token = _CORRELATION_ID.set(correlation_id)
        try:
            set_current_span_attributes({"ai.correlation_id": correlation_id})
            response = await call_next(request)
            response.headers["X-Request-ID"] = correlation_id
            return response
        finally:
            _CORRELATION_ID.reset(token)


def current_correlation_id() -> str | None:
    return _CORRELATION_ID.get()


def configure_langsmith(
    *,
    enabled: bool,
    project: str,
    api_key: str | None,
    endpoint: str | None,
) -> bool:
    """Configure native LangChain/LangGraph tracing without exposing payloads."""
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    if not api_key:
        logger.warning("LangSmith tracing is enabled but LANGSMITH_API_KEY is unavailable.")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGSMITH_API_KEY"] = api_key
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
    # Preserve topology, timing, tags, and safe metadata while raw questions,
    # prompts, evidence bodies, and answers remain local.
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"
    return True


def configure_opentelemetry(
    *,
    app: Any,
    engine: Any,
    enabled: bool,
    service_name: str,
    endpoint: str | None,
) -> bool:
    """Configure OTLP/FastAPI/SQLAlchemy tracing; failures never block the app."""
    global _OTEL_TRACER
    if not enabled:
        _OTEL_TRACER = None
        return False
    if not endpoint:
        logger.warning("OpenTelemetry is enabled but OTEL_EXPORTER_OTLP_ENDPOINT is absent.")
        _OTEL_TRACER = None
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)),
        )
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        SQLAlchemyInstrumentor().instrument(engine=engine, tracer_provider=provider)
        _OTEL_TRACER = provider.get_tracer("footyscout.ai_scout")
        trace.set_tracer_provider(provider)
    except Exception as exc:  # noqa: BLE001 - observability is always fail-open
        logger.warning("OpenTelemetry configuration failed safely: %s", type(exc).__name__)
        _OTEL_TRACER = None
        return False
    return True


@contextmanager
def start_span(
    name: str,
    attributes: Mapping[str, str | bool | int | float | None] | None = None,
) -> Generator[Any, None, None]:
    """Start one optional OTel span without changing application behavior."""
    tracer = _OTEL_TRACER
    if tracer is None:
        yield None
        return
    manager = None
    try:
        manager = tracer.start_as_current_span(name)
        span = manager.__enter__()
        _apply_span_attributes(span, attributes or {})
    except Exception as exc:  # noqa: BLE001 - optional telemetry boundary
        logger.warning("OpenTelemetry span start failed safely: %s", type(exc).__name__)
        yield None
        return
    try:
        yield span
    except BaseException as exc:  # noqa: BLE001 - preserve caller exceptions exactly
        try:
            manager.__exit__(type(exc), exc, exc.__traceback__)
        finally:
            raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - exporter/processor failure isolation
            logger.warning("OpenTelemetry span export failed safely: %s", type(exc).__name__)


def set_current_span_attributes(
    attributes: Mapping[str, str | bool | int | float | None],
) -> None:
    """Attach bounded, low-cardinality attributes to the active OTel span."""
    try:
        from opentelemetry import trace

        _apply_span_attributes(trace.get_current_span(), attributes)
    except Exception:  # noqa: BLE001 - optional telemetry must be fail-open
        return


def langsmith_graph_config(
    *,
    correlation_id: str,
    provider: str,
    model: str,
    planner_prompt_version: str,
    synthesis_prompt_version: str,
    validation_mode: str,
) -> dict[str, object]:
    """Return safe LangGraph trace metadata without request/evidence content."""
    return {
        "run_name": "ai_scout",
        "tags": ["ai_scout", validation_mode],
        "metadata": {
            "ai.workflow": "ai_scout",
            "ai.correlation_id": correlation_id,
            "ai.provider": provider,
            "ai.model": model,
            "ai.planner_prompt_version": planner_prompt_version,
            "ai.synthesis_prompt_version": synthesis_prompt_version,
            "ai.validation_mode": validation_mode,
        },
    }


def _apply_span_attributes(
    span: Any,
    attributes: Mapping[str, str | bool | int | float | None],
) -> None:
    if span is None:
        return
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


def _valid_correlation_id(candidate: str | None) -> str:
    if candidate:
        try:
            return str(UUID(candidate))
        except (ValueError, AttributeError):
            pass
    return str(uuid4())


def set_test_tracer(tracer: Any) -> None:
    """Inject a recorder in offline tests without configuring a global SDK."""
    global _OTEL_TRACER
    _OTEL_TRACER = tracer
