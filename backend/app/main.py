from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.ai.observability.telemetry import (
    CorrelationIdMiddleware,
    configure_langsmith,
    configure_opentelemetry,
)
from app.config import settings
from app.database import engine
from app.routers import (
    action_value_info,
    ai_scout,
    archetypes,
    compare,
    leaderboard,
    meta,
    model_info,
    players,
    teams,
    xg_model_info,
)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Football scouting and analytics API for FootyScout.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(CorrelationIdMiddleware)

configure_langsmith(
    enabled=settings.langsmith_tracing_enabled,
    project=settings.langsmith_project,
    api_key=settings.langsmith_api_key,
    endpoint=settings.langsmith_endpoint,
)
configure_opentelemetry(
    app=app,
    engine=engine,
    enabled=settings.otel_enabled,
    service_name=settings.otel_service_name,
    endpoint=settings.otel_exporter_otlp_endpoint,
)

app.include_router(players.router)
app.include_router(compare.router)
app.include_router(leaderboard.router)
app.include_router(model_info.router)
app.include_router(xg_model_info.router)
app.include_router(action_value_info.router)
app.include_router(ai_scout.router)
app.include_router(archetypes.router)
app.include_router(teams.router)
app.include_router(meta.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}
