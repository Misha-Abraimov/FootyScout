from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.ai.validation import ValidationMode


class AIScoutProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class WebSearchProviderName(str, Enum):
    EXA = "exa"


class Settings(BaseSettings):
    app_name: str = "FootyScout API"
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://scoutlens:scoutlens_dev@localhost:5433/scoutlens"
    cors_origins: list[str] = ["http://localhost:3000"]
    ai_scout_provider: AIScoutProvider = AIScoutProvider.ANTHROPIC
    ai_scout_fallback_provider: AIScoutProvider | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ai_scout_anthropic_model: str = "claude-sonnet-5"
    ai_scout_openai_model: str = "gpt-5-mini"
    ai_scout_synthesis_model: str = "gpt-5-mini"
    ai_scout_synthesis_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    ai_scout_anthropic_timeout_seconds: float | None = Field(default=None, gt=0)
    ai_scout_openai_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    ai_scout_openai_planner_reasoning_effort: Literal[
        "minimal", "low", "medium", "high"
    ] = "low"
    ai_scout_max_retries: int = Field(default=1, ge=0, le=2)
    ai_scout_web_search_enabled: bool = False
    ai_scout_web_search_provider: WebSearchProviderName = WebSearchProviderName.EXA
    exa_api_key: str | None = None
    ai_scout_web_max_results: int = Field(default=5, ge=1, le=5)
    ai_scout_web_timeout_seconds: float = Field(default=15.0, gt=0, le=60)
    ai_scout_web_max_retries: int = Field(default=1, ge=0, le=2)
    ai_scout_max_output_tokens: int = Field(default=4096, ge=256, le=8192)
    ai_scout_web_content_max_age_hours: int = Field(default=24, ge=1, le=168)
    ai_scout_web_current_status_max_age_hours: int = Field(
        default=168, ge=1, le=720
    )
    ai_scout_validation_mode: ValidationMode = ValidationMode.STRICT
    ai_scout_trace_enabled: bool = False
    ai_scout_trace_path: str | None = None
    ai_scout_pricing_registry_path: str | None = None
    langsmith_tracing_enabled: bool = False
    langsmith_project: str = "footyscout-ai-scout"
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    otel_enabled: bool = False
    otel_service_name: str = "footyscout-backend"
    otel_exporter_otlp_endpoint: str | None = None
    ai_scout_judge_enabled: bool = False
    ai_scout_judge_provider: str = "openai"
    ai_scout_judge_model: str | None = None
    ai_scout_judge_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    ai_scout_judge_max_retries: int = Field(default=1, ge=0, le=2)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
