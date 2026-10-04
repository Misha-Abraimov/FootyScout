"""Single configuration boundary for AI Scout planner providers."""

from __future__ import annotations

from app.ai.planner import ScoutPlanner
from app.ai.provider import (
    AnthropicPlannerProvider,
    FallbackPlannerProvider,
    OpenAIPlannerProvider,
    PlannerProvider,
)
from app.config import AIScoutProvider, Settings


def create_planner_provider(settings: Settings) -> PlannerProvider:
    """Instantiate only the explicitly selected provider."""
    primary = _create_provider(settings, settings.ai_scout_provider)
    fallback_name = settings.ai_scout_fallback_provider
    if fallback_name is None:
        return primary
    if fallback_name is settings.ai_scout_provider:
        raise ValueError("AI_SCOUT_FALLBACK_PROVIDER must differ from AI_SCOUT_PROVIDER.")
    return FallbackPlannerProvider(primary, _create_provider(settings, fallback_name))


def _create_provider(settings: Settings, provider: AIScoutProvider) -> PlannerProvider:
    if provider is AIScoutProvider.ANTHROPIC:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when AI_SCOUT_PROVIDER=anthropic.")
        return AnthropicPlannerProvider(
            api_key=settings.anthropic_api_key,
            model=settings.ai_scout_anthropic_model,
            timeout_seconds=settings.ai_scout_anthropic_timeout_seconds,
            max_retries=settings.ai_scout_max_retries,
            max_tokens=settings.ai_scout_max_output_tokens,
        )
    if provider is AIScoutProvider.OPENAI:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_SCOUT_PROVIDER=openai.")
        return OpenAIPlannerProvider(
            api_key=settings.openai_api_key,
            model=settings.ai_scout_openai_model,
            timeout_seconds=settings.ai_scout_openai_timeout_seconds,
            max_retries=settings.ai_scout_max_retries,
            max_output_tokens=settings.ai_scout_max_output_tokens,
            reasoning_effort=settings.ai_scout_openai_planner_reasoning_effort,
        )
    raise ValueError(f"Unsupported AI Scout provider: {provider!r}")


def create_planner(settings: Settings) -> ScoutPlanner:
    """Build the provider-neutral planner from application settings."""
    return ScoutPlanner(create_planner_provider(settings))
