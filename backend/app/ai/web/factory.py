"""Configuration boundary for optional current-web providers."""

from app.ai.web.provider import ExaWebSearchProvider, WebSearchProvider
from app.config import Settings, WebSearchProviderName


def create_web_search_provider(settings: Settings) -> WebSearchProvider | None:
    if not settings.ai_scout_web_search_enabled:
        return None
    if settings.ai_scout_web_search_provider is WebSearchProviderName.EXA:
        if not settings.exa_api_key:
            raise ValueError("EXA_API_KEY is required when Exa web search is enabled.")
        return ExaWebSearchProvider(
            api_key=settings.exa_api_key,
            timeout_seconds=settings.ai_scout_web_timeout_seconds,
            max_retries=settings.ai_scout_web_max_retries,
        )
    raise ValueError(
        f"Unsupported AI Scout web provider: {settings.ai_scout_web_search_provider!r}"
    )
