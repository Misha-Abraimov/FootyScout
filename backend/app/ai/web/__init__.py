"""Bounded current-web context for AI Scout."""

from app.ai.web.provider import (
    ExaWebSearchProvider,
    WebSearchProvider,
    WebSearchProviderError,
    classify_source_quality,
)
from app.ai.web.schemas import (
    CurrentContextCategory,
    CurrentContextRequest,
    SourceQuality,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
    WebSource,
)

__all__ = [
    "CurrentContextCategory",
    "CurrentContextRequest",
    "ExaWebSearchProvider",
    "SourceQuality",
    "WebSearchProvider",
    "WebSearchProviderError",
    "WebSearchRequest",
    "WebSearchResponse",
    "WebSearchResult",
    "WebSource",
    "classify_source_quality",
]
