"""Provider-neutral bounded search protocol and Exa REST adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from app.ai.diagnostics import sanitize_error_message
from app.ai.reliability import (
    ProviderReliabilityPolicy,
    RetryExhaustedError,
    call_with_retry,
    is_transient_provider_error,
    retry_after_from_headers,
)
from app.ai.web.schemas import (
    SourceQuality,
    WebSearchRequest,
    WebSearchResponse,
    WebSearchResult,
)

EXA_SEARCH_URL = "https://api.exa.ai/search"


class WebSearchProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        http_status: int | None = None,
        provider_attempts: int = 1,
        retry_count: int = 0,
        rate_limit_events: int = 0,
    ) -> None:
        super().__init__(sanitize_error_message(message))
        self.error_type = error_type
        self.http_status = http_status
        self.provider_attempts = provider_attempts
        self.retry_count = retry_count
        self.rate_limit_events = rate_limit_events


class WebSearchProvider(Protocol):
    provider: str

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        """Return normalized search results only."""


class ExaWebSearchProvider:
    """One-shot Exa `/search` adapter using bounded highlights, never full pages."""

    provider = "exa"

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        max_retries: int = 1,
    ) -> None:
        if not api_key:
            raise ValueError("EXA_API_KEY is required when Exa web search is enabled.")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._reliability = ProviderReliabilityPolicy(max_retries=max_retries)

    def search(self, request: WebSearchRequest) -> WebSearchResponse:
        retrieved_at = datetime.now(UTC)
        contents: dict[str, Any] = {"highlights": {"maxCharacters": 1200}}
        if request.content_max_age_hours is not None:
            contents["maxAgeHours"] = request.content_max_age_hours
        payload: dict[str, Any] = {
            "query": request.query,
            "type": "auto",
            "numResults": request.max_results,
            "contents": contents,
        }
        if request.publication_filter_start is not None:
            payload["startPublishedDate"] = request.publication_filter_start.isoformat().replace(
                "+00:00", "Z"
            )
        if request.preferred_domains:
            payload["includeDomains"] = list(request.preferred_domains)
        if request.excluded_domains:
            payload["excludeDomains"] = list(request.excluded_domains)
        def perform_request() -> httpx.Response:
            response = httpx.post(
                EXA_SEARCH_URL,
                headers={"x-api-key": self._api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            return response

        try:
            response, retry_outcome = call_with_retry(
                perform_request,
                policy=self._reliability,
                is_retryable=is_transient_provider_error,
                retry_after_seconds=retry_after_from_headers,
            )
            body = response.json()
            if not isinstance(body, dict):
                raise TypeError("The web search provider returned an invalid response shape.")
        except RetryExhaustedError as exhausted:
            exc = exhausted.error
            if isinstance(exc, httpx.HTTPStatusError):
                raise WebSearchProviderError(
                    "The web search provider returned an HTTP error.",
                    error_type=type(exc).__name__,
                    http_status=exc.response.status_code,
                    provider_attempts=exhausted.outcome.attempts,
                    retry_count=exhausted.outcome.retry_count,
                    rate_limit_events=exhausted.outcome.rate_limit_events,
                ) from exc
            raise WebSearchProviderError(
                "The web search provider request failed.",
                error_type=type(exc).__name__,
                provider_attempts=exhausted.outcome.attempts,
                retry_count=exhausted.outcome.retry_count,
                rate_limit_events=exhausted.outcome.rate_limit_events,
            ) from exc
        except Exception as exc:
            raise WebSearchProviderError(
                "The web search provider request failed.",
                error_type=type(exc).__name__,
            ) from exc
        raw_results = body.get("results", [])
        if not isinstance(raw_results, list):
            raise WebSearchProviderError(
                "The web search provider returned an invalid response shape.",
                error_type="ResponseValidationError",
            )
        results = tuple(
            normalized
            for item in raw_results[: request.max_results]
            if (normalized := _normalize_exa_result(item, retrieved_at)) is not None
        )
        return WebSearchResponse(
            query=request.query,
            results=results,
            provider=self.provider,
            retrieved_at=retrieved_at,
            provider_attempts=retry_outcome.attempts,
            retry_count=retry_outcome.retry_count,
            rate_limit_events=retry_outcome.rate_limit_events,
        )


def _normalize_exa_result(
    item: object,
    retrieved_at: datetime,
) -> WebSearchResult | None:
    if not isinstance(item, dict):
        return None
    url = item.get("url")
    title = item.get("title")
    highlights = item.get("highlights")
    snippet = " … ".join(str(value) for value in highlights if value) if isinstance(highlights, list) else ""
    if not snippet:
        snippet = str(item.get("text") or "")[:1200]
    domain = urlparse(str(url)).hostname or ""
    if not url or not title or not domain or not snippet:
        return None
    try:
        published_at = _parse_datetime(item.get("publishedDate"))
        return WebSearchResult(
            title=str(title),
            url=str(url),
            domain=domain.casefold().removeprefix("www."),
            published_at=published_at,
            author=str(item["author"]) if item.get("author") else None,
            snippet=snippet[:1600],
            retrieved_at=retrieved_at,
            provider_result_id=str(item["id"]) if item.get("id") else None,
            provider_score=float(item["score"]) if item.get("score") is not None else None,
            source_quality=classify_source_quality(domain),
        )
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


_OFFICIAL_DOMAINS = frozenset(
    {"bundesliga.com", "uefa.com", "fifa.com", "dfb.de", "premierleague.com", "laliga.com"}
)
_REPUTABLE_MEDIA_DOMAINS = frozenset(
    {
        "reuters.com",
        "bbc.com",
        "espn.com",
        "goal.com",
        "kicker.de",
        "liverpoolecho.co.uk",
        "skysports.com",
        "theathletic.com",
    }
)


def classify_source_quality(domain: str) -> SourceQuality:
    normalized = domain.casefold().removeprefix("www.")
    if any(normalized == item or normalized.endswith(f".{item}") for item in _OFFICIAL_DOMAINS):
        return SourceQuality.OFFICIAL
    if any(
        normalized == item or normalized.endswith(f".{item}")
        for item in _REPUTABLE_MEDIA_DOMAINS
    ):
        return SourceQuality.REPUTABLE_MEDIA
    return SourceQuality.OTHER
