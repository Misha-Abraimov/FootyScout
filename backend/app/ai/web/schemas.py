"""FootyScout-owned schemas for bounded current-web context."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from app.ai.content_hygiene import sanitize_user_facing_text

WEB_QUERY_MAX_LENGTH = 300
WEB_RESULTS_HARD_MAX = 5
WEB_SNIPPET_MAX_LENGTH = 1600


class WebModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CurrentContextCategory(str, Enum):
    CURRENT_CLUB = "current_club"
    INJURY_STATUS = "injury_status"
    AVAILABILITY = "availability"
    RECENT_NEWS = "recent_news"
    TRANSFER_REPORTING = "transfer_reporting"
    CURRENT_TEAM_CONTEXT = "current_team_context"
    CURRENT_MANAGER = "current_manager"
    OTHER_CURRENT_FACT = "other_current_fact"


class SourceQuality(str, Enum):
    OFFICIAL = "official"
    REPUTABLE_MEDIA = "reputable_media"
    OTHER = "other"


class CurrentContextRequest(WebModel):
    required: bool = False
    category: CurrentContextCategory | None = None
    subject: str | None = Field(default=None, max_length=200)
    query: str | None = Field(default=None, max_length=WEB_QUERY_MAX_LENGTH)
    publication_filter_start: datetime | None = None
    content_max_age_hours: int | None = Field(default=None, ge=1, le=168)
    evidence_max_age_hours: int | None = Field(default=None, ge=1, le=720)
    reason: str | None = Field(default=None, max_length=300)
    analytics_requested_by_user: bool = False
    current_only: bool = False


class WebSearchRequest(WebModel):
    query: str = Field(min_length=1, max_length=WEB_QUERY_MAX_LENGTH)
    max_results: int = Field(default=5, ge=1, le=WEB_RESULTS_HARD_MAX)
    publication_filter_start: datetime | None = None
    content_max_age_hours: int | None = Field(default=None, ge=1, le=168)
    preferred_domains: tuple[str, ...] | None = None
    excluded_domains: tuple[str, ...] | None = None


class WebSearchResult(WebModel):
    title: str = Field(min_length=1, max_length=500)
    url: HttpUrl
    domain: str = Field(min_length=1, max_length=253)
    published_at: datetime | None = None
    author: str | None = Field(default=None, max_length=300)
    snippet: str = Field(min_length=1, max_length=WEB_SNIPPET_MAX_LENGTH)
    retrieved_at: datetime
    provider_result_id: str | None = Field(default=None, max_length=300)
    provider_score: float | None = None
    source_quality: SourceQuality = SourceQuality.OTHER

    @field_validator("title", "author", "snippet")
    @classmethod
    def sanitize_external_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return sanitize_user_facing_text(value).strip()


class WebSearchResponse(WebModel):
    query: str
    results: tuple[WebSearchResult, ...] = ()
    provider: str
    retrieved_at: datetime
    warnings: tuple[str, ...] = ()
    provider_attempts: int = Field(default=1, ge=1)
    retry_count: int = Field(default=0, ge=0)
    rate_limit_events: int = Field(default=0, ge=0)


class WebSource(WebModel):
    evidence_id: str
    title: str
    url: HttpUrl
    domain: str
    published_at: datetime | None = None
    source_quality: SourceQuality
