"""Deterministic bounded selection of normalized web results."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.ai.web.schemas import SourceQuality, WebSearchResult

_QUALITY_ORDER = {
    SourceQuality.OFFICIAL: 0,
    SourceQuality.REPUTABLE_MEDIA: 1,
    SourceQuality.OTHER: 2,
}


def select_web_results(
    results: tuple[WebSearchResult, ...],
    *,
    subject: str | None,
    limit: int = 3,
    max_evidence_age_hours: int | None = None,
    now: datetime | None = None,
) -> tuple[WebSearchResult, ...]:
    subject_terms = {
        term.strip(".,'\"")
        for term in (subject or "").casefold().split()
        if len(term.strip(".,'\"")) >= 2
    }
    deduplicated: dict[str, WebSearchResult] = {}
    for result in results:
        url = str(result.url).rstrip("/")
        deduplicated.setdefault(url, result)
    candidates = tuple(
        result
        for result in deduplicated.values()
        if not subject_terms or _entity_matches(result, subject_terms) > 0
    )
    if max_evidence_age_hours is not None:
        reference = now or max(
            (result.retrieved_at for result in candidates),
            default=None,
        )
        if reference is not None:
            cutoff = reference - timedelta(hours=max_evidence_age_hours)
            candidates = tuple(
                result
                for result in candidates
                if result.published_at is not None and result.published_at >= cutoff
            )
    ranked = sorted(
        candidates,
        key=lambda result: (
            _QUALITY_ORDER[result.source_quality],
            -_entity_matches(result, subject_terms),
            -(result.published_at.timestamp() if result.published_at else 0),
            result.domain,
            str(result.url),
        ),
    )
    selected: list[WebSearchResult] = []
    domains: set[str] = set()
    for result in ranked:
        if result.domain in domains and len(selected) >= 1:
            continue
        selected.append(result)
        domains.add(result.domain)
        if len(selected) == limit:
            break
    return tuple(selected)


def _entity_matches(result: WebSearchResult, subject_terms: set[str]) -> int:
    haystack = f"{result.title} {result.snippet}".casefold()
    return sum(term in haystack for term in subject_terms)
