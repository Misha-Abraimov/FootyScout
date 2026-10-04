"""Bounded detection and query construction for explicit current-information needs."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from app.ai.provider_schemas import LLMPlannerDecision
from app.ai.schemas import IntentKind
from app.ai.web.schemas import CurrentContextCategory, CurrentContextRequest

_CURRENT_PATTERNS: tuple[tuple[CurrentContextCategory, re.Pattern[str]], ...] = (
    (CurrentContextCategory.INJURY_STATUS, re.compile(r"\b(injur(?:ed|y)|fitness update)\b", re.IGNORECASE)),
    (
        CurrentContextCategory.AVAILABILITY,
        re.compile(
            r"\b(?:is|was|remains?|currently)\s+[^?!.]{1,100}\s+available\b|"
            r"\bavailability(?:\s+(?:status|update))?\b",
            re.IGNORECASE,
        ),
    ),
    (CurrentContextCategory.CURRENT_CLUB, re.compile(r"\bcurrent(?:ly)?\s+(?:club|play(?:s|ing)? for)\b|\bwhat club\b", re.IGNORECASE)),
    (CurrentContextCategory.TRANSFER_REPORTING, re.compile(r"\b(?:latest|recent) transfer|transfer (?:news|rumou?rs?|report(?:s|ing)?)\b", re.IGNORECASE)),
    (
        CurrentContextCategory.CURRENT_MANAGER,
        re.compile(
            r"\b(?:current manager|who (?:currently )?manages|manager now)\b",
            re.IGNORECASE,
        ),
    ),
    (CurrentContextCategory.RECENT_NEWS, re.compile(r"\b(latest|recent(?:ly)?|current situation|what'?s new|right now|today|now)\b", re.IGNORECASE)),
)
_ANALYTICS_LANGUAGE = re.compile(
    r"\b(pass(?:ing)? profile|pass completion|xpass|expected goals|\bxg\b|"
    r"attacking impact|role fit|similar(?:ity| players)|leaderboard|percentile|"
    r"team intelligence|(?:observed|positional|midfield|defensive|attacking)\s+role|"
    r"best fits?|how well (?:does|would) .+? fit|"
    r"scouting recommendation|historical|statistics?|stats)\b",
    re.IGNORECASE,
)
_FRESHNESS_SENSITIVE_CATEGORIES = frozenset(
    {
        CurrentContextCategory.INJURY_STATUS,
        CurrentContextCategory.AVAILABILITY,
        CurrentContextCategory.RECENT_NEWS,
        CurrentContextCategory.TRANSFER_REPORTING,
    }
)
_INHERENTLY_ANALYTICAL_INTENTS = frozenset(
    {
        IntentKind.PLAYER_COMPARISON,
        IntentKind.SIMILAR_PLAYERS,
        IntentKind.LEADERBOARD,
        IntentKind.TEAM_ANALYSIS,
        IntentKind.ROLE_FIT,
        IntentKind.ROLE_RECOMMENDATIONS,
        IntentKind.METHODOLOGY,
    }
)
_MONTHS = {
    name: index
    for index, name in enumerate(
        (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ),
        start=1,
    )
}


def detect_current_context(
    question: str,
    decision: LLMPlannerDecision | None,
    *,
    normalized_intent: IntentKind | None = None,
    content_max_age_hours: int | None,
    evidence_max_age_hours: int | None = None,
    now: datetime | None = None,
) -> CurrentContextRequest:
    analytics_requested = _ANALYTICS_LANGUAGE.search(question) is not None or bool(
        normalized_intent in _INHERENTLY_ANALYTICAL_INTENTS
        or decision is not None
        and (
            decision.intent in _INHERENTLY_ANALYTICAL_INTENTS
            or any(
                intent in _INHERENTLY_ANALYTICAL_INTENTS
                for intent in decision.requested_analyses
            )
        )
    )
    matched_category = next(
        (category for category, pattern in _CURRENT_PATTERNS if pattern.search(question)),
        None,
    )
    publication_filter_start = _publication_filter_start(
        question,
        now=now or datetime.now(UTC),
    )
    category = matched_category
    if category is None and publication_filter_start is not None:
        category = CurrentContextCategory.RECENT_NEWS
    if category is None:
        return CurrentContextRequest(analytics_requested_by_user=analytics_requested)
    # Prefer the entity explicitly present in the user's current-information clause.
    # Planner metadata is only a fallback for mixed questions such as "...and what is
    # his current situation?", where the clause itself contains only a pronoun.
    subject = _extract_subject(question, category) or _decision_subject(decision, category)
    current_only = not analytics_requested
    return CurrentContextRequest(
        required=True,
        category=category,
        subject=subject,
        publication_filter_start=publication_filter_start,
        content_max_age_hours=(
            content_max_age_hours
            if matched_category in _FRESHNESS_SENSITIVE_CATEGORIES
            else None
        ),
        evidence_max_age_hours=(
            evidence_max_age_hours
            if matched_category in _FRESHNESS_SENSITIVE_CATEGORIES
            else None
        ),
        reason=f"Explicit current-information language detected for {category.value}.",
        analytics_requested_by_user=analytics_requested,
        current_only=current_only,
    )


def build_search_query(request: CurrentContextRequest, *, fallback_subject: str | None = None) -> str | None:
    if not request.required or request.category is None:
        return None
    subject = (request.subject or fallback_subject or "").strip()
    if not subject:
        return None
    suffix = {
        CurrentContextCategory.INJURY_STATUS: "injury status latest",
        CurrentContextCategory.AVAILABILITY: "availability next match latest",
        CurrentContextCategory.CURRENT_CLUB: "current club",
        CurrentContextCategory.TRANSFER_REPORTING: "transfer latest",
        CurrentContextCategory.CURRENT_MANAGER: "current manager",
        CurrentContextCategory.CURRENT_TEAM_CONTEXT: "current team latest",
        CurrentContextCategory.RECENT_NEWS: "latest news",
        CurrentContextCategory.OTHER_CURRENT_FACT: "current latest",
    }[request.category]
    publication_clause = (
        f" since {request.publication_filter_start.date().isoformat()}"
        if request.publication_filter_start is not None
        else ""
    )
    return f"{subject} {suffix}{publication_clause}"[:300]


def _publication_filter_start(question: str, *, now: datetime) -> datetime | None:
    reference = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    relative_days = re.search(
        r"\b(?:in\s+)?(?:the\s+)?(?:last|past)\s+(\d{1,4})\s+days?\b",
        question,
        re.IGNORECASE,
    )
    if relative_days:
        days = int(relative_days.group(1))
        if 1 <= days <= 3650:
            return reference - timedelta(days=days)
    if re.search(
        r"\b(?:in\s+)?(?:the\s+)?(?:last|past)\s+month\b",
        question,
        re.IGNORECASE,
    ):
        return reference - timedelta(days=30)

    iso_date = re.search(r"\bsince\s+(20\d{2})-(\d{1,2})-(\d{1,2})\b", question)
    if iso_date:
        return _safe_datetime(
            int(iso_date.group(1)),
            int(iso_date.group(2)),
            int(iso_date.group(3)),
        )

    named_date = re.search(
        r"\bsince\s+(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:,?\s+(20\d{2}))?\b",
        question,
        re.IGNORECASE,
    )
    if named_date:
        month = _MONTHS[named_date.group(1).casefold()]
        year = int(named_date.group(3)) if named_date.group(3) else reference.year
        candidate = _safe_datetime(year, month, int(named_date.group(2)))
        if candidate is not None and not named_date.group(3) and candidate > reference:
            candidate = candidate.replace(year=year - 1)
        return candidate

    year_window = re.search(
        r"\bnews(?:\s+(?:on|about)\s+.+?)?\s+(?:from|in)\s+(20\d{2})\b",
        question,
        re.IGNORECASE,
    )
    if year_window:
        return datetime(int(year_window.group(1)), 1, 1, tzinfo=UTC)
    return None


def _safe_datetime(year: int, month: int, day: int) -> datetime | None:
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def _decision_subject(
    decision: LLMPlannerDecision | None,
    category: CurrentContextCategory,
) -> str | None:
    if decision is None:
        return None
    if decision.primary_player_name.strip():
        return decision.primary_player_name.strip()
    if category in {
        CurrentContextCategory.CURRENT_MANAGER,
        CurrentContextCategory.CURRENT_TEAM_CONTEXT,
    } and decision.team_name.strip():
        return decision.team_name.strip()
    return None


def _extract_subject(question: str, category: CurrentContextCategory) -> str | None:
    patterns = {
        CurrentContextCategory.INJURY_STATUS: (
            r"\bis\s+(.+?)\s+injured\b",
            r"\binjury(?: status| update)?\s+(?:for|on)\s+(.+?)(?:\?|$)",
        ),
        CurrentContextCategory.AVAILABILITY: (r"\bis\s+(.+?)\s+available\b",),
        CurrentContextCategory.CURRENT_CLUB: (r"\bwhat club does\s+(.+?)\s+currently\b",),
        CurrentContextCategory.TRANSFER_REPORTING: (r"\btransfer.*?\babout\s+(.+?)(?:\?|$)",),
        CurrentContextCategory.CURRENT_MANAGER: (
            r"\bwho (?:currently )?manages\s+(.+?)(?:\s+(?:now|currently))?(?:\?|$)",
        ),
        CurrentContextCategory.RECENT_NEWS: (
            r"\blatest(?: reliable update)?\s+(?:on|about)\s+(.+?)(?:\?|$)",
            r"\bnews\s+(?:on|about)\s+(.+?)\s+(?:in|from|since)\b",
            r"\bwith\s+(.+?)\s+since\b",
        ),
    }.get(category, ())
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            subject = match.group(1).strip(" .?!")
            if not _is_placeholder_subject(subject):
                return subject
    return None


def _is_placeholder_subject(subject: str) -> bool:
    normalized = re.sub(r"\s+", " ", subject.casefold()).strip()
    return normalized in {
        "he",
        "her",
        "him",
        "his",
        "she",
        "that player",
        "that player currently",
        "the player",
        "they",
        "them",
        "their",
    }
