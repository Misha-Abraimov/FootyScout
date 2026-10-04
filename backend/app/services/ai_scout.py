"""Application boundary for the grounded AI Scout workflow and public response."""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.ai.grounding import EvidenceCategory, EvidenceRecord
from app.ai.observability.pricing import (
    PricingRegistry,
    load_default_pricing_registry,
    load_pricing_registry,
)
from app.ai.observability.sinks import JsonlTraceSink, NullTraceSink, TraceSink
from app.ai.observability.telemetry import current_correlation_id
from app.ai.presentation import (
    present_answer_markdown,
    present_numbered_citations,
    suppress_duplicate_limitation_section,
)
from app.ai.provider_factory import create_planner
from app.ai.synthesis import GroundedAnswerStatus, GroundedScoutAnswer, create_synthesizer
from app.ai.web.factory import create_web_search_provider
from app.ai.workflow import AIScoutWorkflow, AIScoutWorkflowResult
from app.config import Settings, settings
from app.schemas import (
    AIScoutResponse,
    AIScoutSourceResponse,
    AIScoutWebSourceResponse,
)


class AIScoutRunner:
    """Create and run the existing bounded workflow for one HTTP request."""

    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings
        self._trace_sink = _configured_trace_sink(app_settings)
        self._pricing_registry = _configured_pricing_registry(app_settings)

    def run(self, session: Session, question: str) -> AIScoutWorkflowResult:
        planner = create_planner(self._settings)
        synthesizer = create_synthesizer(self._settings)
        web_provider = create_web_search_provider(self._settings)
        return AIScoutWorkflow(
            session=session,
            planner=planner,
            synthesizer=synthesizer,
            web_search_provider=web_provider,
            web_search_enabled=self._settings.ai_scout_web_search_enabled,
            web_max_results=self._settings.ai_scout_web_max_results,
            web_content_max_age_hours=(
                self._settings.ai_scout_web_content_max_age_hours
            ),
            web_current_status_max_age_hours=(
                self._settings.ai_scout_web_current_status_max_age_hours
            ),
            validation_mode=self._settings.ai_scout_validation_mode,
            trace_sink=self._trace_sink,
            environment=self._settings.app_env,
            pricing_registry=self._pricing_registry,
        ).run(question, run_id=current_correlation_id())


def get_ai_scout_runner() -> AIScoutRunner:
    return AIScoutRunner(settings)


def _configured_trace_sink(app_settings: Settings) -> TraceSink:
    if app_settings.ai_scout_trace_enabled and app_settings.ai_scout_trace_path:
        return JsonlTraceSink(app_settings.ai_scout_trace_path)
    return NullTraceSink()


def _configured_pricing_registry(app_settings: Settings) -> PricingRegistry:
    configured = app_settings.ai_scout_pricing_registry_path
    return (
        load_pricing_registry(Path(configured))
        if configured
        else load_default_pricing_registry()
    )


def public_ai_scout_response(result: AIScoutWorkflowResult) -> AIScoutResponse:
    """Project a workflow result into the intentionally narrow public contract."""
    answer = result.answer
    if answer.status is not GroundedAnswerStatus.ANSWERED:
        return _public_terminal_response(result.run_id, answer)
    records_by_id = {record.evidence_id: record for record in result.evidence.records}
    cited_records = tuple(
        records_by_id[evidence_id]
        for evidence_id in answer.evidence_ids
        if evidence_id in records_by_id
    )
    public_markdown = present_answer_markdown(answer.answer_markdown)
    public_markdown = suppress_duplicate_limitation_section(
        public_markdown, answer.limitations
    )
    return AIScoutResponse(
        run_id=result.run_id,
        status=answer.status.value,
        answer_markdown=public_markdown,
        evidence_ids=list(answer.evidence_ids),
        methodology_sources=list(answer.methodology_sources),
        web_sources=[
            AIScoutWebSourceResponse(
                evidence_id=source.evidence_id,
                title=source.title,
                url=str(source.url),
                domain=source.domain,
                published_at=(
                    source.published_at.isoformat() if source.published_at else None
                ),
                source_quality=source.source_quality.value,
            )
            for source in answer.web_sources
        ],
        sources=[_public_source(record) for record in cited_records],
        limitations=list(answer.limitations),
    )


def presented_ai_scout_answer(result: AIScoutWorkflowResult) -> str:
    """Return the production-presented answer text used for quality evaluation."""
    response = public_ai_scout_response(result)
    return present_numbered_citations(
        response.answer_markdown,
        tuple(source.evidence_id for source in response.sources),
    )


_TERMINAL_FALLBACKS = {
    GroundedAnswerStatus.CLARIFICATION_REQUIRED: (
        "Please provide a little more detail so AI Scout can identify the correct "
        "player or request."
    ),
    GroundedAnswerStatus.UNSUPPORTED: "That request is not supported by AI Scout.",
    GroundedAnswerStatus.INSUFFICIENT_EVIDENCE: (
        "FootyScout does not have sufficient evidence to answer that request."
    ),
    GroundedAnswerStatus.ERROR: "AI Scout could not complete that request safely.",
}


def _public_terminal_response(
    run_id: str,
    answer: GroundedScoutAnswer,
) -> AIScoutResponse:
    """Fail closed when a terminal status is paired with grounded-answer payload data."""
    contaminated = bool(
        answer.evidence_ids
        or answer.methodology_sources
        or answer.web_sources
        or answer.limitations
        or _contains_grounded_answer_markup(answer.answer_markdown)
    )
    markdown = (
        _TERMINAL_FALLBACKS[answer.status]
        if contaminated
        else present_answer_markdown(answer.answer_markdown).strip()
    )
    return AIScoutResponse(
        run_id=run_id,
        status=answer.status.value,
        answer_markdown=markdown,
        evidence_ids=[],
        methodology_sources=[],
        web_sources=[],
        sources=[],
        limitations=[],
    )


def _contains_grounded_answer_markup(markdown: str) -> bool:
    return (
        re.search(r"(?:[0-9a-f-]{8,}:)?evidence-\d+", markdown, re.IGNORECASE)
        is not None
        or "\\##" in markdown
        or any(line.lstrip().startswith("#") for line in markdown.splitlines())
    )


def _public_source(record: EvidenceRecord) -> AIScoutSourceResponse:
    if record.evidence_category is EvidenceCategory.WEB and record.result is not None:
        return AIScoutSourceResponse(
            evidence_id=record.evidence_id,
            category="web",
            label=str(record.result["title"]),
            url=str(record.result["url"]),
            domain=str(record.result["domain"]),
            published_at=_serialized_date(record.result.get("published_at")),
        )
    if record.evidence_category is EvidenceCategory.METHODOLOGY:
        return AIScoutSourceResponse(
            evidence_id=record.evidence_id,
            category="methodology",
            label=_methodology_source_label(record.methodology_topic),
        )
    return AIScoutSourceResponse(
        evidence_id=record.evidence_id,
        category="analytics",
        label=_analytics_source_label(record),
    )


_ANALYTICS_SOURCE_LABELS = {
    "search_players": "Player profile data",
    "get_player_dossier": "Player analytics",
    "compare_players": "Player comparison",
    "get_similar_players": "Playing-style similarity",
    "get_leaderboard": "FootyScout leaderboard",
    "get_team_intelligence": "Team intelligence",
    "get_role_fit": "Role Fit analytics",
    "get_role_recommendations": "Role recommendations",
}

_METHODOLOGY_SOURCE_LABELS = {
    "xpass": "Expected Pass (xPass) methodology",
    "xg": "Expected Goals (xG) methodology",
    "possession_value": "Possession Value methodology",
    "attacking_impact": "Attacking Impact methodology",
    "player_profiles": "Player Profiles methodology",
    "percentiles": "Same-position percentiles methodology",
    "archetypes": "Player archetypes methodology",
    "similarity": "Playing-style similarity methodology",
    "team_intelligence": "Team intelligence methodology",
    "role_fit": "Role Fit methodology",
    "role_recommendations": "Role recommendations methodology",
}


def _analytics_source_label(record: EvidenceRecord) -> str:
    label = _ANALYTICS_SOURCE_LABELS.get(record.tool_name.value, "FootyScout analytics")
    player_name = _result_player_name(record.result)
    return f"{label} — {player_name}" if player_name else label


def _methodology_source_label(topic: str | None) -> str:
    if topic is None:
        return "FootyScout methodology"
    return _METHODOLOGY_SOURCE_LABELS.get(
        topic,
        f"{topic.replace('_', ' ').title()} methodology",
    )


def _result_player_name(result: dict[str, object] | None) -> str | None:
    if not result:
        return None
    player = result.get("player")
    if not isinstance(player, dict):
        return None
    name = player.get("player_name")
    return str(name) if isinstance(name, str) and name.strip() else None


def _serialized_date(value: object) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat()) if callable(isoformat) else str(value)
