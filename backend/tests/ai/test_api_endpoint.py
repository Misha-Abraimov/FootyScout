"""Safe public API projection for the existing AI Scout workflow."""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.ai.grounding import EvidenceCategory, EvidenceLedger, EvidenceRecord
from app.ai.schemas import (
    ProductionStatus,
    SourceCategory,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.synthesis import GroundedAnswerStatus, GroundedScoutAnswer
from app.ai.web import SourceQuality, WebSource
from app.main import app
from app.services.ai_scout import (
    get_ai_scout_runner,
    presented_ai_scout_answer,
    public_ai_scout_response,
)


class FakeRunner:
    def __init__(self, result: object | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.questions: list[str] = []

    def run(self, session: object, question: str) -> object:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return self.result


def _record(
    order: int,
    category: EvidenceCategory,
    *,
    result: dict[str, object],
    tool_name: ToolName | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"run:evidence-{order}",
        evidence_category=category,
        tool_name=tool_name or (
            ToolName.SEARCH_WEB
            if category is EvidenceCategory.WEB
            else ToolName.GET_METHODOLOGY
            if category is EvidenceCategory.METHODOLOGY
            else ToolName.GET_ROLE_FIT
        ),
        normalized_arguments={},
        execution_status=ToolExecutionStatus.SUCCESS,
        result=result,
        source_category=(
            SourceCategory.EXTERNAL_WEB
            if category is EvidenceCategory.WEB
            else SourceCategory.CURATED_DOCUMENTATION
            if category is EvidenceCategory.METHODOLOGY
            else SourceCategory.POSTGRESQL
        ),
        internal_notes=("never expose this",),
        production_status=(
            ProductionStatus.NOT_APPLICABLE
            if category is EvidenceCategory.WEB
            else ProductionStatus.PRODUCTION
        ),
        execution_order=order,
        methodology_topic="role_fit" if category is EvidenceCategory.METHODOLOGY else None,
        methodology_sources=(
            ("role_fit:primary",)
            if category is EvidenceCategory.METHODOLOGY
            else ()
        ),
    )


def _result(status: GroundedAnswerStatus = GroundedAnswerStatus.ANSWERED) -> object:
    analytics = _record(1, EvidenceCategory.ANALYTICS, result={"role_distance": 0.39})
    methodology = _record(2, EvidenceCategory.METHODOLOGY, result={"summary": "Lower is closer."})
    web = _record(
        3,
        EvidenceCategory.WEB,
        result={
            "title": "Official player update",
            "url": "https://example.com/player-update",
            "domain": "example.com",
            "published_at": "2026-09-29T12:00:00+00:00",
            "source_quality": "official",
            "unsafe_raw_body": "not public",
        },
    )
    evidence_ids = tuple(record.evidence_id for record in (analytics, methodology, web))
    answer = GroundedScoutAnswer(
        answer_markdown=(
            "## Role Fit\n\nGrounded answer [run:evidence-1; run:evidence-2].\n\n"
            "Current report [run:evidence-3]."
            if status is GroundedAnswerStatus.ANSWERED
            else "Please clarify the player."
        ),
        evidence_ids=evidence_ids if status is GroundedAnswerStatus.ANSWERED else (),
        status=status,
        methodology_sources=("role_fit:primary",) if status is GroundedAnswerStatus.ANSWERED else (),
        web_sources=(
            WebSource(
                evidence_id=web.evidence_id,
                title="Official player update",
                url="https://example.com/player-update",
                domain="example.com",
                published_at=datetime(2026, 9, 29, 12, tzinfo=UTC),
                source_quality=SourceQuality.OFFICIAL,
            ),
        )
        if status is GroundedAnswerStatus.ANSWERED
        else (),
        limitations=("Role Fit does not predict future performance.",),
    )
    return SimpleNamespace(
        run_id="run",
        answer=answer,
        evidence=EvidenceLedger(records=(analytics, methodology, web)),
        diagnostics={"validation_outcome": "repair", "hidden_prompt": "secret"},
    )


@pytest.fixture
def api_client(ai_client: TestClient) -> Generator[TestClient, None, None]:
    yield ai_client
    app.dependency_overrides.pop(get_ai_scout_runner, None)


@pytest.mark.parametrize(
    "answer_status",
    [
        GroundedAnswerStatus.ANSWERED,
        GroundedAnswerStatus.CLARIFICATION_REQUIRED,
        GroundedAnswerStatus.UNSUPPORTED,
        GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        GroundedAnswerStatus.ERROR,
    ],
)
def test_application_statuses_are_stable_http_responses(
    api_client: TestClient,
    answer_status: GroundedAnswerStatus,
) -> None:
    runner = FakeRunner(_result(answer_status))
    app.dependency_overrides[get_ai_scout_runner] = lambda: runner
    response = api_client.post("/api/ai-scout", json={"question": "  Explain Role Fit.  "})
    assert response.status_code == 200
    assert response.json()["status"] == answer_status.value
    assert runner.questions == ["Explain Role Fit."]


def test_answered_response_preserves_safe_sources_without_internal_leaks(
    api_client: TestClient,
) -> None:
    app.dependency_overrides[get_ai_scout_runner] = lambda: FakeRunner(_result())
    body = api_client.post(
        "/api/ai-scout",
        json={"question": "How does the player fit?"},
    ).json()
    assert body["run_id"] == "run"
    assert body["methodology_sources"] == ["role_fit:primary"]
    assert body["web_sources"][0]["url"] == "https://example.com/player-update"
    assert [source["category"] for source in body["sources"]] == [
        "analytics",
        "methodology",
        "web",
    ]
    assert body["limitations"] == ["Role Fit does not predict future performance."]
    assert "diagnostics" not in body
    assert "evidence" not in body
    assert "internal_notes" not in response_text(body)
    assert "unsafe_raw_body" not in response_text(body)
    assert "validation_outcome" not in response_text(body)


def test_public_sources_use_descriptive_deterministic_titles() -> None:
    player_result = {
        "player": {"player_id": 3500, "player_name": "Granit Xhaka"}
    }
    dossier = _record(
        1,
        EvidenceCategory.ANALYTICS,
        result=player_result,
        tool_name=ToolName.GET_PLAYER_DOSSIER,
    )
    role_fit = _record(
        2,
        EvidenceCategory.ANALYTICS,
        result=player_result,
        tool_name=ToolName.GET_ROLE_FIT,
    )
    methodology = _record(
        3,
        EvidenceCategory.METHODOLOGY,
        result={"summary": "Lower is closer."},
    )
    evidence_ids = (dossier.evidence_id, role_fit.evidence_id, methodology.evidence_id)
    result = SimpleNamespace(
        run_id="run",
        answer=GroundedScoutAnswer(
            answer_markdown="Grounded answer.",
            evidence_ids=evidence_ids,
            status=GroundedAnswerStatus.ANSWERED,
            methodology_sources=("role_fit:primary",),
        ),
        evidence=EvidenceLedger(records=(dossier, role_fit, methodology)),
    )

    response = public_ai_scout_response(result)

    assert [source.label for source in response.sources] == [
        "Player analytics — Granit Xhaka",
        "Role Fit analytics — Granit Xhaka",
        "Role Fit methodology",
    ]


def test_public_response_formats_metrics_and_suppresses_exact_duplicate_limitations() -> None:
    result = _result()
    result.answer = result.answer.model_copy(
        update={
            "answer_markdown": (
                "## Role Fit\n\nRaw role_distance: 0.38882682605732694.\n\n"
                "## Limitations\n\n- Role Fit does not predict future performance."
            )
        }
    )

    response = public_ai_scout_response(result)

    assert "Raw Role Fit distance: 0.389" in response.answer_markdown
    assert "## Limitations" not in response.answer_markdown
    assert response.limitations == ["Role Fit does not predict future performance."]


def test_presented_answer_uses_public_formatting_and_reader_source_numbers() -> None:
    result = _result()
    result.answer = result.answer.model_copy(
        update={
            "answer_markdown": (
                "expected_completion_rate: 0.923 "
                "[run:evidence-1; run:evidence-2]."
            )
        }
    )

    presented = presented_ai_scout_answer(result)

    assert presented == "Expected completion rate: 92.3% [1] [2]."
    assert "run:evidence" not in presented


def test_validation_repair_is_transparent_to_public_consumer(api_client: TestClient) -> None:
    app.dependency_overrides[get_ai_scout_runner] = lambda: FakeRunner(_result())
    body = api_client.post("/api/ai-scout", json={"question": "Question"}).json()
    assert body["status"] == "answered"
    assert "repair" not in body


def test_clarification_response_fails_closed_when_grounded_payload_is_inconsistent(
    api_client: TestClient,
) -> None:
    result = _result()
    result.answer = result.answer.model_copy(
        update={
            "status": GroundedAnswerStatus.CLARIFICATION_REQUIRED,
            "answer_markdown": (
                r"\## Style overview" + "\n\nComplete analysis [run:evidence-1]."
            ),
        }
    )
    app.dependency_overrides[get_ai_scout_runner] = lambda: FakeRunner(result)

    body = api_client.post(
        "/api/ai-scout",
        json={"question": "Explain the player."},
    ).json()

    assert body["status"] == "clarification_required"
    assert body["answer_markdown"] == (
        "Please provide a little more detail so AI Scout can identify the correct "
        "player or request."
    )
    assert body["evidence_ids"] == []
    assert body["methodology_sources"] == []
    assert body["web_sources"] == []
    assert body["sources"] == []
    assert body["limitations"] == []
    assert ":evidence-" not in response_text(body)
    assert "Complete analysis" not in response_text(body)


def test_unexpected_workflow_exception_is_safe_server_error(api_client: TestClient) -> None:
    app.dependency_overrides[get_ai_scout_runner] = lambda: FakeRunner(
        error=RuntimeError("provider secret should not be exposed")
    )
    response = api_client.post("/api/ai-scout", json={"question": "Question"})
    assert response.status_code == 500
    assert response.json() == {"detail": "AI Scout could not complete this request."}
    assert "provider secret" not in response.text


def test_question_contract_rejects_blank_and_unknown_fields(api_client: TestClient) -> None:
    app.dependency_overrides[get_ai_scout_runner] = lambda: FakeRunner(_result())
    assert api_client.post("/api/ai-scout", json={"question": "   "}).status_code == 422
    assert (
        api_client.post(
            "/api/ai-scout",
            json={"question": "Question", "debug": True},
        ).status_code
        == 422
    )


def response_text(value: object) -> str:
    return repr(value)
