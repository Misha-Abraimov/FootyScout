"""Strict and balanced policy behavior for deterministic answer validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai.content_hygiene import GroundingValidationError
from app.ai.grounding import EvidenceCategory, EvidenceLedger, EvidenceRecord
from app.ai.schemas import (
    ProductionStatus,
    SourceCategory,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.synthesis import (
    GroundedAnswerStatus,
    LLMGroundedAnswer,
    validate_answer_with_policy,
)
from app.ai.validation import (
    ValidationAction,
    ValidationFinding,
    ValidationMode,
    ValidationPolicy,
)
from app.config import Settings


def _record(
    order: int,
    *,
    category: EvidenceCategory,
    tool_name: ToolName,
    result: dict[str, object],
    methodology_topic: str | None = None,
    methodology_sources: tuple[str, ...] = (),
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"run:evidence-{order}",
        evidence_category=category,
        tool_name=tool_name,
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
        production_status=(
            ProductionStatus.NOT_APPLICABLE
            if category is EvidenceCategory.WEB
            else ProductionStatus.PRODUCTION
        ),
        execution_order=order,
        methodology_topic=methodology_topic,
        methodology_sources=methodology_sources,
    )


def _role_fit_record(order: int = 1) -> EvidenceRecord:
    return _record(
        order,
        category=EvidenceCategory.ANALYTICS,
        tool_name=ToolName.GET_ROLE_FIT,
        result={
            "role_distance": 0.3888,
            "feature_gaps": {"pressure_pass_rate": 0.21},
        },
    )


def _methodology_record(order: int = 2) -> EvidenceRecord:
    return _record(
        order,
        category=EvidenceCategory.METHODOLOGY,
        tool_name=ToolName.GET_METHODOLOGY,
        result={
            "summary": (
                "Role Fit measures resemblance to a supported team's observed "
                "positional-role style; lower raw distance means closer resemblance."
            )
        },
        methodology_topic="role_fit",
        methodology_sources=("role_fit:primary",),
    )


def _web_record(order: int = 3) -> EvidenceRecord:
    return _record(
        order,
        category=EvidenceCategory.WEB,
        tool_name=ToolName.SEARCH_WEB,
        result={
            "title": "Official squad profile",
            "url": "https://example.com/squad-profile",
            "domain": "example.com",
            "published_at": "2026-09-29T12:00:00Z",
            "source_quality": "official",
        },
    )


def _answer(markdown: str, *evidence_ids: str) -> LLMGroundedAnswer:
    return LLMGroundedAnswer(
        answer_markdown=markdown,
        evidence_ids=evidence_ids,
        status=GroundedAnswerStatus.ANSWERED,
    )


def _finding(code: str, line: int = 1) -> ValidationFinding:
    return ValidationFinding(
        error_code=code,
        rule=f"{code}_rule",
        message=f"{code} message",
        claim=f"claim at line {line}",
        line=line,
        end_line=line,
    )


def test_validation_mode_defaults_to_strict_and_accepts_supported_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_SCOUT_VALIDATION_MODE", raising=False)
    assert Settings(_env_file=None).ai_scout_validation_mode is ValidationMode.STRICT
    assert (
        Settings(_env_file=None, ai_scout_validation_mode="strict").ai_scout_validation_mode
        is ValidationMode.STRICT
    )
    assert (
        Settings(_env_file=None, ai_scout_validation_mode="balanced").ai_scout_validation_mode
        is ValidationMode.BALANCED
    )


def test_invalid_validation_mode_fails_configuration_validation() -> None:
    with pytest.raises(ValidationError, match="validation_mode"):
        Settings(_env_file=None, ai_scout_validation_mode="permissive")


@pytest.mark.parametrize(
    "error_code",
    [
        "unknown_evidence_id",
        "current_world_claim_requires_web",
        "uncalibrated_role_fit_band",
        "normative_role_fit_inference",
        "unsupported_implementation_claim",
        "unsupported_methodology_strengthening",
    ],
)
def test_strict_policy_blocks_every_existing_rejection(error_code: str) -> None:
    resolution = ValidationPolicy().resolve(
        (_finding(error_code),),
        ValidationMode.STRICT,
    )
    assert resolution.outcome is ValidationAction.BLOCK
    assert resolution.block_count == 1


@pytest.mark.parametrize(
    ("error_code", "expected"),
    [
        ("unknown_evidence_id", ValidationAction.BLOCK),
        ("undeclared_inline_citation", ValidationAction.BLOCK),
        ("uncalibrated_role_fit_band", ValidationAction.REPAIR),
        ("normative_role_fit_inference", ValidationAction.REPAIR),
        ("current_world_claim_requires_web", ValidationAction.REPAIR),
        ("unsupported_implementation_claim", ValidationAction.REPAIR),
        ("unsupported_methodology_strengthening", ValidationAction.REPAIR),
        ("non_fatal_validation_warning", ValidationAction.WARN),
    ],
)
def test_balanced_policy_maps_existing_rules_centrally(
    error_code: str,
    expected: ValidationAction,
) -> None:
    resolution = ValidationPolicy().resolve(
        (_finding(error_code),),
        ValidationMode.BALANCED,
    )
    assert resolution.outcome is expected


def test_no_findings_pass_in_both_modes() -> None:
    policy = ValidationPolicy()
    assert policy.resolve((), ValidationMode.STRICT).outcome is ValidationAction.PASS
    assert policy.resolve((), ValidationMode.BALANCED).outcome is ValidationAction.PASS


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        (
            ("non_fatal_validation_warning", "uncalibrated_role_fit_band"),
            ValidationAction.REPAIR,
        ),
        (
            ("normative_role_fit_inference", "unknown_evidence_id"),
            ValidationAction.BLOCK,
        ),
        (
            ("non_fatal_validation_warning", "undeclared_inline_citation"),
            ValidationAction.BLOCK,
        ),
    ],
)
def test_policy_uses_block_repair_warn_pass_priority(
    codes: tuple[str, ...],
    expected: ValidationAction,
) -> None:
    findings = tuple(_finding(code, line=index) for index, code in enumerate(codes, 1))
    resolution = ValidationPolicy().resolve(findings, ValidationMode.BALANCED)
    assert resolution.outcome is expected
    assert tuple(item.finding for item in resolution.findings) == findings


def test_strict_validation_preserves_unknown_evidence_failure() -> None:
    answer = _answer("Claim [run:evidence-999].", "run:evidence-999")
    with pytest.raises(GroundingValidationError) as captured:
        validate_answer_with_policy(
            answer,
            EvidenceLedger(),
            mode=ValidationMode.STRICT,
        )
    assert captured.value.error_code == "unknown_evidence_id"


@pytest.mark.parametrize(
    ("markdown", "expected_code"),
    [
        (
            "He currently plays for Leipzig [run:evidence-1].",
            "current_world_claim_requires_web",
        ),
        (
            "This is a moderate stylistic fit [run:evidence-1].",
            "uncalibrated_role_fit_band",
        ),
        (
            "Pressure passing is a weakness [run:evidence-1].",
            "normative_role_fit_inference",
        ),
    ],
)
def test_strict_validation_preserves_semantic_rejections(
    markdown: str,
    expected_code: str,
) -> None:
    record = _role_fit_record()
    with pytest.raises(GroundingValidationError) as captured:
        validate_answer_with_policy(
            _answer(markdown, record.evidence_id),
            EvidenceLedger(records=(record,)),
            mode=ValidationMode.STRICT,
        )
    assert captured.value.error_code == expected_code


def test_balanced_repairs_multiple_claims_in_deterministic_source_order() -> None:
    record = _role_fit_record()
    answer = _answer(
        "## Role Fit\n\n"
        "Role Fit distance is 0.389 [run:evidence-1].\n\n"
        "This is a moderate stylistic fit [run:evidence-1].\n\n"
        "Pressure-pass rate is a weakness that needs improvement [run:evidence-1].\n\n"
        "## Current situation\n\n"
        "Seiwald currently plays for Leipzig [run:evidence-1].",
        record.evidence_id,
    )

    result = validate_answer_with_policy(
        answer,
        EvidenceLedger(records=(record,)),
        mode=ValidationMode.BALANCED,
    )

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert "Role Fit distance is 0.389" in result.answer.answer_markdown
    assert "moderate stylistic fit" not in result.answer.answer_markdown
    assert "weakness" not in result.answer.answer_markdown
    assert "currently plays" not in result.answer.answer_markdown
    assert result.repair_applied is True
    assert result.repair_strategy == "remove_unsupported_claims"
    assert result.resolution.outcome is ValidationAction.REPAIR
    assert result.resolution.repair_count == 3
    assert tuple(
        item.finding.line for item in result.resolution.findings
    ) == (5, 7, 11)


def test_balanced_current_only_repair_degrades_to_insufficient_evidence() -> None:
    record = _role_fit_record()
    result = validate_answer_with_policy(
        _answer(
            "## Current situation\n\n"
            "Seiwald currently plays for Leipzig [run:evidence-1].",
            record.evidence_id,
        ),
        EvidenceLedger(records=(record,)),
        mode=ValidationMode.BALANCED,
        current_only=True,
    )
    assert result.answer.status is GroundedAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer.evidence_ids == ()
    assert "Leipzig" not in result.answer.answer_markdown
    assert result.repair_applied is True


def test_balanced_mixed_answer_repairs_only_unsupported_role_fit_band() -> None:
    role_fit = _role_fit_record()
    methodology = _methodology_record()
    web = _web_record()
    ledger = EvidenceLedger(records=(role_fit, methodology, web))
    answer = _answer(
        "## Role Fit\n\n"
        "Role Fit distance is 0.3888 [run:evidence-1].\n\n"
        "Lower distance means closer stylistic resemblance [run:evidence-2].\n\n"
        "This fit is not extreme [run:evidence-1].\n\n"
        "## Current situation\n\n"
        "Current reporting identifies Seiwald as a Leipzig player [run:evidence-3].",
        role_fit.evidence_id,
        methodology.evidence_id,
        web.evidence_id,
    )

    result = validate_answer_with_policy(
        answer,
        ledger,
        mode=ValidationMode.BALANCED,
    )
    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert "Role Fit distance is 0.3888" in result.answer.answer_markdown
    assert "Lower distance means closer" in result.answer.answer_markdown
    assert "Current reporting identifies" in result.answer.answer_markdown
    assert "not extreme" not in result.answer.answer_markdown
    assert result.answer.methodology_sources == ("role_fit:primary",)
    assert len(result.answer.web_sources) == 1
    assert result.resolution.repair_count == 1

    with pytest.raises(GroundingValidationError) as strict_error:
        validate_answer_with_policy(answer, ledger, mode=ValidationMode.STRICT)
    assert strict_error.value.error_code == "uncalibrated_role_fit_band"


def test_balanced_warning_preserves_answer_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = _finding("non_fatal_validation_warning")
    monkeypatch.setattr(
        "app.ai.synthesis.detect_answer_validation_findings",
        lambda answer, ledger: (warning,),
    )
    answer = _answer("Grounded answer.")
    result = validate_answer_with_policy(
        answer,
        EvidenceLedger(),
        mode=ValidationMode.BALANCED,
    )
    assert result.resolution.outcome is ValidationAction.WARN
    assert result.answer.answer_markdown == answer.answer_markdown
    assert result.repair_applied is False


def test_balanced_unknown_evidence_blocks_before_any_repair() -> None:
    record = _role_fit_record()
    answer = _answer(
        "This is a moderate fit [run:evidence-1].\n\n"
        "Invented claim [run:evidence-999].",
        record.evidence_id,
        "run:evidence-999",
    )
    with pytest.raises(GroundingValidationError) as captured:
        validate_answer_with_policy(
            answer,
            EvidenceLedger(records=(record,)),
            mode=ValidationMode.BALANCED,
        )
    assert captured.value.error_code == "unknown_evidence_id"
    assert captured.value.resolution is not None
    assert captured.value.resolution.outcome is ValidationAction.BLOCK


@pytest.mark.parametrize(
    "claim",
    (
        "Player identity used deterministic name matching in the query.",
        "The dossier does not return raw event rows.",
        "The API retrieves profiles from an internal endpoint.",
        "The database stores the profile in PostgreSQL tables.",
        "An internal tool queries the player records.",
    ),
)
def test_strict_mode_blocks_unsupported_internal_mechanics(claim: str) -> None:
    record = _record(
        1,
        category=EvidenceCategory.ANALYTICS,
        tool_name=ToolName.GET_PLAYER_DOSSIER,
        result={"player_name": "Alex Morgan", "matches_observed": 12},
    )
    with pytest.raises(GroundingValidationError) as captured:
        validate_answer_with_policy(
            _answer(f"{claim} [run:evidence-1]", record.evidence_id),
            EvidenceLedger(records=(record,)),
            mode=ValidationMode.STRICT,
        )

    assert captured.value.error_code == "unsupported_implementation_claim"
    assert captured.value.rule == "implementation_claims_require_explicit_cited_evidence"


def test_balanced_mode_removes_only_unsupported_internal_mechanics() -> None:
    record = _record(
        1,
        category=EvidenceCategory.ANALYTICS,
        tool_name=ToolName.GET_PLAYER_DOSSIER,
        result={"player_name": "Alex Morgan", "matches_observed": 12},
    )
    result = validate_answer_with_policy(
        _answer(
            "FootyScout's historical sample records Alex Morgan across 12 matches "
            "[run:evidence-1].\n\n"
            "Player identity used deterministic name matching in the query "
            "[run:evidence-1].",
            record.evidence_id,
        ),
        EvidenceLedger(records=(record,)),
        mode=ValidationMode.BALANCED,
    )

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert "historical sample records" in result.answer.answer_markdown
    assert "deterministic name matching" not in result.answer.answer_markdown
    assert result.repair_applied is True
    assert result.resolution.outcome is ValidationAction.REPAIR
    assert result.resolution.repair_count == 1


def test_explicit_cited_implementation_evidence_is_allowed() -> None:
    statement = "The documented API returns aggregated player profiles."
    record = _record(
        1,
        category=EvidenceCategory.METHODOLOGY,
        tool_name=ToolName.GET_METHODOLOGY,
        result={"content": statement},
        methodology_topic="api_contract",
        methodology_sources=("api_contract:primary",),
    )
    result = validate_answer_with_policy(
        _answer(f"{statement} [run:evidence-1]", record.evidence_id),
        EvidenceLedger(records=(record,)),
        mode=ValidationMode.STRICT,
    )

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert result.resolution.outcome is ValidationAction.PASS


def test_supported_methodology_explanation_is_not_an_implementation_claim() -> None:
    statement = "Role Fit measures resemblance to an observed positional-role style."
    record = _methodology_record()
    result = validate_answer_with_policy(
        _answer(f"{statement} [run:evidence-2]", record.evidence_id),
        EvidenceLedger(records=(record,)),
        mode=ValidationMode.STRICT,
    )

    assert result.answer.status is GroundedAnswerStatus.ANSWERED
    assert result.resolution.outcome is ValidationAction.PASS
