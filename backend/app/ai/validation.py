"""Deterministic policy for grounded-answer validation findings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ValidationMode(str, Enum):
    """Configured answer-validation behavior."""

    STRICT = "strict"
    BALANCED = "balanced"


class ValidationAction(str, Enum):
    """Policy response to one or more deterministic findings."""

    PASS = "pass"
    WARN = "warn"
    REPAIR = "repair"
    BLOCK = "block"


@dataclass(frozen=True)
class ValidationFinding:
    """Safe structured output from deterministic grounding detection."""

    error_code: str
    rule: str
    message: str
    claim: str | None = None
    line: int | None = None
    end_line: int | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedValidationFinding:
    """One finding paired with its centralized policy action."""

    finding: ValidationFinding
    action: ValidationAction


@dataclass(frozen=True)
class ValidationResolution:
    """Deterministic aggregate policy decision in source order."""

    mode: ValidationMode
    outcome: ValidationAction
    findings: tuple[ResolvedValidationFinding, ...] = ()

    @property
    def warning_count(self) -> int:
        return self._count(ValidationAction.WARN)

    @property
    def repair_count(self) -> int:
        return self._count(ValidationAction.REPAIR)

    @property
    def block_count(self) -> int:
        return self._count(ValidationAction.BLOCK)

    def _count(self, action: ValidationAction) -> int:
        return sum(item.action is action for item in self.findings)


_BALANCED_ACTIONS: dict[str, ValidationAction] = {
    "unknown_evidence_id": ValidationAction.BLOCK,
    "undeclared_inline_citation": ValidationAction.BLOCK,
    "unsupported_methodology_term": ValidationAction.REPAIR,
    "uncalibrated_role_fit_band": ValidationAction.REPAIR,
    "unsupported_role_fit_driver_language": ValidationAction.REPAIR,
    "unsupported_role_fit_limitation_expansion": ValidationAction.REPAIR,
    "normative_role_fit_inference": ValidationAction.REPAIR,
    "current_world_claim_requires_web": ValidationAction.REPAIR,
    "unsupported_implementation_claim": ValidationAction.REPAIR,
    "unsupported_methodology_strengthening": ValidationAction.REPAIR,
    # No production detector emits this code yet; it keeps WARN policy testable
    # without weakening or inventing a grounding detector.
    "non_fatal_validation_warning": ValidationAction.WARN,
}
_ACTION_PRIORITY = {
    ValidationAction.PASS: 0,
    ValidationAction.WARN: 1,
    ValidationAction.REPAIR: 2,
    ValidationAction.BLOCK: 3,
}


class ValidationPolicy:
    """Map detection findings to actions without embedding detection logic."""

    def resolve(
        self,
        findings: tuple[ValidationFinding, ...],
        mode: ValidationMode,
    ) -> ValidationResolution:
        resolved = tuple(
            ResolvedValidationFinding(
                finding=finding,
                action=self.action_for(finding, mode),
            )
            for finding in findings
        )
        outcome = max(
            (item.action for item in resolved),
            key=_ACTION_PRIORITY.__getitem__,
            default=ValidationAction.PASS,
        )
        return ValidationResolution(mode=mode, outcome=outcome, findings=resolved)

    def action_for(
        self,
        finding: ValidationFinding,
        mode: ValidationMode,
    ) -> ValidationAction:
        if mode is ValidationMode.STRICT:
            return ValidationAction.BLOCK
        return _BALANCED_ACTIONS.get(finding.error_code, ValidationAction.BLOCK)

    def block(
        self,
        findings: tuple[ValidationFinding, ...],
        mode: ValidationMode,
    ) -> ValidationResolution:
        """Fail closed after the single allowed repair/revalidation pass."""
        return ValidationResolution(
            mode=mode,
            outcome=ValidationAction.BLOCK,
            findings=tuple(
                ResolvedValidationFinding(
                    finding=finding,
                    action=ValidationAction.BLOCK,
                )
                for finding in findings
            ),
        )
