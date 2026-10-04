"""Typed context passed to the synthesis boundary."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from app.ai.grounding import EvidenceRecord


class ContextStatus(str, Enum):
    SUFFICIENT = "sufficient"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class SelectedContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    intent: str
    analytics_evidence: tuple[EvidenceRecord, ...] = ()
    methodology_evidence: tuple[EvidenceRecord, ...] = ()
    web_evidence: tuple[EvidenceRecord, ...] = ()
    limitations: tuple[str, ...] = ()
    status: ContextStatus = ContextStatus.INSUFFICIENT_EVIDENCE

    @property
    def evidence(self) -> tuple[EvidenceRecord, ...]:
        return (*self.analytics_evidence, *self.methodology_evidence, *self.web_evidence)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(record.evidence_id for record in self.evidence)
