"""Immutable ledger entries derived only from deterministic tool executions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ai.schemas import (
    ProductionStatus,
    SourceCategory,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolName,
)


class EvidenceCategory(str, Enum):
    ANALYTICS = "analytics"
    METHODOLOGY = "methodology"
    WEB = "web"


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    evidence_category: EvidenceCategory = EvidenceCategory.ANALYTICS
    tool_name: ToolName
    producer: str = "footyscout_tool_registry"
    normalized_arguments: dict[str, Any]
    execution_status: ToolExecutionStatus
    result: dict[str, Any] | None
    source_category: SourceCategory
    stable_entity_ids: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    internal_notes: tuple[str, ...] = ()
    production_status: ProductionStatus
    execution_order: int = Field(ge=1)
    methodology_topic: str | None = None
    methodology_sources: tuple[str, ...] = ()
    provenance: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_execution(
        cls,
        *,
        run_id: str,
        execution_order: int,
        execution: ToolExecutionResult,
    ) -> EvidenceRecord:
        result = execution.result or {}
        is_methodology = execution.tool_name is ToolName.GET_METHODOLOGY
        sources = result.get("sources", []) if is_methodology else []
        return cls(
            evidence_id=f"{run_id}:evidence-{execution_order}",
            evidence_category=(
                EvidenceCategory.METHODOLOGY if is_methodology else EvidenceCategory.ANALYTICS
            ),
            tool_name=execution.tool_name,
            normalized_arguments=execution.arguments,
            execution_status=execution.status,
            result=execution.result,
            source_category=execution.source_category,
            stable_entity_ids=tuple(execution.entity_ids),
            warnings=tuple(execution.warnings),
            internal_notes=tuple(execution.internal_notes),
            production_status=execution.production_status,
            execution_order=execution_order,
            methodology_topic=result.get("topic") if is_methodology else None,
            methodology_sources=tuple(
                source["source_id"]
                for source in sources
                if isinstance(source, dict) and isinstance(source.get("source_id"), str)
            ),
            provenance={
                "source_category": execution.source_category.value,
                "producer": "footyscout_tool_registry",
            },
        )


class EvidenceLedger(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[EvidenceRecord, ...] = ()

    def append(self, record: EvidenceRecord) -> EvidenceLedger:
        """Return a new ledger, preserving prior ledger instances unchanged."""
        expected_order = len(self.records) + 1
        if record.execution_order != expected_order:
            raise ValueError(f"Evidence order must be {expected_order}.")
        return EvidenceLedger(records=(*self.records, record))

    def extend(self, records: tuple[EvidenceRecord, ...]) -> EvidenceLedger:
        ledger = self
        for record in records:
            ledger = ledger.append(record)
        return ledger
