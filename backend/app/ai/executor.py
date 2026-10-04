"""Bounded, single-pass execution of normalized deterministic tool plans."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.ai.grounding import EvidenceLedger, EvidenceRecord
from app.ai.observability.telemetry import set_current_span_attributes, start_span
from app.ai.plan_normalizer import ALLOWED_TOOLS_BY_INTENT
from app.ai.policy import MAX_TOOL_CALLS
from app.ai.schemas import NormalizedPlan, ToolExecutionStatus
from app.ai.tools.registry import execute_tool


class PlanExecutionStatus(str, Enum):
    COMPLETED = "completed"
    STOPPED_ON_ERROR = "stopped_on_error"


class PlanExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PlanExecutionStatus
    evidence: EvidenceLedger
    stopped_on_call_id: str | None = None


def execute_plan(
    session: Session,
    plan: NormalizedPlan,
    *,
    run_id: str,
) -> PlanExecutionResult:
    """Execute each prevalidated call once, in order, with no dynamic replanning."""
    if len(plan.calls) > MAX_TOOL_CALLS:
        raise ValueError(f"Execution plans may contain at most {MAX_TOOL_CALLS} calls.")
    allowed = ALLOWED_TOOLS_BY_INTENT[plan.intent.kind]
    unauthorized = tuple(call.name for call in plan.calls if call.name not in allowed)
    if unauthorized:
        names = ", ".join(name.value for name in unauthorized)
        raise ValueError(
            f"Execution refused tools not authorized for {plan.intent.kind.value}: {names}."
        )

    ledger = EvidenceLedger()
    for order, call in enumerate(plan.calls, start=1):
        with start_span(
            "ai_scout.tool",
            {
                "ai.workflow": "ai_scout",
                "ai.tool.name": call.name.value,
                "ai.tool.order": order,
            },
        ):
            execution = execute_tool(session, call.name, call.arguments)
            set_current_span_attributes(
                {
                    "ai.tool.success": execution.status is ToolExecutionStatus.SUCCESS,
                    "ai.tool.status": execution.status.value,
                }
            )
        ledger = ledger.append(
            EvidenceRecord.from_execution(
                run_id=run_id,
                execution_order=order,
                execution=execution,
            )
        )
        if execution.status is not ToolExecutionStatus.SUCCESS:
            return PlanExecutionResult(
                status=PlanExecutionStatus.STOPPED_ON_ERROR,
                evidence=ledger,
                stopped_on_call_id=call.call_id,
            )
    return PlanExecutionResult(status=PlanExecutionStatus.COMPLETED, evidence=ledger)
