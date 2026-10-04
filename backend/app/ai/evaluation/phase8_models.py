"""Typed Phase 8 deterministic, judge, and run artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ai.evaluation.cases import GoldenEvalCase, JudgeCriterion
from app.ai.observability import AIScoutRunTrace, AIScoutTraceSummary

EVALUATION_SCHEMA_VERSION = "ai-scout-evaluation-v1"


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArgumentFailure(EvalModel):
    tool_name: str
    argument: str
    operator: str
    expected: Any = None
    actual: Any = None


class DeterministicCaseEvaluation(EvalModel):
    case_id: str
    planned_tools: tuple[str, ...] = ()
    normalized_tools: tuple[str, ...] = ()
    executed_tools: tuple[str, ...] = ()
    unreachable_tools: tuple[str, ...] = ()
    status_correct: bool
    status_conditionally_accepted: bool = False
    status_acceptance_reason: str | None = None
    intent_correct: bool | None
    entity_resolution_correct: bool | None = None
    tool_selection_correct: bool
    required_tools_present: bool
    required_tool_recall: float | None = Field(default=None, ge=0, le=1)
    forbidden_tools_absent: bool
    unexpected_tools: tuple[str, ...] = ()
    tool_argument_correct: bool | None = None
    argument_failures: tuple[ArgumentFailure, ...] = ()
    methodology_retrieval_correct: bool | None = None
    web_routing_correct: bool | None = None
    evidence_category_correct: bool | None = None
    citation_integrity_pass: bool | None = None
    citation_valid_references: int = Field(default=0, ge=0)
    citation_total_references: int = Field(default=0, ge=0)
    grounding_guard_pass: bool
    historical_current_authority_pass: bool | None = None
    unsupported_behavior_pass: bool | None = None
    deterministic_pass: bool
    failure_reasons: tuple[str, ...] = ()


class JudgeUsage(EvalModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    pricing_version: str | None = None


class CriterionJudgment(EvalModel):
    criterion: JudgeCriterion
    label: str
    rationale: str = Field(max_length=600)
    issues: tuple[str, ...] = ()
    prompt_version: str
    usage: JudgeUsage = Field(default_factory=JudgeUsage)


class JudgeError(EvalModel):
    criterion: JudgeCriterion
    error_type: str
    safe_message: str


class JudgeCriterionSkip(EvalModel):
    criterion: JudgeCriterion
    reason: str


class JudgeCaseEvaluation(EvalModel):
    case_id: str
    judge_provider: str | None = None
    judge_model: str | None = None
    eligible_criteria: tuple[JudgeCriterion, ...] = ()
    skipped_criteria: tuple[JudgeCriterionSkip, ...] = ()
    judgments: tuple[CriterionJudgment, ...] = ()
    errors: tuple[JudgeError, ...] = ()


class EvidenceForJudge(EvalModel):
    evidence_id: str
    category: str
    source_label: str
    content: str = Field(max_length=8000)


class AIScoutCaseEvaluation(EvalModel):
    golden_case: GoldenEvalCase
    trace: AIScoutRunTrace
    answer_status: str
    answer_markdown: str = Field(
        description="Backward-compatible alias of internal_answer_markdown."
    )
    internal_answer_markdown: str | None = None
    presented_answer_markdown: str | None = None
    evidence_for_judge: tuple[EvidenceForJudge, ...] = ()
    deterministic: DeterministicCaseEvaluation
    judge: JudgeCaseEvaluation | None = None


class MetricAggregate(EvalModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: float | None = Field(default=None, ge=0, le=1)


class DeterministicAggregates(EvalModel):
    case_count: int = Field(ge=0)
    expected_status_accuracy: MetricAggregate
    planner_intent_accuracy: MetricAggregate
    entity_resolution_accuracy: MetricAggregate
    tool_selection_accuracy: MetricAggregate
    required_tool_recall: MetricAggregate
    forbidden_tool_violation_rate: MetricAggregate
    tool_argument_accuracy: MetricAggregate
    methodology_retrieval_accuracy: MetricAggregate
    web_routing_accuracy: MetricAggregate
    evidence_category_accuracy: MetricAggregate
    citation_validity_case_rate: MetricAggregate
    citation_reference_validity: MetricAggregate
    historical_current_authority_accuracy: MetricAggregate
    unsupported_claim_violation_rate: MetricAggregate
    deterministic_overall_pass_rate: MetricAggregate


class LabelAggregate(EvalModel):
    counts: dict[str, int]
    rates: dict[str, float]
    denominator: int = Field(ge=0)


class JudgeAggregates(EvalModel):
    judge_eligible_criteria: int = Field(default=0, ge=0)
    judge_evaluated_criteria: int = Field(default=0, ge=0)
    judge_skipped_criteria: int = Field(default=0, ge=0)
    judge_coverage: MetricAggregate
    criteria: dict[str, LabelAggregate]
    total_input_tokens: int | None = Field(default=None, ge=0)
    total_output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_total_cost_usd: float | None = Field(default=None, ge=0)


class FailureAnalysis(EvalModel):
    groups: dict[str, tuple[str, ...]]
    judge_quality_warnings: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class EvaluationRunConfiguration(EvalModel):
    evaluation_schema_version: str = EVALUATION_SCHEMA_VERSION
    golden_dataset_version: str
    validation_mode: str
    planner_provider: str
    planner_model: str
    planner_prompt_version: str
    synthesis_provider: str
    synthesis_model: str
    synthesis_prompt_version: str
    judge_enabled: bool
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_prompt_versions: dict[str, str]
    pricing_version: str | None = None


class AIScoutEvaluationRun(EvalModel):
    evaluation_schema_version: str = EVALUATION_SCHEMA_VERSION
    run_id: str
    started_at: datetime
    completed_at: datetime
    configuration: EvaluationRunConfiguration
    golden_case_count: int = Field(ge=0)
    completed_case_ids: tuple[str, ...]
    deterministic_aggregates: DeterministicAggregates
    judge_aggregates: JudgeAggregates
    judge_calibration_status: str
    judge_calibration_report: dict[str, Any] | None = None
    observability_aggregates: AIScoutTraceSummary
    failure_analysis: FailureAnalysis
    product_estimated_cost_usd: float | None = Field(default=None, ge=0)
    judge_estimated_cost_usd: float | None = Field(default=None, ge=0)
    total_evaluation_estimated_cost_usd: float | None = Field(default=None, ge=0)
