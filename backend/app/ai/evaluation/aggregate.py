"""Phase 8 quality aggregates layered over Phase 7 operational summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime

from app.ai.evaluation.calibration import JudgeCalibrationReport
from app.ai.evaluation.phase8_models import (
    AIScoutCaseEvaluation,
    AIScoutEvaluationRun,
    DeterministicAggregates,
    EvaluationRunConfiguration,
    FailureAnalysis,
    JudgeAggregates,
    LabelAggregate,
    MetricAggregate,
)
from app.ai.observability import summarize_traces


def aggregate_evaluation(
    cases: tuple[AIScoutCaseEvaluation, ...],
    *,
    run_id: str,
    started_at: datetime,
    configuration: EvaluationRunConfiguration,
    judge_calibration_status: str,
    judge_calibration_report: JudgeCalibrationReport | None = None,
) -> AIScoutEvaluationRun:
    deterministic = aggregate_deterministic(cases)
    judges = aggregate_judges(cases, enabled=configuration.judge_enabled)
    observability = summarize_traces([case.trace for case in cases])
    failures = failure_analysis(cases)
    product_cost = observability.total_estimated_cost_usd
    judge_cost = judges.estimated_total_cost_usd
    total_cost = product_cost if not configuration.judge_enabled else (
        product_cost + judge_cost
        if product_cost is not None and judge_cost is not None
        else None
    )
    return AIScoutEvaluationRun(
        run_id=run_id,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        configuration=configuration,
        golden_case_count=len(cases),
        completed_case_ids=tuple(case.golden_case.case_id for case in cases),
        deterministic_aggregates=deterministic,
        judge_aggregates=judges,
        judge_calibration_status=judge_calibration_status,
        judge_calibration_report=(
            judge_calibration_report.model_dump(mode="json")
            if judge_calibration_report is not None
            else None
        ),
        observability_aggregates=observability,
        failure_analysis=failures,
        product_estimated_cost_usd=product_cost,
        judge_estimated_cost_usd=judge_cost,
        total_evaluation_estimated_cost_usd=total_cost,
    )


def aggregate_deterministic(
    cases: tuple[AIScoutCaseEvaluation, ...],
) -> DeterministicAggregates:
    results = tuple(case.deterministic for case in cases)
    required_numerator = 0
    required_denominator = 0
    for case, result in zip(cases, results, strict=True):
        required_count = len(set(case.golden_case.required_tools))
        if required_count and result.required_tool_recall is not None:
            required_denominator += required_count
            required_numerator += round(result.required_tool_recall * required_count)
    valid_citations = sum(result.citation_valid_references for result in results)
    total_citations = sum(result.citation_total_references for result in results)
    return DeterministicAggregates(
        case_count=len(cases),
        expected_status_accuracy=_bool_metric(result.status_correct for result in results),
        planner_intent_accuracy=_optional_metric(
            result.intent_correct for result in results
        ),
        entity_resolution_accuracy=_optional_metric(
            result.entity_resolution_correct for result in results
        ),
        tool_selection_accuracy=_bool_metric(
            result.tool_selection_correct for result in results
        ),
        required_tool_recall=_metric(required_numerator, required_denominator),
        forbidden_tool_violation_rate=_violation_metric(
            result.forbidden_tools_absent
            for case, result in zip(cases, results, strict=True)
            if case.golden_case.forbidden_tools
        ),
        tool_argument_accuracy=_optional_metric(
            result.tool_argument_correct for result in results
        ),
        methodology_retrieval_accuracy=_optional_metric(
            result.methodology_retrieval_correct for result in results
        ),
        web_routing_accuracy=_optional_metric(
            result.web_routing_correct for result in results
        ),
        evidence_category_accuracy=_optional_metric(
            result.evidence_category_correct for result in results
        ),
        citation_validity_case_rate=_optional_metric(
            result.citation_integrity_pass for result in results
        ),
        citation_reference_validity=_metric(valid_citations, total_citations),
        historical_current_authority_accuracy=_optional_metric(
            result.historical_current_authority_pass for result in results
        ),
        unsupported_claim_violation_rate=_optional_violation_metric(
            result.unsupported_behavior_pass for result in results
        ),
        deterministic_overall_pass_rate=_bool_metric(
            result.deterministic_pass for result in results
        ),
    )


def aggregate_judges(
    cases: tuple[AIScoutCaseEvaluation, ...],
    *,
    enabled: bool | None = None,
) -> JudgeAggregates:
    judges_enabled = enabled if enabled is not None else any(case.judge for case in cases)
    eligible_count = 0
    skipped_count = 0
    if judges_enabled:
        for case in cases:
            if case.judge is None:
                continue
            if case.judge.eligible_criteria or case.judge.skipped_criteria:
                eligible_count += len(case.judge.eligible_criteria)
                skipped_count += len(case.judge.skipped_criteria)
            else:
                # Backward compatibility for artifacts written before eligibility
                # was recorded explicitly.
                eligible_count += len(case.golden_case.judge_criteria_enabled)
    judgments = tuple(
        judgment
        for case in cases
        if case.judge is not None
        for judgment in case.judge.judgments
    )
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for judgment in judgments:
        labels[judgment.criterion.value][judgment.label] += 1
    criteria = {
        criterion: LabelAggregate(
            counts=dict(sorted(counts.items())),
            rates={
                label: count / sum(counts.values())
                for label, count in sorted(counts.items())
            },
            denominator=sum(counts.values()),
        )
        for criterion, counts in sorted(labels.items())
    }
    usages = [judgment.usage for judgment in judgments]
    input_tokens = [usage.input_tokens for usage in usages if usage.input_tokens is not None]
    output_tokens = [usage.output_tokens for usage in usages if usage.output_tokens is not None]
    total_tokens = [usage.total_tokens for usage in usages if usage.total_tokens is not None]
    costs = [
        usage.estimated_cost_usd
        for usage in usages
        if usage.estimated_cost_usd is not None
    ]
    return JudgeAggregates(
        judge_eligible_criteria=eligible_count,
        judge_evaluated_criteria=len(judgments),
        judge_skipped_criteria=skipped_count,
        judge_coverage=_metric(len(judgments), eligible_count),
        criteria=criteria,
        total_input_tokens=sum(input_tokens) if input_tokens else None,
        total_output_tokens=sum(output_tokens) if output_tokens else None,
        total_tokens=sum(total_tokens) if total_tokens else None,
        estimated_total_cost_usd=sum(costs) if costs else None,
    )


def failure_analysis(cases: tuple[AIScoutCaseEvaluation, ...]) -> FailureAnalysis:
    groups: dict[str, list[str]] = defaultdict(list)
    quality_warnings: dict[str, list[str]] = defaultdict(list)
    mappings = {
        "intent_failures": "intent_correct",
        "entity_resolution_failures": "entity_resolution_correct",
        "tool_selection_failures": "tool_selection_correct",
        "tool_argument_failures": "tool_argument_correct",
        "methodology_failures": "methodology_retrieval_correct",
        "web_routing_failures": "web_routing_correct",
        "citation_failures": "citation_integrity_pass",
        "authority_failures": "historical_current_authority_pass",
        "unsupported_claim_failures": "unsupported_behavior_pass",
    }
    for case in cases:
        for group, attribute in mappings.items():
            if getattr(case.deterministic, attribute) is False:
                groups[group].append(case.golden_case.case_id)
        if case.trace.provider_failure:
            groups["provider_failures"].append(case.golden_case.case_id)
        if case.judge is not None:
            for judgment in case.judge.judgments:
                if judgment.label.startswith("partially_"):
                    quality_warnings[
                        f"judge_{judgment.criterion.value}_{judgment.label}"
                    ].append(case.golden_case.case_id)
                if judgment.label in {
                    "irrelevant",
                    "unfaithful",
                    "incomplete",
                    "unclear",
                    "verbose",
                    "not_useful",
                }:
                    groups[f"judge_{judgment.criterion.value}_failures"].append(
                        case.golden_case.case_id
                    )
    return FailureAnalysis(
        groups={name: tuple(ids) for name, ids in sorted(groups.items())},
        judge_quality_warnings={
            name: tuple(ids) for name, ids in sorted(quality_warnings.items())
        },
    )


def _bool_metric(values) -> MetricAggregate:
    rows = tuple(values)
    return _metric(sum(rows), len(rows))


def _optional_metric(values) -> MetricAggregate:
    rows = tuple(value for value in values if value is not None)
    return _metric(sum(rows), len(rows))


def _violation_metric(passes) -> MetricAggregate:
    rows = tuple(passes)
    return _metric(sum(not value for value in rows), len(rows))


def _optional_violation_metric(passes) -> MetricAggregate:
    rows = tuple(value for value in passes if value is not None)
    return _metric(sum(not value for value in rows), len(rows))


def _metric(numerator: int, denominator: int) -> MetricAggregate:
    return MetricAggregate(
        numerator=numerator,
        denominator=denominator,
        rate=numerator / denominator if denominator else None,
    )
