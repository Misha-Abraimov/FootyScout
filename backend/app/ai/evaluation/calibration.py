"""Human-approved calibration data and criterion-level judge metrics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.evaluation.cases import JudgeCriterion
from app.ai.evaluation.judges import (
    CRITERION_LABELS,
    JUDGE_PROMPT_VERSIONS,
    JudgeProvider,
    JudgeRequest,
)
from app.ai.evaluation.phase8_models import EvidenceForJudge
from app.ai.observability import EMPTY_PRICING_REGISTRY, PricingRegistry, estimate_cost_usd

JUDGE_CALIBRATION_VERSION = "ai-scout-judge-calibration-v2"
JUDGE_CALIBRATION_SCHEMA_VERSION = "ai-scout-judge-calibration-report-v2"
DEFAULT_CALIBRATION_PATH = (
    Path(__file__).parents[3] / "evals" / "ai_scout_judge_calibration.jsonl"
)
MIN_APPROVED_DEVELOPMENT_EXAMPLES = 2


class CalibrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CalibrationSplit(str, Enum):
    DEVELOPMENT = "development"
    HELD_OUT = "held_out"


class CalibrationStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    INSUFFICIENT = "insufficient"
    DEVELOPMENT_ONLY = "development_only"
    HELD_OUT_EVALUATED = "held_out_evaluated"


class JudgeCalibrationExample(CalibrationModel):
    example_id: str = Field(pattern=r"^JC-[A-Z]{2}-\d{3}$")
    criterion: JudgeCriterion
    question: str
    response: str
    evidence: tuple[dict[str, str], ...] = ()
    expected_label: str
    human_approved: bool = False
    notes: str = ""
    split: CalibrationSplit
    previously_evaluated: bool = False


class JudgeCalibrationPrediction(CalibrationModel):
    example_id: str
    criterion: JudgeCriterion
    expected_label: str
    predicted_label: str | None = None
    error_type: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class ClassMetrics(CalibrationModel):
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    support: int = Field(ge=0)


class CriterionCalibrationMetrics(CalibrationModel):
    criterion: JudgeCriterion
    split: CalibrationSplit
    approved_examples: int = Field(ge=0)
    evaluated_examples: int = Field(ge=0)
    accuracy: float | None = None
    macro_f1: float | None = None
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    metric_computable: bool
    example_count: int = Field(ge=0)
    label_support: dict[str, int]
    label_coverage: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_sufficiency_field(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.pop("sufficient_examples", None)
        data.setdefault("metric_computable", bool(data.get("evaluated_examples", 0)))
        data.setdefault("example_count", data.get("evaluated_examples", 0))
        support = data.setdefault(
            "label_support",
            {
                label: (
                    details.support
                    if isinstance(details, ClassMetrics)
                    else details.get("support", 0)
                )
                for label, details in data.get("per_class", {}).items()
            },
        )
        data.setdefault(
            "label_coverage",
            (
                sum(count > 0 for count in support.values()) / len(support)
                if support
                else 0.0
            ),
        )
        return data


class JudgeCalibrationReport(CalibrationModel):
    calibration_schema_version: str = JUDGE_CALIBRATION_SCHEMA_VERSION
    dataset_version: str = JUDGE_CALIBRATION_VERSION
    status: CalibrationStatus
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_prompt_versions: dict[str, str] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    approved_development_example_count: int = Field(default=0, ge=0)
    approved_held_out_example_count: int = Field(default=0, ge=0)
    judge_input_tokens: int | None = Field(default=None, ge=0)
    judge_output_tokens: int | None = Field(default=None, ge=0)
    judge_total_tokens: int | None = Field(default=None, ge=0)
    judge_total_latency_ms: float | None = Field(default=None, ge=0)
    pricing_version: str | None = None
    estimated_judge_cost_usd: float | None = Field(default=None, ge=0)
    metrics: tuple[CriterionCalibrationMetrics, ...]
    predictions: tuple[JudgeCalibrationPrediction, ...]


def load_calibration_examples(
    path: Path = DEFAULT_CALIBRATION_PATH,
) -> tuple[JudgeCalibrationExample, ...]:
    examples: list[JudgeCalibrationExample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            example = JudgeCalibrationExample.model_validate_json(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid calibration example at {path}:{line_number}.") from exc
        if example.expected_label not in CRITERION_LABELS[example.criterion]:
            raise ValueError(
                f"Invalid expected label for {example.example_id}: {example.expected_label}."
            )
        examples.append(example)
    ids = [example.example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("Judge calibration example IDs must be unique.")
    return tuple(examples)


def run_calibration(
    examples: tuple[JudgeCalibrationExample, ...],
    provider: JudgeProvider,
    *,
    pricing_registry: PricingRegistry = EMPTY_PRICING_REGISTRY,
) -> JudgeCalibrationReport:
    started_at = datetime.now(UTC)
    predictions: list[JudgeCalibrationPrediction] = []
    for example in examples:
        if not example.human_approved:
            continue
        request_started = perf_counter()
        try:
            response = provider.judge(
                JudgeRequest(
                    criterion=example.criterion,
                    question=example.question,
                    answer=(
                        None
                        if example.criterion is JudgeCriterion.CONTEXT_RELEVANCE
                        else example.response
                    ),
                    evidence=tuple(
                        EvidenceForJudge(
                            evidence_id=item.get("evidence_id", "calibration-evidence"),
                            category=item.get("category", "calibration"),
                            source_label=item.get("category", "calibration"),
                            content=item.get("content", ""),
                        )
                        for item in example.evidence
                    ),
                )
            )
            label = response.output.label
            if label not in CRITERION_LABELS[example.criterion]:
                raise ValueError(f"Invalid judge label: {label}.")
            usage = response.usage
            pricing = pricing_registry.find(provider.provider, provider.model)
            predictions.append(
                JudgeCalibrationPrediction(
                    example_id=example.example_id,
                    criterion=example.criterion,
                    expected_label=example.expected_label,
                    predicted_label=label,
                    prompt_version=JUDGE_PROMPT_VERSIONS[example.criterion],
                    input_tokens=usage.input_tokens if usage else None,
                    output_tokens=usage.output_tokens if usage else None,
                    total_tokens=usage.total_tokens if usage else None,
                    latency_ms=max(0.0, (perf_counter() - request_started) * 1000),
                    estimated_cost_usd=estimate_cost_usd(
                        input_tokens=usage.input_tokens if usage else None,
                        output_tokens=usage.output_tokens if usage else None,
                        pricing=pricing,
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - calibration provider boundary
            predictions.append(
                JudgeCalibrationPrediction(
                    example_id=example.example_id,
                    criterion=example.criterion,
                    expected_label=example.expected_label,
                    error_type=type(exc).__name__,
                    prompt_version=JUDGE_PROMPT_VERSIONS[example.criterion],
                    latency_ms=max(0.0, (perf_counter() - request_started) * 1000),
                )
            )
    return calibration_report(
        examples,
        tuple(predictions),
        judge_provider=provider.provider,
        judge_model=provider.model,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        pricing_registry=pricing_registry,
    )


def calibration_report(
    examples: tuple[JudgeCalibrationExample, ...],
    predictions: tuple[JudgeCalibrationPrediction, ...],
    *,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    pricing_registry: PricingRegistry = EMPTY_PRICING_REGISTRY,
) -> JudgeCalibrationReport:
    approved_by_id = {item.example_id: item for item in examples if item.human_approved}
    prediction_by_id = {item.example_id: item for item in predictions}
    metrics = tuple(
        _criterion_metrics(
            criterion,
            split,
            tuple(
                item
                for item in approved_by_id.values()
                if item.criterion is criterion and item.split is split
            ),
            prediction_by_id,
        )
        for criterion in JudgeCriterion
        for split in CalibrationSplit
    )
    status = _calibration_status(metrics, examples)
    approved_development = sum(
        item.human_approved and item.split is CalibrationSplit.DEVELOPMENT
        for item in examples
    )
    approved_held_out = sum(
        item.human_approved and item.split is CalibrationSplit.HELD_OUT
        for item in examples
    )
    return JudgeCalibrationReport(
        status=status,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_prompt_versions={
            criterion.value: version
            for criterion, version in JUDGE_PROMPT_VERSIONS.items()
        },
        started_at=started_at,
        completed_at=completed_at,
        approved_development_example_count=approved_development,
        approved_held_out_example_count=approved_held_out,
        judge_input_tokens=_complete_sum(predictions, "input_tokens"),
        judge_output_tokens=_complete_sum(predictions, "output_tokens"),
        judge_total_tokens=_complete_sum(predictions, "total_tokens"),
        judge_total_latency_ms=_complete_sum(predictions, "latency_ms"),
        pricing_version=pricing_registry.version,
        estimated_judge_cost_usd=_complete_sum(predictions, "estimated_cost_usd"),
        metrics=metrics,
        predictions=predictions,
    )


def recompute_calibration_report(
    examples: tuple[JudgeCalibrationExample, ...],
    historical: JudgeCalibrationReport,
) -> JudgeCalibrationReport:
    """Recompute metrics from frozen labels/predictions without invoking a provider."""
    if historical.dataset_version != JUDGE_CALIBRATION_VERSION:
        raise ValueError("Historical calibration dataset version is incompatible.")
    examples_by_id = {example.example_id: example for example in examples}
    prediction_ids = [prediction.example_id for prediction in historical.predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Historical calibration predictions must have unique IDs.")
    for prediction in historical.predictions:
        example = examples_by_id.get(prediction.example_id)
        if example is None:
            raise ValueError(
                f"Historical prediction has no frozen example: {prediction.example_id}."
            )
        if (
            prediction.criterion is not example.criterion
            or prediction.expected_label != example.expected_label
        ):
            raise ValueError(
                f"Historical prediction does not match frozen label: {prediction.example_id}."
            )

    recomputed = calibration_report(
        examples,
        historical.predictions,
        judge_provider=historical.judge_provider,
        judge_model=historical.judge_model,
        started_at=historical.started_at,
        completed_at=historical.completed_at,
    )
    return recomputed.model_copy(
        update={
            "calibration_schema_version": historical.calibration_schema_version,
            "dataset_version": historical.dataset_version,
            "judge_prompt_versions": historical.judge_prompt_versions,
            "judge_input_tokens": historical.judge_input_tokens,
            "judge_output_tokens": historical.judge_output_tokens,
            "judge_total_tokens": historical.judge_total_tokens,
            "judge_total_latency_ms": historical.judge_total_latency_ms,
            "pricing_version": historical.pricing_version,
            "estimated_judge_cost_usd": historical.estimated_judge_cost_usd,
        }
    )


def _criterion_metrics(
    criterion: JudgeCriterion,
    split: CalibrationSplit,
    approved: tuple[JudgeCalibrationExample, ...],
    predictions: dict[str, JudgeCalibrationPrediction],
) -> CriterionCalibrationMetrics:
    labels = CRITERION_LABELS[criterion]
    evaluated = tuple(
        (example.expected_label, predictions[example.example_id].predicted_label)
        for example in approved
        if example.example_id in predictions
        and predictions[example.example_id].predicted_label is not None
    )
    confusion = {expected: {actual: 0 for actual in labels} for expected in labels}
    for expected, actual in evaluated:
        assert actual is not None
        confusion[expected][actual] += 1
    class_metrics: dict[str, ClassMetrics] = {}
    label_support = {
        label: sum(expected == label for expected, _ in evaluated)
        for label in labels
    }
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in labels if other != label)
        false_negative = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = _ratio(2 * true_positive, f1_denominator)
        class_metrics[label] = ClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )
    supported_f1_values = [
        item.f1
        for item in class_metrics.values()
        if item.support > 0 and item.f1 is not None
    ]
    correct = sum(expected == actual for expected, actual in evaluated)
    return CriterionCalibrationMetrics(
        criterion=criterion,
        split=split,
        approved_examples=len(approved),
        evaluated_examples=len(evaluated),
        accuracy=_ratio(correct, len(evaluated)),
        macro_f1=(
            sum(supported_f1_values) / len(supported_f1_values)
            if supported_f1_values
            else None
        ),
        per_class=class_metrics,
        confusion_matrix=confusion,
        metric_computable=bool(evaluated),
        example_count=len(evaluated),
        label_support=label_support,
        label_coverage=(
            sum(count > 0 for count in label_support.values()) / len(labels)
        ),
    )


def _calibration_status(
    metrics: tuple[CriterionCalibrationMetrics, ...],
    examples: tuple[JudgeCalibrationExample, ...],
) -> CalibrationStatus:
    if any(
        item.split is CalibrationSplit.HELD_OUT and not item.human_approved
        for item in examples
    ):
        return CalibrationStatus.NEEDS_HUMAN_REVIEW
    evaluated = [metric for metric in metrics if metric.evaluated_examples]
    if not evaluated:
        return CalibrationStatus.UNREVIEWED
    development = [
        metric for metric in metrics if metric.split is CalibrationSplit.DEVELOPMENT
    ]
    if not all(
        metric.example_count >= MIN_APPROVED_DEVELOPMENT_EXAMPLES
        for metric in development
    ):
        return CalibrationStatus.INSUFFICIENT
    held_out = [metric for metric in metrics if metric.split is CalibrationSplit.HELD_OUT]
    if all(metric.metric_computable for metric in held_out):
        return CalibrationStatus.HELD_OUT_EVALUATED
    return CalibrationStatus.DEVELOPMENT_ONLY


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _complete_sum(
    predictions: tuple[JudgeCalibrationPrediction, ...],
    field: str,
) -> int | float | None:
    if not predictions:
        return None
    values = [getattr(item, field) for item in predictions]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
