"""Run resumable full-workflow AI Scout golden evaluation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.ai.evaluation.aggregate import aggregate_evaluation
from app.ai.evaluation.artifacts import EvaluationArtifactStore
from app.ai.evaluation.calibration import (
    JUDGE_CALIBRATION_VERSION,
    CalibrationStatus,
    JudgeCalibrationReport,
)
from app.ai.evaluation.cases import DEFAULT_GOLDEN_PATH, load_golden_dataset
from app.ai.evaluation.deterministic import evaluate_deterministic_case
from app.ai.evaluation.judge_factory import create_judge_provider
from app.ai.evaluation.judges import (
    JUDGE_PROMPT_VERSIONS,
    build_judge_evidence,
    evaluate_with_judges,
)
from app.ai.evaluation.phase8_models import (
    AIScoutCaseEvaluation,
    EvaluationRunConfiguration,
)
from app.ai.observability import (
    InMemoryTraceSink,
    PricingRegistry,
)
from app.ai.observability.pricing import (
    load_default_pricing_registry,
    load_pricing_registry,
)
from app.ai.prompts import PLANNER_PROMPT_VERSION
from app.ai.provider_factory import create_planner
from app.ai.synthesis import SYNTHESIS_PROMPT_VERSION, create_synthesizer
from app.ai.validation import ValidationMode
from app.ai.web.factory import create_web_search_provider
from app.ai.workflow import AIScoutWorkflow
from app.config import Settings
from app.database import SessionLocal
from app.services.ai_scout import presented_ai_scout_answer

DEFAULT_RESULTS_ROOT = Path(__file__).parents[1] / "evals" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--category")
    parser.add_argument("--tag")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--validation-mode",
        choices=[mode.value for mode in ValidationMode],
        default=ValidationMode.STRICT.value,
    )
    parser.add_argument("--enable-judges", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--pricing-registry", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive.")
    settings = Settings()
    pricing_registry = _pricing_registry(args.pricing_registry)
    dataset = load_golden_dataset(args.dataset)
    selected = tuple(
        case
        for case in dataset.cases
        if (not args.case_id or case.case_id in set(args.case_id))
        and (args.category is None or case.category == args.category)
        and (args.tag is None or args.tag in case.tags)
    )
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No golden cases matched the requested filters.")

    planner = create_planner(settings)
    synthesizer = create_synthesizer(settings)
    web_provider = create_web_search_provider(settings)
    judges_requested = args.enable_judges and not args.deterministic_only
    judge_provider = None
    judge_error = None
    if judges_requested:
        try:
            judge_provider = create_judge_provider(
                settings,
                model_override=args.judge_model,
            )
        except ValueError as exc:
            judge_error = str(exc)

    configuration = EvaluationRunConfiguration(
        golden_dataset_version=dataset.version,
        validation_mode=args.validation_mode,
        planner_provider=planner.provider,
        planner_model=planner.model,
        planner_prompt_version=PLANNER_PROMPT_VERSION,
        synthesis_provider=synthesizer.provider,
        synthesis_model=synthesizer.model,
        synthesis_prompt_version=SYNTHESIS_PROMPT_VERSION,
        judge_enabled=judges_requested,
        judge_provider=(judge_provider.provider if judge_provider else None),
        judge_model=(judge_provider.model if judge_provider else args.judge_model),
        judge_prompt_versions={
            criterion.value: version for criterion, version in JUDGE_PROMPT_VERSIONS.items()
        },
        pricing_version=pricing_registry.version,
    )
    timestamp = datetime.now(UTC)
    run_id = f"eval-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    output_dir = args.output_dir or DEFAULT_RESULTS_ROOT / run_id
    store = EvaluationArtifactStore(output_dir)
    store.initialize(configuration, resume=args.resume)
    existing = store.completed_cases() if args.resume else {}
    completed = list(existing.values())
    memory_sink = InMemoryTraceSink()

    with SessionLocal() as session:
        workflow = AIScoutWorkflow(
            session=session,
            planner=planner,
            synthesizer=synthesizer,
            web_search_provider=web_provider,
            web_search_enabled=settings.ai_scout_web_search_enabled,
            web_max_results=settings.ai_scout_web_max_results,
            web_content_max_age_hours=settings.ai_scout_web_content_max_age_hours,
            web_current_status_max_age_hours=(
                settings.ai_scout_web_current_status_max_age_hours
            ),
            validation_mode=ValidationMode(args.validation_mode),
            trace_sink=memory_sink,
            environment=settings.app_env,
            pricing_registry=pricing_registry,
        )
        for case in selected:
            if case.case_id in existing:
                continue
            result = workflow.run(case.question, case_id=case.case_id)
            trace = memory_sink.traces[-1]
            deterministic = evaluate_deterministic_case(case, result, trace)
            presented_answer = presented_ai_scout_answer(result)
            judge = (
                evaluate_with_judges(
                    case,
                    result,
                    judge_provider,
                    pricing_registry=pricing_registry,
                )
                if judges_requested
                else None
            )
            evaluated = AIScoutCaseEvaluation(
                golden_case=case,
                trace=trace,
                answer_status=result.answer.status.value,
                answer_markdown=result.answer.answer_markdown,
                internal_answer_markdown=result.answer.answer_markdown,
                presented_answer_markdown=presented_answer,
                evidence_for_judge=build_judge_evidence(
                    result.evidence.records,
                    answer_markdown=result.answer.answer_markdown,
                ),
                deterministic=deterministic,
                judge=judge,
            )
            store.append_case(evaluated)
            completed.append(evaluated)

    calibration_report = _calibration_report(args.calibration_report)
    calibration_status = (
        calibration_report.status.value
        if calibration_report is not None
        else CalibrationStatus.UNREVIEWED.value
    )
    report = aggregate_evaluation(
        tuple(completed),
        run_id=run_id,
        started_at=timestamp,
        configuration=configuration,
        judge_calibration_status=calibration_status,
        judge_calibration_report=calibration_report,
    )
    store.finalize(report)
    _print_summary(report)
    if judge_error:
        print(f"Judge unavailable; deterministic evaluation completed: {judge_error}")
    print(f"Saved: {output_dir}")


def _calibration_report(path: Path | None) -> JudgeCalibrationReport | None:
    if path is None:
        return None
    report = JudgeCalibrationReport.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    expected_prompts = {
        criterion.value: version
        for criterion, version in JUDGE_PROMPT_VERSIONS.items()
    }
    if report.dataset_version != JUDGE_CALIBRATION_VERSION:
        raise ValueError("Calibration report dataset version is incompatible.")
    if report.judge_prompt_versions != expected_prompts:
        raise ValueError("Calibration report judge prompt versions are incompatible.")
    return report


def _pricing_registry(path: Path | None) -> PricingRegistry:
    if path is None:
        return load_default_pricing_registry()
    registry = load_pricing_registry(path)
    if not registry.version:
        raise ValueError("A pricing registry file requires an explicit dated version.")
    return registry


def _print_summary(report) -> None:
    metrics = report.deterministic_aggregates
    print("AI Scout Evaluation")
    print("===================")
    print(f"Cases: {report.golden_case_count}")
    print(
        "Deterministic pass: "
        f"{metrics.deterministic_overall_pass_rate.numerator}/"
        f"{metrics.deterministic_overall_pass_rate.denominator}"
    )
    for label, metric in (
        ("Intent accuracy", metrics.planner_intent_accuracy),
        ("Entity accuracy", metrics.entity_resolution_accuracy),
        ("Tool-selection accuracy", metrics.tool_selection_accuracy),
        ("Tool-argument accuracy", metrics.tool_argument_accuracy),
        ("Methodology accuracy", metrics.methodology_retrieval_accuracy),
        ("Web-routing accuracy", metrics.web_routing_accuracy),
        ("Citation-validity rate", metrics.citation_validity_case_rate),
        ("Unsupported-claim violation rate", metrics.unsupported_claim_violation_rate),
    ):
        print(f"{label}: {_percent(metric.rate)}")
    judges = report.judge_aggregates
    print(f"Judge eligible criteria: {judges.judge_eligible_criteria}")
    print(f"Judge evaluated criteria: {judges.judge_evaluated_criteria}")
    print(f"Judge skipped criteria: {judges.judge_skipped_criteria}")
    print(f"Judge coverage: {_percent(judges.judge_coverage.rate)}")
    print(f"Latency p50: {report.observability_aggregates.total_latency.p50}")
    print(f"Latency p95: {report.observability_aggregates.total_latency.p95}")
    print(f"Tokens: {report.observability_aggregates.total_tokens}")
    print(f"Cost: {report.total_evaluation_estimated_cost_usd}")


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    main()
