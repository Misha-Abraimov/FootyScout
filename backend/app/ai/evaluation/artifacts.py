"""Safe, resumable Phase 8 evaluation artifacts and Markdown reporting."""

from __future__ import annotations

import json
from pathlib import Path

from app.ai.evaluation.phase8_models import (
    AIScoutCaseEvaluation,
    AIScoutEvaluationRun,
    EvaluationRunConfiguration,
    JudgeCaseEvaluation,
)


class EvaluationArtifactStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.configuration_path = self.output_dir / "run_config.json"
        self.cases_path = self.output_dir / "cases.jsonl"
        self.traces_path = self.output_dir / "traces.jsonl"
        self.judges_path = self.output_dir / "judge_results.jsonl"

    def initialize(
        self,
        configuration: EvaluationRunConfiguration,
        *,
        resume: bool,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.configuration_path.exists():
            existing = EvaluationRunConfiguration.model_validate_json(
                self.configuration_path.read_text(encoding="utf-8")
            )
            if existing != configuration:
                raise ValueError(
                    "Evaluation output configuration is incompatible; use a new output directory."
                )
            if not resume:
                raise ValueError("Evaluation output already exists; pass --resume or use a new directory.")
            return
        if resume and any(self.output_dir.iterdir()):
            raise ValueError("Cannot resume an output directory without compatible run_config.json.")
        self.configuration_path.write_text(
            configuration.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def completed_cases(self) -> dict[str, AIScoutCaseEvaluation]:
        if not self.cases_path.exists():
            return {}
        return {
            case.golden_case.case_id: case
            for case in _read_jsonl(self.cases_path, AIScoutCaseEvaluation)
        }

    def append_case(self, case: AIScoutCaseEvaluation) -> None:
        _append_jsonl(self.cases_path, case.model_dump_json())
        _append_jsonl(self.traces_path, case.trace.model_dump_json())
        if case.judge is not None:
            _append_jsonl(self.judges_path, case.judge.model_dump_json())

    def finalize(self, run: AIScoutEvaluationRun) -> None:
        (self.output_dir / "evaluation.json").write_text(
            run.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "summary.json").write_text(
            run.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (self.output_dir / "summary.md").write_text(
            render_summary_markdown(run),
            encoding="utf-8",
        )
        for path in (self.cases_path, self.traces_path, self.judges_path):
            path.touch(exist_ok=True)


def render_summary_markdown(run: AIScoutEvaluationRun) -> str:
    config = run.configuration
    deterministic = run.deterministic_aggregates
    judge = run.judge_aggregates
    operations = run.observability_aggregates
    lines = [
        "# AI Scout Evaluation",
        "",
        "## Configuration",
        "",
        f"- Golden dataset: `{config.golden_dataset_version}`",
        f"- Cases: {run.golden_case_count}",
        f"- Validation mode: `{config.validation_mode}`",
        f"- Planner: `{config.planner_provider}/{config.planner_model}`",
        f"- Synthesis: `{config.synthesis_provider}/{config.synthesis_model}`",
        f"- Judge: `{config.judge_provider or 'disabled'}/{config.judge_model or 'unavailable'}`",
        "",
        "## Deterministic Quality",
        "",
    ]
    deterministic_rows = {
        "Expected status accuracy": deterministic.expected_status_accuracy,
        "Planner intent accuracy": deterministic.planner_intent_accuracy,
        "Entity resolution accuracy": deterministic.entity_resolution_accuracy,
        "Tool selection accuracy": deterministic.tool_selection_accuracy,
        "Tool argument accuracy": deterministic.tool_argument_accuracy,
        "Methodology retrieval accuracy": deterministic.methodology_retrieval_accuracy,
        "Web routing accuracy": deterministic.web_routing_accuracy,
        "Citation validity": deterministic.citation_validity_case_rate,
        "Historical/current authority": deterministic.historical_current_authority_accuracy,
        "Unsupported-claim violation rate": deterministic.unsupported_claim_violation_rate,
        "Overall deterministic pass rate": deterministic.deterministic_overall_pass_rate,
    }
    lines.extend(
        f"- {label}: {_metric(metric.rate)} ({metric.numerator}/{metric.denominator})"
        for label, metric in deterministic_rows.items()
    )
    lines.extend(
        [
            "",
            "## LLM-as-a-Judge",
            "",
            f"- Judge eligible criteria: {judge.judge_eligible_criteria}",
            f"- Judge evaluated criteria: {judge.judge_evaluated_criteria}",
            f"- Judge skipped criteria: {judge.judge_skipped_criteria}",
            (
                f"- Judge coverage: {_metric(judge.judge_coverage.rate)} "
                f"({judge.judge_coverage.numerator}/{judge.judge_coverage.denominator})"
            ),
            f"- Calibration status: `{run.judge_calibration_status}`",
        ]
    )
    for criterion, aggregate in sorted(judge.criteria.items()):
        labels = ", ".join(
            f"{label}={count} ({aggregate.rates[label]:.1%})"
            for label, count in aggregate.counts.items()
        )
        lines.append(f"- {criterion}: {labels}")
    lines.extend(["", "## Judge Calibration", ""])
    calibration = run.judge_calibration_report
    if calibration is None:
        lines.append("- No reviewed calibration report was attached to this run.")
    else:
        lines.append(f"- Dataset: `{calibration['dataset_version']}`")
        for metric in calibration["metrics"]:
            if metric["approved_examples"]:
                lines.append(
                    f"- {metric['criterion']} / {metric['split']}: "
                    f"approved={metric['approved_examples']}, "
                    f"evaluated={metric['evaluated_examples']}, "
                    f"accuracy={_metric(metric['accuracy'])}, "
                    f"macro-F1={_metric(metric['macro_f1'])}"
                )
    lines.extend(
        [
            "",
            "## Operations",
            "",
            f"- Workflow-completed runs: {operations.workflow_completed_runs}/{operations.total_runs}",
            f"- Total latency p50/p95: {_number(operations.total_latency.p50)} / {_number(operations.total_latency.p95)} ms",
            f"- Total tokens: {operations.total_tokens if operations.total_tokens is not None else 'unavailable'}",
            f"- Product estimated cost: {_cost(run.product_estimated_cost_usd)}",
            f"- Judge estimated cost: {_cost(run.judge_estimated_cost_usd)}",
            "",
            "## Failure Analysis",
            "",
        ]
    )
    if run.failure_analysis.groups:
        lines.extend(
            f"- {name}: {', '.join(case_ids)}"
            for name, case_ids in run.failure_analysis.groups.items()
        )
    else:
        lines.append("- No failures in the completed subset.")
    lines.extend(["", "## Judge Quality Warnings", ""])
    if run.failure_analysis.judge_quality_warnings:
        lines.extend(
            f"- {name}: {', '.join(case_ids)}"
            for name, case_ids in run.failure_analysis.judge_quality_warnings.items()
        )
    else:
        lines.append("- No partial judge outcomes in the completed subset.")
    lines.extend(
        [
            "",
            "> LLM-as-a-judge is a proxy for human evaluation, not an objective metric like deterministic accuracy.",
            "",
        ]
    )
    return "\n".join(lines)


def load_judge_results(path: str | Path) -> tuple[JudgeCaseEvaluation, ...]:
    return tuple(_read_jsonl(Path(path), JudgeCaseEvaluation))


def _append_jsonl(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{line}\n")


def _read_jsonl(path: Path, model_type):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model_type.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
    return rows


def _metric(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2f}"


def _cost(value: float | None) -> str:
    return "unavailable" if value is None else f"${value:.6f}"
