"""Explicit developer command for the real-provider AI Scout golden evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.ai.evaluation import PlannerEvaluationReport, evaluate_runs, load_golden_cases
from app.ai.provider_factory import create_planner
from app.ai.run import run_ai_scout
from app.config import settings
from app.database import SessionLocal

RESULTS_DIR = Path(__file__).parents[1] / "evals" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N checked-in cases (useful for a paid smoke test).",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Print semantic mismatches and sanitized failures after evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_golden_cases()
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive.")
        cases = cases[: args.limit]

    try:
        planner = create_planner(settings)
    except ValueError as exc:
        print(f"{exc} Evaluation skipped with no provider calls.")
        return
    with SessionLocal() as session:
        runs = [run_ai_scout(session, question=case.question, planner=planner) for case in cases]
    report = evaluate_runs(cases, runs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"planner-{timestamp}.json"
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(f"Provider: {report.provider}")
    print(f"Model: {report.model}")
    print(f"Prompt version: {report.prompt_version}")
    print(f"Cases: {report.metrics.case_count}")
    print("Request-understanding metrics:")
    print(report.request_understanding_metrics.model_dump_json(indent=2))
    print("End-to-end metrics:")
    print(report.metrics.model_dump_json(indent=2))
    print(
        "Token usage: "
        f"input={report.input_tokens}, output={report.output_tokens}, total={report.total_tokens}"
    )
    if args.verbose_errors:
        print_verbose_diagnostics(report)
    print(f"Saved: {output_path}")


def print_verbose_diagnostics(report: PlannerEvaluationReport) -> None:
    """Print safe semantic differences even when provider execution succeeded."""
    noteworthy = [
        case for case in report.cases if case.diagnostics.mismatches or case.failure is not None
    ]
    if not noteworthy:
        print("Semantic mismatches and sanitized failures: none")
        return

    print("Semantic mismatches and sanitized failures:")
    for case in noteworthy:
        if case.diagnostics.mismatches:
            print(f"{case.case_id} semantic mismatches:")
            for category, values in case.diagnostics.mismatches.items():
                print(f"  {category}:")
                print(f"    expected: {_compact_json(values['expected'])}")
                print(f"    actual:   {_compact_json(values['actual'])}")
        if case.failure is not None:
            print(f"{case.case_id} sanitized failure:")
            print(case.failure.model_dump_json(indent=2))


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
