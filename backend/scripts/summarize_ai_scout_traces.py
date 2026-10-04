"""Summarize operational AI Scout trace JSONL without quality scoring."""

from __future__ import annotations

import argparse

from app.ai.observability import load_jsonl_traces, summarize_traces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an AI Scout trace JSONL file.")
    parser.add_argument("--json", action="store_true", help="Print the typed summary as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_traces(load_jsonl_traces(args.input))
    if args.json:
        print(summary.model_dump_json(indent=2))
        return
    print(_human_summary(summary))


def _human_summary(summary) -> str:
    latency = summary.total_latency
    lines = [
        "AI Scout Observability Summary",
        "==============================",
        f"Runs: {summary.total_runs}",
        f"Successful: {summary.successful_runs}",
        f"Error rate: {_percent(summary.error_rate)}",
        f"Repair rate: {_percent(summary.repair_rate)}",
        f"Block rate: {_percent(summary.block_rate)}",
        "",
        "Latency",
        f"  median total: {_number(latency.p50, ' ms')}",
        f"  p95 total: {_number(latency.p95, ' ms')}",
        f"  median planner: {_number(summary.planner_latency.p50, ' ms')}",
        f"  median synthesis: {_number(summary.synthesis_latency.p50, ' ms')}",
        "",
        "Usage",
        f"  total tokens: {_number(summary.total_tokens)}",
        f"  median tokens/run: {_number(summary.median_tokens_per_run)}",
        "",
        "Cost",
        f"  estimated total: {_money(summary.total_estimated_cost_usd)}",
        f"  estimated mean/run: {_money(summary.mean_estimated_cost_per_run_usd)}",
        "",
        "Retrieval",
        f"  web search used: {_percent(summary.web_search_usage_rate)}",
        f"  web failure rate: {_percent(summary.web_failure_rate)}",
        "",
        "Validation",
        f"  pass rate (reached only): {_percent(summary.validation_pass_rate)}",
    ]
    lines.extend(
        f"  {name}: {count}"
        for name, count in summary.validation_outcome_counts.items()
    )
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1%}"


def _number(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    rendered = f"{value:.2f}" if isinstance(value, float) else str(value)
    return f"{rendered}{suffix}"


def _money(value: float | None) -> str:
    return "unavailable" if value is None else f"${value:.6f}"


if __name__ == "__main__":
    main()
