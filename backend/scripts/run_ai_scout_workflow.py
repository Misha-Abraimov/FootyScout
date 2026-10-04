"""Run one explicit, real-provider grounded AI Scout smoke test."""

from __future__ import annotations

import argparse

from app.ai.observability import JsonlTraceSink, NullTraceSink
from app.ai.provider_factory import create_planner
from app.ai.synthesis import create_synthesizer
from app.ai.web.factory import create_web_search_provider
from app.ai.workflow import AIScoutWorkflow
from app.config import Settings
from app.database import SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="One supported scouting question.")
    parser.add_argument("--case-id", help="Optional evaluation correlation ID.")
    parser.add_argument("--trace-path", help="Optional JSONL destination for this run trace.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    planner = create_planner(settings)
    synthesizer = create_synthesizer(settings)
    web_search_provider = create_web_search_provider(settings)
    trace_sink = JsonlTraceSink(args.trace_path) if args.trace_path else NullTraceSink()
    with SessionLocal() as session:
        result = AIScoutWorkflow(
            session=session,
            planner=planner,
            synthesizer=synthesizer,
            web_search_provider=web_search_provider,
            web_search_enabled=settings.ai_scout_web_search_enabled,
            web_max_results=settings.ai_scout_web_max_results,
            web_content_max_age_hours=settings.ai_scout_web_content_max_age_hours,
            web_current_status_max_age_hours=(
                settings.ai_scout_web_current_status_max_age_hours
            ),
            validation_mode=settings.ai_scout_validation_mode,
            trace_sink=trace_sink,
            environment=settings.app_env,
        ).run(args.question, case_id=args.case_id)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
