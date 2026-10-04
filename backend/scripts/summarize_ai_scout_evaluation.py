"""Print a concise summary for one completed Phase 8 evaluation run."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ai.evaluation.artifacts import render_summary_markdown
from app.ai.evaluation.phase8_models import AIScoutEvaluationRun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run = AIScoutEvaluationRun.model_validate_json(
        parse_args().input.read_text(encoding="utf-8")
    )
    print(render_summary_markdown(run))


if __name__ == "__main__":
    main()
