"""Inspect candidate judge labels; this command makes no provider calls."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.ai.evaluation.calibration import (
    DEFAULT_CALIBRATION_PATH,
    JUDGE_CALIBRATION_VERSION,
    load_calibration_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CALIBRATION_PATH)
    return parser.parse_args()


def main() -> None:
    examples = load_calibration_examples(parse_args().dataset)
    approved = tuple(example for example in examples if example.human_approved)
    fresh_held_out = tuple(
        example
        for example in examples
        if example.split.value == "held_out"
        and not example.previously_evaluated
    )
    print(f"Dataset: {JUDGE_CALIBRATION_VERSION}")
    print(f"Candidate examples: {len(examples)}")
    print(f"Human-approved examples: {len(approved)}")
    print(f"Fresh held-out candidates: {len(fresh_held_out)}")
    print(
        "Fresh held-out human-approved: "
        f"{sum(example.human_approved for example in fresh_held_out)}"
    )
    print(
        "By criterion: "
        f"{dict(sorted(Counter(example.criterion.value for example in examples).items()))}"
    )
    print(
        "By split: "
        f"{dict(sorted(Counter(example.split.value for example in examples).items()))}"
    )
    if not approved:
        print("Calibration status: unreviewed (candidate labels are not trusted metrics)")


if __name__ == "__main__":
    main()
