"""Validate the versioned 85-case AI Scout golden dataset without provider calls."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.ai.evaluation.cases import DEFAULT_GOLDEN_PATH, load_golden_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_GOLDEN_PATH)
    return parser.parse_args()


def main() -> None:
    dataset = load_golden_dataset(parse_args().dataset)
    categories = Counter(case.category for case in dataset.cases)
    tags = Counter(tag for case in dataset.cases for tag in case.tags)
    print(f"Dataset: {dataset.version}")
    print(f"Cases: {len(dataset.cases)}")
    print(f"Categories: {dict(sorted(categories.items()))}")
    print(f"Tags: {dict(sorted(tags.items()))}")
    print("Golden dataset: valid")


if __name__ == "__main__":
    main()
