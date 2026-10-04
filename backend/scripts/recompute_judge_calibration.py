"""Recompute judge-calibration metrics from frozen recorded predictions offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ai.evaluation.calibration import (
    DEFAULT_CALIBRATION_PATH,
    JudgeCalibrationReport,
    load_calibration_examples,
    recompute_calibration_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corrected = recompute_file(args.dataset, args.input, args.output)
    print(f"Calibration status: {corrected.status.value}")
    print(f"Saved: {args.output}")


def recompute_file(
    dataset_path: Path,
    input_path: Path,
    output_path: Path,
) -> JudgeCalibrationReport:
    """Recompute and write a distinct artifact without any provider dependency."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite the historical calibration artifact.")
    examples = load_calibration_examples(dataset_path)
    historical = JudgeCalibrationReport.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    corrected = recompute_calibration_report(examples, historical)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(corrected.model_dump_json(indent=2), encoding="utf-8")
    return corrected


if __name__ == "__main__":
    main()
