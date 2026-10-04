"""Run explicitly human-approved judge calibration examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ai.evaluation.calibration import (
    DEFAULT_CALIBRATION_PATH,
    load_calibration_examples,
    run_calibration,
)
from app.ai.evaluation.judge_factory import create_judge_provider
from app.ai.observability import EMPTY_PRICING_REGISTRY, PricingRegistry
from app.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--pricing-registry", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    examples = load_calibration_examples(args.dataset)
    provider = create_judge_provider(settings, model_override=args.judge_model)
    pricing_registry = (
        PricingRegistry.model_validate_json(
            args.pricing_registry.read_text(encoding="utf-8")
        )
        if args.pricing_registry is not None
        else EMPTY_PRICING_REGISTRY
    )
    if args.pricing_registry is not None and not pricing_registry.version:
        raise ValueError("A pricing registry file requires an explicit dated version.")
    report = run_calibration(
        examples,
        provider,
        pricing_registry=pricing_registry,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"Calibration status: {report.status.value}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
