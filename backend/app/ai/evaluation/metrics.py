"""Programmatic planner metrics; no LLM-as-judge is used."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_count: int = Field(ge=0)
    intent_accuracy: float = Field(ge=0, le=1)
    tool_selection_exact_match: float = Field(ge=0, le=1)
    required_tool_recall: float = Field(ge=0, le=1)
    forbidden_tool_violation_rate: float = Field(ge=0, le=1)
    normalized_argument_accuracy: float = Field(ge=0, le=1)
    entity_resolution_accuracy: float = Field(ge=0, le=1)
    clarification_accuracy: float = Field(ge=0, le=1)
    unsupported_request_accuracy: float = Field(ge=0, le=1)
    structured_output_validity: float = Field(ge=0, le=1)
    plan_length_violation_rate: float = Field(ge=0, le=1)


def mean(values: list[bool | float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def is_subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return all(key in actual and actual[key] == value for key, value in expected.items())
