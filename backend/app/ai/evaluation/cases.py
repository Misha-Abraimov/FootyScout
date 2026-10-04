"""Versioned typed schema and validation for the checked-in 85-case golden set."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ai.schemas import IntentKind, MethodologyTopic, ToolName
from app.ai.synthesis import GroundedAnswerStatus

GOLDEN_DATASET_VERSION = "ai-scout-golden-v2"
DEFAULT_GOLDEN_PATH = Path(__file__).parents[3] / "evals" / "ai_scout_golden.jsonl"
EXPECTED_GOLDEN_CASES = 85
TERMINAL_STATUS_TAXONOMY = {
    GroundedAnswerStatus.ANSWERED: (
        "Governed evidence directly resolves the request, including a definitive "
        "negative result."
    ),
    GroundedAnswerStatus.CLARIFICATION_REQUIRED: (
        "Missing or ambiguous user input can realistically be supplied to allow execution."
    ),
    GroundedAnswerStatus.INSUFFICIENT_EVIDENCE: (
        "The request and inputs are supported, but available evidence cannot establish "
        "the answer."
    ),
    GroundedAnswerStatus.UNSUPPORTED: (
        "The requested capability, operation, guarantee, prediction, or recalculation "
        "is outside governed functionality."
    ),
    GroundedAnswerStatus.ERROR: "The workflow or system failed.",
}
VALID_GOLDEN_CATEGORIES = frozenset(
    {
        "entity_resolution",
        "profile_comparison",
        "similarity_archetype",
        "team_role",
        "methodology",
        "unsupported_adversarial",
        "current_world",
        "mixed",
    }
)
VALID_GOLDEN_TAGS = VALID_GOLDEN_CATEGORIES | frozenset(
    {
        "availability",
        "current_club",
        "current_manager",
        "historical_current",
        "injury",
        "recent_news",
        "role_fit",
        "team_intelligence",
        "transfer_reporting",
        "intentional_duplicate",
        "balanced_guardrail",
    }
)


class WebExpectation(str, Enum):
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"


class ArgumentOperator(str, Enum):
    EXACT = "exact"
    SET_EQUAL = "set_equal"
    CASE_INSENSITIVE = "case_insensitive"
    NUMERIC_RANGE = "numeric_range"
    IS_NULL = "is_null"
    CONTAINS = "contains"


class JudgeCriterion(str, Enum):
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    FAITHFULNESS = "faithfulness"
    CLARITY = "clarity"
    CONCISION = "concision"
    USEFULNESS = "usefulness"
    CONTEXT_RELEVANCE = "context_relevance"


class ArgumentAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: ToolName
    argument: str = Field(min_length=1, max_length=100)
    operator: ArgumentOperator
    expected: Any = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_operator_payload(self) -> ArgumentAssertion:
        if self.operator is ArgumentOperator.NUMERIC_RANGE:
            if self.minimum is None and self.maximum is None:
                raise ValueError("numeric_range requires minimum and/or maximum.")
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            ):
                raise ValueError("numeric_range minimum cannot exceed maximum.")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError("minimum/maximum are valid only for numeric_range.")
        if self.operator is ArgumentOperator.IS_NULL and self.expected is not None:
            raise ValueError("is_null does not accept an expected value.")
        return self


class GoldenEvalCase(BaseModel):
    """Canonical Phase 8 case; the before-validator adapts the original dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]*-\d{3}$")
    question: str = Field(min_length=1, max_length=1000)
    category: str
    tags: tuple[str, ...]
    expected_status: GroundedAnswerStatus
    expected_primary_intent: IntentKind
    expected_entity_behavior: str

    expected_player: str | None = None
    expected_player_id: int | None = Field(default=None, gt=0)
    expected_team: str | None = None
    expected_team_id: int | None = Field(default=None, gt=0)
    expected_position_group: str | None = None

    required_tools: tuple[ToolName, ...] = ()
    optional_tools: tuple[ToolName, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    expected_tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    argument_assertions: tuple[ArgumentAssertion, ...] = ()

    required_methodology_topics: tuple[MethodologyTopic, ...] = ()
    optional_methodology_topics: tuple[MethodologyTopic, ...] = ()
    forbidden_methodology_topics: tuple[MethodologyTopic, ...] = ()
    methodology_topics_exact: bool = False
    expected_methodology_source: str | None = None
    web_expectation: WebExpectation = WebExpectation.FORBIDDEN
    expected_web_category: str | None = None
    required_evidence_categories: tuple[str, ...] = ()
    forbidden_evidence_categories: tuple[str, ...] = ()

    required_answer_behaviors: tuple[str, ...] = ()
    forbidden_answer_behaviors: tuple[str, ...] = ()
    judge_criteria_enabled: tuple[JudgeCriterion, ...] = tuple(JudgeCriterion)
    reference_answer: str | None = None
    reference_key_points: tuple[str, ...] = ()
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_original_case(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data["case_id"] = data.pop("id", data.get("case_id", None))
        data["expected_primary_intent"] = data.pop(
            "expected_intent", data.get("expected_primary_intent", None)
        )
        allowed = tuple(data.pop("allowed_tools", ()))
        required = tuple(data.get("required_tools", ()))
        forbidden = tuple(data.get("forbidden_tools", ()))
        data.setdefault(
            "optional_tools",
            tuple(tool for tool in allowed if tool not in required and tool not in forbidden),
        )
        expected_arguments = data.pop(
            "expected_normalized_arguments",
            data.get("expected_tool_arguments", {}),
        )
        data["expected_tool_arguments"] = expected_arguments
        if "argument_assertions" not in data:
            data["argument_assertions"] = tuple(
                {
                    "tool_name": tool,
                    "argument": argument,
                    "operator": ArgumentOperator.EXACT.value,
                    "expected": expected,
                }
                for tool, arguments in expected_arguments.items()
                for argument, expected in arguments.items()
            )
        clarification = bool(data.pop("clarification_expected", False))
        insufficient = bool(data.pop("insufficient_evidence_expected", False))
        if "expected_status" not in data:
            if clarification:
                data["expected_status"] = GroundedAnswerStatus.CLARIFICATION_REQUIRED.value
            elif insufficient:
                data["expected_status"] = (
                    GroundedAnswerStatus.UNSUPPORTED.value
                    if not required
                    else GroundedAnswerStatus.INSUFFICIENT_EVIDENCE.value
                )
            else:
                data["expected_status"] = GroundedAnswerStatus.ANSWERED.value
        forbidden_claims = tuple(data.pop("forbidden_claims", ()))
        data.setdefault("forbidden_answer_behaviors", forbidden_claims)
        data.setdefault("tags", (data.get("category", "uncategorized"),))
        if "required_methodology_topics" not in data:
            method_args = expected_arguments.get(ToolName.GET_METHODOLOGY.value, {})
            topic = method_args.get("topic")
            data["required_methodology_topics"] = (topic,) if topic else ()
        if "required_evidence_categories" not in data:
            categories: list[str] = []
            if any(tool != ToolName.GET_METHODOLOGY.value for tool in required):
                categories.append("analytics")
            if ToolName.GET_METHODOLOGY.value in required:
                categories.append("methodology")
            if data.get("web_expectation") == WebExpectation.REQUIRED.value:
                categories.append("web")
            data["required_evidence_categories"] = tuple(categories)
        _derive_entity_expectations(data, expected_arguments)
        return data

    @model_validator(mode="after")
    def reject_conflicting_expectations(self) -> GoldenEvalCase:
        required = set(self.required_tools)
        optional = set(self.optional_tools)
        forbidden = set(self.forbidden_tools)
        supported_forbidden = {tool.value for tool in ToolName} | {"run_sql"}
        if forbidden - supported_forbidden:
            raise ValueError("forbidden_tools contains an unknown governed tool name.")
        if {tool.value for tool in required} & forbidden:
            raise ValueError("required_tools and forbidden_tools conflict.")
        if {tool.value for tool in optional} & forbidden or optional & required:
            raise ValueError("optional_tools must be disjoint from required/forbidden tools.")
        required_topics = set(self.required_methodology_topics)
        optional_topics = set(self.optional_methodology_topics)
        forbidden_topics = set(self.forbidden_methodology_topics)
        if required_topics & forbidden_topics or optional_topics & forbidden_topics:
            raise ValueError("Allowed and forbidden methodology topics conflict.")
        if required_topics & optional_topics:
            raise ValueError("Required and optional methodology topics must be disjoint.")
        if self.methodology_topics_exact and optional_topics:
            raise ValueError("Exact methodology matching cannot allow optional topics.")
        if set(self.required_evidence_categories) & set(self.forbidden_evidence_categories):
            raise ValueError("Required and forbidden evidence categories conflict.")
        return self

    @property
    def id(self) -> str:
        return self.case_id

    @property
    def expected_intent(self) -> IntentKind:
        return self.expected_primary_intent

    @property
    def allowed_tools(self) -> list[str]:
        return [tool.value for tool in (*self.required_tools, *self.optional_tools)]

    @property
    def expected_normalized_arguments(self) -> dict[str, dict[str, Any]]:
        return self.expected_tool_arguments

    @property
    def clarification_expected(self) -> bool:
        return self.expected_status is GroundedAnswerStatus.CLARIFICATION_REQUIRED

    @property
    def insufficient_evidence_expected(self) -> bool:
        return self.expected_status in {
            GroundedAnswerStatus.UNSUPPORTED,
            GroundedAnswerStatus.INSUFFICIENT_EVIDENCE,
        }

    @property
    def forbidden_claims(self) -> list[str]:
        return list(self.forbidden_answer_behaviors)


GoldenPlannerCase = GoldenEvalCase


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = GOLDEN_DATASET_VERSION
    cases: tuple[GoldenEvalCase, ...]


def load_golden_dataset(path: Path = DEFAULT_GOLDEN_PATH) -> GoldenDataset:
    cases: list[GoldenEvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(GoldenEvalCase.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid golden case at {path}:{line_number}.") from exc
    validate_golden_cases(tuple(cases))
    return GoldenDataset(cases=tuple(cases))


def load_golden_cases(path: Path = DEFAULT_GOLDEN_PATH) -> list[GoldenEvalCase]:
    return list(load_golden_dataset(path).cases)


def validate_golden_cases(
    cases: tuple[GoldenEvalCase, ...] | list[GoldenEvalCase],
    *,
    expected_count: int = EXPECTED_GOLDEN_CASES,
) -> None:
    if len(cases) != expected_count:
        raise ValueError(f"Golden dataset must contain exactly {expected_count} cases.")
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Golden case IDs must be unique.")
    questions = [case.question.casefold().strip() for case in cases]
    unknown_categories = {case.category for case in cases} - VALID_GOLDEN_CATEGORIES
    if unknown_categories:
        raise ValueError(f"Unknown golden categories: {sorted(unknown_categories)}.")
    unknown_tags = {tag for case in cases for tag in case.tags} - VALID_GOLDEN_TAGS
    if unknown_tags:
        raise ValueError(f"Unknown golden tags: {sorted(unknown_tags)}.")
    duplicates = {question for question in questions if questions.count(question) > 1}
    for duplicate in duplicates:
        matching = [case for case in cases if case.question.casefold().strip() == duplicate]
        if not all("intentional_duplicate" in case.tags for case in matching):
            raise ValueError("Duplicate golden questions require intentional_duplicate tags.")
    _validate_stable_order(ids)


def _derive_entity_expectations(
    data: dict[str, Any], expected_arguments: dict[str, dict[str, Any]]
) -> None:
    for arguments in expected_arguments.values():
        player_ids = arguments.get("player_ids")
        if isinstance(player_ids, list) and player_ids:
            data.setdefault("expected_player_id", player_ids[0])
        if isinstance(arguments.get("player_id"), int):
            data.setdefault("expected_player_id", arguments["player_id"])
        if isinstance(arguments.get("target_team_id"), int):
            data.setdefault("expected_team_id", arguments["target_team_id"])
        elif isinstance(arguments.get("team_id"), int):
            data.setdefault("expected_team_id", arguments["team_id"])
        if isinstance(arguments.get("team"), str):
            data.setdefault("expected_team", arguments["team"])
        if isinstance(arguments.get("position_group"), str):
            data.setdefault("expected_position_group", arguments["position_group"])


def _validate_stable_order(ids: list[str]) -> None:
    previous_number: dict[str, int] = {}
    closed_prefixes: set[str] = set()
    active_prefix: str | None = None
    for case_id in ids:
        match = re.fullmatch(r"([A-Z]{2})-(\d{3})", case_id)
        if match is None:
            raise ValueError(f"Invalid stable case ID: {case_id}.")
        prefix, number_text = match.groups()
        number = int(number_text)
        if prefix != active_prefix:
            if active_prefix is not None:
                closed_prefixes.add(active_prefix)
            if prefix in closed_prefixes:
                raise ValueError("Golden case prefixes must occupy deterministic blocks.")
            active_prefix = prefix
        expected = previous_number.get(prefix, 0) + 1
        if number != expected:
            raise ValueError(f"Golden case IDs for {prefix} must be sequential.")
        previous_number[prefix] = number
