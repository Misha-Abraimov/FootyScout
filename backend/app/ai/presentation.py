"""Deterministic presentation rules for user-facing AI Scout content."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricPresentation:
    label: str
    format: str
    digits: int = 1
    unit: str | None = None


METRIC_PRESENTATION: dict[str, MetricPresentation] = {
    "role_distance": MetricPresentation("Role Fit distance", "decimal", 3),
    "centroid_distance": MetricPresentation("Archetype distance", "decimal", 3),
    "second_centroid_distance": MetricPresentation(
        "Second-closest archetype distance", "decimal", 3
    ),
    "actual_completion_rate": MetricPresentation("Actual pass completion", "percentage", 2),
    "expected_completion_rate": MetricPresentation("Expected completion rate", "percentage", 2),
    "completion_above_expected_pp": MetricPresentation(
        "Actual vs. expected passing", "percentage_points", 2
    ),
    "pressure_pass_rate": MetricPresentation("Passes under pressure", "percentage", 2),
    "progressive_pass_rate": MetricPresentation(
        "Progressive passing rate", "percentage", 2
    ),
    "long_pass_rate": MetricPresentation("Long-pass rate", "percentage", 2),
    "final_third_entries_per_100_passes": MetricPresentation(
        "Final-third entries per 100 passes", "decimal", 2, "per 100 passes"
    ),
    "positive_forward_distance_per_100_passes": MetricPresentation(
        "Positive forward distance per 100 passes",
        "decimal",
        1,
        None,
    ),
    "average_forward_distance": MetricPresentation(
        "Average forward distance per pass",
        "decimal",
        2,
        "StatsBomb pitch-coordinate units",
    ),
    "carry_share_of_actions": MetricPresentation("Carry involvement", "percentage", 2),
    "pass_attempts": MetricPresentation("Pass attempts", "integer", 0),
    "matches_observed": MetricPresentation("Matches observed", "integer", 0),
    "attacking_value": MetricPresentation("Attacking impact", "decimal", 3),
    "attacking_value_per_100_actions": MetricPresentation(
        "Attacking impact per 100 actions", "decimal", 2, "per 100 actions"
    ),
    "pass_value_per_100_actions": MetricPresentation(
        "Pass impact per 100 actions", "decimal", 2
    ),
    "pass_value_per_100_passes": MetricPresentation(
        "Pass impact per 100 passes", "decimal", 2, "per 100 passes"
    ),
    "carry_value_per_100_actions": MetricPresentation(
        "Carry impact per 100 actions", "decimal", 2
    ),
    "carry_value_per_100_carries": MetricPresentation(
        "Carry impact per 100 carries", "decimal", 2, "per 100 carries"
    ),
    "positive_value_action_rate": MetricPresentation(
        "Positive-value action rate", "percentage", 2
    ),
    "similarity_score": MetricPresentation("Style similarity", "decimal", 1),
}


def metric_presentation_payload() -> dict[str, dict[str, str | int | None]]:
    """Return the governed labels and formats supplied to grounded synthesis."""
    return {
        name: {
            "label": presentation.label,
            "format": presentation.format,
            "digits": presentation.digits,
            "unit": presentation.unit,
        }
        for name, presentation in METRIC_PRESENTATION.items()
    }


def metric_label(name: str) -> str:
    presentation = METRIC_PRESENTATION.get(name)
    if presentation is not None:
        return presentation.label
    return name.replace("_", " ").strip().capitalize()


def format_metric_value(name: str, value: float) -> str:
    """Format a known metric without changing its stored numeric value."""
    presentation = METRIC_PRESENTATION.get(name)
    if presentation is None:
        return f"{value:g}"
    if presentation.format == "integer":
        return f"{round(value):,}"
    if presentation.format == "percentage":
        return f"{_trimmed(value * 100, presentation.digits)}%"
    if presentation.format == "percentage_points":
        sign = "+" if value > 0 else ""
        return f"{sign}{_trimmed(value, presentation.digits)} pp"
    return _trimmed(value, presentation.digits)


def present_answer_markdown(markdown: str) -> str:
    """Apply governed metric semantics and formatting at the public boundary."""
    presented = normalize_markdown_headings(markdown)
    presented = re.sub(
        r"\b(?:single\s+concise\s+)?driver\s*=\s*closest\s+observed\s+dimension\b",
        "closest observed dimension",
        presented,
        flags=re.IGNORECASE,
    )
    for name in sorted(METRIC_PRESENTATION, key=len, reverse=True):
        presentation = METRIC_PRESENTATION[name]
        label_pattern = rf"(?:{re.escape(name)}|{re.escape(presentation.label)})"
        presented = re.sub(
            rf"(?P<label>\b{label_pattern}\b)"
            r"(?P<separator>[ \t]*(?:(?:is|was|of)[ \t]*)?[:=]?[ \t]*)"
            r"(?P<value>-?\d+(?:\.\d+)?)(?!\d|\.\d|\s*%)",
            lambda match, metric=name, source=presented: _present_metric_match(
                source,
                match,
                metric,
            ),
            presented,
            flags=re.IGNORECASE,
        )
        presented = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
            presentation.label,
            presented,
        )
    presented = re.sub(
        r"(?P<prefix>\b(?:raw\s+)?role fit distance(?:\s+(?:is|of))?\s*[:=]?\s*)"
        r"(?P<value>-?\d+\.\d{4,})",
        _round_role_fit_match,
        presented,
        flags=re.IGNORECASE,
    )
    presented = re.sub(
        r"\brole fit distance"
        r"(?P<centroid>\s+(?:to|from)\s+(?:the\s+)?[^\n.!?]{1,120}\bcentroid\b)",
        lambda match: f"Archetype distance{match.group('centroid')}",
        presented,
        flags=re.IGNORECASE,
    )
    presented = _remove_metric_reference_arrays(presented)
    presented = re.sub(r"(?<=\bpp)\s+percentage_points\b", "", presented)
    presented = re.sub(r"\bpp(?:\s+pp)+\b", "pp", presented, flags=re.IGNORECASE)
    presented = re.sub(r"\bper_100_passes\b", "per 100 passes", presented)
    presented = re.sub(r"[ \t]+([.,;:])", r"\1", presented)
    return presented


def _present_metric_match(
    markdown: str,
    match: re.Match[str],
    metric: str,
) -> str:
    presentation = METRIC_PRESENTATION[metric]
    value = float(match.group("value"))
    formatted = (
        match.group("value")
        if presentation.format == "percentage"
        and _is_unitless_role_fit_value(markdown, match.start())
        else format_metric_value(metric, value)
    )
    return f"{presentation.label}{match.group('separator')}{formatted}"


def _is_unitless_role_fit_value(markdown: str, offset: int) -> bool:
    """Avoid adding percent semantics to raw Role Fit gaps/contributions."""
    context = markdown[max(0, offset - 300) : offset]
    return bool(
        re.search(
            r"\b(?:distance\s+contributions?|feature\s+gaps?|"
            r"standardized\s+gaps?|role\s+fit\s+(?:drivers?|dimensions?))\b",
            context,
            re.IGNORECASE,
        )
    )


def present_numbered_citations(
    markdown: str,
    evidence_ids: tuple[str, ...] | list[str],
) -> str:
    """Replace internal ledger citations with stable reader-facing source numbers."""
    index_by_id = {
        evidence_id: index
        for index, evidence_id in enumerate(evidence_ids, start=1)
    }

    def replace_group(match: re.Match[str]) -> str:
        indexes = [
            index_by_id[evidence_id.strip()]
            for evidence_id in match.group("ids").split(";")
            if evidence_id.strip() in index_by_id
        ]
        return " ".join(f"[{index}]" for index in indexes)

    return re.sub(
        r"\[(?P<ids>[^\[\]]*evidence-\d+(?:\s*;\s*[^\[\]]*evidence-\d+)*)\]",
        replace_group,
        markdown,
        flags=re.IGNORECASE,
    )


def normalize_markdown_headings(markdown: str) -> str:
    """Undo only a provider-escaped ATX marker at the start of a Markdown line."""
    return re.sub(
        r"(?m)^(?P<indent>[ \t]{0,3})\\(?P<heading>#{1,6}[ \t]+)",
        lambda match: f"{match.group('indent')}{match.group('heading')}",
        markdown,
    )


def suppress_duplicate_limitation_section(
    markdown: str,
    limitations: tuple[str, ...] | list[str],
) -> str:
    """Remove only an exact, fully duplicated generated Limitations section."""
    normalized_limitations = {_normalize_limitation(item) for item in limitations}
    if not normalized_limitations:
        return markdown

    lines = markdown.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if not re.fullmatch(
            r"\s*#{1,6}\s+limitations\s*#*\s*", lines[index], re.IGNORECASE
        ):
            output.append(lines[index])
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not re.match(r"\s*#{1,6}\s+", lines[end]):
            end += 1
        body = [line for line in lines[index + 1 : end] if line.strip()]
        normalized_body = {
            _normalize_limitation(re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line))
            for line in body
        }
        if body and normalized_body and normalized_body.issubset(normalized_limitations):
            while output and not output[-1].strip():
                output.pop()
            index = end
            continue
        output.extend(lines[index:end])
        index = end
    return "\n".join(output).strip()


def _round_role_fit_match(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}{float(match.group('value')):.3f}"


def _trimmed(value: float, digits: int) -> str:
    if digits == 0:
        return str(round(value))
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _normalize_limitation(value: str) -> str:
    return " ".join(value.casefold().strip().rstrip(".").split())


def _remove_metric_reference_arrays(markdown: str) -> str:
    known = {
        value.casefold()
        for name, presentation in METRIC_PRESENTATION.items()
        for value in (name, presentation.label)
    }

    def replace(match: re.Match[str]) -> str:
        labels = re.findall(r'"([^"\r\n]+)"', match.group(0))
        return "" if labels and all(label.casefold() in known for label in labels) else match.group(0)

    return re.sub(r"\[(?:\s*\"[^\"\r\n]+\"\s*,?)+\]", replace, markdown)
