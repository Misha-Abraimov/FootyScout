"""Deterministic user-facing presentation contracts for AI Scout."""

from app.ai.presentation import (
    format_metric_value,
    metric_label,
    metric_presentation_payload,
    normalize_markdown_headings,
    present_answer_markdown,
    present_numbered_citations,
    suppress_duplicate_limitation_section,
)


def test_metric_names_have_governed_human_readable_labels() -> None:
    assert metric_label("role_distance") == "Role Fit distance"
    assert metric_label("centroid_distance") == "Archetype distance"
    assert (
        metric_label("second_centroid_distance")
        == "Second-closest archetype distance"
    )
    assert (
        metric_label("positive_forward_distance_per_100_passes")
        == "Positive forward distance per 100 passes"
    )
    assert metric_label("carry_share_of_actions") == "Carry involvement"
    assert metric_label("expected_completion_rate") == "Expected completion rate"
    assert metric_label("pressure_pass_rate") == "Passes under pressure"
    assert metric_label("progressive_pass_rate") == "Progressive passing rate"
    assert metric_label("long_pass_rate") == "Long-pass rate"
    assert metric_label("completion_above_expected_pp") == (
        "Actual vs. expected passing"
    )


def test_metric_values_are_formatted_for_readers_without_mutating_values() -> None:
    assert format_metric_value("role_distance", 0.38882682605732694) == "0.389"
    assert format_metric_value("expected_completion_rate", 0.923) == "92.3%"
    assert format_metric_value("pressure_pass_rate", 0.9096) == "90.96%"
    assert format_metric_value("pass_attempts", 1234) == "1,234"
    assert format_metric_value("completion_above_expected_pp", 2.003) == "+2 pp"


def test_positive_forward_distance_preserves_semantics_without_guessing_units() -> None:
    contract = metric_presentation_payload()[
        "positive_forward_distance_per_100_passes"
    ]
    assert contract["unit"] is None
    presented = present_answer_markdown(
        "positive_forward_distance_per_100_passes: 328.86"
    )
    assert "Positive forward distance per 100 passes" in presented
    assert "StatsBomb" not in presented
    assert "yards" not in presented.casefold()
    assert "meters" not in presented.casefold()


def test_public_markdown_humanizes_metrics_and_rounds_role_fit_distance() -> None:
    presented = present_answer_markdown(
        "Raw role_distance: 0.38882682605732694; expected_completion_rate: 0.923; "
        "closest: pressure_pass_rate."
    )
    assert presented == (
        "Raw Role Fit distance: 0.389; Expected completion rate: 92.3%; "
        "closest: Passes under pressure."
    )


def test_metric_formatting_does_not_consume_the_next_ordered_list_number() -> None:
    evidence_ids = ("run:evidence-1", "run:evidence-2")
    presented = present_numbered_citations(
        present_answer_markdown(
            "Style similarity shown; single concise driver = closest observed dimension:\n\n"
            "1. First player — distance 0.3888 [run:evidence-1]\n"
            "2. Second player — completion 42% [run:evidence-2]\n"
            "3. Third player — closest observed dimension: expected_completion_rate\n"
            "4. Fourth player — closest observed dimension: expected_completion_rate\n"
            "5. Fifth player — closest observed dimension: pressure_pass_rate\n"
            "6. Sixth player — closest observed dimension: carry_share_of_actions\n"
            "7. Seventh player — closest observed dimension: long_pass_rate\n"
            "8. Eighth player — closest observed dimension: progressive_pass_rate\n"
            "9. Ninth player — score 69.3 [run:evidence-1]\n"
            "10. Tenth player — score 69.3 [run:evidence-2]"
        ),
        evidence_ids,
    )

    assert "single concise driver" not in presented
    assert "closest observed dimension" in presented
    for index in range(1, 11):
        assert f"\n{index}. " in f"\n{presented}"
    assert "distance 0.3888 [1]" in presented
    assert "completion 42% [2]" in presented
    assert "score 69.3 [1]" in presented
    assert "300%." not in presented
    assert "1000%." not in presented


def test_role_fit_unitless_contribution_is_not_converted_to_percentage() -> None:
    presented = present_answer_markdown(
        "## Distance contributions\n\n- expected_completion_rate: 0.307"
    )
    assert "Expected completion rate: 0.307" in presented
    assert "30.7%" not in presented


def test_explicit_rate_and_percentage_point_metrics_keep_governed_units() -> None:
    presented = present_answer_markdown(
        "Expected completion rate: 0.307. completion_above_expected_pp: 2.4."
    )
    assert "Expected completion rate: 30.7%" in presented
    assert "Actual vs. expected passing: +2.4 pp" in presented


def test_percentage_point_unit_is_rendered_only_once() -> None:
    assert present_answer_markdown("Passing: 1.2 pp pp") == "Passing: 1.2 pp"


def test_unitless_role_fit_feature_gap_remains_raw_decimal() -> None:
    presented = present_answer_markdown(
        "## Feature gaps\n\n- pressure_pass_rate: 0.1919"
    )
    assert "Passes under pressure: 0.1919" in presented
    assert "%" not in presented


def test_public_markdown_removes_machine_metric_references_and_units() -> None:
    presented = present_answer_markdown(
        'Expected completion rate: 0.9096 ["Expected completion rate"]. '
        "Actual vs. expected passing: 1.34 percentage_points. "
        "Final-third entries per 100 passes: 12.76 per_100_passes."
    )

    assert '["Expected completion rate"]' not in presented
    assert "percentage_points" not in presented
    assert "per_100_passes" not in presented
    assert "Expected completion rate: 90.96%" in presented
    assert "Actual vs. expected passing: +1.34 pp" in presented
    assert "Final-third entries per 100 passes: 12.76 per 100 passes" in presented


def test_provider_escaped_atx_heading_is_normalized_without_stripping_backslashes() -> None:
    assert normalize_markdown_headings(r"\## Style overview") == "## Style overview"
    assert normalize_markdown_headings(r"Path C:\FootyScout remains") == (
        r"Path C:\FootyScout remains"
    )


def test_reader_citations_remove_internal_run_ids_and_preserve_source_order() -> None:
    presented = present_numbered_citations(
        "Claim [run-uuid:evidence-2; run-uuid:evidence-3].",
        ("run-uuid:evidence-2", "run-uuid:evidence-3"),
    )

    assert presented == "Claim [1] [2]."
    assert "run-uuid" not in presented


def test_archetype_centroid_and_team_role_fit_use_distinct_terms() -> None:
    presented = present_answer_markdown(
        "centroid_distance: 0.45678. "
        "Role Fit distance to the Safe Circulator centroid is 0.45678. "
        "role_distance: 0.388826826."
    )
    assert "Archetype distance: 0.457" in presented
    assert "Archetype distance to the Safe Circulator centroid" in presented
    assert "Role Fit distance to the Safe Circulator centroid" not in presented
    assert "Role Fit distance: 0.389" in presented


def test_only_exact_duplicate_limitation_sections_are_suppressed() -> None:
    limitation = "Role Fit does not predict future performance."
    duplicated = (
        "## Role Fit\n\nSupported answer.\n\n"
        "## Limitations\n\n- Role Fit does not predict future performance."
    )
    assert suppress_duplicate_limitation_section(duplicated, (limitation,)) == (
        "## Role Fit\n\nSupported answer."
    )
    distinct = "## Limitations\n\n- This distinct caveat belongs in the answer."
    assert suppress_duplicate_limitation_section(distinct, (limitation,)) == distinct
