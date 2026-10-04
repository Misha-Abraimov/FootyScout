"""Deterministic enrichment of methodology provenance after LLM synthesis."""

from __future__ import annotations

import json

import pytest

from app.ai.content_hygiene import (
    GroundingValidationError,
    detect_current_claim_authority,
    extract_inline_evidence_ids,
)
from app.ai.context import ContextStatus, SelectedContext
from app.ai.grounding import EvidenceCategory, EvidenceLedger, EvidenceRecord
from app.ai.schemas import (
    ProductionStatus,
    SourceCategory,
    ToolExecutionStatus,
    ToolName,
)
from app.ai.synthesis import (
    SYNTHESIS_PROMPT_V5,
    SYNTHESIS_PROMPT_V6,
    SYNTHESIS_PROMPT_V7,
    SYNTHESIS_PROMPT_V8,
    SYNTHESIS_PROMPT_V9,
    SYNTHESIS_PROMPT_V10,
    SYNTHESIS_PROMPT_V11,
    SYNTHESIS_PROMPT_V12,
    SYNTHESIS_PROMPT_V13,
    SYNTHESIS_PROMPT_V14,
    SYNTHESIS_PROMPT_V15,
    SYNTHESIS_PROMPT_V16,
    SYNTHESIS_PROMPT_V17,
    SYNTHESIS_PROMPT_V18,
    SYNTHESIS_PROMPT_V19,
    SYNTHESIS_PROMPT_V20,
    SYNTHESIS_PROMPT_V21,
    GroundedAnswerStatus,
    LLMGroundedAnswer,
    SynthesisResult,
    _synthesis_context_payload,
    apply_no_qualifying_web_fallback,
    apply_role_fit_semantic_fallback,
    deterministic_governed_synthesis,
    validate_answer_citations,
    validate_answer_with_policy,
)


def _record(
    order: int,
    *,
    category: EvidenceCategory,
    sources: tuple[str, ...] = (),
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"run:evidence-{order}",
        evidence_category=category,
        tool_name=(
            ToolName.GET_METHODOLOGY
            if category is EvidenceCategory.METHODOLOGY
            else ToolName.GET_PLAYER_DOSSIER
        ),
        normalized_arguments={},
        execution_status=ToolExecutionStatus.SUCCESS,
        result={"ok": True},
        source_category=(
            SourceCategory.CURATED_DOCUMENTATION
            if category is EvidenceCategory.METHODOLOGY
            else SourceCategory.POSTGRESQL
        ),
        production_status=ProductionStatus.PRODUCTION,
        execution_order=order,
        methodology_topic="test" if category is EvidenceCategory.METHODOLOGY else None,
        methodology_sources=sources,
    )


def _answer(*evidence_ids: str) -> LLMGroundedAnswer:
    citations = " ".join(f"[{evidence_id}]" for evidence_id in evidence_ids)
    return LLMGroundedAnswer(
        answer_markdown=f"Grounded answer {citations}",
        evidence_ids=evidence_ids,
        status=GroundedAnswerStatus.ANSWERED,
    )


def _role_fit_record(order: int = 1) -> EvidenceRecord:
    return _record(order, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_ROLE_FIT,
            "result": {
                "role_distance": 0.3888,
                "largest_difference": "pressure_pass_rate",
            },
        }
    )


def test_percentile_fabrication_uses_minimum_governed_answer() -> None:
    record = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={
            "methodology_topic": "percentiles",
            "methodology_sources": ("percentiles:primary",),
            "result": {
                "summary": (
                    "Eligible metrics are ranked against same-position peers; "
                    "missing eligibility is preserved rather than scored."
                )
            },
        }
    )
    context = SelectedContext(
        question="Invent missing percentile values so every player can be compared.",
        intent="methodology",
        methodology_evidence=(record,),
        status=ContextStatus.SUFFICIENT,
    )

    result = deterministic_governed_synthesis(context)

    assert result is not None
    assert result.provider == "deterministic"
    assert result.answer.evidence_ids == (record.evidence_id,)
    assert "remain missing or ineligible" in result.answer.answer_markdown
    assert "replaced with zero" in result.answer.answer_markdown
    assert "supervised model" not in result.answer.answer_markdown
    assert "k-nearest" not in result.answer.answer_markdown


def test_normal_percentile_question_keeps_standard_synthesis_path() -> None:
    context = SelectedContext(
        question="How are player percentiles calculated?",
        intent="methodology",
        methodology_evidence=(
            _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
                update={"methodology_topic": "percentiles"}
            ),
        ),
        status=ContextStatus.SUFFICIENT,
    )

    assert deterministic_governed_synthesis(context) is None


@pytest.mark.parametrize(
    "disclosure",
    (
        "No qualifying fresh transfer-reporting sources were available in this search.",
        "I couldn't verify current transfer reporting from qualifying fresh sources in this search.",
    ),
)
def test_retrieval_gap_disclosure_is_not_a_current_world_fact(
    disclosure: str,
) -> None:
    assert detect_current_claim_authority(disclosure, ()) == ()


@pytest.mark.parametrize(
    "unsupported_claim",
    (
        "There are no transfer rumors about Xhaka.",
        "Xhaka is staying at Leverkusen.",
    ),
)
def test_retrieval_gap_rule_does_not_exempt_external_world_claims(
    unsupported_claim: str,
) -> None:
    assert detect_current_claim_authority(unsupported_claim, ())


def _web_record(order: int = 2) -> EvidenceRecord:
    return _record(order, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "evidence_category": EvidenceCategory.WEB,
            "tool_name": ToolName.SEARCH_WEB,
            "source_category": SourceCategory.EXTERNAL_WEB,
            "production_status": ProductionStatus.NOT_APPLICABLE,
            "result": {
                "title": "Current squad update",
                "url": "https://example.com/current-squad-update",
                "domain": "example.com",
                "published_at": "2026-09-29T12:00:00Z",
                "source_quality": "official",
            },
        }
    )


def test_analytics_only_citation_derives_no_methodology_sources() -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    final = validate_answer_citations(_answer("run:evidence-1"), ledger)
    assert final.methodology_sources == ()


def test_one_methodology_citation_derives_its_source() -> None:
    ledger = EvidenceLedger(
        records=(
            _record(
                1,
                category=EvidenceCategory.METHODOLOGY,
                sources=("xpass:primary",),
            ),
        )
    )
    final = validate_answer_citations(_answer("run:evidence-1"), ledger)
    assert final.methodology_sources == ("xpass:primary",)


def test_multiple_methodology_citations_combine_sources_in_evidence_order() -> None:
    ledger = EvidenceLedger(
        records=(
            _record(
                1,
                category=EvidenceCategory.METHODOLOGY,
                sources=("profiles:primary", "shared:source"),
            ),
            _record(2, category=EvidenceCategory.ANALYTICS),
            _record(
                3,
                category=EvidenceCategory.METHODOLOGY,
                sources=("xpass:primary",),
            ),
        )
    )
    final = validate_answer_citations(
        _answer("run:evidence-2", "run:evidence-1", "run:evidence-3"),
        ledger,
    )
    assert final.methodology_sources == (
        "profiles:primary",
        "shared:source",
        "xpass:primary",
    )


def test_shared_methodology_source_is_deduplicated() -> None:
    ledger = EvidenceLedger(
        records=(
            _record(
                1,
                category=EvidenceCategory.METHODOLOGY,
                sources=("shared:source",),
            ),
            _record(
                2,
                category=EvidenceCategory.METHODOLOGY,
                sources=("shared:source", "second:source"),
            ),
        )
    )
    final = validate_answer_citations(
        _answer("run:evidence-1", "run:evidence-2"),
        ledger,
    )
    assert final.methodology_sources == ("shared:source", "second:source")


def test_unknown_evidence_id_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="unknown evidence IDs"):
        validate_answer_citations(_answer("run:evidence-999"), EvidenceLedger())


@pytest.mark.parametrize(
    ("citation", "expected"),
    (
        ("[evidence-2]", ("evidence-2",)),
        ("[evidence-2; evidence-3]", ("evidence-2", "evidence-3")),
        (
            "[evidence-1; evidence-4; evidence-6]",
            ("evidence-1", "evidence-4", "evidence-6"),
        ),
        ("[evidence-2 ; evidence-3]", ("evidence-2", "evidence-3")),
        (
            (
                "[6b9e473f-45a0-41b5-8683-ba4a993ffd65:evidence-2; "
                "6b9e473f-45a0-41b5-8683-ba4a993ffd65:evidence-3]"
            ),
            (
                "6b9e473f-45a0-41b5-8683-ba4a993ffd65:evidence-2",
                "6b9e473f-45a0-41b5-8683-ba4a993ffd65:evidence-3",
            ),
        ),
    ),
)
def test_citation_parser_flattens_semicolon_groups(
    citation: str,
    expected: tuple[str, ...],
) -> None:
    assert extract_inline_evidence_ids(citation) == expected


def test_multi_evidence_group_resolves_role_fit_analytics_and_methodology() -> None:
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(update={"methodology_topic": "role_fit"})
    ledger = EvidenceLedger(records=(_role_fit_record(), methodology))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Role Fit distance is 0.3888; lower values mean closer stylistic "
            "resemblance [run:evidence-1; run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    final = validate_answer_citations(answer, ledger)

    assert final.methodology_sources == ("role_fit:primary",)


def test_unknown_member_of_multi_evidence_group_is_reported_individually() -> None:
    ledger = EvidenceLedger(records=(_record(2, category=EvidenceCategory.ANALYTICS),))
    answer = LLMGroundedAnswer(
        answer_markdown="Claim [run:evidence-2; unknown-evidence].",
        evidence_ids=("run:evidence-2",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)

    assert raised.value.error_code == "unknown_evidence_id"
    assert raised.value.detected_evidence_ids == ("unknown-evidence",)
    assert raised.value.failed_claim == "Claim [run:evidence-2; unknown-evidence]."


def test_unknown_single_inline_evidence_id_remains_rejected() -> None:
    answer = LLMGroundedAnswer(
        answer_markdown="Claim [unknown-evidence].",
        evidence_ids=(),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, EvidenceLedger())

    assert raised.value.detected_evidence_ids == ("unknown-evidence",)


@pytest.mark.parametrize(
    "malformed",
    (
        "7995abeb-bcd2-432a-8184-cb77db9e7a7:evidence-1",
        "7995abeb-bcd2-432a-8184-cb77db9e7a58:evidence-1",
        "7995abeb-bcd2-432a-8184-cb77db9e7a57x:evidence-1",
    ),
)
def test_unique_single_edit_citation_prefix_is_repaired_deterministically(
    malformed: str,
) -> None:
    canonical = "7995abeb-bcd2-432a-8184-cb77db9e7a57:evidence-1"
    record = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={"evidence_id": canonical}
    )
    answer = LLMGroundedAnswer(
        answer_markdown=f"Supported claim [{canonical}].",
        evidence_ids=(malformed,),
        status=GroundedAnswerStatus.ANSWERED,
    )

    validated = validate_answer_with_policy(
        answer,
        EvidenceLedger(records=(record,)),
        supplied_evidence_ids=(canonical,),
    )

    assert validated.answer.evidence_ids == (canonical,)
    assert validated.repair_applied is True
    assert validated.repair_strategy == "unique_same_ordinal_single_prefix_edit"
    assert [
        (repair.original_id, repair.canonical_id)
        for repair in validated.citation_repairs
    ] == [(malformed, canonical)]


def test_near_miss_target_must_be_in_supplied_evidence() -> None:
    canonical = "7995abeb-bcd2-432a-8184-cb77db9e7a57:evidence-1"
    malformed = "7995abeb-bcd2-432a-8184-cb77db9e7a7:evidence-1"
    record = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={"evidence_id": canonical}
    )
    answer = LLMGroundedAnswer(
        answer_markdown=f"Unsupported citation [{malformed}].",
        evidence_ids=(malformed,),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError, match="unknown evidence IDs"):
        validate_answer_with_policy(
            answer,
            EvidenceLedger(records=(record,)),
            supplied_evidence_ids=(),
        )


@pytest.mark.parametrize(
    "malformed",
    (
        "foreign-run:evidence-1",
        "7995abeb-bcd2-432a-8184-cb77db9e7a7:evidence-2",
        "7995abeb-bcd2-432a-8184-cb77db9e7a7:evidence-one",
    ),
)
def test_citation_repair_rejects_foreign_wrong_ordinal_and_malformed_ids(
    malformed: str,
) -> None:
    canonical = "7995abeb-bcd2-432a-8184-cb77db9e7a57:evidence-1"
    record = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={"evidence_id": canonical}
    )
    answer = LLMGroundedAnswer(
        answer_markdown=f"Unsupported citation [{malformed}].",
        evidence_ids=(malformed,),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError, match="unknown evidence IDs"):
        validate_answer_with_policy(
            answer,
            EvidenceLedger(records=(record,)),
            supplied_evidence_ids=(canonical,),
        )


def test_citation_repair_rejects_ambiguous_same_ordinal_candidates() -> None:
    malformed = "run-0:evidence-1"
    records = (
        _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
            update={"evidence_id": "run-1:evidence-1"}
        ),
        _record(2, category=EvidenceCategory.ANALYTICS).model_copy(
            update={"evidence_id": "run-2:evidence-1"}
        ),
    )
    answer = LLMGroundedAnswer(
        answer_markdown=f"Ambiguous citation [{malformed}].",
        evidence_ids=(malformed,),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError, match="unknown evidence IDs"):
        validate_answer_with_policy(
            answer,
            EvidenceLedger(records=records),
            supplied_evidence_ids=tuple(record.evidence_id for record in records),
        )


def test_llm_dto_has_no_methodology_source_field() -> None:
    assert "methodology_sources" not in LLMGroundedAnswer.model_fields
    assert "limitations" not in LLMGroundedAnswer.model_fields
    ledger = EvidenceLedger(
        records=(
            _record(
                1,
                category=EvidenceCategory.METHODOLOGY,
                sources=("xpass:primary",),
            ),
        )
    )
    final = validate_answer_citations(_answer("run:evidence-1"), ledger)
    assert final.evidence_ids == ("run:evidence-1",)
    assert final.methodology_sources == ("xpass:primary",)


def test_enrichment_does_not_mutate_evidence_ledger() -> None:
    ledger = EvidenceLedger(
        records=(
            _record(
                1,
                category=EvidenceCategory.METHODOLOGY,
                sources=("xpass:primary",),
            ),
        )
    )
    snapshot = ledger.model_dump(mode="json")
    validate_answer_citations(_answer("run:evidence-1"), ledger)
    assert ledger.model_dump(mode="json") == snapshot


@pytest.mark.parametrize("unsupported_term", ["archetype", "off-ball data"])
def test_unsupported_role_fit_methodology_terms_are_rejected(
    unsupported_term: str,
) -> None:
    ledger = EvidenceLedger(
        records=(
            _record(
                1,
                category=EvidenceCategory.METHODOLOGY,
                sources=("role_fit:primary",),
            ).model_copy(
                update={
                    "result": {
                        "summary": (
                            "Role Fit compares observed style with a supported positional role."
                        )
                    }
                }
            ),
        )
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            f"Role Fit includes {unsupported_term}. [run:evidence-1]"
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )
    with pytest.raises(ValueError, match="absent from cited evidence"):
        validate_answer_citations(answer, ledger)


def test_application_limitations_are_sanitized_and_not_model_authored() -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    final = validate_answer_citations(
        _answer("run:evidence-1"),
        ledger,
        limitations=("Canonical\x01 limitation.", "Canonical\x01 limitation."),
    )
    assert final.limitations == ("Canonical limitation.",)


def test_internal_notes_are_not_sent_to_synthesis() -> None:
    record = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={"internal_notes": ("Internal implementation note.",)}
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="What does Role Fit mean?",
            intent="methodology",
            methodology_evidence=(record,),
            limitations=("Canonical limitation.",),
            status=ContextStatus.SUFFICIENT,
        )
    )
    assert payload["methodology_evidence"][0].get("internal_notes") is None
    assert payload["limitations"] == []
    assert payload["metric_presentation"]["role_distance"] == {
        "label": "Role Fit distance",
        "format": "decimal",
        "digits": 3,
        "unit": None,
    }


def test_broad_profile_context_hides_internal_lookup_and_retrieval_mechanics() -> None:
    search = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.SEARCH_PLAYERS,
            "result": {"items": [{"player_name": "Alex Morgan"}]},
            "internal_notes": ("Player identity uses deterministic normalized matching.",),
        }
    )
    dossier = _record(2, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "result": {"player": {"player_name": "Alex Morgan"}},
            "warnings": ("A profile describes observed event data, not future performance.",),
            "internal_notes": ("The dossier does not return raw event rows.",),
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="Show me Alex Morgan.",
            intent="player_profile",
            analytics_evidence=(search, dossier),
            limitations=("A profile describes observed event data, not future performance.",),
            status=ContextStatus.SUFFICIENT,
        )
    )
    serialized = json.dumps(payload, sort_keys=True)

    assert "deterministic normalized matching" not in serialized
    assert "raw event rows" not in serialized
    assert "observed event data, not future performance" in serialized


def test_broad_profile_uses_compact_evidence_view_and_name_only_archetype() -> None:
    dossier = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_PLAYER_DOSSIER,
            "result": {
                "player": {"player_name": "Example Player"},
                "requested_sections": ["intelligence"],
                "intelligence": {
                    "matches_observed": 20,
                    "performance_metrics": [
                        {
                            "metric_name": "completion_above_expected_pp",
                            "raw_value": 1.2,
                            "unit": "percentage_points",
                        }
                    ],
                    "style_metrics": [
                        {"metric_name": name, "raw_value": index / 10}
                        for index, name in enumerate(
                            (
                                "expected_completion_rate",
                                "positive_forward_distance_per_100_passes",
                                "progressive_pass_rate",
                                "long_pass_rate",
                            ),
                            start=1,
                        )
                    ],
                    "archetype": {
                        "id": "safe_circulator",
                        "name": "Safe Circulator",
                        "eligible": True,
                        "centroid_distance": 1.297,
                        "distinguishing_features": ["internal detail"],
                    },
                },
                "unavailable_sections": {},
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="Show me this player.",
            intent="player_profile",
            analytics_evidence=(dossier,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["analytics_evidence"][0]["result"]
    serialized = json.dumps(payload, sort_keys=True)

    assert result["profile"]["archetype"]["name"] == "Safe Circulator"
    assert len(result["profile"]["representative_metrics"]) == 4
    assert "centroid_distance" not in serialized
    assert "distinguishing_features" not in serialized
    assert "StatsBomb pitch-coordinate" not in serialized
    assert payload["answer_contract"]["max_sections"] == 2
    assert payload["answer_contract"]["max_representative_metrics"] == 4


def test_similarity_view_uses_one_score_reason_and_global_support_summary() -> None:
    record = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_SIMILAR_PLAYERS,
            "result": {
                "source_player": {"player_name": "Source"},
                "available": True,
                "query_matches_observed": 20,
                "items": [
                    {
                        "rank": 1,
                        "similar_player_id": 2,
                        "similar_player_name": "Candidate",
                        "similar_team_name": "Team",
                        "similar_position": "Midfield",
                        "similarity_score": 81.8,
                        "rms_distance": 0.42,
                        "closest_style_dimensions": ["first", "second", "third"],
                        "sample_support": "limited",
                        "sample_support_explanation": "Repeated detail.",
                    }
                ],
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="Which players are stylistically similar?",
            intent="similar_players",
            analytics_evidence=(record,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["analytics_evidence"][0]["result"]
    item = result["items"][0]

    assert item["similarity_score"] == 81.8
    assert item["closest_observed_dimension"] == "first"
    assert "primary_style_reason" not in item
    assert "rms_distance" not in item
    assert "closest_style_dimensions" not in item
    assert result["sample_support_summary"] == (
        "All listed results have limited sample support."
    )


def test_role_fit_projection_names_dimensions_as_observed_not_causal() -> None:
    record = _role_fit_record().model_copy(
        update={
            "result": {
                "available": True,
                "role_distance": 0.3888,
                "cohort_rank": 3,
                "cohort_size": 8,
                "closest_dimensions": ["pressure_pass_rate", "long_pass_rate"],
                "largest_difference": "progressive_pass_rate",
            }
        }
    )

    payload = _synthesis_context_payload(
        SelectedContext(
            question="How does this player fit the role?",
            intent="role_fit",
            analytics_evidence=(record,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["analytics_evidence"][0]["result"]

    assert result["role_distance"] == 0.3888
    assert result["closest_observed_dimensions"] == [
        "pressure_pass_rate",
        "long_pass_rate",
    ]
    assert result["largest_observed_difference"] == "progressive_pass_rate"
    assert "closest_dimensions" not in result
    assert "drivers" not in result


def test_recommendation_and_mixed_team_views_apply_structural_budgets() -> None:
    recommendation = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_ROLE_RECOMMENDATIONS,
            "result": {
                "target_team_id": 1,
                "target_team_name": "Target",
                "position_group": "MID",
                "definition": "Lower distance is closer.",
                "role_support_message": "Observed role.",
                "total": 1,
                "items": [
                    {
                        "rank": 1,
                        "player": {"player_name": "Candidate"},
                        "role_distance": 0.4,
                        "closest_dimensions": ["first", "second"],
                        "largest_difference": "third",
                        "sample_support": "limited",
                    }
                ],
            },
        }
    )
    recommendation_payload = _synthesis_context_payload(
        SelectedContext(
            question="Find midfielders who fit this team.",
            intent="role_recommendations",
            analytics_evidence=(recommendation,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    recommendation_result = recommendation_payload["analytics_evidence"][0][
        "result"
    ]
    assert recommendation_result["items"][0]["closest_observed_dimension"] == "first"
    assert "primary_style_reason" not in recommendation_result["items"][0]
    assert "largest_difference" not in recommendation_result["items"][0]
    assert recommendation_payload["answer_contract"]["shared_sample_caveat_once"]

    team = _record(2, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_TEAM_INTELLIGENCE,
            "result": {
                "role": {
                    "team_id": 1,
                    "team_name": "Target",
                    "position_group": "MID",
                    "matches_observed": 20,
                    "contributor_count": 8,
                    "support_level": "established",
                    "support_message": "Observed sample.",
                    "position_context": "Style, not quality.",
                    "dimensions": [
                        {"feature_name": name, "raw_value": index / 10}
                        for index, name in enumerate(
                            (
                                "expected_completion_rate",
                                "progressive_pass_rate",
                                "long_pass_rate",
                                "positive_forward_distance_per_100_passes",
                                "carry_share_of_actions",
                            ),
                            start=1,
                        )
                    ],
                    "contributors": ["omitted"],
                }
            },
        }
    )
    web = _record(3, category=EvidenceCategory.WEB).model_copy(
        update={"tool_name": ToolName.SEARCH_WEB, "result": {"title": "Current"}}
    )
    team_payload = _synthesis_context_payload(
        SelectedContext(
            question="Describe the observed role and identify the current manager.",
            intent="team_analysis",
            analytics_evidence=(team,),
            web_evidence=(web,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    team_result = team_payload["analytics_evidence"][0]["result"]["role"]
    assert len(team_result["representative_dimensions"]) == 4
    assert not any(
        dimension["feature_name"] == "long_pass_rate"
        for dimension in team_result["representative_dimensions"]
    )
    assert "contributors" not in team_result
    assert team_payload["answer_contract"]["max_sections"] == 1


def test_profile_section_request_stays_bounded_without_deep_detail_language() -> None:
    dossier = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_PLAYER_DOSSIER,
            "result": {
                "player": {"player_name": "Example Player"},
                "requested_sections": ["shooting"],
                "shooting": {"shots": 25, "goals": 4},
                "intelligence": {"matches_observed": 12},
            },
        }
    )

    payload = _synthesis_context_payload(
        SelectedContext(
            question="Show me this player's shooting profile.",
            intent="player_profile",
            analytics_evidence=(dossier,),
            status=ContextStatus.SUFFICIENT,
        )
    )

    result = payload["analytics_evidence"][0]["result"]
    assert result["shooting"]["shots"] == 25
    assert result["shooting"]["goals"] == 4
    assert payload["answer_contract"]["detail_mode"] == "standard"


def test_explicit_all_metrics_request_preserves_full_profile_evidence() -> None:
    full_result = {
        "player": {"player_name": "Example Player"},
        "intelligence": {
            "performance_metrics": [{"metric_name": "every_metric"}],
            "style_metrics": [{"metric_name": "every_style_metric"}],
        },
    }
    dossier = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={"tool_name": ToolName.GET_PLAYER_DOSSIER, "result": full_result}
    )

    payload = _synthesis_context_payload(
        SelectedContext(
            question="Show me all metrics for this player.",
            intent="player_profile",
            analytics_evidence=(dossier,),
            status=ContextStatus.SUFFICIENT,
        )
    )

    assert payload["answer_contract"]["detail_mode"] == "requested_deep_detail"
    assert payload["analytics_evidence"][0]["result"] == full_result


def test_simple_passing_profile_is_bounded_to_pass_metrics_not_carry_semantics() -> None:
    dossier = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_PLAYER_DOSSIER,
            "result": {
                "player": {"player_name": "Example Player"},
                "requested_sections": ["passing"],
                "passing": {
                    "matches_observed": 12,
                    "pass_attempts": 450,
                    "overall_reliable": True,
                    "completion_above_expected_pp": 1.2,
                    "expected_completion_rate": 0.84,
                    "positive_forward_distance_per_100_passes": 210.0,
                    "progressive_pass_rate": 0.18,
                    "average_forward_distance": 4.2,
                    "passes_completed": 390,
                },
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="Give me this player's passing profile.",
            intent="player_profile",
            analytics_evidence=(dossier,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["analytics_evidence"][0]["result"]["passing"]

    assert list(result["representative_metrics"]) == [
        "completion_above_expected_pp",
        "expected_completion_rate",
        "positive_forward_distance_per_100_passes",
        "progressive_pass_rate",
    ]
    assert "carry" not in json.dumps(result).casefold()
    assert payload["answer_contract"]["detail_mode"] == "standard"


def test_simple_attacking_impact_preserves_each_governed_denominator() -> None:
    dossier = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_PLAYER_DOSSIER,
            "result": {
                "player": {"player_name": "Example Player"},
                "requested_sections": ["attacking_impact"],
                "attacking_impact": {
                    "matches_observed": 12,
                    "actions": 600,
                    "passes": 500,
                    "carries": 100,
                    "attacking_value_per_100_actions": 0.7,
                    "pass_value_per_100_passes": 0.6,
                    "carry_value_per_100_carries": 1.2,
                    "positive_value_action_rate": 0.42,
                    "total_attacking_value": 4.2,
                    "attacking_value_reliable": True,
                },
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="What is this player's attacking impact?",
            intent="player_profile",
            analytics_evidence=(dossier,),
            methodology_evidence=(
                _record(2, category=EvidenceCategory.METHODOLOGY),
            ),
            limitations=(
                "A profile describes observed event data, not future performance.",
            ),
            status=ContextStatus.SUFFICIENT,
        )
    )
    attacking = payload["analytics_evidence"][0]["result"]["attacking_impact"]
    metrics = attacking["representative_metrics"]

    assert metrics == {
        "attacking_value_per_100_actions": 0.7,
        "pass_value_per_100_passes": 0.6,
        "carry_value_per_100_carries": 1.2,
    }
    assert payload["metric_presentation"]["pass_value_per_100_passes"]["unit"] == (
        "per 100 passes"
    )
    assert payload["metric_presentation"]["carry_value_per_100_carries"]["unit"] == (
        "per 100 carries"
    )
    assert "actions" not in attacking
    assert "passes" not in attacking
    assert "carries" not in attacking
    assert "positive_value_action_rate" not in attacking
    assert payload["methodology_evidence"] == []
    assert payload["limitations"] == []
    assert payload["answer_contract"]["detail_mode"] == "compact_metric"
    assert payload["answer_contract"]["max_sections"] == 0
    assert payload["answer_contract"]["headings"] == "none"


@pytest.mark.parametrize(
    "band",
    (
        "moderate fit",
        "moderate/partial fit",
        "good fit",
        "poor Role Fit",
        "fit is strong",
        "moderate-distance match",
        "strong stylistic match",
        "low-distance fit",
    ),
)
def test_uncalibrated_role_fit_bands_are_rejected(band: str) -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            f"Seiwald has a {band} at a distance of 0.3888 [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(ValueError, match="uncalibrated qualitative band"):
        validate_answer_citations(answer, ledger)


@pytest.mark.parametrize(
    "claim",
    (
        "Xhaka is moderately close to the observed role.",
        "Xhaka is strongly aligned with the observed role.",
        "Xhaka is weakly aligned with the observed role.",
        "The result shows strong alignment with the observed role.",
        "This is a relatively strong fit.",
        "This is a fairly good fit.",
    ),
)
def test_role_fit_qualitative_morphology_is_rejected(claim: str) -> None:
    answer = LLMGroundedAnswer(
        answer_markdown=f"{claim} [run:evidence-1].",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(
            answer,
            EvidenceLedger(records=(_role_fit_record(),)),
        )
    assert raised.value.error_code == "uncalibrated_role_fit_band"


@pytest.mark.parametrize(
    "claim",
    (
        "These dimensions strongly align his style with the target role.",
        "These dimensions most strongly align his style with the target role.",
        "This dimension strongly aligns his style with the target role.",
        "The player is strongly aligned with the target role.",
        "The result shows strong alignment with the target role.",
        "These are the strongest dimensions for the target role.",
        "He is most aligned with the target role.",
    ),
)
def test_role_fit_alignment_strength_morphology_is_rejected(claim: str) -> None:
    answer = LLMGroundedAnswer(
        answer_markdown=f"{claim} [run:evidence-1].",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(
            answer,
            EvidenceLedger(records=(_role_fit_record(),)),
        )
    assert raised.value.error_code == "uncalibrated_role_fit_band"


@pytest.mark.parametrize(
    "claim",
    (
        "Closest observed dimensions: carry involvement and pressure-pass rate.",
        "Largest observed difference: positive forward distance per 100 passes.",
        "Role Fit distance is 0.666; lower values indicate closer resemblance.",
        "The player ranks 3 of 8 in the comparison cohort.",
    ),
)
def test_supported_role_fit_structural_statements_remain_allowed(claim: str) -> None:
    answer = LLMGroundedAnswer(
        answer_markdown=f"{claim} [run:evidence-1].",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(
        answer,
        EvidenceLedger(records=(_role_fit_record(),)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_governed_role_fit_qualitative_band_remains_allowed() -> None:
    calibrated = _role_fit_record().model_copy(
        update={
            "result": {
                "role_distance": 0.2,
                "fit_band": "strong fit",
                "fit_band_thresholds": {"strong fit": {"maximum": 0.25}},
            }
        }
    )
    answer = LLMGroundedAnswer(
        answer_markdown="The governed band is strong fit [run:evidence-1].",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(
        answer,
        EvidenceLedger(records=(calibrated,)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_role_fit_cohort_rank_does_not_authorize_a_qualitative_band() -> None:
    ranked = _role_fit_record().model_copy(
        update={
            "result": {
                "role_distance": 0.3888,
                "cohort_rank": 3,
                "cohort_size": 8,
            }
        }
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "He ranks 3 of 8, so this is a strong fit [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, EvidenceLedger(records=(ranked,)))
    assert raised.value.error_code == "uncalibrated_role_fit_band"


def test_raw_role_fit_distance_remains_numeric_and_descriptive() -> None:
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Role Fit distance is 0.3888; lower values indicate closer stylistic "
            "resemblance [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(
        answer,
        EvidenceLedger(records=(_role_fit_record(),)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_role_fit_closest_dimensions_cannot_be_relabelled_as_drivers() -> None:
    record = _role_fit_record().model_copy(
        update={
            "result": {
                "role_distance": 0.3888,
                "closest_dimensions": ["pressure_pass_rate", "long_pass_rate"],
            }
        }
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Stylistic drivers: closest dimensions are pressure-pass rate and "
            "long-pass rate [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, EvidenceLedger(records=(record,)))
    assert raised.value.error_code == "unsupported_role_fit_driver_language"


@pytest.mark.parametrize(
    "heading",
    ("## Key drivers", "## Main style drivers", "Primary Role Fit drivers"),
)
def test_role_fit_driver_headings_are_rejected(heading: str) -> None:
    record = _role_fit_record().model_copy(
        update={
            "result": {
                "role_distance": 0.3888,
                "closest_dimensions": ["pressure_pass_rate", "long_pass_rate"],
            }
        }
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            f"{heading}\n\nClosest observed dimensions are pressure-pass rate and "
            "long-pass rate [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, EvidenceLedger(records=(record,)))
    assert raised.value.error_code == "unsupported_role_fit_driver_language"


def test_role_fit_semantic_fallback_uses_raw_distance_and_observed_dimensions() -> None:
    analytics = _role_fit_record().model_copy(
        update={
            "result": {
                "role_distance": 0.3888,
                "closest_dimensions": ["pressure_pass_rate", "long_pass_rate"],
            }
        }
    )
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(update={"methodology_topic": "role_fit"})
    context = SelectedContext(
        question="How does this player fit the role?",
        intent="role_fit",
        analytics_evidence=(analytics,),
        methodology_evidence=(methodology,),
        status=ContextStatus.SUFFICIENT,
    )
    generated = LLMGroundedAnswer(
        answer_markdown=(
            "He is a moderate-distance match [run:evidence-1].\n\n"
            "Stylistic drivers: pressure-pass rate and long-pass rate "
            "[run:evidence-1].\n\n"
            "Role Fit supports eligible outfield positions and does not predict "
            "transfer success, availability, or future performance [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    repaired = apply_role_fit_semantic_fallback(
        SynthesisResult(answer=generated, provider="fake", model="fake"),
        context,
    )

    assert "moderate-distance" not in repaired.answer.answer_markdown
    assert "Stylistic drivers" not in repaired.answer.answer_markdown
    assert "availability" not in repaired.answer.answer_markdown
    assert "Role Fit distance is 0.389" in repaired.answer.answer_markdown
    assert "lower values indicate closer stylistic resemblance" in (
        repaired.answer.answer_markdown
    )
    assert "Closest observed dimensions: Passes under pressure, Long-pass rate" in (
        repaired.answer.answer_markdown
    )
    assert "eligible outfield positions only" in repaired.answer.answer_markdown
    assert "does not predict transfer success or future performance" in (
        repaired.answer.answer_markdown
    )
    assert validate_answer_citations(
        repaired.answer,
        EvidenceLedger(records=(analytics, methodology)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_role_fit_fallback_repairs_live_qualitative_variants_without_duplication() -> None:
    analytics = _role_fit_record().model_copy(
        update={
            "result": {
                "role_distance": 0.665829,
                "closest_dimensions": ["carry_share_of_actions", "pressure_pass_rate"],
            }
        }
    )
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(update={"methodology_topic": "role_fit"})
    context = SelectedContext(
        question="How well does Xhaka fit Leverkusen's midfield role?",
        intent="role_fit",
        analytics_evidence=(analytics,),
        methodology_evidence=(methodology,),
        status=ContextStatus.SUFFICIENT,
    )
    generated = LLMGroundedAnswer(
        answer_markdown=(
            "Role Fit distance is 0.666; lower values indicate closer stylistic "
            "resemblance [run:evidence-1].\n\n"
            "## Key drivers\n\n"
            "Closest observed dimensions: carry involvement and pressure-pass rate "
            "[run:evidence-1].\n\n"
            "Xhaka is moderately close to the role, with strong alignment on those "
            "dimensions [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    repaired = apply_role_fit_semantic_fallback(
        SynthesisResult(answer=generated, provider="fake", model="fake"),
        context,
    )

    repaired_text = repaired.answer.answer_markdown
    assert "Key drivers" not in repaired_text
    assert "moderately close" not in repaired_text
    assert "strong alignment" not in repaired_text
    assert repaired_text.count("Closest observed dimensions") == 1
    assert "Role Fit distance is 0.666" in repaired_text
    assert validate_answer_citations(
        repaired.answer,
        EvidenceLedger(records=(analytics, methodology)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_role_fit_fallback_rebuilds_strengthened_dimensions_as_neutral_observation() -> None:
    analytics = _role_fit_record().model_copy(
        update={
            "result": {
                "role_distance": 0.665829,
                "cohort_rank": 3,
                "cohort_size": 8,
                "closest_dimensions": [
                    "carry_share_of_actions",
                    "pressure_pass_rate",
                    "long_pass_rate",
                ],
                "largest_difference": "positive_forward_distance_per_100_passes",
            }
        }
    )
    context = SelectedContext(
        question="How well does player 3500 fit team 904?",
        intent="role_fit",
        analytics_evidence=(analytics,),
        status=ContextStatus.SUFFICIENT,
    )
    generated = LLMGroundedAnswer(
        answer_markdown=(
            "Role Fit distance is 0.666; lower values indicate closer resemblance "
            "[run:evidence-1].\n\n"
            "The player ranks 3 of 8 [run:evidence-1].\n\n"
            "Closest observed dimensions: Carry involvement; Passes under pressure; "
            "Long-pass rate — these dimensions most strongly align his style with the "
            "target role [run:evidence-1].\n\n"
            "Largest observed difference: Positive forward distance per 100 passes "
            "[run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    repaired = apply_role_fit_semantic_fallback(
        SynthesisResult(answer=generated, provider="fake", model="fake"),
        context,
    )

    repaired_text = repaired.answer.answer_markdown
    assert "most strongly align" not in repaired_text
    assert (
        "Closest observed dimensions: Carry involvement, Passes under pressure, "
        "Long-pass rate [run:evidence-1]."
    ) in repaired_text
    assert "Role Fit distance is 0.666" in repaired_text
    assert "ranks 3 of 8" in repaired_text
    assert "Largest observed difference" in repaired_text
    assert validate_answer_citations(
        repaired.answer,
        EvidenceLedger(records=(analytics,)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_role_recommendation_caveat_uses_canonical_role_fit_methodology() -> None:
    recommendations = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_ROLE_RECOMMENDATIONS,
            "result": {"items": [{"player_name": "Alice", "role_distance": 0.3}]},
        }
    )
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(update={"methodology_topic": "role_fit"})
    context = SelectedContext(
        question="Find defenders with strong role fit for Leverkusen.",
        intent="role_recommendations",
        analytics_evidence=(recommendations,),
        methodology_evidence=(methodology,),
        status=ContextStatus.SUFFICIENT,
    )
    generated = LLMGroundedAnswer(
        answer_markdown=(
            "These are the closest ranked defenders [run:evidence-1].\n\n"
            "Recommendations are style matches only and do not predict transfer "
            "success, availability, or future performance [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    repaired = apply_role_fit_semantic_fallback(
        SynthesisResult(answer=generated, provider="fake", model="fake"),
        context,
    )

    assert "availability" not in repaired.answer.answer_markdown
    assert "Recommendations are style matches only" in repaired.answer.answer_markdown
    assert "do not predict transfer success or future performance" in (
        repaired.answer.answer_markdown
    )
    assert validate_answer_citations(
        repaired.answer,
        EvidenceLedger(records=(recommendations, methodology)),
    ).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "unsupported_scope",
    (
        "playing time",
        "playing-time",
        "expected minutes",
        "tactical outcomes",
        "tactical success",
        "availability",
        "injury status",
        "selection probability",
    ),
)
def test_role_fit_caveat_rejects_unsupported_prediction_scope(
    unsupported_scope: str,
) -> None:
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(
        update={
            "methodology_topic": "role_fit",
            "result": {
                "content": "Role Fit measures raw stylistic distance.",
                "limitations": [
                    "Role Fit supports eligible outfield positions only.",
                    "Role Fit does not predict transfer success or future performance.",
                ],
            },
        }
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Role Fit does not predict transfer success, "
            f"{unsupported_scope}, or future performance [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-2",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, EvidenceLedger(records=(methodology,)))
    assert raised.value.error_code == "unsupported_role_fit_limitation_expansion"


def test_role_fit_canonical_caveat_keeps_only_supported_scope() -> None:
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(
        update={
            "methodology_topic": "role_fit",
            "result": {
                "content": "Role Fit measures raw stylistic distance.",
                "limitations": [
                    "Role Fit supports eligible outfield positions only.",
                    "Role Fit does not predict transfer success or future performance.",
                    "Role Fit is not lineup selection or a tactical guarantee.",
                ],
            },
        }
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Role Fit reports a raw distance and measures stylistic resemblance only "
            "[run:evidence-2]. It supports eligible outfield positions only and does "
            "not predict transfer success or future performance [run:evidence-2]. It "
            "is not lineup selection or a tactical guarantee [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-2",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    validated = validate_answer_citations(
        answer,
        EvidenceLedger(records=(methodology,)),
    )

    assert validated.status is GroundedAnswerStatus.ANSWERED
    assert "availability" not in validated.answer_markdown
    assert "raw distance" in validated.answer_markdown
    assert "stylistic resemblance" in validated.answer_markdown
    assert "transfer success" in validated.answer_markdown
    assert "future performance" in validated.answer_markdown
    assert "lineup selection" in validated.answer_markdown
    assert "tactical guarantee" in validated.answer_markdown


def test_role_fit_fallback_reconstructs_limitation_from_governed_methodology() -> None:
    analytics = _role_fit_record()
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(
        update={
            "methodology_topic": "role_fit",
            "result": {
                "content": (
                    "Role Fit measures raw distance from an eligible outfield player's "
                    "observed style to a supported positional-role style."
                ),
                "limitations": [
                    "Role Fit supports eligible outfield positions only.",
                    "Role Fit does not predict transfer success or future performance.",
                    "Role Fit is not lineup selection or a tactical guarantee.",
                ],
            },
        }
    )
    context = SelectedContext(
        question="How well does player 3500 fit team 904?",
        intent="role_fit",
        analytics_evidence=(analytics,),
        methodology_evidence=(methodology,),
        status=ContextStatus.SUFFICIENT,
    )
    generated = LLMGroundedAnswer(
        answer_markdown=(
            "Role Fit reports a raw distance [run:evidence-1].\n\n"
            "Role Fit does not predict transfer success, playing-time, or tactical "
            "outcomes [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    repaired = apply_role_fit_semantic_fallback(
        SynthesisResult(answer=generated, provider="fake", model="fake"),
        context,
    )

    repaired_text = repaired.answer.answer_markdown
    assert "playing-time" not in repaired_text
    assert "tactical outcomes" not in repaired_text
    assert "raw distance" in repaired_text
    assert "stylistic resemblance" in repaired_text
    assert "does not predict transfer success or future performance" in repaired_text
    assert "not lineup selection or a tactical guarantee" in repaired_text
    assert validate_answer_citations(
        repaired.answer,
        EvidenceLedger(records=(analytics, methodology)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_role_fit_difference_is_descriptive_not_a_deficiency() -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    unsupported = LLMGroundedAnswer(
        answer_markdown=(
            "His pressure-pass rate would require improvement [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )
    with pytest.raises(ValueError, match="style difference as a quality deficiency"):
        validate_answer_citations(unsupported, ledger)

    descriptive = LLMGroundedAnswer(
        answer_markdown=(
            "His largest stylistic difference is pressure-pass rate "
            "[run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )
    assert validate_answer_citations(descriptive, ledger).status is (
        GroundedAnswerStatus.ANSWERED
    )


def test_raw_role_fit_distance_interpretation_remains_valid() -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "His Role Fit distance is 0.3888; lower values indicate closer stylistic "
            "resemblance [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_needs_to_improve_is_rejected_for_role_fit_style_evidence() -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Seiwald needs to improve his pressure passing [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(ValueError, match="style difference as a quality deficiency"):
        validate_answer_citations(answer, ledger)


@pytest.mark.parametrize(
    "claim",
    (
        "These are stylistic differences, not deficiencies.",
        "This does not indicate a weakness.",
        "This is not evidence of a weakness.",
        "The gap does not mean he needs improvement.",
        (
            "The feature gap should not be interpreted as something that needs "
            "improvement."
        ),
        "Role Fit does not measure better or worse performance.",
        (
            "These quantify the stylistic distance per feature; they describe style "
            "differences, not deficiencies to be corrected."
        ),
    ),
)
def test_negated_role_fit_quality_interpretations_are_allowed(claim: str) -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    answer = LLMGroundedAnswer(
        answer_markdown=f"{claim} [run:evidence-1]",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "This is a deficiency.",
        "Pressure passing is a weakness.",
        "He needs to improve his progressive passing.",
        "His passing is worse.",
    ),
)
def test_affirmative_role_fit_quality_judgments_remain_rejected(claim: str) -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    answer = LLMGroundedAnswer(
        answer_markdown=f"{claim} [run:evidence-1]",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(ValueError, match="style difference as a quality deficiency"):
        validate_answer_citations(answer, ledger)


def test_current_club_claim_requires_web_citation_on_the_claim() -> None:
    ledger = EvidenceLedger(
        records=(
            _record(1, category=EvidenceCategory.ANALYTICS),
            _web_record(),
        )
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Seiwald is an RB Leipzig player [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )
    with pytest.raises(ValueError, match="Current-world claims require a web"):
        validate_answer_citations(answer, ledger)

    web_grounded = answer.model_copy(
        update={
            "answer_markdown": (
                "## Current situation\n"
                "Current reporting identifies Seiwald as an RB Leipzig player "
                "[run:evidence-2]."
            )
        }
    )
    assert validate_answer_citations(web_grounded, ledger).web_sources[0].domain == (
        "example.com"
    )


@pytest.mark.parametrize(
    "heading",
    (
        "Current situation",
        "**Current situation (external reporting)**",
        "Current situation (web and analytics evidence)",
        "Current situation (web evidence)",
        "Current status (public reporting)",
        "Latest update (external reporting)",
        "## Latest update",
    ),
)
def test_current_headings_need_no_citation_when_factual_sentence_is_web_cited(
    heading: str,
) -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(), _web_record()))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            f"{heading}\n"
            "Current reporting indicates Seiwald remains an RB Leipzig player "
            "[run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "heading",
    (
        "# Conclusion",
        "## Conclusion",
        "## Concise conclusion",
        "## Current situation",
        "## Role Fit",
        "Concise conclusion",
        "Conclusion",
        "Role Fit",
        "Current situation",
    ),
)
def test_structural_and_safe_legacy_headings_are_not_factual_claims(
    heading: str,
) -> None:
    answer = LLMGroundedAnswer(
        answer_markdown=heading,
        evidence_ids=(),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, EvidenceLedger()).status is (
        GroundedAnswerStatus.ANSWERED
    )


def test_synthesis_v5_requires_structural_markdown_section_headings() -> None:
    assert "format every section title as an ATX\nMarkdown heading" in SYNTHESIS_PROMPT_V5
    assert "Never emit a plain, unmarked section title" in SYNTHESIS_PROMPT_V5


def test_synthesis_v6_uses_product_language_concision_and_canonical_limitations() -> None:
    assert "metric_presentation labels instead of raw snake_case" in SYNTHESIS_PROMPT_V6
    assert "lead Role\nFit with cohort rank" in SYNTHESIS_PROMPT_V6
    assert "three to six substantive bullets or short sections" in SYNTHESIS_PROMPT_V6
    assert "do not generate a dedicated\n`Limitations`" in SYNTHESIS_PROMPT_V6
    assert "guessing between real-world measurement units" in SYNTHESIS_PROMPT_V6


def test_synthesis_v7_separates_archetype_centroids_from_team_role_fit() -> None:
    assert "K-means playing-style centroid" in SYNTHESIS_PROMPT_V7
    assert "supported team's observed positional-role style" in SYNTHESIS_PROMPT_V7
    assert "Never call an archetype centroid distance Role Fit" in SYNTHESIS_PROMPT_V7


def test_synthesis_v8_prevents_directional_and_implementation_overclaims() -> None:
    assert "Neutral\nor below-peer values" in SYNTHESIS_PROMPT_V8
    assert "never turn them into positive\nqualitative strengths" in SYNTHESIS_PROMPT_V8
    assert "Do not infer implementation or data-access limitations" in (
        SYNTHESIS_PROMPT_V8
    )
    assert "unless supplied methodology evidence states them explicitly" in (
        SYNTHESIS_PROMPT_V8
    )
    assert "long_pass_rate" not in SYNTHESIS_PROMPT_V8
    assert "raw event rows" not in SYNTHESIS_PROMPT_V8


def test_synthesis_v8_scales_broad_profiles_without_limiting_requested_detail() -> None:
    assert "broad, simple player-profile request" in SYNTHESIS_PROMPT_V8
    assert "four to six of the most\ninformative metrics" in SYNTHESIS_PROMPT_V8
    assert "State sample counts once" in SYNTHESIS_PROMPT_V8
    assert "Explicit requests for a deep dive, comparison" in SYNTHESIS_PROMPT_V8
    assert "detailed scouting report may receive the requested detail" in (
        SYNTHESIS_PROMPT_V8
    )


def test_synthesis_v5_separates_historical_analytics_from_current_web_context() -> None:
    assert (
        "Treat analytics and database evidence as historical FootyScout observation evidence"
        in SYNTHESIS_PROMPT_V5
    )
    assert "Team, club, and position fields in analytics establish\nonly what the historical" in (
        SYNTHESIS_PROMPT_V5
    )
    assert "`recent analytics`, `current analytics`, or `latest FootyScout data`" in (
        SYNTHESIS_PROMPT_V5
    )
    assert "Current\nclub, injury, availability, transfer, and recent-status claims" in (
        SYNTHESIS_PROMPT_V5
    )


@pytest.mark.parametrize(
    "claim",
    (
        "Current club: RB Leipzig",
        "Current status: injured",
        "Conclusion: Seiwald currently plays for Leipzig",
    ),
)
def test_factual_label_lines_are_not_treated_as_headings(claim: str) -> None:
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=(),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, EvidenceLedger())

    assert raised.value.error_code == "current_world_claim_requires_web"
    assert raised.value.failed_claim == claim


def test_full_structured_mixed_answer_passes_all_grounding_checks() -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(), _web_record()))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "## Role Fit\n\n"
            "FootyScout's historical sample records a Role Fit distance of 0.3888; "
            "lower values mean closer stylistic resemblance [run:evidence-1].\n\n"
            "## Current situation\n\n"
            "Current reporting identifies Seiwald as a Leipzig player "
            "[run:evidence-2].\n\n"
            "## Conclusion\n\n"
            "The historical Role Fit distance is 0.3888 [run:evidence-1]. Current "
            "reporting identifies him as a Leipzig player [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        (
            "In FootyScout's historical sample, Seiwald was recorded with RB Leipzig "
            "[run:evidence-1]."
        ),
        "Within the observed dataset, Seiwald had 64 pass attempts [run:evidence-1].",
        (
            "FootyScout's historical sample records a completion rate of 0.8125 "
            "[run:evidence-1]."
        ),
    ),
)
def test_historical_analytics_wording_remains_valid(claim: str) -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_current_affiliation_wording_passes_with_web_evidence() -> None:
    ledger = EvidenceLedger(records=(_web_record(1),))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Recent public reporting describes Seiwald as an RB Leipzig player "
            "[run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "Seiwald is an RB Leipzig player [run:evidence-1].",
        "Seiwald currently plays for RB Leipzig [run:evidence-1].",
        "Recent analytics show Seiwald plays for RB Leipzig [run:evidence-1].",
    ),
)
def test_current_affiliation_wording_still_rejects_analytics_only_evidence(
    claim: str,
) -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)

    assert raised.value.error_code == "current_world_claim_requires_web"


def test_v9_requires_general_historical_scope_for_analytics_identity_metadata() -> None:
    assert "player identity, team, club, position, or affiliation" in SYNTHESIS_PROMPT_V9
    assert "historical observed-sample\ncontext" in SYNTHESIS_PROMPT_V9
    assert "Do not convert those fields into present-tense or current-world claims" in (
        SYNTHESIS_PROMPT_V9
    )
    assert "without repetitively restating it" in SYNTHESIS_PROMPT_V9
    assert "still require current web evidence" in SYNTHESIS_PROMPT_V9


def test_v10_keeps_broad_profiles_concise_without_implementation_claims() -> None:
    assert "one short historical\nobserved-sample identity sentence" in SYNTHESIS_PROMPT_V10
    assert "approximately four to six high-value metrics" in SYNTHESIS_PROMPT_V10
    assert "one concise\nstyle or archetype interpretation" in SYNTHESIS_PROMPT_V10
    assert "second-closest centroid distance, separation margin" in SYNTHESIS_PROMPT_V10
    assert "unless the user asks for archetype detail" in SYNTHESIS_PROMPT_V10
    assert "Do not add a separate\nconclusion or bottom-line section" in SYNTHESIS_PROMPT_V10
    assert "does not establish implementation or API behavior" in SYNTHESIS_PROMPT_V10
    assert "does\nnot return raw event rows unless supplied evidence explicitly states" in (
        SYNTHESIS_PROMPT_V10
    )
    assert "historical observed-sample\ncontext" in SYNTHESIS_PROMPT_V10


def test_v11_hard_bans_unsupported_implementation_claims_generally() -> None:
    assert (
        "Never state implementation, retrieval, entity-matching, API, storage, database, "
        "or tool-\nbehavior claims"
    ) in SYNTHESIS_PROMPT_V11
    assert "unless supplied evidence explicitly states the specific claim" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "whether it is deterministic" in SYNTHESIS_PROMPT_V11
    assert "whether raw rows are returned" in SYNTHESIS_PROMPT_V11
    assert "how data\nis stored or retrieved" in SYNTHESIS_PROMPT_V11
    assert "Omit implementation\ncommentary from normal player profiles" in (
        SYNTHESIS_PROMPT_V11
    )


def test_v11_requires_explicit_support_for_directional_style_interpretation() -> None:
    assert "Do not call metrics or style dimensions `distinguishing`" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "unless supplied evidence explicitly\nidentifies the feature as distinguishing" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "supplies the stated direction or quantitative\nbasis" in SYNTHESIS_PROMPT_V11
    assert "presence of a metric name or style dimension alone is not support" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "prefer direct raw metrics and governed peer percentiles" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "archetype name\nalone is normally sufficient style context" in (
        SYNTHESIS_PROMPT_V11
    )


def test_v11_bounds_simple_broad_profile_structure_without_losing_temporal_scope() -> None:
    assert "target roughly 120 to 180 words and at most three\nlogical blocks" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "about four\nrepresentative, evidence-backed metrics" in SYNTHESIS_PROMPT_V11
    assert "Omit centroid diagnostics unless archetype detail was requested" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "Do not add\na `Bottom line` or `Conclusion` section" in SYNTHESIS_PROMPT_V11
    assert "do not repeat precise metrics as qualitative summary" in (
        SYNTHESIS_PROMPT_V11
    )
    assert "historical observed-sample\ncontext" in SYNTHESIS_PROMPT_V11
    assert "still require current web evidence" in SYNTHESIS_PROMPT_V11


def test_v12_uses_least_context_instead_of_implementation_blacklist() -> None:
    assert (
        "Do not describe internal system implementation unless the user asks about it and "
        "supplied\nevidence supports the statement"
    ) in SYNTHESIS_PROMPT_V12
    assert "deterministic name matching" not in SYNTHESIS_PROMPT_V12
    assert "raw event rows" not in SYNTHESIS_PROMPT_V12
    assert "API behavior" not in SYNTHESIS_PROMPT_V12


def test_v12_broad_profile_contract_is_compact_and_evidence_led() -> None:
    assert "target roughly 100 to 150 words and at most two\nheadings" in (
        SYNTHESIS_PROMPT_V12
    )
    assert "three or four representative,\nevidence-backed metrics" in (
        SYNTHESIS_PROMPT_V12
    )
    assert "State observed-sample scope once" in SYNTHESIS_PROMPT_V12
    assert "Omit a separate scope section, centroid\ndiagnostics" in SYNTHESIS_PROMPT_V12
    assert "explicit\ngoverned directional fields" in SYNTHESIS_PROMPT_V12
    assert "Current club, position, availability, injury" in SYNTHESIS_PROMPT_V12


def test_v13_broad_profiles_keep_archetype_name_and_omit_centroid_distance() -> None:
    assert "report the supported archetype name only" in SYNTHESIS_PROMPT_V13
    assert "omit centroid-distance\nvalues" in SYNTHESIS_PROMPT_V13
    assert "only when the user explicitly requests archetype details" in (
        SYNTHESIS_PROMPT_V13
    )


def test_v13_defines_centroid_distance_without_boundary_or_quality_semantics() -> None:
    assert (
        "centroid distance measures proximity to the archetype centroid: smaller values mean "
        "closer to\nthat centroid"
    ) in SYNTHESIS_PROMPT_V13
    assert "centroid boundary" not in SYNTHESIS_PROMPT_V13.casefold()
    assert "not a measure of quality, fit, confidence, probability, or transfer" in (
        SYNTHESIS_PROMPT_V13
    )


def test_v14_requires_web_citations_for_each_current_world_claim() -> None:
    assert "every\nsubstantive claim about current injury or availability" in (
        SYNTHESIS_PROMPT_V14
    )
    assert "current club, a recent transfer or\nreport, current manager" in (
        SYNTHESIS_PROMPT_V14
    )
    assert "does not support a requested\ncurrent fact" in SYNTHESIS_PROMPT_V14
    assert "available evidence is insufficient" in SYNTHESIS_PROMPT_V14


def test_v14_separates_historical_recommendations_from_current_world_facts() -> None:
    assert "historical FootyScout analytics and current public reporting in distinct" in (
        SYNTHESIS_PROMPT_V14
    )
    assert "historical shortlist into a present-tense affiliation" in (
        SYNTHESIS_PROMPT_V14
    )
    assert "club should sign a player now" in SYNTHESIS_PROMPT_V14
    assert "current-world facts with web evidence" in SYNTHESIS_PROMPT_V14


def test_v14_forbids_inferred_units_and_bounds_simple_methodology_answers() -> None:
    assert "Never infer a metric's unit from football-domain familiarity" in (
        SYNTHESIS_PROMPT_V14
    )


def test_v15_preserves_numeric_units_and_requires_current_web_citations() -> None:
    assert "raw unitless distance contribution" in SYNTHESIS_PROMPT_V15
    assert "Never multiply it by\n100 or add `%`" in SYNTHESIS_PROMPT_V15
    assert "attach a relevant web citation to each current claim" in (
        SYNTHESIS_PROMPT_V15
    )
    assert "sufficiently fresh evidence is unavailable" in SYNTHESIS_PROMPT_V15


def test_v15_uses_intent_sensitive_concise_answer_contracts() -> None:
    assert "For a player profile" in SYNTHESIS_PROMPT_V15
    assert "For\nsimilarity, give the top results" in SYNTHESIS_PROMPT_V15
    assert "For Role Fit, give distance or rank" in SYNTHESIS_PROMPT_V15
    assert "For recommendations, give\nranked candidates" in SYNTHESIS_PROMPT_V15
    assert "Methodology answers may be longer" in SYNTHESIS_PROMPT_V15
    assert "state a caveat once" in SYNTHESIS_PROMPT_V15
    assert "omit a\n`Bottom line` or `Conclusion`" in SYNTHESIS_PROMPT_V15


def test_v16_uses_structural_contract_and_forbids_incompatible_comparisons() -> None:
    assert "`answer_contract` is an application-controlled structural budget" in (
        SYNTHESIS_PROMPT_V16
    )
    assert "compatible\nunits, denominators, populations, and event families" in (
        SYNTHESIS_PROMPT_V16
    )
    assert "possession-oriented" in SYNTHESIS_PROMPT_V16
    assert "explicitly\nsupplies that governed label" in SYNTHESIS_PROMPT_V16
    assert "feature gap has no `StatsBomb units`" in SYNTHESIS_PROMPT_V14
    assert "For a simple methodology question" in SYNTHESIS_PROMPT_V14
    assert "Do not describe a production model as hurdle-based or\nhurdle-aware" in (
        SYNTHESIS_PROMPT_V14
    )


def test_v17_bounds_standard_methodology_answers_without_hiding_deep_detail() -> None:
    assert "state the target, name the production model" in SYNTHESIS_PROMPT_V17
    assert "include only one\nor two of the most decision-relevant supported" in (
        SYNTHESIS_PROMPT_V17
    )
    assert "Every retained fact must still appear explicitly" in SYNTHESIS_PROMPT_V17

    methodology = _record(1, category=EvidenceCategory.METHODOLOGY)
    standard = _synthesis_context_payload(
        SelectedContext(
            question="How is possession value modeled?",
            intent="methodology",
            methodology_evidence=(methodology,),
            status=ContextStatus.SUFFICIENT,
        )
    )["answer_contract"]
    deep = _synthesis_context_payload(
        SelectedContext(
            question="Give a detailed breakdown of the possession-value model internals.",
            intent="methodology",
            methodology_evidence=(methodology,),
            status=ContextStatus.SUFFICIENT,
        )
    )["answer_contract"]

    assert standard["detail_mode"] == "compact_methodology"
    assert standard["content_order"] == [
        "direct_definition",
        "requested_interpretation_or_implementation_fact",
        "material_limitation_if_needed",
    ]
    assert standard["target_sentence_range"] == {"minimum": 2, "maximum": 5}
    assert standard["omit_corpus_inventory"] is True
    assert standard["omit_exact_hyperparameter_lists"] is True
    assert deep["detail_mode"] == "requested_deep_detail"
    assert "max_performance_or_limitation_facts" not in deep


def test_v20_keeps_simple_methodology_concise_and_evidence_exact() -> None:
    assert "150–250 word target" in SYNTHESIS_PROMPT_V18
    assert "A row/state count is not `unique`" in SYNTHESIS_PROMPT_V18
    assert "Configured\nround values do not establish that they were tuned" in (
        SYNTHESIS_PROMPT_V18
    )
    assert "Never\noffer stored logs, metric tables" in SYNTHESIS_PROMPT_V18

    methodology = _record(1, category=EvidenceCategory.METHODOLOGY)
    standard = _synthesis_context_payload(
        SelectedContext(
            question="How is possession value modeled?",
            intent="methodology",
            methodology_evidence=(methodology,),
            status=ContextStatus.SUFFICIENT,
        )
    )["answer_contract"]

    assert "deterministically projected" in SYNTHESIS_PROMPT_V19
    assert "do not reconstruct\ndetails" in SYNTHESIS_PROMPT_V19
    assert "minimum sufficient grounded\nanswer" in SYNTHESIS_PROMPT_V20
    assert standard["detail_mode"] == "compact_methodology"
    assert standard["target_word_maximum"] == 110
    assert standard["target_sentence_range"] == {"minimum": 2, "maximum": 5}
    assert standard["max_sections"] == 0
    assert standard["headings"] == "none"
    assert standard["omit_corpus_inventory"] is True
    assert standard["omit_corpus_counts"] is True
    assert standard["omit_hurdle_configuration"] is True
    assert standard["omit_detailed_fold_mechanics"] is True
    assert standard["offer_deeper_detail"] is False
    assert standard["offer_internal_artifacts"] is False


def test_v19_deep_methodology_detail_remains_available_when_requested() -> None:
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY)
    deep = _synthesis_context_payload(
        SelectedContext(
            question="Give a detailed breakdown of the possession-value model internals.",
            intent="methodology",
            methodology_evidence=(methodology,),
            status=ContextStatus.SUFFICIENT,
        )
    )["answer_contract"]

    assert deep["detail_mode"] == "requested_deep_detail"
    assert "target_word_range" not in deep
    assert "omit_corpus_counts" not in deep
    assert "omit_hurdle_configuration" not in deep


def test_standard_possession_methodology_gets_a_bounded_synthesis_view() -> None:
    full_result = {
        "topic": "possession_value",
        "production_status": "production",
        "current_production_model": "xgboost",
        "experimental_models": ["gru", "pytorch_causal_transformer"],
        "summary": "Production state-value methodology.",
        "sources": [{"source_id": "possession_value:primary"}],
        "structured_metadata": {
            "task_name": "Attacking state value",
            "target": "future_oof_xg_same_possession",
            "target_interpretation": "sum of later OOF xG in the same possession",
            "state_convention": "pre-event state",
            "selected_horizon": "all remaining eligible same-possession shots",
            "feature_columns": [
                "ball_x",
                "ball_y",
                "distance_to_goal",
                "angle_to_goal",
                "period",
                "minute",
                "possession_action_number",
                "under_pressure",
                "current_play_pattern",
            ],
            "leakage_protection": "OOF xG labels and no future model inputs",
            "split_methodology": {
                "method": "reused frozen match-grouped 80/10/10 allocation",
                "train_states": 534490,
                "validation_states": 63763,
                "test_states": 68809,
            },
            "nested_cross_fitting": {
                "state_oof": "five outer folds with nested grouped OOF xG",
                "cache_directory": "data/processed/cache",
            },
            "selection": {
                "model": "xgboost_reg_pseudohubererror",
                "objective": "reg:pseudohubererror",
                "primary_metric": "validation_rmse",
                "reason": "lowest validation RMSE",
                "rounds": {"regressor": 300},
            },
            "limitations": [
                "observational/model-derived, not causal",
                "on-ball attacking possession value only",
                "defensive value is outside scope",
            ],
            "hurdle": {"rounds": {"classifier": 221, "regressor": 36}},
            "training_corpus": {
                "states": 667062,
                "competitions": ["Bundesliga", "World Cup"],
            },
            "candidates": [{"name": "candidate", "parameters": {"depth": 4}}],
            "validation_metrics": {"rmse": 0.05},
            "out_of_fold_metrics": {"rmse": 0.05163},
            "untouched_test_metrics": {"buckets": [1, 2, 3]},
        },
        "limitations": ["Production-status note"],
    }
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={"methodology_topic": "possession_value", "result": full_result}
    )

    payload = _synthesis_context_payload(
        SelectedContext(
            question="How does possession value work?",
            intent="methodology",
            methodology_evidence=(methodology,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    record = payload["methodology_evidence"][0]
    result = record["result"]
    metadata = result["structured_metadata"]

    assert record["evidence_id"] == "run:evidence-1"
    assert record["evidence_category"] == "methodology"
    assert result["current_production_model"] == "xgboost"
    assert metadata["target"] == "future_oof_xg_same_possession"
    assert metadata["state_convention"] == "pre-event state"
    assert metadata["selected_horizon"] == "all remaining eligible same-possession shots"
    assert metadata["representative_feature_columns"] == [
        "ball_x",
        "ball_y",
        "distance_to_goal",
        "angle_to_goal",
        "possession_action_number",
        "under_pressure",
    ]
    assert metadata["leakage_protection"] == "OOF xG labels and no future model inputs"
    assert metadata["split_methodology"] == {
        "method": "reused frozen match-grouped 80/10/10 allocation"
    }
    assert "selection" not in metadata
    assert result["current_production_model"] == "xgboost"
    assert metadata["limitations"] == [
        "observational/model-derived, not causal",
        "on-ball attacking possession value only",
    ]
    assert "experimental_models" not in result
    assert "feature_columns" not in metadata
    for excluded in (
        "hurdle",
        "training_corpus",
        "candidates",
        "validation_metrics",
        "out_of_fold_metrics",
        "untouched_test_metrics",
        "nested_cross_fitting",
    ):
        assert excluded not in metadata


@pytest.mark.parametrize(
    "question",
    (
        "What were the hurdle classifier and regressor rounds?",
        "What competitions were in the possession-value corpus?",
        "Show all OOF metrics.",
        "Explain the nested cross-fitting procedure.",
        "What exact feature columns are used?",
        "Show the full supporting sample statistics.",
    ),
)
def test_explicit_methodology_detail_requests_receive_full_evidence(question: str) -> None:
    full_result = {
        "topic": "possession_value",
        "structured_metadata": {
            "hurdle": {"rounds": {"classifier": 221, "regressor": 36}},
            "training_corpus": {"states": 667062, "competitions": ["Bundesliga"]},
            "out_of_fold_metrics": {"rmse": 0.05163},
            "nested_cross_fitting": {"state_oof": "five outer folds"},
            "feature_columns": ["ball_x", "ball_y", "distance_to_goal"],
        },
    }
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={"methodology_topic": "possession_value", "result": full_result}
    )

    payload = _synthesis_context_payload(
        SelectedContext(
            question=question,
            intent="methodology",
            methodology_evidence=(methodology,),
            status=ContextStatus.SUFFICIENT,
        )
    )

    assert payload["answer_contract"]["detail_mode"] == "requested_deep_detail"
    assert payload["methodology_evidence"][0]["result"] == full_result


def test_v20_simple_current_fact_gets_a_compact_answer_contract() -> None:
    payload = _synthesis_context_payload(
        SelectedContext(
            question="What club does Xhaka currently play for?",
            intent="player_profile",
            web_evidence=(_web_record(),),
            status=ContextStatus.SUFFICIENT,
        )
    )

    assert payload["answer_contract"]["detail_mode"] == "compact_fact"
    assert payload["answer_contract"]["target_sentence_range"] == {
        "minimum": 1,
        "maximum": 3,
    }
    assert payload["answer_contract"]["max_sections"] == 0
    assert payload["answer_contract"]["headings"] == "none"
    assert payload["answer_contract"]["omit_unrequested_background"] is True


def test_v21_team_intelligence_projects_metrics_once_without_role_dump() -> None:
    team = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_TEAM_INTELLIGENCE,
            "result": {
                "intelligence": {
                    "team": {
                        "team_id": 904,
                        "team_name": "Bayer Leverkusen",
                        "matches_observed": 34,
                        "sample_scope": "Observed team style across 34 matches.",
                        "actions": 43000,
                        "contributors": 24,
                        "metrics": {
                            "expected_completion_rate": 0.872,
                            "carry_share_of_actions": 0.461,
                            "positive_forward_distance_per_100_passes": 677.6,
                            "progressive_pass_rate": 0.127,
                            "shots_per_match": 18.3,
                        },
                    },
                    "roles": [{"all_role_details": "omitted"}],
                }
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="Describe Bayer Leverkusen's team intelligence.",
            intent="team_analysis",
            analytics_evidence=(team,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["analytics_evidence"][0]["result"]["intelligence"]

    assert result["team"]["representative_metrics"] == {
        "expected_completion_rate": 0.872,
        "carry_share_of_actions": 0.461,
        "positive_forward_distance_per_100_passes": 677.6,
        "progressive_pass_rate": 0.127,
    }
    assert "roles" not in result
    assert "actions" not in result["team"]
    assert "contributors" not in result["team"]
    assert payload["answer_contract"]["max_interpretations"] == 1
    assert payload["answer_contract"]["do_not_restate_metric_summary"] is True


def test_v21_archetype_methodology_is_one_direct_unbranded_concept() -> None:
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={
            "methodology_topic": "archetypes",
            "result": {
                "topic": "archetypes",
                "production_status": "production",
                "current_production_model": None,
                "summary": (
                    "Archetypes are governed K-means playing-style groups in a "
                    "position-normalized feature space."
                ),
                "limitations": [
                    "Archetypes group playing style; they are not player-quality ratings."
                ],
                "sources": [{"source_id": "archetypes:primary"}],
                "structured_metadata": {"centroid_details": "not requested"},
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="How are player archetypes produced?",
            intent="methodology",
            methodology_evidence=(methodology,),
            limitations=("Duplicate application limitation.",),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["methodology_evidence"][0]["result"]

    assert result == {
        "topic": "archetypes",
        "summary": (
            "Archetypes are governed K-means playing-style groups in a "
            "position-normalized feature space."
        ),
        "limitations": [
            "Archetypes group playing style; they are not player-quality ratings."
        ],
    }
    assert "FootyScout" not in json.dumps(result)
    assert "centroid" not in json.dumps(result).casefold()
    assert "pairwise" not in json.dumps(result).casefold()
    assert payload["limitations"] == []
    assert payload["answer_contract"]["max_sections"] == 0
    assert payload["answer_contract"]["do_not_repeat_definition_or_limitation"]


def test_v21_explicit_production_model_question_retains_model_information() -> None:
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={
            "result": {
                "topic": "xg",
                "production_status": "production",
                "current_production_model": "xgboost",
                "summary": "Non-penalty expected-goals model.",
                "limitations": ["Penalties are handled separately."],
                "sources": [{"source_id": "xg:primary"}],
            }
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="What production model is used for expected goals?",
            intent="methodology",
            methodology_evidence=(methodology,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["methodology_evidence"][0]["result"]

    assert result["production_status"] == "production"
    assert result["current_production_model"] == "xgboost"


def test_v21_compact_metric_keeps_material_caveat_but_omits_generic_one() -> None:
    dossier = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_PLAYER_DOSSIER,
            "result": {
                "player": {"player_name": "Example Player"},
                "attacking_impact": {
                    "attacking_value_per_100_actions": 0.03,
                    "pass_value_per_100_passes": -0.02,
                    "carry_value_per_100_carries": 0.08,
                    "attacking_value_reliable": False,
                },
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="What is this player's attacking impact?",
            intent="player_profile",
            analytics_evidence=(dossier,),
            limitations=(
                "Limited sample: interpret this estimate cautiously.",
                "A profile describes observed event data, not future performance.",
            ),
            status=ContextStatus.SUFFICIENT,
        )
    )

    assert payload["limitations"] == [
        "Limited sample: interpret this estimate cautiously."
    ]
    attacking = payload["analytics_evidence"][0]["result"]["attacking_impact"]
    assert attacking["material_reliability_warnings"] == {
        "attacking_value_reliable": False
    }


def test_v20_broad_comparison_projects_only_representative_metrics() -> None:
    comparison = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.COMPARE_PLAYERS,
            "result": {
                "comparison": {"same_position": True},
                "players": [
                    {
                        "player_id": 1,
                        "player_name": "First",
                        "team_name": "Team",
                        "position": "Midfield",
                        "matches_observed": 12,
                        "pass_attempts": 450,
                        "overall_reliable": True,
                        "completion_above_expected_pp": 1.2,
                        "expected_completion_rate": 0.84,
                        "positive_forward_distance_per_100_passes": 210.0,
                        "progressive_pass_rate": 0.18,
                        "long_pass_rate": 0.09,
                    }
                ],
                "attacking": {"every_metric": "omitted"},
                "intelligence": {"every_metric": "omitted"},
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="Compare these players.",
            intent="player_comparison",
            analytics_evidence=(comparison,),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["analytics_evidence"][0]["result"]

    assert list(result["players"][0]["representative_metrics"]) == [
        "completion_above_expected_pp",
        "expected_completion_rate",
        "positive_forward_distance_per_100_passes",
        "progressive_pass_rate",
    ]
    assert "attacking" not in result
    assert "intelligence" not in result
    assert payload["answer_contract"]["max_representative_metrics_per_player"] == 4


def test_v20_requested_profile_metric_survives_compact_projection() -> None:
    dossier = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_PLAYER_DOSSIER,
            "result": {
                "player": {"player_name": "Example Player"},
                "intelligence": {
                    "matches_observed": 12,
                    "performance_metrics": [],
                    "style_metrics": [
                        {"metric_name": "long_pass_rate", "raw_value": 0.12},
                        {
                            "metric_name": "completion_above_expected_pp",
                            "raw_value": 1.0,
                        },
                        {
                            "metric_name": "expected_completion_rate",
                            "raw_value": 0.82,
                        },
                        {
                            "metric_name": "progressive_pass_rate",
                            "raw_value": 0.18,
                        },
                    ],
                },
            },
        }
    )
    payload = _synthesis_context_payload(
        SelectedContext(
            question="What is this player's long-pass rate?",
            intent="player_profile",
            analytics_evidence=(dossier,),
            limitations=("Observed data does not predict future performance.",),
            status=ContextStatus.SUFFICIENT,
        )
    )
    result = payload["analytics_evidence"][0]["result"]
    names = [
        row["metric_name"] for row in result["profile"]["representative_metrics"]
    ]

    assert names[0] == "long_pass_rate"
    assert "long_pass_rate" in payload["metric_presentation"]
    assert payload["limitations"] == []


def test_v20_simple_recommendation_contract_omits_methodology_dump() -> None:
    payload = _synthesis_context_payload(
        SelectedContext(
            question="Which midfielders should I inspect for this role?",
            intent="role_recommendations",
            status=ContextStatus.SUFFICIENT,
        )
    )

    assert payload["answer_contract"]["detail_mode"] == "standard"
    assert payload["answer_contract"]["max_reasons_per_candidate"] == 1
    assert payload["answer_contract"]["max_sections"] == 2
    assert payload["answer_contract"]["offer_more_detail"] is False


def test_v20_prompt_encodes_minimal_clarification_and_semantic_preservation() -> None:
    assert "Ask one direct clarification question" in SYNTHESIS_PROMPT_V20
    assert "Keep\nexpected and observed completion distinct" in SYNTHESIS_PROMPT_V20
    assert "forward-pass distance and carry distance\ndistinct" in SYNTHESIS_PROMPT_V20
    assert "descriptive style relationship is not causal" in SYNTHESIS_PROMPT_V20
    assert "Preserve player and team names exactly as supplied" in SYNTHESIS_PROMPT_V20
    assert "render each unit once" in SYNTHESIS_PROMPT_V20
    assert "sample support is incorporated into a score" in SYNTHESIS_PROMPT_V20
    assert "omitted flags were normalized to false" in SYNTHESIS_PROMPT_V20
    assert "does not mean boolean features were removed" in SYNTHESIS_PROMPT_V20
    assert "Do not offer to provide more detail" in SYNTHESIS_PROMPT_V20


def test_v21_prompt_requires_minimum_sufficient_unbranded_nonrepetitive_answers() -> None:
    assert "Stop as soon as the answer has supplied the requested fact" in (
        SYNTHESIS_PROMPT_V21
    )
    assert "one-component answer uses one compact paragraph and no heading" in (
        SYNTHESIS_PROMPT_V21
    )
    assert "never\nfollow a metric summary with prose that merely restates" in (
        SYNTHESIS_PROMPT_V21
    )
    assert "do not inject product-brand or system-identity attribution" in (
        SYNTHESIS_PROMPT_V21
    )
    assert "without a second\nsection that repeats its method or caveat" in (
        SYNTHESIS_PROMPT_V21
    )


@pytest.mark.parametrize(
    ("unsupported_claim", "supported_fragment"),
    (
        ("This is his primary style driver", "primary driver"),
        ("The model uses pairwise distance", "pairwise distance"),
        ("His net forward carry is 12", "net forward carry"),
        ("The value is nearer the centroid boundary", "centroid boundary"),
    ),
)
def test_v20_high_confidence_semantic_strengthening_requires_explicit_evidence(
    unsupported_claim: str,
    supported_fragment: str,
) -> None:
    unsupported_record = _record(1, category=EvidenceCategory.METHODOLOGY)
    unsupported_ledger = EvidenceLedger(records=(unsupported_record,))
    answer = LLMGroundedAnswer(
        answer_markdown=f"{unsupported_claim} [run:evidence-1].",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, unsupported_ledger)
    assert raised.value.error_code == "unsupported_methodology_term"

    supported_record = unsupported_record.model_copy(
        update={"result": {"methodology": supported_fragment}}
    )
    supported_ledger = EvidenceLedger(records=(supported_record,))
    assert validate_answer_citations(answer, supported_ledger).status is (
        GroundedAnswerStatus.ANSWERED
    )


def _methodology_ledger(result: dict[str, object]) -> EvidenceLedger:
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={"methodology_topic": "possession_value", "result": result}
    )
    return EvidenceLedger(records=(methodology,))


def test_fixed_round_configuration_does_not_support_tuning_language() -> None:
    ledger = _methodology_ledger({"hurdle": {"rounds": {"classifier": 221}}})
    supported = LLMGroundedAnswer(
        answer_markdown="The classifier used 221 rounds [run:evidence-1].",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )
    unsupported = supported.model_copy(
        update={"answer_markdown": "The classifier rounds were tuned [run:evidence-1]."}
    )

    assert validate_answer_citations(supported, ledger).status is GroundedAnswerStatus.ANSWERED
    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(unsupported, ledger)
    assert raised.value.error_code == "unsupported_methodology_strengthening"


def test_explicit_validation_search_supports_tuning_language() -> None:
    ledger = _methodology_ledger(
        {
            "hurdle": {"rounds": {"classifier": 221}},
            "tuning_process": "Rounds were selected by validation search.",
        }
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "The classifier rounds were tuned using validation search "
            "[run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_fixed_parameter_does_not_support_optimized_language() -> None:
    ledger = _methodology_ledger({"parameters": {"objective": "pseudohuber"}})
    answer = LLMGroundedAnswer(
        answer_markdown="The objective was optimized [run:evidence-1].",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)
    assert raised.value.error_code == "unsupported_methodology_strengthening"


def test_historical_recommendation_wording_does_not_require_web_authority() -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Within FootyScout's historical analytics sample, the role-recommendation "
            "model ranks this player first [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_current_transfer_recommendation_still_requires_web_authority() -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Bayer Leverkusen should sign this player now [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)

    assert raised.value.error_code == "current_world_claim_requires_web"


@pytest.mark.parametrize(
    ("claim", "category"),
    (
        pytest.param(
            "The current player-search result contains Xhaka in FootyScout's "
            "observed dataset [run:evidence-1].",
            EvidenceCategory.ANALYTICS,
            id="ER-003-historical-player-search",
        ),
        pytest.param(
            "The current filter returns midfielders recorded for Bayern Munich "
            "in the observed dataset [run:evidence-1].",
            EvidenceCategory.ANALYTICS,
            id="ER-007-historical-filtered-search",
        ),
        pytest.param(
            "The current leaderboard ranks observed pass attempts from the "
            "historical dataset [run:evidence-1].",
            EvidenceCategory.ANALYTICS,
            id="ER-009-historical-leaderboard",
        ),
        pytest.param(
            "The current profile records 3,299 pass attempts in FootyScout's "
            "observed sample [run:evidence-1].",
            EvidenceCategory.ANALYTICS,
            id="PC-015-profile-analytics",
        ),
        pytest.param(
            "The current similarity method compares profile dimensions from the "
            "frozen dataset [run:evidence-1].",
            EvidenceCategory.METHODOLOGY,
            id="SA-010-similarity-methodology",
        ),
        pytest.param(
            "The current recommendation output ranks eligible midfielders from "
            "the frozen analytics snapshot [run:evidence-1].",
            EvidenceCategory.ANALYTICS,
            id="TR-008-recommendation-analytics",
        ),
        pytest.param(
            "The current recommendation result contains ten historical forward "
            "profiles [run:evidence-1].",
            EvidenceCategory.ANALYTICS,
            id="TR-010-recommendation-result",
        ),
        pytest.param(
            "The current-team Role Fit methodology uses leave-self-out role "
            "construction [run:evidence-1].",
            EvidenceCategory.METHODOLOGY,
            id="TR-017-role-fit-methodology",
        ),
        pytest.param(
            "The current percentile methodology ranks players against "
            "same-position peers [run:evidence-1].",
            EvidenceCategory.METHODOLOGY,
            id="ME-008-percentile-methodology",
        ),
        pytest.param(
            "The current player-profile methodology uses out-of-fold predictions "
            "rather than only the test set [run:evidence-1].",
            EvidenceCategory.METHODOLOGY,
            id="ME-013-profile-methodology",
        ),
    ),
)
def test_canonical_current_world_false_positive_patterns_do_not_require_web(
    claim: str,
    category: EvidenceCategory,
) -> None:
    record = _record(1, category=category)

    assert detect_current_claim_authority(claim, (record,)) == ()


@pytest.mark.parametrize(
    "claim",
    (
        pytest.param(
            "Granit Xhaka currently plays for Sunderland [run:evidence-1].",
            id="UA-003-current-club",
        ),
        pytest.param(
            "The current manager is Kasper Hjulmand [run:evidence-1].",
            id="current-manager",
        ),
        pytest.param(
            "Granit Xhaka is currently injured [run:evidence-1].",
            id="current-injury",
        ),
        pytest.param(
            "Granit Xhaka is available for today's match [run:evidence-1].",
            id="SA-011-current-availability",
        ),
        pytest.param(
            "Current transfer reporting links him with a move "
            "[run:evidence-1].",
            id="current-transfer",
        ),
        pytest.param(
            "Recent news identifies a change in his playing status "
            "[run:evidence-1].",
            id="recent-news",
        ),
        pytest.param(
            "## Current situation\n\nClub: Sunderland [run:evidence-1].",
            id="current-section-club-label",
        ),
    ),
)
def test_external_mutable_world_state_still_requires_web(claim: str) -> None:
    analytics = _record(1, category=EvidenceCategory.ANALYTICS)

    findings = detect_current_claim_authority(claim, (analytics,))

    assert len(findings) == 1
    assert findings[0].error_code == "current_world_claim_requires_web"


@pytest.mark.parametrize(
    "claim",
    (
        (
            "The profile records a Role Fit distance of 0.39 and Xhaka currently "
            "plays for Sunderland [run:evidence-1]."
        ),
        (
            "Role Fit measures style resemblance, and Xhaka is currently injured "
            "[run:evidence-1]."
        ),
        (
            "The current production model uses XGBoost, and Leverkusen's current "
            "manager is Kasper Hjulmand [run:evidence-1]."
        ),
    ),
)
def test_mixed_internal_and_external_claims_still_require_web(claim: str) -> None:
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY)

    findings = detect_current_claim_authority(claim, (methodology,))

    assert len(findings) == 1
    assert findings[0].error_code == "current_world_claim_requires_web"


@pytest.mark.parametrize(
    "claim",
    (
        "XGBoost is the current production model [run:evidence-1].",
        "The current production model uses pseudohuber loss [run:evidence-1].",
    ),
)
def test_governed_current_product_state_does_not_require_web(claim: str) -> None:
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY)

    assert detect_current_claim_authority(claim, (methodology,)) == ()


def test_product_state_heading_does_not_create_external_current_section() -> None:
    methodology = _record(1, category=EvidenceCategory.METHODOLOGY)
    answer = (
        "## Current production methodology\n\n"
        "Availability: model metadata is available [run:evidence-1]."
    )

    assert detect_current_claim_authority(answer, (methodology,)) == ()


def test_governed_production_model_statement_uses_methodology_authority() -> None:
    methodology = _record(
        1,
        category=EvidenceCategory.METHODOLOGY,
        sources=("possession_value:primary",),
    ).model_copy(update={"methodology_topic": "possession_value"})
    ledger = EvidenceLedger(records=(methodology,))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "The current production possession-value model uses XGBoost "
            "[run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "The selected model uses XGBoost [run:evidence-1].",
        "The current event belongs to the future state [run:evidence-1].",
        "Nested cross-fitting groups training by match [run:evidence-1].",
        "The model architecture uses pre-event game state [run:evidence-1].",
    ),
)
def test_governed_internal_methodology_vocabulary_does_not_require_web(
    claim: str,
) -> None:
    methodology = _record(
        1,
        category=EvidenceCategory.METHODOLOGY,
        sources=("possession_value:primary",),
    ).model_copy(update={"methodology_topic": "possession_value"})
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(
        answer,
        EvidenceLedger(records=(methodology,)),
    ).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "The model family is XGBoost [run:evidence-1].",
        "The production regressor uses the pseudohuber objective [run:evidence-1].",
        "The model was trained for 300 boosting rounds [run:evidence-1].",
        "The selected model was chosen by validation RMSE [run:evidence-1].",
        "Out-of-fold RMSE was 0.05163 [run:evidence-1].",
        "The documented limitation is that the estimate is not causal [run:evidence-1].",
        "Methodology version and configuration are governed [run:evidence-1].",
    ),
)
def test_current_methodology_section_uses_methodology_authority(claim: str) -> None:
    methodology = _record(
        1,
        category=EvidenceCategory.METHODOLOGY,
        sources=("possession_value:primary",),
    ).model_copy(update={"methodology_topic": "possession_value"})
    answer = LLMGroundedAnswer(
        answer_markdown=f"## Current production methodology\n\n{claim}",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(
        answer,
        EvidenceLedger(records=(methodology,)),
    ).status is GroundedAnswerStatus.ANSWERED


def test_me005_like_methodology_answer_passes_strict_validation_without_web() -> None:
    methodology = _record(
        1,
        category=EvidenceCategory.METHODOLOGY,
        sources=("possession_value:primary",),
    ).model_copy(update={"methodology_topic": "possession_value"})
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "## Target\n\n"
            "The model estimates future out-of-fold xG in the same possession "
            "[run:evidence-1].\n\n"
            "## Current production methodology\n\n"
            "The production regressor uses XGBoost with the pseudohuber objective "
            "and 300 boosting rounds [run:evidence-1]. Model selection uses validation "
            "RMSE [run:evidence-1]. Out-of-fold RMSE was 0.05163 "
            "[run:evidence-1].\n\n"
            "## Limitations\n\n"
            "The documented limitation is that the estimate is observational rather "
            "than causal [run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(
        answer,
        EvidenceLedger(records=(methodology,)),
    ).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "Xhaka currently plays for Bayer Leverkusen [run:evidence-1].",
        "Xhaka is currently injured [run:evidence-1].",
        "Leverkusen's current manager is Kasper Hjulmand [run:evidence-1].",
        "Player X recently transferred to Bayer Leverkusen [run:evidence-1].",
        "Player X is available for today's match [run:evidence-1].",
        (
            "FootyScout's current model uses XGBoost, and Xhaka currently plays "
            "for Bayer Leverkusen [run:evidence-1]."
        ),
    ),
)
def test_methodology_authority_does_not_exempt_external_current_state(
    claim: str,
) -> None:
    methodology = _record(
        1,
        category=EvidenceCategory.METHODOLOGY,
        sources=("possession_value:primary",),
    ).model_copy(update={"methodology_topic": "possession_value"})
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, EvidenceLedger(records=(methodology,)))

    assert raised.value.error_code == "current_world_claim_requires_web"


@pytest.mark.parametrize(
    "claim",
    (
        "The current manager is Kasper Hjulmand [run:evidence-1].",
        "The current injury status is unavailable [run:evidence-1].",
        "The current club is Bayer Leverkusen [run:evidence-1].",
        "Recent transfer reporting links him elsewhere [run:evidence-1].",
    ),
)
def test_external_current_facts_cannot_use_methodology_authority(claim: str) -> None:
    methodology = _record(
        1,
        category=EvidenceCategory.METHODOLOGY,
        sources=("possession_value:primary",),
    )
    ledger = EvidenceLedger(records=(methodology,))
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)

    assert raised.value.error_code == "current_world_claim_requires_web"


@pytest.mark.parametrize(
    "claim",
    (
        (
            "In FootyScout's observed sample, Granit Xhaka was listed for Bayer "
            "Leverkusen as a midfielder [run:evidence-1]."
        ),
        (
            "Within FootyScout's historical dataset, Alex Morgan was listed for "
            "Orlando Pride as a forward [run:evidence-1]."
        ),
        (
            "In the observed sample, he was listed as a midfielder "
            "[run:evidence-1]."
        ),
    ),
)
def test_general_historical_identity_team_and_position_wording_is_valid(
    claim: str,
) -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "Xhaka currently plays for Bayer Leverkusen [run:evidence-1].",
        "Xhaka plays for Bayer Leverkusen [run:evidence-1].",
        "Morgan plays for Orlando Pride [run:evidence-1].",
    ),
)
def test_general_present_tense_affiliation_still_requires_web(claim: str) -> None:
    ledger = EvidenceLedger(records=(_record(1, category=EvidenceCategory.ANALYTICS),))
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)

    assert raised.value.error_code == "current_world_claim_requires_web"


def test_realistic_role_fit_current_and_historical_answer_passes() -> None:
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(
        update={
            "methodology_topic": "role_fit",
            "result": {
                "summary": "Lower Role Fit distance means closer stylistic resemblance."
            },
        }
    )
    ledger = EvidenceLedger(records=(_role_fit_record(), methodology, _web_record(3)))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "## Role Fit\n\n"
            "FootyScout's historical sample records a Role Fit distance of 0.3888 "
            "[run:evidence-1]. Lower distance means closer stylistic resemblance "
            "[run:evidence-2].\n\n"
            "## Current situation\n\n"
            "Current public reporting describes Seiwald as an RB Leipzig player "
            "[run:evidence-3]. Current reporting describes a rotational role "
            "[run:evidence-3].\n\n"
            "## Historical FootyScout context\n\n"
            "Within the observed dataset, Seiwald had 64 pass attempts "
            "[run:evidence-1].\n\n"
            "## Conclusion\n\n"
            "The historical sample records a Role Fit distance of 0.3888 "
            "[run:evidence-1; run:evidence-2], while current reporting describes "
            "him as an RB Leipzig player [run:evidence-3]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2", "run:evidence-3"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    final = validate_answer_citations(answer, ledger)

    assert final.status is GroundedAnswerStatus.ANSWERED
    assert final.methodology_sources == ("role_fit:primary",)
    assert final.web_sources[0].evidence_id == "run:evidence-3"


@pytest.mark.parametrize(
    "claim",
    (
        "Current reporting says he remains at Leipzig [run:evidence-2].",
        "Current reporting says he remains at Leipzig. [run:evidence-2]",
        "- Current club: RB Leipzig [run:evidence-2]",
        "Current reporting says he remains at Leipzig.\n[run:evidence-2]",
    ),
)
def test_normal_markdown_web_citation_placement_is_accepted(claim: str) -> None:
    ledger = EvidenceLedger(records=(_web_record(1),))
    answer = LLMGroundedAnswer(
        answer_markdown=claim.replace("evidence-2", "evidence-1"),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_current_claim_may_cite_both_analytics_and_web() -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(), _web_record()))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Current reporting identifies him as a Leipzig player "
            "[run:evidence-1] [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_current_claim_multi_evidence_group_passes_with_web_member() -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(), _web_record()))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Current reporting identifies him as a Leipzig player "
            "[run:evidence-1; run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_current_claim_multi_evidence_group_still_requires_web_member() -> None:
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    )
    ledger = EvidenceLedger(records=(_role_fit_record(), methodology))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Current reporting identifies him as a Leipzig player "
            "[run:evidence-1; run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)

    assert raised.value.error_code == "current_world_claim_requires_web"
    assert raised.value.detected_evidence_ids == (
        "run:evidence-1",
        "run:evidence-2",
    )


@pytest.mark.parametrize(
    "claim",
    (
        "He currently plays for RB Leipzig [run:evidence-1].",
        "He is currently injured [run:evidence-1].",
        "He is currently unavailable [run:evidence-1].",
        "The current manager is Kasper Hjulmand [run:evidence-1].",
        "Current reporting says he is leaving Leipzig.",
        "Current situation is that Seiwald plays for Leipzig.",
        "Current status: Seiwald is injured.",
        "Latest update: Seiwald has joined Bayern.",
    ),
)
def test_current_claims_without_web_authority_are_rejected(claim: str) -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(ValueError, match="Current-world claims require a web"):
        validate_answer_citations(answer, ledger)


@pytest.mark.parametrize(
    "claim",
    (
        "Current reporting identifies his club [run:evidence-1].",
        "Current reporting says he is available [run:evidence-1].",
        "Recent transfer reporting links him with a move [run:evidence-1].",
        "Current reporting identifies the manager [run:evidence-1].",
    ),
)
def test_current_claims_pass_with_claim_local_web_authority(claim: str) -> None:
    ledger = EvidenceLedger(records=(_web_record(1),))
    answer = LLMGroundedAnswer(
        answer_markdown=claim,
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_historical_club_wording_may_cite_analytics_in_a_current_section() -> None:
    ledger = EvidenceLedger(
        records=(
            _record(1, category=EvidenceCategory.ANALYTICS),
            _web_record(),
        )
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "## Current situation\n"
            "FootyScout's historical dataset records Seiwald with RB Leipzig "
            "[run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_historical_sample_wording_may_cite_analytics() -> None:
    ledger = EvidenceLedger(records=(_role_fit_record(),))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "Current situation\n"
            "The historical FootyScout sample observed him across two matches "
            "[run:evidence-1]."
        ),
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "In FootyScout's historical sample, Seiwald was recorded with RB Leipzig.",
        (
            "In the FootyScout dataset, Seiwald was observed as a right defensive "
            "midfielder."
        ),
        "Club affiliation in the analytics sample: Seiwald is recorded with RB Leipzig.",
        "Seiwald was recorded with RB Leipzig in the historical FootyScout sample.",
        "FootyScout's observed dataset lists Seiwald with RB Leipzig.",
        "The analytics sample records Seiwald with RB Leipzig.",
    ),
)
def test_historical_dataset_claims_pass_inside_current_section(claim: str) -> None:
    ledger = EvidenceLedger(
        records=(_record(1, category=EvidenceCategory.ANALYTICS),)
    )
    answer = LLMGroundedAnswer(
        answer_markdown=f"## Current situation\n\n{claim} [run:evidence-1]",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


def test_exact_live_historical_dataset_claim_passes_with_analytics() -> None:
    ledger = EvidenceLedger(
        records=(
            _record(1, category=EvidenceCategory.ANALYTICS),
            _record(2, category=EvidenceCategory.ANALYTICS),
        )
    )
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "## Current situation\n\n"
            "Club affiliation in the analytics sample: Seiwald is recorded with "
            "RB Leipzig in the FootyScout dataset (historical observation) "
            "[run:evidence-1] [run:evidence-2]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    assert validate_answer_citations(answer, ledger).status is GroundedAnswerStatus.ANSWERED


@pytest.mark.parametrize(
    "claim",
    (
        "Seiwald currently plays for RB Leipzig.",
        "Seiwald is currently with RB Leipzig.",
        "His current club is RB Leipzig.",
        "Recent reporting places him at RB Leipzig.",
        "Today he plays for RB Leipzig.",
        "In the historical dataset, Seiwald currently plays for RB Leipzig.",
        "Historical FootyScout data shows his current club is RB Leipzig.",
        "Although the sample is historical, Seiwald is now at RB Leipzig.",
    ),
)
def test_current_language_overrides_historical_scope_markers(claim: str) -> None:
    ledger = EvidenceLedger(
        records=(_record(1, category=EvidenceCategory.ANALYTICS),)
    )
    answer = LLMGroundedAnswer(
        answer_markdown=f"{claim} [run:evidence-1]",
        evidence_ids=("run:evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_citations(answer, ledger)

    assert raised.value.error_code == "current_world_claim_requires_web"
    assert raised.value.failed_claim == f"{claim} [run:evidence-1]"


def test_full_historical_and_current_scope_regression() -> None:
    methodology = _record(
        2,
        category=EvidenceCategory.METHODOLOGY,
        sources=("role_fit:primary",),
    ).model_copy(
        update={
            "methodology_topic": "role_fit",
            "result": {
                "summary": "Lower Role Fit distance means closer stylistic resemblance."
            },
        }
    )
    ledger = EvidenceLedger(records=(_role_fit_record(), methodology, _web_record(3)))
    answer = LLMGroundedAnswer(
        answer_markdown=(
            "## Role Fit\n\n"
            "FootyScout's historical sample records a Role Fit distance of 0.3888 "
            "[run:evidence-1]. Lower distance means closer stylistic resemblance "
            "[run:evidence-2].\n\n"
            "## Historical FootyScout context\n\n"
            "Seiwald was recorded with RB Leipzig in the historical FootyScout sample "
            "[run:evidence-1].\n\n"
            "## Current situation\n\n"
            "Recent public reporting describes his current situation "
            "[run:evidence-3].\n\n"
            "## Conclusion\n\n"
            "The historical sample records Seiwald with RB Leipzig "
            "[run:evidence-1], while recent reporting provides his current situation "
            "[run:evidence-3]."
        ),
        evidence_ids=("run:evidence-1", "run:evidence-2", "run:evidence-3"),
        status=GroundedAnswerStatus.ANSWERED,
    )

    final = validate_answer_citations(answer, ledger)

    assert final.status is GroundedAnswerStatus.ANSWERED
    assert final.methodology_sources == ("role_fit:primary",)
    assert final.web_sources[0].evidence_id == "run:evidence-3"


def test_xg_evaluation_uses_governed_procedure_and_metrics() -> None:
    record = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={
            "methodology_topic": "xg",
            "result": {
                "topic": "xg",
                "summary": "Expected goals methodology.",
                "structured_metadata": {
                    "selected_model": "xgboost",
                    "selection": {
                        "primary_metric": "validation_log_loss",
                        "test_metrics_used": False,
                    },
                    "split_methodology": {
                        "method": "match-grouped deterministic approximately 80/10/10 split",
                        "train": {"match_count": 186},
                        "validation": {"match_count": 23},
                        "test": {"match_count": 24},
                    },
                    "validation_metrics": {
                        "selected_effective": {
                            "log_loss": 0.2858636,
                            "brier_score": 0.0805818,
                            "roc_auc": 0.7219372,
                        }
                    },
                    "untouched_test_metrics": {
                        "selected_effective": {
                            "log_loss": 0.2507239,
                            "brier_score": 0.0692584,
                            "roc_auc": 0.7686973,
                        }
                    },
                    "out_of_fold_metrics": {
                            "log_loss": 0.2703254,
                            "brier_score": 0.0760300,
                            "roc_auc": 0.7528267,
                    },
                    "oof": {"fold_count": 5, "prediction_count": 5545},
                },
            },
        }
    )
    context = SelectedContext(
        question="How is expected goals evaluated?",
        intent="methodology",
        methodology_evidence=(record,),
        status=ContextStatus.SUFFICIENT,
    )

    result = deterministic_governed_synthesis(context)
    payload = _synthesis_context_payload(context)

    assert result is not None
    assert "186 training, 23 validation, and 24 test matches" in result.answer.answer_markdown
    assert "validation log loss" in result.answer.answer_markdown
    assert "untouched test" in result.answer.answer_markdown
    assert "5-fold grouped out-of-fold" in result.answer.answer_markdown
    assert "not available" not in result.answer.answer_markdown
    metadata = payload["methodology_evidence"][0]["result"]["structured_metadata"]
    assert metadata["selection"]["test_metrics_used"] is False
    assert metadata["metrics"]["untouched_test_metrics"]["log_loss"] == pytest.approx(
        0.2507239
    )


def test_xpass_test_selection_answers_no_from_governed_metadata() -> None:
    record = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={
            "methodology_topic": "xpass",
            "result": {
                "topic": "xpass",
                "structured_metadata": {
                    "selection": {
                        "source": "validation metrics only",
                        "primary_metric": "validation log loss",
                        "test_metrics_used": False,
                    }
                },
            },
        }
    )
    context = SelectedContext(
        question="Were test matches used to pick the xPass model?",
        intent="methodology",
        methodology_evidence=(record,),
        status=ContextStatus.SUFFICIENT,
    )

    result = deterministic_governed_synthesis(context)

    assert result is not None
    assert result.answer.answer_markdown.startswith("No.")
    assert "validation metrics only" in result.answer.answer_markdown
    assert "held out from model selection" in result.answer.answer_markdown


@pytest.mark.parametrize(
    "selection",
    (
        {},
        {
            "source": "validation metrics only",
            "primary_metric": "validation log loss",
            "test_metrics_used": True,
        },
    ),
)
def test_xpass_test_selection_without_complete_consistent_evidence_stays_cautious(
    selection: dict[str, object],
) -> None:
    record = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={
            "methodology_topic": "xpass",
            "result": {
                "topic": "xpass",
                "structured_metadata": {"selection": selection},
            },
        }
    )
    context = SelectedContext(
        question="Were test matches used to pick the xPass model?",
        intent="methodology",
        methodology_evidence=(record,),
        status=ContextStatus.SUFFICIENT,
    )

    assert deterministic_governed_synthesis(context) is None


def test_xg_evaluation_without_structured_metadata_stays_on_cautious_path() -> None:
    record = _record(1, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={
            "methodology_topic": "xg",
            "result": {"topic": "xg", "summary": "Expected goals methodology."},
        }
    )
    context = SelectedContext(
        question="How is expected goals evaluated?",
        intent="methodology",
        methodology_evidence=(record,),
        status=ContextStatus.SUFFICIENT,
    )

    assert deterministic_governed_synthesis(context) is None


def test_pressure_passing_answer_preserves_rate_and_missing_execution_value() -> None:
    record = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={
            "tool_name": ToolName.GET_PLAYER_DOSSIER,
            "result": {
                "player": {"player_id": 3500, "player_name": "Granit Xhaka"},
                "requested_sections": ["passing"],
                "passing": {
                    "pressure_pass_rate": 0.1600484995,
                    "pressure_above_expected_pp": None,
                    "pass_attempts": 3299,
                    "matches_observed": 33,
                },
            },
        }
    )
    context = SelectedContext(
        question="How has Xhaka performed under pressure as a passer?",
        intent="player_profile",
        analytics_evidence=(record,),
        status=ContextStatus.SUFFICIENT,
    )

    result = deterministic_governed_synthesis(context)
    payload = _synthesis_context_payload(context)

    assert result is not None
    assert "16.00%" in result.answer.answer_markdown
    assert "cannot directly quantify his execution quality" in result.answer.answer_markdown
    assert "do not include a pressure-specific" not in result.answer.answer_markdown
    metrics = payload["analytics_evidence"][0]["result"]["passing"][
        "representative_metrics"
    ]
    assert metrics["pressure_pass_rate"] == pytest.approx(0.1600484995)
    assert metrics["pressure_above_expected_pp"] is None


def test_unique_category_placeholder_is_repaired_to_canonical_id() -> None:
    record = _record(1, category=EvidenceCategory.ANALYTICS)
    answer = LLMGroundedAnswer(
        answer_markdown="Supported claim [analytics evidence].",
        evidence_ids=(),
        status=GroundedAnswerStatus.ANSWERED,
    )

    validated = validate_answer_with_policy(
        answer,
        EvidenceLedger(records=(record,)),
        supplied_evidence_ids=(record.evidence_id,),
    )

    assert validated.answer.answer_markdown == "Supported claim [run:evidence-1]."
    assert validated.answer.evidence_ids == ("run:evidence-1",)
    assert validated.citation_repairs[0].strategy == "unique_category_placeholder"


@pytest.mark.parametrize(
    ("placeholder", "record"),
    (
        (
            "methodology evidence",
            _record(1, category=EvidenceCategory.METHODOLOGY),
        ),
        ("web evidence", _web_record(1)),
        ("evidence", _record(1, category=EvidenceCategory.ANALYTICS)),
        ("source", _record(1, category=EvidenceCategory.ANALYTICS)),
    ),
)
def test_unique_placeholder_categories_repair_to_ledger_ids(
    placeholder: str,
    record: EvidenceRecord,
) -> None:
    answer = LLMGroundedAnswer(
        answer_markdown=f"Supported claim [{placeholder}].",
        evidence_ids=(),
        status=GroundedAnswerStatus.ANSWERED,
    )

    validated = validate_answer_with_policy(
        answer,
        EvidenceLedger(records=(record,)),
        supplied_evidence_ids=(record.evidence_id,),
    )

    assert f"[{record.evidence_id}]" in validated.answer.answer_markdown
    assert validated.answer.evidence_ids == (record.evidence_id,)


def test_ambiguous_category_placeholder_remains_blocked() -> None:
    records = (
        _record(1, category=EvidenceCategory.ANALYTICS),
        _record(2, category=EvidenceCategory.ANALYTICS),
    )
    answer = LLMGroundedAnswer(
        answer_markdown="Ambiguous claim [analytics evidence].",
        evidence_ids=(),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_with_policy(
            answer,
            EvidenceLedger(records=records),
            supplied_evidence_ids=tuple(record.evidence_id for record in records),
        )

    assert raised.value.error_code == "invalid_placeholder_citation"


def test_unique_short_evidence_suffix_is_canonicalized() -> None:
    canonical = "run-unique:evidence-1"
    record = _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
        update={"evidence_id": canonical}
    )
    answer = LLMGroundedAnswer(
        answer_markdown="Supported claim [evidence-1].",
        evidence_ids=("evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    validated = validate_answer_with_policy(
        answer,
        EvidenceLedger(records=(record,)),
        supplied_evidence_ids=(canonical,),
    )

    assert validated.answer.answer_markdown == f"Supported claim [{canonical}]."
    assert validated.answer.evidence_ids == (canonical,)
    assert validated.citation_repairs[0].strategy == "unique_supplied_ordinal_suffix"


def test_ambiguous_short_evidence_suffix_remains_blocked() -> None:
    records = (
        _record(1, category=EvidenceCategory.ANALYTICS).model_copy(
            update={"evidence_id": "run-a:evidence-1"}
        ),
        _record(2, category=EvidenceCategory.METHODOLOGY).model_copy(
            update={"evidence_id": "run-b:evidence-1"}
        ),
    )
    answer = LLMGroundedAnswer(
        answer_markdown="Ambiguous claim [evidence-1].",
        evidence_ids=("evidence-1",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_with_policy(
            answer,
            EvidenceLedger(records=records),
            supplied_evidence_ids=tuple(record.evidence_id for record in records),
        )

    assert raised.value.error_code == "unknown_evidence_id"


def test_nonexistent_short_evidence_suffix_remains_blocked() -> None:
    record = _record(1, category=EvidenceCategory.ANALYTICS)
    answer = LLMGroundedAnswer(
        answer_markdown="Unsupported claim [evidence-999].",
        evidence_ids=("evidence-999",),
        status=GroundedAnswerStatus.ANSWERED,
    )

    with pytest.raises(GroundingValidationError) as raised:
        validate_answer_with_policy(
            answer,
            EvidenceLedger(records=(record,)),
            supplied_evidence_ids=(record.evidence_id,),
        )

    assert raised.value.error_code == "unknown_evidence_id"


def test_canonical_evidence_id_is_left_unchanged() -> None:
    record = _record(1, category=EvidenceCategory.ANALYTICS)
    validated = validate_answer_with_policy(
        _answer(record.evidence_id),
        EvidenceLedger(records=(record,)),
        supplied_evidence_ids=(record.evidence_id,),
    )

    assert validated.answer.evidence_ids == (record.evidence_id,)
    assert validated.citation_repairs == ()


def test_degraded_web_answer_repairs_unique_analytics_placeholder() -> None:
    analytics = _role_fit_record()
    methodology = _record(2, category=EvidenceCategory.METHODOLOGY).model_copy(
        update={"methodology_topic": "role_fit"}
    )
    context = SelectedContext(
        question="How well does Xhaka fit Leverkusen and what is the latest transfer news?",
        intent="role_fit",
        analytics_evidence=(analytics,),
        methodology_evidence=(methodology,),
        status=ContextStatus.SUFFICIENT,
    )
    generated = SynthesisResult(
        answer=LLMGroundedAnswer(
            answer_markdown=(
                "Role Fit distance is 0.3888 [analytics evidence].\n\n"
                "Current reporting says Xhaka will transfer."
            ),
            evidence_ids=(),
            status=GroundedAnswerStatus.ANSWERED,
        ),
        provider="fake",
        model="fake",
    )

    degraded = apply_no_qualifying_web_fallback(
        generated,
        context,
        category="transfer_reporting",
    )
    validated = validate_answer_with_policy(
        degraded.answer,
        EvidenceLedger(records=context.evidence),
        supplied_evidence_ids=context.evidence_ids,
    )

    assert "[analytics evidence]" not in validated.answer.answer_markdown
    assert f"[{analytics.evidence_id}]" in validated.answer.answer_markdown
    assert "couldn't verify current transfer reporting" in validated.answer.answer_markdown
