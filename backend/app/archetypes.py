"""Presentation and catalogue services for V3.2C player-style archetypes."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Player, PlayerArchetype, PlayerAttackingProfile
from app.runtime_metadata import PLAYER_ARCHETYPE_METADATA_PATH
from app.schemas import (
    ArchetypeCatalogueResponse,
    ArchetypeDefinitionResponse,
    ArchetypeDistinguishingFeatureResponse,
    ArchetypePositionCompositionResponse,
    ArchetypeRepresentativeResponse,
    ArchetypeStyleDimensionResponse,
    PlayerArchetypeResponse,
)

METADATA_PATH = PLAYER_ARCHETYPE_METADATA_PATH
PASS_THRESHOLD = 50
CARRY_THRESHOLD = 29
METHODOLOGY_VERSION = "V3.2C"
SEPARATION_INTERPRETATION = (
    "Distance-based style separation, not a probability. Smaller values mean the "
    "player lies nearer the boundary between the two archetypes."
)
FEATURE_COLUMNS = {
    "expected_completion_rate": "expected_completion_rate_position_z",
    "pressure_pass_rate": "pressure_pass_rate_position_z",
    "progressive_pass_rate": "progressive_pass_rate_position_z",
    "long_pass_rate": "long_pass_rate_position_z",
    "positive_forward_distance_per_100_passes": (
        "positive_forward_distance_per_100_passes_position_z"
    ),
    "carry_share_of_actions": "carry_share_of_actions_position_z",
}
FEATURE_LABELS = {
    "expected_completion_rate": "Expected completion",
    "pressure_pass_rate": "Under-pressure pass rate",
    "progressive_pass_rate": "Progressive-pass rate",
    "long_pass_rate": "Long-pass rate",
    "positive_forward_distance_per_100_passes": "Positive forward distance / 100 passes",
    "carry_share_of_actions": "Carry share of actions",
}


@lru_cache(maxsize=1)
def load_archetype_metadata() -> dict[str, Any]:
    """Load the controlled production metadata without exposing its file location."""
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _ineligibility_reason(session: Session, player: Player) -> str:
    if player.position_group == "GK":
        return "Goalkeepers are excluded from the outfield archetype model."
    if player.pass_attempts < PASS_THRESHOLD:
        return f"Limited sample for archetype analysis: requires {PASS_THRESHOLD} passes."
    attacking = session.get(PlayerAttackingProfile, player.player_id)
    carries = attacking.carries if attacking is not None else 0
    if carries < CARRY_THRESHOLD:
        return f"Limited sample for archetype analysis: requires {CARRY_THRESHOLD} carries."
    return "Limited sample for archetype analysis: required style features are unavailable."


def player_archetype_response(session: Session, player: Player) -> PlayerArchetypeResponse:
    assignment = session.get(PlayerArchetype, player.player_id)
    if assignment is None:
        return PlayerArchetypeResponse(
            id=None,
            name=None,
            eligible=False,
            eligibility_reason=_ineligibility_reason(session, player),
            position_group=player.position_group,
            separation_interpretation=SEPARATION_INTERPRETATION,
            style_dimensions=[],
            distinguishing_features=[],
            methodology_version=METHODOLOGY_VERSION,
        )

    metadata = load_archetype_metadata()
    definition = next(
        item for item in metadata["definitions"] if item["id"] == assignment.archetype_id
    )
    dimensions = [
        ArchetypeStyleDimensionResponse(
            feature_name=feature,
            label=FEATURE_LABELS[feature],
            position_z=float(getattr(assignment, column)),
        )
        for feature, column in FEATURE_COLUMNS.items()
    ]
    distinguishing = [
        ArchetypeDistinguishingFeatureResponse.model_validate(item)
        for item in definition["distinguishing_features"]
    ]
    return PlayerArchetypeResponse(
        id=assignment.archetype_id,
        name=assignment.archetype_name,
        eligible=True,
        eligibility_reason=None,
        position_group=assignment.position_group,
        centroid_distance=assignment.centroid_distance,
        second_centroid_distance=assignment.second_centroid_distance,
        separation_margin=assignment.separation_margin,
        separation_interpretation=SEPARATION_INTERPRETATION,
        style_dimensions=dimensions,
        distinguishing_features=distinguishing,
        methodology_version=assignment.model_version,
    )


def archetype_catalogue_response(session: Session) -> ArchetypeCatalogueResponse:
    """Combine governed centroid metadata with live PostgreSQL assignment counts."""
    metadata = load_archetype_metadata()
    definitions: list[ArchetypeDefinitionResponse] = []
    for stored in metadata["definitions"]:
        archetype_id = stored["id"]
        count = session.scalar(
            select(func.count())
            .select_from(PlayerArchetype)
            .where(PlayerArchetype.archetype_id == archetype_id)
        ) or 0
        grouped = session.execute(
            select(PlayerArchetype.position_group, func.count())
            .where(PlayerArchetype.archetype_id == archetype_id)
            .group_by(PlayerArchetype.position_group)
        ).all()
        counts = {str(position): int(value) for position, value in grouped}
        composition = {
            position: ArchetypePositionCompositionResponse(
                count=counts.get(position, 0),
                percentage=(100.0 * counts.get(position, 0) / count if count else 0.0),
            )
            for position in ("DEF", "MID", "FWD")
        }
        representative_rows = session.execute(
            select(PlayerArchetype, Player)
            .join(Player, Player.player_id == PlayerArchetype.player_id)
            .where(PlayerArchetype.archetype_id == archetype_id)
            .order_by(PlayerArchetype.centroid_distance, Player.player_name, Player.player_id)
            .limit(5)
        ).all()
        representatives = [
            ArchetypeRepresentativeResponse(
                player_id=player.player_id,
                player_name=player.player_name,
                team_name=player.team_name,
                position_group=assignment.position_group,
                centroid_distance=assignment.centroid_distance,
            )
            for assignment, player in representative_rows
        ]
        definitions.append(
            ArchetypeDefinitionResponse(
                id=archetype_id,
                name=stored["name"],
                description=stored["description"],
                centroid=stored["centroid"],
                distinguishing_features=stored["distinguishing_features"],
                player_count=int(count),
                position_composition=composition,
                representative_players=representatives,
                separation_distribution=stored["separation_distribution"],
            )
        )
    sensitivity = metadata["selection_history"]["six_feature_sensitivity"]
    resample = sensitivity.get("resample_stability", {})
    methodology = {
        "method": metadata["method"],
        "k": metadata["k"],
        "cohort": metadata["cohort"],
        "feature_order": metadata["feature_order"],
        "thresholds": metadata["thresholds"],
        "normalization": metadata["normalization"]["method"],
        "clustering_space": "Six position-normalized feature dimensions",
        "separation_formula": metadata["separation_formula"],
        "separation_interpretation": metadata["separation_interpretation"],
        "production_refinement": metadata["omitted_feature"],
        "selection_history": {
            "methods_evaluated": metadata["selection_history"]["methods_evaluated"],
            "k_values_evaluated": metadata["selection_history"]["k_values_evaluated"],
            "selected_before_production_refinement": metadata["selection_history"][
                "selected_before_production_refinement"
            ],
            "six_feature_diagnostics": {
                "silhouette": sensitivity.get("silhouette"),
                "davies_bouldin": sensitivity.get("davies_bouldin"),
                "calinski_harabasz": sensitivity.get("calinski_harabasz"),
                "assignment_ari_vs_seven_features": sensitivity.get(
                    "assignment_ari_vs_seven_features"
                ),
                "position_adjusted_rand": sensitivity.get("position_adjusted_rand"),
                "position_normalized_mutual_information": sensitivity.get(
                    "position_normalized_mutual_information"
                ),
                "repeated_seed_stability": sensitivity.get(
                    "repeated_seed_stability"
                ),
                "resample_stability": {
                    key: value
                    for key in (
                        "resamples",
                        "sample_fraction",
                        "sample_size",
                        "mean_ari",
                        "median_ari",
                        "minimum_ari",
                        "cluster_assignment_stability",
                        "player_assignment_stability_summary",
                    )
                    if (value := resample.get(key)) is not None
                },
            },
            "interpretation_caution": metadata["selection_history"][
                "interpretation_caution"
            ],
        },
    }
    return ArchetypeCatalogueResponse(
        methodology_version=metadata["methodology_version"],
        purpose=metadata["purpose"],
        definitions=definitions,
        methodology=methodology,
    )
