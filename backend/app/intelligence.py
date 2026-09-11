"""Database-to-API presentation for V3.1 player intelligence."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from analytics.player_feature_registry import FEATURE_REGISTRY, RADAR_FEATURES
from app.archetypes import player_archetype_response
from app.models import Player, PlayerIntelligenceProfile, PlayerPercentile
from app.presenters import player_identity
from app.schemas import PlayerIntelligenceMetricResponse, PlayerIntelligenceResponse


def get_intelligence_response(
    session: Session, player: Player
) -> PlayerIntelligenceResponse | None:
    profile = session.get(PlayerIntelligenceProfile, player.player_id)
    if profile is None:
        return None
    stored = {
        row.metric_name: row
        for row in session.scalars(
            select(PlayerPercentile).where(PlayerPercentile.player_id == player.player_id)
        ).all()
    }
    metrics: list[PlayerIntelligenceMetricResponse] = []
    for definition in FEATURE_REGISTRY:
        row = stored.get(definition.feature_name)
        if row is None:
            continue
        metrics.append(
            PlayerIntelligenceMetricResponse(
                metric_name=row.metric_name,
                label=definition.label,
                family=row.family,
                raw_value=row.raw_value,
                unit=definition.unit,
                percentile=row.percentile,
                peer_position_group=row.peer_position_group,
                peer_count=row.peer_count,
                sample_count=row.sample_count,
                minimum_sample=definition.minimum_sample,
                eligible=row.eligible,
                eligibility_reason=row.eligibility_reason,
                directionality=definition.directionality,
                stability_note=definition.stability_note,
            )
        )
    style = [metric for metric in metrics if metric.family == "style"]
    performance = [metric for metric in metrics if metric.family == "performance"]
    radar = [
        metric
        for metric in metrics
        if metric.metric_name in RADAR_FEATURES and metric.percentile is not None
    ]
    if player.position_group == "GK":
        radar = []
        radar_status = (
            "Limited goalkeeper profile: FootyScout does not yet have enough "
            "goalkeeper-specific metrics for a meaningful radar."
        )
    elif len(radar) < 3:
        radar = []
        radar_status = "Insufficient eligible position metrics for a meaningful radar."
    else:
        radar_status = (
            f"Percentiles among eligible {player.position_group} peers; higher style "
            "percentiles describe tendency, not quality."
        )
    return PlayerIntelligenceResponse(
        player=player_identity(player),
        matches_observed=profile.matches_observed,
        position_group=profile.position_group,
        style_metrics=style,
        performance_metrics=performance,
        radar_metrics=radar,
        radar_status=radar_status,
        percentile_context=(
            "Each percentile is an empirical rank among eligible players in the same "
            "broad position group. Percentiles are not ratings."
        ),
        archetype=player_archetype_response(session, player),
    )
