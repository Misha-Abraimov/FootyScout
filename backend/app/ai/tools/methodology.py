"""Deterministic methodology tool backed by governed metadata and source refs."""

from app.ai.policy import (
    ARCHETYPE_LIMITATIONS,
    METHODOLOGY_SOURCE_REFERENCES,
    POSSESSION_VALUE_LIMITATIONS,
    ROLE_FIT_LIMITATIONS,
    SIMILARITY_LIMITATIONS,
)
from app.ai.schemas import (
    MethodologyInput,
    MethodologyResponse,
    MethodologySource,
    MethodologyTopic,
    ProductionStatus,
)
from app.services import methodology

SUMMARIES = {
    MethodologyTopic.XPASS: (
        "Expected pass completion is produced by the governed production xPass "
        "model from pre-outcome pass characteristics."
    ),
    MethodologyTopic.XG: (
        "Expected goals is produced for eligible non-penalty shots by the governed "
        "production xG model from pre-outcome shot characteristics."
    ),
    MethodologyTopic.POSSESSION_VALUE: (
        "The production XGBoost state model estimates expected attacking value from "
        "pre-event game state; GRU and Transformer variants remain experiments."
    ),
    MethodologyTopic.ATTACKING_IMPACT: (
        "Attacking impact is the change in modeled possession value across a pass or "
        "carry, using persisted model outputs rather than AI-generated values."
    ),
    MethodologyTopic.PLAYER_PROFILES: (
        "Player profiles combine governed passing, shooting, attacking-impact, and "
        "playing-style outputs while preserving eligibility and unavailable metrics."
    ),
    MethodologyTopic.PERCENTILES: (
        "Eligible player metrics are ranked against metric-specific peers in the "
        "same position group; missing eligibility is preserved rather than scored."
    ),
    MethodologyTopic.ARCHETYPES: (
        "Archetypes are governed K-means playing-style groups in the production "
        "position-normalized feature space, not assessments of quality."
    ),
    MethodologyTopic.SIMILARITY: (
        "Similarity ranks eligible same-position players in the frozen six-feature "
        "style space and includes explicit sample support."
    ),
    MethodologyTopic.TEAM_INTELLIGENCE: (
        "Team intelligence summarizes observed team event data, with supported "
        "positional-role profiles pooled from eligible outfield actions."
    ),
    MethodologyTopic.ROLE_FIT: (
        "Role Fit measures raw distance from an eligible outfield player's observed "
        "style to a supported team's observed positional-role style; lower raw "
        "distance means closer resemblance."
    ),
    MethodologyTopic.ROLE_RECOMMENDATIONS: (
        "Scouting recommendations order eligible external same-position players by "
        "their persisted Role Fit distance to a supported observed team role."
    ),
}


def get_methodology(request: MethodologyInput) -> MethodologyResponse:
    topic = request.topic
    metadata = None
    production_model = None
    experimental_models: list[str] = []
    limitations: list[str] = []

    if topic is MethodologyTopic.XPASS:
        metadata = methodology.get_pass_model_info()
        production_model = metadata.selected_model
    elif topic is MethodologyTopic.XG:
        metadata = methodology.get_xg_model_info()
        production_model = metadata.selected_model
    elif topic in {
        MethodologyTopic.POSSESSION_VALUE,
        MethodologyTopic.ATTACKING_IMPACT,
    }:
        metadata = methodology.get_action_value_model_info()
        production_model = "xgboost"
        experimental_models = ["gru", "pytorch_causal_transformer"]
        limitations.extend(POSSESSION_VALUE_LIMITATIONS)
    elif topic is MethodologyTopic.ARCHETYPES:
        limitations.extend(ARCHETYPE_LIMITATIONS)
    elif topic is MethodologyTopic.SIMILARITY:
        limitations.extend(SIMILARITY_LIMITATIONS)
    elif topic is MethodologyTopic.ROLE_FIT:
        limitations.extend(ROLE_FIT_LIMITATIONS)
    elif topic is MethodologyTopic.PLAYER_PROFILES:
        limitations.append(
            "A profile describes observed event data, not future performance."
        )
    elif topic is MethodologyTopic.TEAM_INTELLIGENCE:
        limitations.extend(
            [
                "Team intelligence describes observed event data, not coaching intent.",
                "Current production coverage is limited to qualified teams and outfield roles.",
            ]
        )
    elif topic is MethodologyTopic.ROLE_RECOMMENDATIONS:
        limitations.append(
            "Recommendations are style matches, not predictions of transfer success."
        )

    path, section = METHODOLOGY_SOURCE_REFERENCES[topic.value]
    sources = [
        MethodologySource(
            source_id=f"{topic.value}:primary",
            path=path,
            section=section,
            status=ProductionStatus.PRODUCTION,
        )
    ]
    if topic is MethodologyTopic.ATTACKING_IMPACT:
        sources.insert(
            0,
            MethodologySource(
                source_id="attacking_impact:production_metadata",
                path="backend/app/runtime_metadata/action_value_model_metadata.json",
                section="production possession-value metadata",
                status=ProductionStatus.PRODUCTION,
            ),
        )
    if topic is MethodologyTopic.POSSESSION_VALUE:
        sources.extend(
            [
                MethodologySource(
                    source_id="possession_value:gru_experiment",
                    path="docs/experiments/gru_possession_value.md",
                    section="Offline GRU experiment",
                    status=ProductionStatus.EXPERIMENTAL,
                ),
                MethodologySource(
                    source_id="possession_value:transformer_experiment",
                    path="docs/experiments/transformer_possession_value.md",
                    section="Offline causal-Transformer experiment",
                    status=ProductionStatus.EXPERIMENTAL,
                ),
            ]
        )

    return MethodologyResponse(
        topic=topic,
        production_status=ProductionStatus.PRODUCTION,
        current_production_model=production_model,
        experimental_models=experimental_models,
        summary=SUMMARIES[topic],
        structured_metadata=metadata,
        sources=sources,
        limitations=limitations,
    )
