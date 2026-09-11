"""Validated public request values and response shapes for the FootyScout API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class PositionGroup(str, Enum):
    GK = "GK"
    DEF = "DEF"
    MID = "MID"
    FWD = "FWD"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class PlayerSortField(str, Enum):
    PLAYER_NAME = "player_name"
    PASS_ATTEMPTS = "pass_attempts"
    ACTUAL_COMPLETION_RATE = "actual_completion_rate"
    EXPECTED_COMPLETION_RATE = "expected_completion_rate"
    COMPLETION_ABOVE_EXPECTED_PP = "completion_above_expected_pp"
    PROGRESSIVE_PASS_RATE = "progressive_pass_rate"
    PRESSURE_ABOVE_EXPECTED_PP = "pressure_above_expected_pp"
    FINAL_THIRD_ENTRIES_PER_100_PASSES = "final_third_entries_per_100_passes"


class LeaderboardMetric(str, Enum):
    COMPLETION_ABOVE_EXPECTED_PP = "completion_above_expected_pp"
    PRESSURE_ABOVE_EXPECTED_PP = "pressure_above_expected_pp"
    PROGRESSIVE_ABOVE_EXPECTED_PP = "progressive_above_expected_pp"
    LONG_PASS_ABOVE_EXPECTED_PP = "long_pass_above_expected_pp"
    FINAL_THIRD_ENTRIES_PER_100_PASSES = "final_third_entries_per_100_passes"
    EXPECTED_COMPLETION_RATE = "expected_completion_rate"
    PROGRESSIVE_PASS_RATE = "progressive_pass_rate"
    ATTACKING_VALUE_PER_100_ACTIONS = "attacking_value_per_100_actions"
    PASS_VALUE_PER_100_PASSES = "pass_value_per_100_passes"
    CARRY_VALUE_PER_100_CARRIES = "carry_value_per_100_carries"
    PROGRESSIVE_VALUE_PER_100_ACTIONS = "progressive_value_per_100_actions"
    PRESSURE_VALUE_PER_100_ACTIONS = "pressure_value_per_100_actions"


class PlayerIdentity(ApiModel):
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    position: str
    position_group: str


class PlayerSummary(PlayerIdentity):
    matches_observed: int
    pass_attempts: int
    actual_completion_rate: float
    expected_completion_rate: float
    completion_above_expected_pp: float
    progressive_pass_rate: float
    pressure_pass_rate: float
    final_third_entries_per_100_passes: float
    overall_reliable: bool


class PlayerListResponse(ApiModel):
    total: int
    limit: int
    offset: int
    items: list[PlayerSummary]


class PlayerProfileResponse(PlayerIdentity):
    matches_observed: int
    pass_attempts: int
    overall_reliable: bool

    passes_completed: int
    actual_completion_rate: float
    expected_completions: float
    expected_completion_rate: float
    completions_above_expected: float
    completion_above_expected_pp: float

    pressure_attempts: int
    pressure_completed: int
    pressure_actual_completion_rate: float | None
    pressure_expected_completion_rate: float | None
    pressure_completions_above_expected: float | None
    pressure_above_expected_pp: float | None
    pressure_pass_rate: float

    progressive_attempts: int
    progressive_completed: int
    progressive_actual_completion_rate: float | None
    progressive_expected_completion_rate: float | None
    progressive_completions_above_expected: float | None
    progressive_above_expected_pp: float | None
    progressive_pass_rate: float

    long_pass_attempts: int
    long_pass_completed: int
    long_pass_actual_completion_rate: float | None
    long_pass_expected_completion_rate: float | None
    long_pass_completions_above_expected: float | None
    long_pass_above_expected_pp: float | None

    average_forward_distance: float
    net_forward_distance_per_100_passes: float
    positive_forward_distance_per_100_passes: float
    final_third_entries: int
    final_third_entries_per_100_passes: float

    pressure_reliable: bool
    progressive_reliable: bool
    long_pass_reliable: bool


class SimilarPlayerResponse(ApiModel):
    similar_player_id: int
    similar_player_name: str
    similar_team_name: str
    similar_position: str
    similar_position_group: str
    rank: int
    rms_distance: float = Field(ge=0)
    similarity_score: float = Field(
        ge=0,
        le=100,
        description=(
            "A 0–100 playing-style similarity index; this is not a probability or ability rating."
        ),
    )
    same_position_group: bool
    closest_feature_1: str
    closest_feature_2: str
    closest_feature_3: str
    closest_style_dimensions: list[str]
    query_matches_observed: int
    candidate_matches_observed: int
    pair_support_matches: int
    sample_support: str
    sample_support_explanation: str
    methodology_version: str


class SimilarPlayersResponse(ApiModel):
    source_player: PlayerIdentity
    available: bool
    unavailable_reason: str | None
    methodology_version: str
    query_matches_observed: int
    query_sample_support: str
    eligibility_requirements: list[str]
    total: int
    limit: int
    items: list[SimilarPlayerResponse]


class PassResponse(ApiModel):
    pass_index: int
    match_id: int
    player_id: int | None
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    pass_length: float
    pass_angle: float
    forward_distance: float
    lateral_distance: float
    distance_to_goal_before: float
    distance_to_goal_after: float
    distance_toward_goal: float
    completed: bool
    expected_completion: float = Field(ge=0, le=1)
    under_pressure: bool
    progressive: bool
    pass_height: str
    body_part: str
    pass_type: str
    start_zone: str
    end_zone: str
    fold: int
    attacking_value: float | None = None
    state_value_before: float | None = Field(default=None, ge=0)
    state_value_after: float | None = Field(default=None, ge=0)
    pass_risk: float | None = Field(default=None, ge=0, le=1)
    risk_reward_category: str | None = None


class PassListResponse(ApiModel):
    total: int
    limit: int
    offset: int
    items: list[PassResponse]

class ShootingProfileResponse(ApiModel):
    player_id: int
    shots: int
    goals: int
    total_xg: float
    xg_per_shot: float
    goals_minus_xg: float
    goals_per_shot: float
    matches_observed: int
    shooting_reliable: bool


class ShotResponse(ApiModel):
    shot_id: str
    match_id: int
    player_id: int | None
    period: int
    minute: int
    second: int
    start_x: float
    start_y: float
    distance: float
    angle: float
    goal: bool
    expected_goal: float | None = Field(default=None, ge=0, le=1)
    body_part: str
    shot_type: str
    technique: str
    play_pattern: str
    under_pressure: bool
    first_time: bool
    one_on_one: bool
    open_goal: bool
    penalty: bool
    penalty_shootout: bool
    model_eligible: bool


class ShotListResponse(ApiModel):
    total: int
    limit: int
    offset: int
    items: list[ShotResponse]


class AttackingProfileResponse(ApiModel):
    player_id: int
    matches_observed: int
    actions: int
    passes: int
    carries: int
    total_attacking_value: float
    attacking_value_per_100_actions: float
    total_pass_value: float
    pass_value_per_100_passes: float | None
    total_carry_value: float
    carry_value_per_100_carries: float | None
    positive_value_actions: int
    positive_value_action_rate: float
    progressive_action_value: float
    progressive_value_per_100_actions: float
    pressure_action_value: float
    pressure_value_per_100_actions: float
    attacking_value_reliable: bool
    pass_value_reliable: bool
    carry_value_reliable: bool


class AttackingActionResponse(ApiModel):
    action_id: str
    pass_index: int | None
    match_id: int
    possession_id: int
    event_index: int
    player_id: int | None
    team_id: int
    action_type: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    state_value_before: float = Field(ge=0)
    state_value_after: float = Field(ge=0)
    attacking_value: float
    success: bool
    under_pressure: bool
    progressive: bool
    expected_completion: float | None = Field(default=None, ge=0, le=1)
    pass_risk: float | None = Field(default=None, ge=0, le=1)
    risk_reward_category: str | None
    fold: int


class PlayerIntelligenceMetricResponse(ApiModel):
    metric_name: str
    label: str
    family: str
    raw_value: float | None
    unit: str
    percentile: float | None = Field(default=None, ge=0, le=100)
    peer_position_group: str
    peer_count: int
    sample_count: int
    minimum_sample: int
    eligible: bool
    eligibility_reason: str
    directionality: str
    stability_note: str | None = None


class ArchetypeStyleDimensionResponse(ApiModel):
    feature_name: str
    label: str
    position_z: float


class ArchetypeDistinguishingFeatureResponse(ArchetypeStyleDimensionResponse):
    direction: str


class PlayerArchetypeResponse(ApiModel):
    id: str | None
    name: str | None
    eligible: bool
    eligibility_reason: str | None
    position_group: str
    centroid_distance: float | None = Field(default=None, ge=0)
    second_centroid_distance: float | None = Field(default=None, ge=0)
    separation_margin: float | None = Field(default=None, ge=0, le=1)
    separation_interpretation: str
    style_dimensions: list[ArchetypeStyleDimensionResponse]
    distinguishing_features: list[ArchetypeDistinguishingFeatureResponse]
    methodology_version: str


class PlayerIntelligenceResponse(ApiModel):
    player: PlayerIdentity
    matches_observed: int
    position_group: str
    style_metrics: list[PlayerIntelligenceMetricResponse]
    performance_metrics: list[PlayerIntelligenceMetricResponse]
    radar_metrics: list[PlayerIntelligenceMetricResponse]
    radar_status: str
    percentile_context: str
    archetype: PlayerArchetypeResponse


class ArchetypeRepresentativeResponse(ApiModel):
    player_id: int
    player_name: str
    team_name: str
    position_group: str
    centroid_distance: float = Field(ge=0)


class ArchetypePositionCompositionResponse(ApiModel):
    count: int
    percentage: float = Field(ge=0, le=100)


class ArchetypeDefinitionResponse(ApiModel):
    id: str
    name: str
    description: str
    centroid: dict[str, float]
    distinguishing_features: list[ArchetypeDistinguishingFeatureResponse]
    player_count: int
    position_composition: dict[str, ArchetypePositionCompositionResponse]
    representative_players: list[ArchetypeRepresentativeResponse]
    separation_distribution: dict[str, float]


class ArchetypeCatalogueResponse(ApiModel):
    methodology_version: str
    purpose: str
    definitions: list[ArchetypeDefinitionResponse]
    methodology: dict[str, Any]


class AttackingActionListResponse(ApiModel):
    total: int
    limit: int
    offset: int
    items: list[AttackingActionResponse]


class ActionValueModelInfoResponse(ApiModel):
    task_name: str
    target: str
    target_interpretation: str
    state_convention: str
    selected_horizon: str
    feature_columns: list[str]
    leakage_protection: str
    preprocessing: dict[str, Any]
    split_methodology: dict[str, Any]
    nested_cross_fitting: dict[str, Any]
    training_corpus: dict[str, Any]
    target_distribution: dict[str, Any]
    baseline: dict[str, Any]
    candidates: list[dict[str, Any]]
    hurdle: dict[str, Any]
    selection: dict[str, Any]
    validation_metrics: dict[str, Any]
    untouched_test_metrics: dict[str, Any]
    out_of_fold_metrics: dict[str, Any]
    oof: dict[str, Any]
    transition_rules: dict[str, Any]
    limitations: list[str]


class ComparisonMetadata(ApiModel):
    player_ids: list[int]
    same_position_group: bool


class ComparisonResponse(ApiModel):
    comparison: ComparisonMetadata
    players: list[PlayerProfileResponse]
    attacking: list[AttackingProfileResponse | None] = Field(default_factory=list)
    intelligence: list[PlayerIntelligenceResponse | None] = Field(default_factory=list)


class LeaderboardEntry(PlayerIdentity):
    rank: int
    pass_attempts: int
    relevant_attempts: int
    metric: LeaderboardMetric
    metric_value: float
    overall_reliable: bool


class LeaderboardResponse(ApiModel):
    metric: LeaderboardMetric
    total: int
    limit: int
    minimum_pass_attempts: int
    items: list[LeaderboardEntry]


class ModelMetrics(ApiModel):
    roc_auc: float
    log_loss: float
    brier_score: float
    accuracy: float
    expected_calibration_error: float


class DatasetSummary(ApiModel):
    pass_count: int
    match_count: int
    completion_rate: float


class SplitPartition(ApiModel):
    match_count: int
    pass_count: int
    completion_rate: float


class SplitSummary(ApiModel):
    method: str
    random_seed: int
    train: SplitPartition
    validation: SplitPartition
    test: SplitPartition


class PreprocessingSummary(ApiModel):
    evaluation_fit_split: str
    oof_fit_policy: str
    numeric: str
    boolean: str
    categorical: str
    encoded_feature_count: int


class ModelSelectionSummary(ApiModel):
    model: str
    source: str
    reason: str
    primary_metric: str
    tie_breaker: str
    supporting_metric: str
    test_metrics_used: bool


class TemperatureScalingSummary(ApiModel):
    method: str
    fit_split: str
    temperature: float
    retained: bool
    reason: str
    uncalibrated_validation_metrics: ModelMetrics
    calibrated_validation_metrics: ModelMetrics


class ProductionModelSummary(ApiModel):
    model: str
    training_rows: int
    encoded_feature_count: int
    epochs: int | None = None
    boosting_rounds: int | None = None
    temperature: float
    temperature_source: str
    evaluation_use: str


class ModelMethodology(ApiModel):
    model_selection: str
    test_set: str
    player_profiles: str
    production_model: str


class XGBoostFeatureImportance(ApiModel):
    feature: str
    gain: float
    normalized_gain: float


class XGBoostSummary(ApiModel):
    version: str
    random_seed: int
    selection_metric: str
    selected_parameters: dict[str, str | int | float]
    best_iteration: int
    boosting_rounds: int
    calibration: TemperatureScalingSummary
    feature_importance_type: str
    feature_importance_interpretation: str
    feature_importance: list[XGBoostFeatureImportance]


class ModelInfoResponse(ApiModel):
    task_name: str
    task_description: str
    selected_model: str
    models_evaluated: list[str]
    architecture: list[str]
    feature_columns: list[str]
    preprocessing: PreprocessingSummary
    dataset: DatasetSummary
    split_methodology: SplitSummary
    selection: ModelSelectionSummary
    validation_metrics: dict[str, ModelMetrics]
    untouched_test_metrics: dict[str, ModelMetrics]
    out_of_fold_metrics: ModelMetrics
    calibration: TemperatureScalingSummary
    xgboost: XGBoostSummary
    oof_fold_count: int
    production: ProductionModelSummary
    methodology: ModelMethodology


class XGCorpusEntry(ApiModel):
    competition_id: int
    season_id: int
    matches: int
    shots: int
    goals: int


class XGDatasetSummary(ApiModel):
    matches: int
    shots: int
    goals: int
    penalties: int
    penalty_shootout_shots: int
    eligible_non_penalty_shots: int
    eligible_non_penalty_goals: int
    corpus: list[XGCorpusEntry]


class XGSplitPartition(ApiModel):
    match_count: int
    shot_count: int
    goal_count: int
    goal_rate: float


class XGSplitSummary(ApiModel):
    method: str
    random_seed: int
    train: XGSplitPartition
    validation: XGSplitPartition
    test: XGSplitPartition


class XGCalibrationSummary(ApiModel):
    method: str
    fit_split: str
    temperature: float
    retained: bool
    retention_rule: str
    reason: str
    uncalibrated_validation_metrics: ModelMetrics
    calibrated_validation_metrics: ModelMetrics


class XGSelectionSummary(ApiModel):
    primary_metric: str
    secondary_metrics: list[str]
    reason: str
    test_metrics_used: bool


class XGPreprocessingSummary(ApiModel):
    numeric: str
    boolean: str
    categorical: str
    evaluation_fit_split: str
    oof_fit_policy: str
    encoded_feature_count: int


class XGOOFSummary(ApiModel):
    fold_count: int
    prediction_count: int
    unique_shots: int
    missing_predictions: int
    duplicate_shots: int
    probability_min: float
    probability_max: float
    group_integrity: bool


class XGModelInfoResponse(ApiModel):
    task_name: str
    selected_model: str
    feature_columns: list[str]
    dataset: XGDatasetSummary
    preprocessing: XGPreprocessingSummary
    split_methodology: XGSplitSummary
    selection: XGSelectionSummary
    validation_metrics: dict[str, ModelMetrics]
    untouched_test_metrics: dict[str, ModelMetrics]
    out_of_fold_metrics: ModelMetrics
    calibration: XGCalibrationSummary
    selected_parameters: dict[str, str | int | float]
    selected_boosting_rounds: int | None
    oof: XGOOFSummary
    penalty_policy: str
    feature_importance: list[XGBoostFeatureImportance]


class MetaResponse(ApiModel):
    teams: list[str]
    position_groups: list[str]
    player_count: int
    reliable_player_count: int


class TeamStyleResponse(ApiModel):
    team_id: int
    team_name: str
    methodology_version: str
    sample_scope: str
    matches_observed: int
    contributors: int
    passes: int
    carries: int
    actions: int
    shots: int
    metrics: dict[str, float]


class TeamRoleContributorResponse(ApiModel):
    player_id: int
    player_name: str
    actions: int
    action_share: float = Field(ge=0, le=1)


class TeamRoleDimensionResponse(ApiModel):
    feature_name: str
    label: str
    raw_value: float
    position_z: float


class TeamRoleResponse(ApiModel):
    team_id: int
    team_name: str
    position_group: str
    methodology_version: str
    aggregation_method: str
    matches_observed: int
    contributor_count: int
    contributors: list[TeamRoleContributorResponse]
    passes: int
    carries: int
    actions: int
    shots: int
    support_level: str
    support_message: str
    dimensions: list[TeamRoleDimensionResponse]
    position_context: str


class TeamRolesResponse(ApiModel):
    team_id: int
    team_name: str
    roles: list[TeamRoleResponse]


class TeamIntelligenceResponse(ApiModel):
    team: TeamStyleResponse
    roles: list[TeamRoleResponse]
    fit_definition: str


class PlayerRoleFitResponse(ApiModel):
    player: PlayerIdentity
    available: bool
    unavailable_reason: str | None
    target_team_id: int
    target_team_name: str
    position_group: str
    is_target_team_player: bool
    calculation_scope: str | None
    role_distance: float | None = Field(default=None, ge=0)
    closest_dimensions: list[str]
    largest_difference: str | None
    feature_gaps: dict[str, float]
    distance_contributions: dict[str, float]
    player_matches_observed: int
    sample_support: str | None
    sample_support_message: str | None
    role_matches_observed: int | None
    role_contributor_count: int | None
    role_actions: int | None
    role_support_message: str | None
    methodology_version: str
    interpretation: str


class ScoutingRecommendationResponse(ApiModel):
    rank: int
    player: PlayerIdentity
    role_distance: float = Field(ge=0)
    closest_dimensions: list[str]
    largest_difference: str
    sample_support: str
    sample_support_message: str
    player_matches_observed: int
    archetype_name: str | None


class ScoutingRecommendationsResponse(ApiModel):
    target_team_id: int
    target_team_name: str
    position_group: str
    methodology_version: str
    definition: str
    disclaimer: str
    role_support_message: str
    total: int
    limit: int
    items: list[ScoutingRecommendationResponse]
