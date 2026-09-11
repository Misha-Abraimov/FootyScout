export type PositionGroup = "GK" | "DEF" | "MID" | "FWD";
export type SortOrder = "asc" | "desc";

export type PlayerSortField =
  | "player_name"
  | "pass_attempts"
  | "actual_completion_rate"
  | "expected_completion_rate"
  | "completion_above_expected_pp"
  | "progressive_pass_rate"
  | "pressure_above_expected_pp"
  | "final_third_entries_per_100_passes";

export type LeaderboardMetric =
  | "completion_above_expected_pp"
  | "pressure_above_expected_pp"
  | "progressive_above_expected_pp"
  | "long_pass_above_expected_pp"
  | "final_third_entries_per_100_passes"
  | "expected_completion_rate"
  | "progressive_pass_rate"
  | "attacking_value_per_100_actions"
  | "pass_value_per_100_passes"
  | "carry_value_per_100_carries"
  | "progressive_value_per_100_actions"
  | "pressure_value_per_100_actions";

export interface PlayerIdentity {
  player_id: number;
  player_name: string;
  team_id: number;
  team_name: string;
  position: string;
  position_group: PositionGroup;
}

export interface PlayerSummary extends PlayerIdentity {
  matches_observed: number;
  pass_attempts: number;
  actual_completion_rate: number;
  expected_completion_rate: number;
  completion_above_expected_pp: number;
  progressive_pass_rate: number;
  pressure_pass_rate: number;
  final_third_entries_per_100_passes: number;
  overall_reliable: boolean;
}

export interface PlayerListResponse {
  total: number;
  limit: number;
  offset: number;
  items: PlayerSummary[];
}

export interface PlayerProfileResponse extends PlayerIdentity {
  matches_observed: number;
  pass_attempts: number;
  overall_reliable: boolean;
  passes_completed: number;
  actual_completion_rate: number;
  expected_completions: number;
  expected_completion_rate: number;
  completions_above_expected: number;
  completion_above_expected_pp: number;
  pressure_attempts: number;
  pressure_completed: number;
  pressure_actual_completion_rate: number | null;
  pressure_expected_completion_rate: number | null;
  pressure_completions_above_expected: number | null;
  pressure_above_expected_pp: number | null;
  pressure_pass_rate: number;
  progressive_attempts: number;
  progressive_completed: number;
  progressive_actual_completion_rate: number | null;
  progressive_expected_completion_rate: number | null;
  progressive_completions_above_expected: number | null;
  progressive_above_expected_pp: number | null;
  progressive_pass_rate: number;
  long_pass_attempts: number;
  long_pass_completed: number;
  long_pass_actual_completion_rate: number | null;
  long_pass_expected_completion_rate: number | null;
  long_pass_completions_above_expected: number | null;
  long_pass_above_expected_pp: number | null;
  average_forward_distance: number;
  net_forward_distance_per_100_passes: number;
  positive_forward_distance_per_100_passes: number;
  final_third_entries: number;
  final_third_entries_per_100_passes: number;
  pressure_reliable: boolean;
  progressive_reliable: boolean;
  long_pass_reliable: boolean;
}

export interface SimilarPlayerResponse {
  similar_player_id: number;
  similar_player_name: string;
  similar_team_name: string;
  similar_position: string;
  similar_position_group: PositionGroup;
  rank: number;
  rms_distance: number;
  similarity_score: number;
  same_position_group: boolean;
  closest_feature_1: string;
  closest_feature_2: string;
  closest_feature_3: string;
  closest_style_dimensions: string[];
  query_matches_observed: number;
  candidate_matches_observed: number;
  pair_support_matches: number;
  sample_support: "limited" | "higher";
  sample_support_explanation: string;
  methodology_version: "V3.3B";
}

export interface SimilarPlayersResponse {
  source_player: PlayerIdentity;
  available: boolean;
  unavailable_reason: string | null;
  methodology_version: "V3.3B";
  query_matches_observed: number;
  query_sample_support: "limited" | "higher";
  eligibility_requirements: string[];
  total: number;
  limit: number;
  items: SimilarPlayerResponse[];
}

export interface PassResponse {
  pass_index: number;
  match_id: number;
  player_id: number | null;
  start_x: number;
  start_y: number;
  end_x: number;
  end_y: number;
  pass_length: number;
  pass_angle: number;
  forward_distance: number;
  lateral_distance: number;
  distance_to_goal_before: number;
  distance_to_goal_after: number;
  distance_toward_goal: number;
  completed: boolean;
  expected_completion: number;
  under_pressure: boolean;
  progressive: boolean;
  pass_height: string;
  body_part: string;
  pass_type: string;
  start_zone: string;
  end_zone: string;
  fold: number;
  attacking_value?: number | null;
  state_value_before?: number | null;
  state_value_after?: number | null;
  pass_risk?: number | null;
  risk_reward_category?: string | null;
}

export interface AttackingProfileResponse {
  player_id: number;
  matches_observed: number;
  actions: number;
  passes: number;
  carries: number;
  total_attacking_value: number;
  attacking_value_per_100_actions: number;
  total_pass_value: number;
  pass_value_per_100_passes: number | null;
  total_carry_value: number;
  carry_value_per_100_carries: number | null;
  positive_value_actions: number;
  positive_value_action_rate: number;
  progressive_action_value: number;
  progressive_value_per_100_actions: number;
  pressure_action_value: number;
  pressure_value_per_100_actions: number;
  attacking_value_reliable: boolean;
  pass_value_reliable: boolean;
  carry_value_reliable: boolean;
}

export interface AttackingActionResponse {
  action_id: string;
  pass_index: number | null;
  match_id: number;
  possession_id: number;
  event_index: number;
  player_id: number | null;
  team_id: number;
  action_type: "Pass" | "Carry";
  start_x: number;
  start_y: number;
  end_x: number;
  end_y: number;
  state_value_before: number;
  state_value_after: number;
  attacking_value: number;
  success: boolean;
  under_pressure: boolean;
  progressive: boolean;
  expected_completion: number | null;
  pass_risk: number | null;
  risk_reward_category: string | null;
  fold: number;
}

export interface AttackingActionListResponse {
  total: number;
  limit: number;
  offset: number;
  items: AttackingActionResponse[];
}

export interface ActionValueModelInfoResponse {
  task_name: string;
  target: string;
  target_interpretation: string;
  state_convention: string;
  selected_horizon: string;
  feature_columns: string[];
  leakage_protection: string;
  preprocessing: Record<string, unknown>;
  split_methodology: Record<string, string | number>;
  nested_cross_fitting: {
    upstream_model: string;
    upstream_model_selection_changed: boolean;
    validation: string;
    test: string;
    state_oof: string;
    heldout_outcomes_used_for_training: boolean;
    cache_directory: string;
  };
  training_corpus: {
    matches: number;
    events: number;
    states: number;
    possessions: number;
    competitions: Array<{
      name: string;
      competition_id: number;
      season_id: number;
      matches: number;
      events: number;
    }>;
  };
  target_distribution: Record<string, unknown>;
  baseline: Record<string, unknown>;
  candidates: Array<Record<string, unknown>>;
  hurdle: Record<string, unknown>;
  selection: { model: string; objective: string; primary_metric: string; reason: string; test_metrics_used: boolean; rounds: Record<string, number> };
  validation_metrics: Record<string, ValueModelMetrics>;
  untouched_test_metrics: Record<string, ValueModelMetrics>;
  out_of_fold_metrics: ValueModelMetrics;
  oof: Record<string, number | boolean>;
  transition_rules: Record<string, string>;
  limitations: string[];
}

export interface ValueModelMetrics {
  mae: number;
  rmse: number;
  r2: number;
  spearman: number;
  positive_target_mae: number;
  positive_target_rmse: number;
  prediction_min: number;
  prediction_max: number;
}

export interface PassListResponse {
  total: number;
  limit: number;
  offset: number;
  items: PassResponse[];
}

export interface ShootingProfileResponse {
  player_id: number;
  shots: number;
  goals: number;
  total_xg: number;
  xg_per_shot: number;
  goals_minus_xg: number;
  goals_per_shot: number;
  matches_observed: number;
  shooting_reliable: boolean;
}

export interface ShotResponse {
  shot_id: string;
  match_id: number;
  player_id: number | null;
  period: number;
  minute: number;
  second: number;
  start_x: number;
  start_y: number;
  distance: number;
  angle: number;
  goal: boolean;
  expected_goal: number | null;
  body_part: string;
  shot_type: string;
  technique: string;
  play_pattern: string;
  under_pressure: boolean;
  first_time: boolean;
  one_on_one: boolean;
  open_goal: boolean;
  penalty: boolean;
  penalty_shootout: boolean;
  model_eligible: boolean;
}

export interface ShotListResponse {
  total: number;
  limit: number;
  offset: number;
  items: ShotResponse[];
}

export interface ComparisonResponse {
  comparison: { player_ids: number[]; same_position_group: boolean };
  players: PlayerProfileResponse[];
  attacking: Array<AttackingProfileResponse | null>;
  intelligence: Array<PlayerIntelligenceResponse | null>;
}

export interface PlayerIntelligenceMetric {
  metric_name: string;
  label: string;
  family: "style" | "performance";
  raw_value: number | null;
  unit: string;
  percentile: number | null;
  peer_position_group: PositionGroup;
  peer_count: number;
  sample_count: number;
  minimum_sample: number;
  eligible: boolean;
  eligibility_reason: string;
  directionality: string;
  stability_note: string | null;
}

export interface ArchetypeStyleDimension {
  feature_name: string;
  label: string;
  position_z: number;
}

export interface ArchetypeDistinguishingFeature extends ArchetypeStyleDimension {
  direction: "higher" | "lower";
}

export interface PlayerArchetype {
  id: "direct_progressor" | "safe_circulator" | null;
  name: string | null;
  eligible: boolean;
  eligibility_reason: string | null;
  position_group: PositionGroup;
  centroid_distance: number | null;
  second_centroid_distance: number | null;
  separation_margin: number | null;
  separation_interpretation: string;
  style_dimensions: ArchetypeStyleDimension[];
  distinguishing_features: ArchetypeDistinguishingFeature[];
  methodology_version: string;
}

export interface PlayerIntelligenceResponse {
  player: PlayerIdentity;
  matches_observed: number;
  position_group: PositionGroup;
  style_metrics: PlayerIntelligenceMetric[];
  performance_metrics: PlayerIntelligenceMetric[];
  radar_metrics: PlayerIntelligenceMetric[];
  radar_status: string;
  percentile_context: string;
  archetype: PlayerArchetype;
}

export interface ArchetypeRepresentative {
  player_id: number;
  player_name: string;
  team_name: string;
  position_group: "DEF" | "MID" | "FWD";
  centroid_distance: number;
}

export interface ArchetypeDefinition {
  id: "direct_progressor" | "safe_circulator";
  name: string;
  description: string;
  centroid: Record<string, number>;
  distinguishing_features: ArchetypeDistinguishingFeature[];
  player_count: number;
  position_composition: Record<"DEF" | "MID" | "FWD", {
    count: number;
    percentage: number;
  }>;
  representative_players: ArchetypeRepresentative[];
  separation_distribution: Record<string, number>;
}

export interface ArchetypeCatalogueResponse {
  methodology_version: string;
  purpose: string;
  definitions: ArchetypeDefinition[];
  methodology: Record<string, unknown>;
}

export interface LeaderboardEntry extends PlayerIdentity {
  rank: number;
  pass_attempts: number;
  relevant_attempts: number;
  metric: LeaderboardMetric;
  metric_value: number;
  overall_reliable: boolean;
}

export interface LeaderboardResponse {
  metric: LeaderboardMetric;
  total: number;
  limit: number;
  minimum_pass_attempts: number;
  items: LeaderboardEntry[];
}

export interface ModelMetrics {
  roc_auc: number;
  log_loss: number;
  brier_score: number;
  accuracy: number;
  expected_calibration_error: number;
}

export interface ModelMetricGroups {
  logistic_regression?: ModelMetrics;
  pytorch_mlp_uncalibrated?: ModelMetrics;
  pytorch_mlp_effective?: ModelMetrics;
  xgboost_uncalibrated?: ModelMetrics;
  xgboost_effective?: ModelMetrics;
  selected_model?: ModelMetrics;
  [group: string]: ModelMetrics | undefined;
}

export interface ModelInfoResponse {
  task_name: string;
  task_description: string;
  selected_model: string;
  models_evaluated: string[];
  architecture: string[];
  feature_columns: string[];
  preprocessing: {
    evaluation_fit_split: string;
    oof_fit_policy: string;
    numeric: string;
    boolean: string;
    categorical: string;
    encoded_feature_count: number;
  };
  dataset: { pass_count: number; match_count: number; completion_rate: number };
  split_methodology: {
    method: string;
    random_seed: number;
    train: ModelSplit;
    validation: ModelSplit;
    test: ModelSplit;
  };
  selection: {
    model: string;
    source: string;
    reason: string;
    primary_metric: string;
    tie_breaker: string;
    supporting_metric: string;
    test_metrics_used: boolean;
  };
  validation_metrics: ModelMetricGroups;
  untouched_test_metrics: ModelMetricGroups;
  out_of_fold_metrics: ModelMetrics;
  calibration: {
    method: string;
    fit_split: string;
    temperature: number;
    retained: boolean;
    reason: string;
    uncalibrated_validation_metrics: ModelMetrics;
    calibrated_validation_metrics: ModelMetrics;
  };
  xgboost: {
    version: string;
    random_seed: number;
    selection_metric: string;
    selected_parameters: Record<string, string | number>;
    best_iteration: number;
    boosting_rounds: number;
    calibration: ModelInfoResponse["calibration"];
    feature_importance_type: string;
    feature_importance_interpretation: string;
    feature_importance: Array<{
      feature: string;
      gain: number;
      normalized_gain: number;
    }>;
  };
  oof_fold_count: number;
  production: {
    model: string;
    training_rows: number;
    encoded_feature_count: number;
    epochs: number | null;
    boosting_rounds: number | null;
    temperature: number;
    temperature_source: string;
    evaluation_use: string;
  };
  methodology: {
    model_selection: string;
    test_set: string;
    player_profiles: string;
    production_model: string;
  };
}

export interface ModelSplit {
  match_count: number;
  pass_count: number;
  completion_rate: number;
}

export interface XGModelInfoResponse {
  task_name: string;
  selected_model: string;
  feature_columns: string[];
  dataset: {
    matches: number;
    shots: number;
    goals: number;
    penalties: number;
    penalty_shootout_shots: number;
    eligible_non_penalty_shots: number;
    eligible_non_penalty_goals: number;
    corpus: Array<{
      competition_id: number;
      season_id: number;
      matches: number;
      shots: number;
      goals: number;
    }>;
  };
  preprocessing: {
    numeric: string;
    boolean: string;
    categorical: string;
    evaluation_fit_split: string;
    oof_fit_policy: string;
    encoded_feature_count: number;
  };
  split_methodology: {
    method: string;
    random_seed: number;
    train: XGModelSplit;
    validation: XGModelSplit;
    test: XGModelSplit;
  };
  selection: {
    primary_metric: string;
    secondary_metrics: string[];
    reason: string;
    test_metrics_used: boolean;
  };
  validation_metrics: Record<string, ModelMetrics | undefined>;
  untouched_test_metrics: Record<string, ModelMetrics | undefined>;
  out_of_fold_metrics: ModelMetrics;
  calibration: {
    method: string;
    fit_split: string;
    temperature: number;
    retained: boolean;
    retention_rule: string;
    reason: string;
    uncalibrated_validation_metrics: ModelMetrics;
    calibrated_validation_metrics: ModelMetrics;
  };
  selected_parameters: Record<string, string | number>;
  selected_boosting_rounds: number | null;
  oof: {
    fold_count: number;
    prediction_count: number;
    unique_shots: number;
    missing_predictions: number;
    duplicate_shots: number;
    probability_min: number;
    probability_max: number;
    group_integrity: boolean;
  };
  penalty_policy: string;
  feature_importance: Array<{
    feature: string;
    gain: number;
    normalized_gain: number;
  }>;
}

export interface XGModelSplit {
  match_count: number;
  shot_count: number;
  goal_count: number;
  goal_rate: number;
}

export interface MetaResponse {
  teams: string[];
  position_groups: PositionGroup[];
  player_count: number;
  reliable_player_count: number;
}

export interface PlayerQuery {
  search?: string;
  team?: string;
  position_group?: PositionGroup;
  min_pass_attempts?: number;
  sort_by?: PlayerSortField;
  sort_order?: SortOrder;
  limit?: number;
  offset?: number;
}

export interface PassQuery {
  match_id?: number;
  completed?: boolean;
  under_pressure?: boolean;
  progressive?: boolean;
  min_expected_completion?: number;
  max_expected_completion?: number;
  limit?: number;
  offset?: number;
}

export interface ShotQuery {
  match_id?: number;
  goal?: boolean;
  model_eligible?: boolean;
  limit?: number;
  offset?: number;
}

export interface AttackingActionQuery {
  action_type?: "Pass" | "Carry";
  match_id?: number;
  positive_only?: boolean;
  under_pressure?: boolean;
  progressive?: boolean;
  min_value?: number;
  max_value?: number;
  limit?: number;
  offset?: number;
}

export interface LeaderboardQuery {
  metric?: LeaderboardMetric;
  position_group?: PositionGroup;
  team?: string;
  limit?: number;
}

export interface TeamStyleResponse {
  team_id: number;
  team_name: string;
  methodology_version: string;
  sample_scope: string;
  matches_observed: number;
  contributors: number;
  passes: number;
  carries: number;
  actions: number;
  shots: number;
  metrics: Record<string, number>;
}

export interface TeamRoleContributor {
  player_id: number;
  player_name: string;
  actions: number;
  action_share: number;
}

export interface TeamRoleDimension {
  feature_name: string;
  label: string;
  raw_value: number;
  position_z: number;
}

export interface TeamRoleResponse {
  team_id: number;
  team_name: string;
  position_group: "DEF" | "MID" | "FWD";
  methodology_version: string;
  aggregation_method: string;
  matches_observed: number;
  contributor_count: number;
  contributors: TeamRoleContributor[];
  passes: number;
  carries: number;
  actions: number;
  shots: number;
  support_level: string;
  support_message: string;
  dimensions: TeamRoleDimension[];
  position_context: string;
}

export interface TeamIntelligenceResponse {
  team: TeamStyleResponse;
  roles: TeamRoleResponse[];
  fit_definition: string;
}

export interface PlayerRoleFitResponse {
  player: PlayerIdentity;
  available: boolean;
  unavailable_reason: string | null;
  target_team_id: number;
  target_team_name: string;
  position_group: PositionGroup;
  is_target_team_player: boolean;
  calculation_scope: string | null;
  role_distance: number | null;
  closest_dimensions: string[];
  largest_difference: string | null;
  feature_gaps: Record<string, number>;
  distance_contributions: Record<string, number>;
  player_matches_observed: number;
  sample_support: "limited" | "higher" | null;
  sample_support_message: string | null;
  role_matches_observed: number | null;
  role_contributor_count: number | null;
  role_actions: number | null;
  role_support_message: string | null;
  methodology_version: string;
  interpretation: string;
}

export interface ScoutingRecommendation {
  rank: number;
  player: PlayerIdentity;
  role_distance: number;
  closest_dimensions: string[];
  largest_difference: string;
  sample_support: "limited" | "higher";
  sample_support_message: string;
  player_matches_observed: number;
  archetype_name: string | null;
}

export interface ScoutingRecommendationsResponse {
  target_team_id: number;
  target_team_name: string;
  position_group: "DEF" | "MID" | "FWD";
  methodology_version: string;
  definition: string;
  disclaimer: string;
  role_support_message: string;
  total: number;
  limit: number;
  items: ScoutingRecommendation[];
}
