import type { ModelInfoResponse } from "@/lib/types";

const logisticValidation = {
  roc_auc: 0.8875629163559875,
  log_loss: 0.3117504545850485,
  brier_score: 0.0989089551707705,
  accuracy: 0.8532690569448629,
  expected_calibration_error: 0.032613520447920194,
};

const mlpUncalibratedValidation = {
  roc_auc: 0.9159341984600109,
  log_loss: 0.2701541682479213,
  brier_score: 0.08499338124737954,
  accuracy: 0.8755649291955409,
  expected_calibration_error: 0.013521299485603246,
};

const mlpEffectiveValidation = {
  roc_auc: 0.9159341984600109,
  log_loss: 0.2696740307484722,
  brier_score: 0.08496100417313532,
  accuracy: 0.8755649291955409,
  expected_calibration_error: 0.010925198198891251,
};

const selectedTest = {
  roc_auc: 0.9017512160513677,
  log_loss: 0.25457277609389467,
  brier_score: 0.07475449343286988,
  accuracy: 0.8974414104493658,
  expected_calibration_error: 0.018107230744421332,
};

/** Fixture copied from the public shape produced by GET /api/model. */
export const modelInfoFixture: ModelInfoResponse = {
  task_name: "Expected pass completion",
  task_description:
    "Supervised binary classification of whether a pass is completed, returning an expected-completion probability from pre-outcome features.",
  selected_model: "xgboost",
  models_evaluated: ["logistic_regression", "pytorch_mlp", "xgboost"],
  architecture: [
    "XGBClassifier (315 boosting rounds)",
    "max_depth=4",
    "learning_rate=0.05",
    "objective=binary:logistic · tree_method=hist",
  ],
  feature_columns: [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "pass_length",
    "pass_angle",
    "forward_distance",
    "lateral_distance",
    "distance_to_goal_before",
    "distance_to_goal_after",
    "distance_toward_goal",
    "under_pressure",
    "pass_height",
    "body_part",
    "pass_type",
    "start_zone",
    "end_zone",
    "progressive",
  ],
  preprocessing: {
    evaluation_fit_split: "train only",
    oof_fit_policy: "fit separately inside each outer training fold",
    numeric: "median imputation then StandardScaler",
    boolean: "explicit 0/1 conversion then most-frequent imputation",
    categorical:
      "most-frequent imputation then OneHotEncoder(handle_unknown='ignore', sparse_output=False)",
    encoded_feature_count: 50,
  },
  dataset: {
    pass_count: 39214,
    match_count: 34,
    completion_rate: 0.8500025501096548,
  },
  split_methodology: {
    method: "grouped by match_id; whole matches; approximately 80/10/10",
    random_seed: 42,
    train: { match_count: 27, pass_count: 31244, completion_rate: 0.8531558059147356 },
    validation: { match_count: 3, pass_count: 3319, completion_rate: 0.8189213618559807 },
    test: { match_count: 4, pass_count: 4651, completion_rate: 0.8509997849924748 },
  },
  selection: {
    model: "xgboost",
    source: "validation metrics only",
    reason: "lower validation log loss by 0.008164 versus pytorch_mlp",
    primary_metric: "validation log loss",
    tie_breaker: "validation Brier score",
    supporting_metric: "validation ROC-AUC",
    test_metrics_used: false,
  },
  validation_metrics: {
    logistic_regression: logisticValidation,
    pytorch_mlp_uncalibrated: mlpUncalibratedValidation,
    pytorch_mlp_effective: mlpEffectiveValidation,
    xgboost_uncalibrated: {
      roc_auc: 0.920994442669135,
      log_loss: 0.26150971182908933,
      brier_score: 0.08101450894377414,
      accuracy: 0.8840012051822839,
      expected_calibration_error: 0.010204766953319663,
    },
    xgboost_effective: {
      roc_auc: 0.920994442669135,
      log_loss: 0.26150971182908933,
      brier_score: 0.08101450894377414,
      accuracy: 0.8840012051822839,
      expected_calibration_error: 0.010204766953319663,
    },
  },
  untouched_test_metrics: {
    logistic_regression: {
      roc_auc: 0.8786334798209483,
      log_loss: 0.27779373933666335,
      brier_score: 0.08244803345010615,
      accuracy: 0.889701139539884,
      expected_calibration_error: 0.0067375412111102565,
    },
    pytorch_mlp_uncalibrated: {
      roc_auc: 0.9017512160513677,
      log_loss: 0.2539883666826108,
      brier_score: 0.07496435292531953,
      accuracy: 0.8974414104493658,
      expected_calibration_error: 0.017260222633092348,
    },
    pytorch_mlp_effective: selectedTest,
    xgboost_uncalibrated: {
      roc_auc: 0.9157716266104342,
      log_loss: 0.2353630151368052,
      brier_score: 0.06908827786780011,
      accuracy: 0.9064717265104278,
      expected_calibration_error: 0.007250777055062633,
    },
    xgboost_effective: {
      roc_auc: 0.9157716266104342,
      log_loss: 0.2353630151368052,
      brier_score: 0.06908827786780011,
      accuracy: 0.9064717265104278,
      expected_calibration_error: 0.007250777055062633,
    },
    selected_model: {
      roc_auc: 0.9157716266104342,
      log_loss: 0.2353630151368052,
      brier_score: 0.06908827786780011,
      accuracy: 0.9064717265104278,
      expected_calibration_error: 0.007250777055062633,
    },
  },
  out_of_fold_metrics: {
    roc_auc: 0.9220761851555327,
    log_loss: 0.23377613072664288,
    brier_score: 0.0705766414683141,
    accuracy: 0.9022797980313153,
    expected_calibration_error: 0.0040878181998831595,
  },
  calibration: {
    method: "temperature_scaling_on_raw_margins",
    fit_split: "validation only",
    temperature: 0.9848703145980835,
    retained: false,
    reason: "not retained because validation log loss, Brier score, and ECE did not jointly satisfy the conservative improvement rule",
    uncalibrated_validation_metrics: {
      roc_auc: 0.920994442669135, log_loss: 0.26150971182908933, brier_score: 0.08101450894377414, accuracy: 0.8840012051822839, expected_calibration_error: 0.010204766953319663,
    },
    calibrated_validation_metrics: {
      roc_auc: 0.920994442669135, log_loss: 0.2614787680799094, brier_score: 0.08103043108693495, accuracy: 0.8840012051822839, expected_calibration_error: 0.0090113222168584,
    },
  },
  xgboost: {
    version: "3.4.1",
    random_seed: 42,
    selection_metric: "validation log loss",
    selected_parameters: { objective: "binary:logistic", tree_method: "hist", max_depth: 4, learning_rate: 0.05 },
    best_iteration: 314,
    boosting_rounds: 315,
    calibration: {
      method: "temperature_scaling_on_raw_margins", fit_split: "validation only", temperature: 0.9848703145980835, retained: false, reason: "Brier score did not improve", uncalibrated_validation_metrics: { roc_auc: 0.920994442669135, log_loss: 0.26150971182908933, brier_score: 0.08101450894377414, accuracy: 0.8840012051822839, expected_calibration_error: 0.010204766953319663 }, calibrated_validation_metrics: { roc_auc: 0.920994442669135, log_loss: 0.2614787680799094, brier_score: 0.08103043108693495, accuracy: 0.8840012051822839, expected_calibration_error: 0.0090113222168584 },
    },
    feature_importance_type: "gain",
    feature_importance_interpretation: "model feature importance; not a causal claim",
    feature_importance: [{ feature: "pass_height_Ground Pass", gain: 413.6780700683594, normalized_gain: 0.3411688509590019 }],
  },
  oof_fold_count: 5,
  production: {
    model: "xgboost",
    training_rows: 39214,
    encoded_feature_count: 50,
    epochs: null,
    boosting_rounds: 315,
    temperature: 1,
    temperature_source: "not used",
    evaluation_use: "none; full-data model is for future inference only",
  },
  methodology: {
    model_selection: "Model selection was performed using validation data.",
    test_set: "The test data was not used for model selection.",
    player_profiles: "Player profiles use grouped out-of-fold predictions.",
    production_model:
      "The production full-data model is for future inference and is not used for evaluation claims.",
  },
};
