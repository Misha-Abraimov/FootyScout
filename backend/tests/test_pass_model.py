from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analytics.pass_features import KNOWN_LEAKAGE_COLUMNS, MODEL_FEATURE_COLUMNS
from analytics.pass_model import (
    MODEL_BOOLEAN_COLUMNS,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    PassCompletionMLP,
    TrainingConfig,
    XGBoostConfig,
    assert_disjoint_match_ids,
    build_preprocessor,
    build_xgboost_model,
    evaluate_temperature_scaling,
    fit_preprocessor,
    grouped_match_folds,
    load_preprocessor,
    load_torch_model,
    load_torch_temperature,
    load_xgboost_model,
    predict_mlp_probabilities,
    predict_xgboost_probabilities,
    save_preprocessor,
    save_torch_model,
    save_xgboost_model,
    select_model,
    split_by_match,
    validate_model_contract,
)
from analytics.pass_oof import (
    OOF_PREDICTION_COLUMNS,
    generate_oof_predictions,
    validate_oof_predictions,
)


def sample_feature_data(match_count: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for match_id in range(1, match_count + 1):
        for pass_index in range(3):
            start_x = float(10 + match_id + pass_index)
            start_y = float(15 + pass_index)
            end_x = start_x + 20.0
            end_y = start_y + 5.0
            before = float(np.hypot(120.0 - start_x, 40.0 - start_y))
            after = float(np.hypot(120.0 - end_x, 40.0 - end_y))
            rows.append(
                {
                    "match_id": match_id,
                    "player_id": 100 + pass_index,
                    "player_name": f"Player {pass_index}",
                    "team_id": 200 + match_id % 2,
                    "team_name": f"Team {match_id % 2}",
                    "position": "Center Midfield",
                    "start_x": start_x,
                    "start_y": start_y,
                    "end_x": end_x,
                    "end_y": end_y,
                    "pass_length": float(np.hypot(20.0, 5.0)),
                    "pass_angle": 0.25,
                    "forward_distance": 20.0,
                    "lateral_distance": 5.0,
                    "distance_to_goal_before": before,
                    "distance_to_goal_after": after,
                    "distance_toward_goal": before - after,
                    "under_pressure": pass_index == 0,
                    "pass_height": "Ground Pass",
                    "body_part": "Right Foot",
                    "pass_type": "Regular",
                    "start_zone": "defensive_left",
                    "end_zone": "defensive_centre",
                    "progressive": False,
                    "completed": int(pass_index != 2),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_splits_are_deterministic_and_do_not_overlap() -> None:
    data = sample_feature_data()
    first = split_by_match(data, random_seed=42)
    second = split_by_match(data, random_seed=42)

    assert_disjoint_match_ids(first)
    assert first.train_match_ids == second.train_match_ids
    assert first.validation_match_ids == second.validation_match_ids
    assert first.test_match_ids == second.test_match_ids
    assert len(first.train) + len(first.validation) + len(first.test) == len(data)


def test_grouped_oof_folds_never_overlap_and_cover_every_row_once() -> None:
    data = sample_feature_data(match_count=15)
    folds = grouped_match_folds(data, n_splits=5, random_seed=42)
    heldout_indices: list[int] = []

    for fold in folds:
        assert not set(fold.train_match_ids).intersection(fold.heldout_match_ids)
        heldout_indices.extend(fold.heldout_indices.tolist())

    assert len(folds) == 5
    assert sorted(heldout_indices) == list(range(len(data)))
    assert len(heldout_indices) == len(set(heldout_indices))


def test_preprocessor_fits_training_data_and_ignores_unseen_categories() -> None:
    splits = split_by_match(sample_feature_data(), random_seed=42)
    splits.validation.loc[:, "pass_height"] = "Unseen Validation Height"
    preprocessor = build_preprocessor()

    transformed = fit_preprocessor(preprocessor, splits)

    assert transformed.train.shape[0] == len(splits.train)
    assert transformed.validation.shape[0] == len(splits.validation)
    assert transformed.test.shape[0] == len(splits.test)
    assert transformed.train.shape[1] == transformed.validation.shape[1]
    height_categories = preprocessor.named_transformers_["categorical"].named_steps[
        "one_hot"
    ].categories_[0]
    assert "Unseen Validation Height" not in height_categories


def test_model_contract_excludes_prohibited_leakage_columns() -> None:
    configured = MODEL_NUMERIC_COLUMNS + MODEL_BOOLEAN_COLUMNS + MODEL_CATEGORICAL_COLUMNS

    assert set(configured) == set(MODEL_FEATURE_COLUMNS)
    assert not KNOWN_LEAKAGE_COLUMNS.intersection(configured)
    validate_model_contract()


def test_model_selection_uses_validation_metrics_not_test_metrics() -> None:
    logistic_validation = {"log_loss": 0.20, "brier_score": 0.06, "roc_auc": 0.91}
    pytorch_validation = {"log_loss": 0.25, "brier_score": 0.07, "roc_auc": 0.95}
    contradictory_test_metrics = {
        "logistic_regression": {"log_loss": 0.40, "brier_score": 0.12},
        "pytorch_mlp": {"log_loss": 0.18, "brier_score": 0.05},
    }

    selected, reason = select_model(logistic_validation, pytorch_validation)

    assert contradictory_test_metrics["pytorch_mlp"]["log_loss"] < 0.20
    assert selected == "logistic_regression"
    assert "validation" in reason


def test_xgboost_can_win_only_by_validation_policy() -> None:
    logistic = {"log_loss": 0.31, "brier_score": 0.10}
    pytorch = {"log_loss": 0.27, "brier_score": 0.085}
    xgboost = {"log_loss": 0.26, "brier_score": 0.081}

    selected, reason = select_model(logistic, pytorch, xgboost)

    assert selected == "xgboost"
    assert "validation log loss" in reason


def test_xgboost_construction_probability_bounds_and_serialization(tmp_path: Path) -> None:
    config = XGBoostConfig(3, 0.1, 8, 1.0, 0.9, 0.9, 1.0, 0.0)
    model = build_xgboost_model(config, early_stopping=False)
    features = np.array([[0.0], [1.0], [0.2], [0.8]], dtype=np.float32)
    target = np.array([0, 1, 0, 1])
    model.fit(features, target, verbose=False)
    probabilities = predict_xgboost_probabilities(model, features)
    path = tmp_path / "model.json"
    save_xgboost_model(model, path)
    loaded = load_xgboost_model(path)

    assert model.get_params()["objective"] == "binary:logistic"
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(loaded.predict_proba(features), model.predict_proba(features))


def test_xgboost_margin_temperature_calibration_is_validation_only() -> None:
    result = evaluate_temperature_scaling(
        np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32),
        np.array([0, 0, 1, 1], dtype=np.float32),
    )

    assert result.temperature > 0
    assert set(result.uncalibrated_metrics) == {
        "roc_auc", "log_loss", "brier_score", "accuracy", "expected_calibration_error"
    }


def test_pytorch_forward_shape_and_probability_bounds() -> None:
    model = PassCompletionMLP(input_dim=12)
    features = np.zeros((5, 12), dtype=np.float32)

    logits = model(torch.from_numpy(features))
    probabilities = predict_mlp_probabilities(model, features)

    assert logits.shape == (5, 1)
    assert probabilities.shape == (5,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_saved_pytorch_model_can_be_loaded(tmp_path: Path) -> None:
    model = PassCompletionMLP(input_dim=7)
    model_path = tmp_path / "model.pt"
    save_torch_model(model, model_path)

    loaded = load_torch_model(model_path)

    inputs = torch.randn(4, 7)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(inputs), loaded(inputs))


def test_final_production_artifacts_can_be_loaded(tmp_path: Path) -> None:
    splits = split_by_match(sample_feature_data(), random_seed=42)
    preprocessor = build_preprocessor()
    fit_preprocessor(preprocessor, splits)
    preprocessor_path = tmp_path / "pass_preprocessor_final.joblib"
    model_path = tmp_path / "pass_completion_model_final.pt"
    model = PassCompletionMLP(input_dim=len(preprocessor.get_feature_names_out()))
    save_preprocessor(preprocessor, preprocessor_path)
    save_torch_model(model, model_path, temperature=1.25)

    loaded_preprocessor = load_preprocessor(preprocessor_path)
    loaded_model = load_torch_model(model_path)

    assert len(loaded_preprocessor.get_feature_names_out()) == loaded_model.input_dim
    assert load_torch_temperature(model_path) == 1.25


def test_saved_preprocessor_can_be_loaded(tmp_path: Path) -> None:
    splits = split_by_match(sample_feature_data(), random_seed=42)
    preprocessor = build_preprocessor()
    transformed = fit_preprocessor(preprocessor, splits)
    preprocessor_path = tmp_path / "preprocessor.joblib"
    save_preprocessor(preprocessor, preprocessor_path)

    loaded = load_preprocessor(preprocessor_path)
    loaded_test = loaded.transform(splits.test[MODEL_FEATURE_COLUMNS])

    np.testing.assert_allclose(loaded_test, transformed.test)


def test_oof_predictions_are_complete_unique_and_bounded() -> None:
    data = sample_feature_data(match_count=10).reset_index(drop=True)
    predictions = data[
        [
            "match_id",
            "player_id",
            "player_name",
            "team_id",
            "team_name",
            "position",
            "completed",
            "under_pressure",
            "progressive",
            "pass_length",
            "forward_distance",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
    ].copy()
    predictions.insert(0, "pass_index", np.arange(len(data)))
    predictions["expected_completion"] = np.linspace(0.05, 0.95, len(data))
    predictions["fold"] = predictions["match_id"].mod(5).add(1)
    predictions = predictions[OOF_PREDICTION_COLUMNS]

    validate_oof_predictions(predictions, expected_rows=len(data))

    assert len(predictions) == len(data)
    assert predictions["pass_index"].is_unique
    assert predictions["expected_completion"].between(0.0, 1.0).all()


def test_xgboost_grouped_oof_integrity() -> None:
    data = sample_feature_data(match_count=15)
    result = generate_oof_predictions(
        data,
        selected_model="xgboost",
        training_config=TrainingConfig(),
        mlp_epochs=1,
        use_temperature_scaling=False,
        xgboost_config=XGBoostConfig(2, 0.1, 6, 1.0, 1.0, 1.0, 1.0, 0.0),
        xgboost_rounds=6,
        n_splits=5,
    )

    validate_oof_predictions(result.predictions, len(data))
    assert len(result.fold_summaries) == 5
    assert result.predictions["expected_completion"].between(0.0, 1.0).all()
