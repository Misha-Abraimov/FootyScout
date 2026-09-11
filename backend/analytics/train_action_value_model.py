"""Train, evaluate, OOF-score, and persist FootyScout attacking state value."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from analytics.action_value_features import (
    ACTION_VALUE_FEATURE_COLUMNS,
    TARGET_COLUMN,
    horizon_audit,
    save_parquet,
)
from analytics.action_value_model import (
    REGRESSOR_CANDIDATES,
    build_preprocessor,
    build_regressor,
    config_dict,
    evaluate_values,
    feature_frame,
    fit_hurdle,
    grouped_folds,
    nonnegative,
    predict_hurdle,
)
from analytics.nested_xg_labels import (
    FrozenXGSpec,
    apply_nested_state_targets,
    load_frozen_xg_spec,
    nested_xg_predictions,
)
from analytics.pass_features import is_progressive_pass
from analytics.player_attacking_profiles import build_player_attacking_profiles

ROOT = Path(__file__).resolve().parents[2]
STATES_PATH = ROOT / "data" / "processed" / "possession_states.parquet"
EVENTS_PATH = ROOT / "data" / "processed" / "events.parquet"
SHOTS_PATH = ROOT / "data" / "processed" / "shots.parquet"
PASS_OOF_PATH = ROOT / "data" / "processed" / "pass_oof_predictions.parquet"
PASS_FEATURES_PATH = ROOT / "data" / "processed" / "pass_features.parquet"
STATE_OOF_PATH = ROOT / "data" / "processed" / "state_oof_predictions.parquet"
ACTIONS_PATH = ROOT / "data" / "processed" / "attacking_actions.parquet"
PROFILES_PATH = ROOT / "data" / "processed" / "player_attacking_profiles.parquet"
MODEL_PATH = ROOT / "models" / "action_value_model.json"
CLASSIFIER_PATH = ROOT / "models" / "action_value_hurdle_classifier.json"
PREPROCESSOR_PATH = ROOT / "models" / "action_value_preprocessor.joblib"
METADATA_PATH = ROOT / "models" / "action_value_model_metadata.json"
XG_METADATA_PATH = ROOT / "models" / "xg_model_metadata.json"
NESTED_XG_CACHE_DIR = ROOT / "data" / "processed" / "action_value_xg_cache"

CORPUS_NAMES = {
    (9, 281): "Germany — 1. Bundesliga 2023/24",
    (43, 106): "FIFA World Cup 2022",
    (55, 282): "UEFA Euro 2024",
    (223, 282): "Copa América 2024",
    (1267, 107): "African Cup of Nations 2023",
}


def training_corpus(states: pd.DataFrame) -> dict[str, Any]:
    """Describe the audited full-event corpus without exposing split match IDs."""
    events = pd.read_parquet(
        EVENTS_PATH,
        columns=["match_id", "competition_id", "season_id"],
    )
    competitions = []
    for (competition_id, season_id), group in events.groupby(
        ["competition_id", "season_id"], sort=True
    ):
        key = (int(competition_id), int(season_id))
        competitions.append(
            {
                "name": CORPUS_NAMES[key],
                "competition_id": key[0],
                "season_id": key[1],
                "matches": int(group["match_id"].nunique()),
                "events": len(group),
            }
        )
    return {
        "matches": int(events["match_id"].nunique()),
        "events": len(events),
        "states": len(states),
        "possessions": int(states[["match_id", "possession_id"]].drop_duplicates().shape[0]),
        "competitions": competitions,
    }


def fixed_split(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reuse the frozen V2.1 match allocation exactly."""
    metadata = json.loads(XG_METADATA_PATH.read_text(encoding="utf-8"))
    split = metadata["split"]
    train_ids = set(split["train_match_ids"])
    validation_ids = set(split["validation_match_ids"])
    test_ids = set(split["test_match_ids"])
    corpus = set(states["match_id"].astype(int).unique())
    if train_ids | validation_ids | test_ids != corpus:
        raise ValueError("State corpus does not exactly match the frozen V2.1 split IDs")
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Fixed match splits overlap")
    return (
        states.loc[states.match_id.isin(train_ids)].copy(),
        states.loc[states.match_id.isin(validation_ids)].copy(),
        states.loc[states.match_id.isin(test_ids)].copy(),
        {
            "method": "reused frozen V2.1 match-grouped 80/10/10 allocation",
            "train_match_ids": sorted(train_ids),
            "validation_match_ids": sorted(validation_ids),
            "test_match_ids": sorted(test_ids),
        },
    )


def _fit_selected(
    model_name: str,
    config_index: int,
    rounds: dict[str, int],
    matrix: np.ndarray,
    target: np.ndarray,
) -> Any:
    if model_name == "hurdle":
        classifier, regressor, _, _ = fit_hurdle(
            matrix,
            target,
            classifier_rounds=rounds["classifier"],
            regressor_rounds=rounds["regressor"],
        )
        return classifier, regressor
    return build_regressor(REGRESSOR_CANDIDATES[config_index], rounds["regressor"]).fit(
        matrix, target, verbose=False
    )


def select_model_by_validation(validation_metrics: dict[str, dict[str, Any]]) -> str:
    """Select solely by the predeclared validation RMSE criterion."""
    if not validation_metrics:
        raise ValueError("At least one validation result is required")
    return min(validation_metrics, key=lambda name: validation_metrics[name]["rmse"])


def _predict(model_name: str, model: Any, matrix: np.ndarray) -> np.ndarray:
    if model_name == "hurdle":
        return predict_hurdle(model[0], model[1], matrix)
    return nonnegative(model.predict(matrix))


def generate_nested_oof(
    states: pd.DataFrame,
    shots: pd.DataFrame,
    xg_spec: FrozenXGSpec,
    model_name: str,
    config_index: int,
    rounds: dict[str, int],
) -> pd.DataFrame:
    output = states[
        ["event_id", "match_id", "possession_id", "event_index"]
    ].copy()
    output["observed_future_oof_xg"] = np.nan
    output["predicted_state_value"] = np.nan
    output["fold"] = 0
    for fold, (train_index, heldout_index) in enumerate(grouped_folds(states), start=1):
        train_base = states.iloc[train_index]
        heldout_base = states.iloc[heldout_index]
        train_ids = set(train_base["match_id"].astype(int))
        heldout_ids = set(heldout_base["match_id"].astype(int))
        if train_ids.intersection(heldout_ids):
            raise AssertionError(f"Fold {fold} contains match leakage")
        nested_xg = nested_xg_predictions(
            shots,
            train_ids,
            heldout_ids,
            xg_spec,
            outer_name=f"state_oof_fold_{fold}",
            cache_path=NESTED_XG_CACHE_DIR / f"state_oof_fold_{fold}.parquet",
        )
        train = apply_nested_state_targets(
            train_base,
            nested_xg.loc[nested_xg["match_id"].isin(train_ids)],
        )
        heldout = apply_nested_state_targets(
            heldout_base,
            nested_xg.loc[nested_xg["match_id"].isin(heldout_ids)],
        )
        preprocessor = build_preprocessor()
        train_matrix = preprocessor.fit_transform(feature_frame(train))
        heldout_matrix = preprocessor.transform(feature_frame(heldout))
        model = _fit_selected(
            model_name,
            config_index,
            rounds,
            train_matrix,
            train[TARGET_COLUMN].to_numpy(dtype=float),
        )
        prediction = _predict(model_name, model, heldout_matrix)
        output.loc[output.index[heldout_index], "predicted_state_value"] = prediction
        output.loc[output.index[heldout_index], "observed_future_oof_xg"] = heldout[
            TARGET_COLUMN
        ].to_numpy()
        output.loc[output.index[heldout_index], "fold"] = fold
    if output.event_id.duplicated().any() or len(output) != len(states):
        raise AssertionError("Every state must receive exactly one OOF prediction")
    if output.predicted_state_value.isna().any() or not np.isfinite(output.predicted_state_value).all():
        raise AssertionError("OOF state predictions must be complete and finite")
    if output.predicted_state_value.lt(0).any():
        raise AssertionError("OOF state predictions must be non-negative")
    output["fold"] = output["fold"].astype(int)
    return output


def _validated_product_xpass(states: pd.DataFrame) -> pd.DataFrame:
    pass_oof = pd.read_parquet(PASS_OOF_PATH).sort_values("pass_index").reset_index(drop=True)
    features = pd.read_parquet(PASS_FEATURES_PATH).reset_index(drop=True)
    if len(pass_oof) != len(features) or not pass_oof.pass_index.equals(
        pd.Series(np.arange(len(pass_oof)), name="pass_index")
    ):
        raise ValueError("Frozen xPass artifacts are no longer positionally aligned")
    source = features[["match_id", "player_id", "start_x", "start_y", "end_x", "end_y"]].copy()
    source["pass_index"] = pass_oof["pass_index"].to_numpy()
    source["pass_ordinal"] = source.groupby("match_id").cumcount()
    source["expected_completion"] = pass_oof["expected_completion"].to_numpy()
    product_passes = states.loc[
        states.is_product_cohort & states.event_type.eq("Pass"),
        ["event_id", "match_id", "pass_ordinal", "player_id", "start_x", "start_y", "end_x", "end_y"],
    ].copy()
    merged = product_passes.merge(
        source,
        on=["match_id", "pass_ordinal"],
        how="left",
        suffixes=("_event", "_xpass"),
        validate="one_to_one",
    )
    if merged.expected_completion.isna().any() or len(merged) != len(product_passes):
        raise ValueError("Every eligible product pass must map to one frozen xPass prediction")
    for column in ["start_x", "start_y", "end_x", "end_y"]:
        if not np.allclose(
            merged[f"{column}_event"].astype(float),
            merged[f"{column}_xpass"].astype(float),
            equal_nan=True,
        ):
            raise ValueError(f"xPass event alignment failed for {column}")
    return merged[["event_id", "pass_index", "expected_completion"]]


def build_attacking_actions(
    states: pd.DataFrame,
    oof: pd.DataFrame,
    xpass: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = states.merge(
        oof[["event_id", "predicted_state_value", "fold"]],
        on="event_id",
        validate="one_to_one",
    ).sort_values(["match_id", "possession_id", "event_index"], kind="stable")
    group = merged.groupby(["match_id", "possession_id"], sort=False)
    merged["next_state_value"] = group["predicted_state_value"].shift(-1)
    merged["next_fold"] = group["fold"].shift(-1)
    action_mask = merged.event_type.isin(["Pass", "Carry"]) & merged.end_x.notna() & merged.end_y.notna()
    actions = merged.loc[action_mask].copy()
    continues = actions.event_success & actions.next_state_value.notna()
    actions["state_value_before"] = actions.predicted_state_value
    actions["state_value_after"] = np.where(continues, actions.next_state_value, 0.0)
    actions["attacking_value"] = actions.state_value_after - actions.state_value_before
    actions["success"] = actions.event_success.astype(bool)
    actions["progressive"] = [
        is_progressive_pass(
            np.hypot(120 - float(sx), 40 - float(sy)),
            np.hypot(120 - float(ex), 40 - float(ey)),
        )
        for sx, sy, ex, ey in zip(
            actions.start_x, actions.start_y, actions.end_x, actions.end_y, strict=True
        )
    ]
    if not (actions.loc[continues, "fold"] == actions.loc[continues, "next_fold"]).all():
        raise AssertionError("Before and after states must come from the same held-out fold")
    actions["expected_completion"] = np.nan
    if xpass is None:
        xpass = _validated_product_xpass(states)
    actions = actions.merge(xpass, on="event_id", how="left", suffixes=("", "_joined"), validate="one_to_one")
    actions["expected_completion"] = actions.expected_completion_joined.combine_first(
        actions.expected_completion
    )
    actions = actions.drop(columns=["expected_completion_joined"])
    product = actions.loc[actions.is_product_cohort].copy()
    product["pass_risk"] = np.where(
        product.event_type.eq("Pass"), 1 - product.expected_completion, np.nan
    )
    pass_rows = product.event_type.eq("Pass")
    risk_threshold = float(product.loc[pass_rows, "pass_risk"].median())
    reward_threshold = float(product.loc[pass_rows, "attacking_value"].median())
    product["risk_reward_category"] = pd.NA
    high_risk = product.pass_risk.ge(risk_threshold)
    high_reward = product.attacking_value.ge(reward_threshold)
    product.loc[pass_rows & high_risk & high_reward, "risk_reward_category"] = "difficult_high_value"
    product.loc[pass_rows & high_risk & ~high_reward, "risk_reward_category"] = "difficult_low_value"
    product.loc[pass_rows & ~high_risk & high_reward, "risk_reward_category"] = "routine_high_value"
    product.loc[pass_rows & ~high_risk & ~high_reward, "risk_reward_category"] = "routine_low_value"
    output = product.rename(columns={"event_id": "action_id", "event_type": "action_type"})
    columns = [
        "action_id", "pass_index", "match_id", "possession_id", "event_index", "player_id", "team_id",
        "action_type", "start_x", "start_y", "end_x", "end_y", "state_value_before",
        "state_value_after", "attacking_value", "success", "under_pressure", "progressive",
        "expected_completion", "pass_risk", "risk_reward_category", "fold",
    ]
    return output[columns].reset_index(drop=True)


def target_distribution(states: pd.DataFrame) -> dict[str, Any]:
    target = states[TARGET_COLUMN].astype(float)
    summary: dict[str, Any] = {
        "total_states": len(states),
        "total_possessions": states[["match_id", "possession_id"]].drop_duplicates().shape[0],
        "zero_target_percentage": float(target.eq(0).mean()),
        "positive_target_states": int(target.gt(0).sum()),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "standard_deviation": float(target.std()),
        "p90": float(target.quantile(0.90)),
        "p95": float(target.quantile(0.95)),
        "p99": float(target.quantile(0.99)),
        "maximum": float(target.max()),
    }
    thirds = pd.cut(states.ball_x, [-np.inf, 40, 80, np.inf], labels=["defensive", "middle", "attacking"], right=False)
    summary["by_pitch_third"] = _distribution_groups(states.assign(group=thirds), "group")
    summary["by_event_type"] = _distribution_groups(states.assign(group=states.event_type), "group")
    action_band = pd.cut(states.possession_action_number, [0,1,3,5,10,np.inf], labels=["1","2-3","4-5","6-10","11+"])
    summary["by_possession_action_number"] = _distribution_groups(states.assign(group=action_band), "group")
    return summary


def _distribution_groups(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    rows = []
    for name, group in frame.groupby(column, observed=False):
        values = group[TARGET_COLUMN].astype(float)
        rows.append({"group": str(name), "states": len(group), "positive_rate": float(values.gt(0).mean()), "mean_target": float(values.mean())})
    return rows


def run() -> dict[str, Any]:
    states = pd.read_parquet(STATES_PATH)
    shots = pd.read_parquet(SHOTS_PATH)
    train_base, validation_base, test_base, split_metadata = fixed_split(states)
    xg_spec = load_frozen_xg_spec(XG_METADATA_PATH)
    train_ids = set(train_base["match_id"].astype(int))
    validation_ids = set(validation_base["match_id"].astype(int))
    test_ids = set(test_base["match_id"].astype(int))
    validation_nested_xg = nested_xg_predictions(
        shots,
        train_ids,
        validation_ids,
        xg_spec,
        outer_name="fixed_validation",
        cache_path=NESTED_XG_CACHE_DIR / "fixed_validation.parquet",
    )
    train = apply_nested_state_targets(
        train_base,
        validation_nested_xg.loc[validation_nested_xg["match_id"].isin(train_ids)],
    )
    validation = apply_nested_state_targets(
        validation_base,
        validation_nested_xg.loc[
            validation_nested_xg["match_id"].isin(validation_ids)
        ],
    )
    preprocessor = build_preprocessor()
    train_x = preprocessor.fit_transform(feature_frame(train))
    validation_x = preprocessor.transform(feature_frame(validation))
    train_y = train[TARGET_COLUMN].to_numpy(dtype=float)
    validation_y = validation[TARGET_COLUMN].to_numpy(dtype=float)

    baseline_value = float(train_y.mean())
    validation_results: dict[str, Any] = {
        "global_mean": evaluate_values(validation_y, np.full(len(validation_y), baseline_value))
    }
    candidates: list[dict[str, Any]] = []
    fitted: list[XGBRegressor] = []
    for index, config in enumerate(REGRESSOR_CANDIDATES, start=1):
        model = build_regressor(config)
        model.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
        rounds = int(model.best_iteration) + 1
        metrics = evaluate_values(validation_y, model.predict(validation_x))
        name = f"xgboost_{config.objective.replace(':', '_')}"
        validation_results[name] = metrics
        candidates.append({"name": name, "parameters": config_dict(config), "boosting_rounds": rounds, "metrics": metrics})
        fitted.append(model)

    zero_rate = float(states[TARGET_COLUMN].eq(0).mean())
    hurdle = None
    hurdle_rounds = {"classifier": 0, "regressor": 0}
    if zero_rate >= 0.75:
        classifier, positive_regressor, classifier_rounds, regressor_rounds = fit_hurdle(
            train_x, train_y, validation_x, validation_y
        )
        hurdle = (classifier, positive_regressor)
        hurdle_rounds = {"classifier": classifier_rounds, "regressor": regressor_rounds}
        validation_results["hurdle"] = evaluate_values(
            validation_y, predict_hurdle(classifier, positive_regressor, validation_x)
        )

    selected_name = select_model_by_validation(validation_results)
    if selected_name == "global_mean":
        raise RuntimeError("Global mean unexpectedly won; investigate before producing actions")
    selected_index = next(
        (
            index
            for index, candidate in enumerate(candidates)
            if candidate["name"] == selected_name
        ),
        0,
    )
    rounds = (
        hurdle_rounds
        if selected_name == "hurdle"
        else {"regressor": candidates[selected_index]["boosting_rounds"]}
    )
    non_test_ids = train_ids | validation_ids
    test_nested_xg = nested_xg_predictions(
        shots,
        non_test_ids,
        test_ids,
        xg_spec,
        outer_name="fixed_untouched_test",
        cache_path=NESTED_XG_CACHE_DIR / "fixed_untouched_test.parquet",
    )
    non_test_base = states.loc[states["match_id"].isin(non_test_ids)]
    non_test = apply_nested_state_targets(
        non_test_base,
        test_nested_xg.loc[test_nested_xg["match_id"].isin(non_test_ids)],
    )
    test = apply_nested_state_targets(
        test_base,
        test_nested_xg.loc[test_nested_xg["match_id"].isin(test_ids)],
    )
    test_preprocessor = build_preprocessor()
    non_test_x = test_preprocessor.fit_transform(feature_frame(non_test))
    test_x = test_preprocessor.transform(feature_frame(test))
    non_test_y = non_test[TARGET_COLUMN].to_numpy(dtype=float)
    test_y = test[TARGET_COLUMN].to_numpy(dtype=float)
    test_model = _fit_selected(
        selected_name,
        selected_index,
        rounds,
        non_test_x,
        non_test_y,
    )
    test_predictions = _predict(selected_name, test_model, test_x)
    test_baseline_value = float(non_test_y.mean())
    test_results = {
        "global_mean": evaluate_values(
            test_y, np.full(len(test_y), test_baseline_value)
        ),
        "selected_model": evaluate_values(test_y, test_predictions),
    }

    oof = generate_nested_oof(
        states,
        shots,
        xg_spec,
        selected_name,
        selected_index,
        rounds,
    )
    save_parquet(oof, STATE_OOF_PATH)
    oof_results = evaluate_values(
        oof.observed_future_oof_xg.to_numpy(dtype=float),
        oof.predicted_state_value.to_numpy(dtype=float),
    )
    actions = build_attacking_actions(states, oof)
    profiles, reliability = build_player_attacking_profiles(actions)
    save_parquet(actions, ACTIONS_PATH)
    save_parquet(profiles, PROFILES_PATH)

    final_preprocessor = build_preprocessor()
    full_x = final_preprocessor.fit_transform(feature_frame(states))
    final_model = _fit_selected(
        selected_name,
        selected_index,
        rounds,
        full_x,
        states[TARGET_COLUMN].to_numpy(dtype=float),
    )
    joblib.dump(final_preprocessor, PREPROCESSOR_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if selected_name == "hurdle":
        final_model[0].save_model(CLASSIFIER_PATH)
        final_model[1].save_model(MODEL_PATH)
    else:
        final_model.save_model(MODEL_PATH)

    metadata = {
        "task_name": "Attacking state value",
        "target": TARGET_COLUMN,
        "target_interpretation": "sum of later FootyScout OOF xG in the same possession; an expected-goal quantity, not a scoring probability",
        "state_convention": "state immediately before an on-ball event; the current event has not occurred and belongs to the future",
        "selected_horizon": "all remaining eligible shots in the same possession",
        "horizon_candidates": horizon_audit(states),
        "features": ACTION_VALUE_FEATURE_COLUMNS,
        "leakage_protection": "OOF xG only for labels; no future fields, outcomes, provider xG, identities, or next actions in model inputs",
        "preprocessing": {"numeric": "median imputation and standard scaling", "boolean": "most-frequent imputation", "categorical": "Unknown fill and one-hot encoding with unseen values ignored", "encoded_feature_count": int(full_x.shape[1])},
        "split": {**split_metadata, "train_states": len(train), "validation_states": len(validation), "test_states": len(test)},
        "training_corpus": training_corpus(states),
        "target_distribution": target_distribution(states),
        "nested_cross_fitting": {
            "upstream_model": "frozen V2.1 FootyScout xG specification",
            "upstream_model_selection_changed": False,
            "validation": "inner grouped xG OOF on fixed train matches; held-out validation xG predicted from fixed train only",
            "test": "inner grouped xG OOF on all non-test matches; held-out test xG predicted from non-test only",
            "state_oof": "five outer match folds; inner grouped xG OOF for outer train and outer-train-only xG for outer holdout",
            "heldout_outcomes_used_for_training": False,
            "cache_directory": str(NESTED_XG_CACHE_DIR.relative_to(ROOT)),
        },
        "baseline": {"validation_training_global_mean": baseline_value, "test_non_test_global_mean": test_baseline_value, "validation": validation_results["global_mean"], "test": test_results["global_mean"]},
        "candidates": candidates,
        "hurdle": {"evaluated": hurdle is not None, "reason": f"evaluated because zero-target rate was {zero_rate:.2%}", "rounds": hurdle_rounds, "validation_metrics": validation_results.get("hurdle")},
        "selection": {"model": selected_name, "objective": "two-stage" if selected_name == "hurdle" else REGRESSOR_CANDIDATES[selected_index].objective, "primary_metric": "validation_rmse", "reason": "lowest validation RMSE among the global mean, direct XGBoost objectives, and justified hurdle candidate", "test_metrics_used": False, "rounds": rounds},
        "validation_metrics": validation_results,
        "untouched_test_metrics": test_results,
        "out_of_fold_metrics": oof_results,
        "oof": {"fold_count": 5, "prediction_count": len(oof), "unique_states": int(oof.event_id.nunique()), "missing_predictions": int(oof.predicted_state_value.isna().sum()), "duplicate_states": int(oof.event_id.duplicated().sum()), "prediction_min": float(oof.predicted_state_value.min()), "prediction_max": float(oof.predicted_state_value.max()), "group_integrity": True},
        "actions": {"product_actions": len(actions), "passes": int(actions.action_type.eq("Pass").sum()), "carries": int(actions.action_type.eq("Carry").sum()), "risk_threshold": float(actions.loc[actions.action_type.eq("Pass"), "pass_risk"].median()), "reward_threshold": float(actions.loc[actions.action_type.eq("Pass"), "attacking_value"].median())},
        "player_profiles": {"count": len(profiles), **reliability},
        "transition_rules": {"successful_pass_or_carry": "next same-possession pre-event state from the same held-out model", "failed_or_ending_action": "zero for the original team", "opponent_possession": "never used as the after-state"},
        "limitations": ["observational/model-derived, not causal", "on-ball attacking possession value only", "off-ball movement is weakly represented", "defensive value is outside V2.2", "FootyScout xG has documented upper-tail compression", "sparse rare states are less certain", "product profiles cover the available Bundesliga 2023/24 cohort"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"selection": metadata["selection"], "split": metadata["split"], "target": metadata["target_distribution"], "oof": metadata["out_of_fold_metrics"], "actions": metadata["actions"], "profiles": metadata["player_profiles"]}, indent=2))
    return metadata


if __name__ == "__main__":
    run()
