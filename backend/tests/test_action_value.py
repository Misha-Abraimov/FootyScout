from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics import nested_xg_labels
from analytics.action_value_features import (
    ACTION_VALUE_FEATURE_COLUMNS,
    KNOWN_ACTION_VALUE_LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    build_possession_states,
    horizon_audit,
    normalize_event,
    validate_event_order,
)
from analytics.action_value_model import build_preprocessor, grouped_folds, nonnegative
from analytics.nested_xg_labels import FrozenXGSpec, nested_xg_predictions
from analytics.player_attacking_profiles import build_player_attacking_profiles
from analytics.train_action_value_model import (
    build_attacking_actions,
    select_model_by_validation,
)
from analytics.xg_model import XGBoostConfig


def raw_event(
    event_id: str,
    index: int,
    event_type: str,
    *,
    possession: int = 1,
    team: int = 10,
    location: list[float] | None = None,
    end: list[float] | None = None,
    outcome: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    key = event_type.casefold().replace(" ", "_").replace("*", "")
    if end is not None:
        payload["end_location"] = end
    if outcome is not None:
        payload["outcome"] = {"name": outcome}
    event: dict[str, object] = {
        "id": event_id,
        "index": index,
        "match_id": 100,
        "period": 1,
        "minute": index,
        "second": 0,
        "timestamp": f"00:00:{index:02d}.000",
        "type": {"name": event_type},
        "possession": possession,
        "possession_team": {"id": 10, "name": "A"},
        "team": {"id": team, "name": "A" if team == 10 else "B"},
        "player": {"id": index, "name": f"Player {index}"},
        "play_pattern": {"name": "Regular Play"},
        "location": location or [60.0 + index, 40.0],
    }
    if payload:
        event[key] = payload
    return event


def test_event_normalization_uses_verified_pass_semantics() -> None:
    completed = normalize_event(
        raw_event("p1", 1, "Pass", end=[70, 40]),
        competition_id=9,
        season_id=281,
        product_cohort=True,
    )
    failed = normalize_event(
        raw_event("p2", 2, "Pass", end=[70, 40], outcome="Incomplete"),
        competition_id=9,
        season_id=281,
        product_cohort=True,
    )
    assert completed["event_success"] is True
    assert failed["event_success"] is False
    assert completed["event_index"] == 1


def test_pre_event_future_target_order_and_score_state() -> None:
    events = [
        raw_event("pass", 1, "Pass", end=[70, 40]),
        raw_event("carry", 2, "Carry", end=[80, 40]),
        raw_event("shot", 3, "Shot", outcome="Goal"),
        raw_event("later", 4, "Pass", possession=2, end=[65, 40]),
    ]
    frame = pd.DataFrame(
        [
            normalize_event(e, competition_id=9, season_id=281, product_cohort=True)
            for e in events
        ]
    )
    oof = pd.DataFrame({"shot_id": ["shot"], "expected_goal": [0.2]})
    states, audit = build_possession_states(frame, oof)
    assert audit["mapped_oof_shots"] == 1
    assert states.loc[states.event_id.isin(["pass", "carry", "shot"]), TARGET_COLUMN].eq(0.2).all()
    assert states.loc[states.event_id.eq("later"), TARGET_COLUMN].item() == 0
    assert states.loc[states.event_id.eq("later"), "score_difference"].item() == 1
    assert states.sort_values("event_index").event_index.is_monotonic_increasing


def test_order_validation_rejects_duplicate_match_index() -> None:
    frame = pd.DataFrame(
        {"event_id": ["a", "b"], "event_index": [1, 1], "match_id": [1, 1], "possession_id": [1, 1]}
    )
    with pytest.raises(ValueError, match="unique within a match"):
        validate_event_order(frame)


def test_horizons_include_current_future_event_and_keep_matches_grouped() -> None:
    events = []
    for index in range(1, 7):
        event_type = "Shot" if index in {1, 6} else "Pass"
        event = raw_event(
            f"event-{index}",
            index,
            event_type,
            end=[70, 40] if event_type == "Pass" else None,
        )
        events.append(
            normalize_event(
                event,
                competition_id=9,
                season_id=281,
                product_cohort=True,
            )
        )
    states, _ = build_possession_states(
        pd.DataFrame(events[::-1]),
        pd.DataFrame(
            {"shot_id": ["event-1", "event-6"], "expected_goal": [0.2, 0.3]}
        ),
    )
    first = states.loc[states.event_id.eq("event-1")].iloc[0]
    assert first[TARGET_COLUMN] == pytest.approx(0.5)
    assert first["future_oof_xg_next_5"] == pytest.approx(0.2)
    assert first["future_oof_xg_next_10"] == pytest.approx(0.5)
    assert horizon_audit(states)["remaining_possession"]["positive_states"] == 6
    assert states.event_index.tolist() == [1, 2, 3, 4, 5, 6]


def test_unmatched_oof_shot_is_rejected() -> None:
    event = normalize_event(
        raw_event("pass", 1, "Pass", end=[70, 40]),
        competition_id=9,
        season_id=281,
        product_cohort=True,
    )
    with pytest.raises(ValueError, match="exactly one normalized event"):
        build_possession_states(
            pd.DataFrame([event]),
            pd.DataFrame({"shot_id": ["missing-shot"], "expected_goal": [0.2]}),
        )


def test_feature_contract_excludes_future_and_identity() -> None:
    assert not KNOWN_ACTION_VALUE_LEAKAGE_COLUMNS.intersection(ACTION_VALUE_FEATURE_COLUMNS)


def test_grouped_folds_keep_matches_disjoint() -> None:
    frame = pd.DataFrame({"match_id": np.repeat(np.arange(10), 2)})
    for train, heldout in grouped_folds(frame):
        assert not set(frame.iloc[train].match_id).intersection(frame.iloc[heldout].match_id)


def test_model_selection_uses_validation_rmse() -> None:
    assert select_model_by_validation(
        {
            "global_mean": {"rmse": 0.06},
            "xgboost_reg_squarederror": {"rmse": 0.05},
            "hurdle": {"rmse": 0.055},
        }
    ) == "xgboost_reg_squarederror"


def test_nested_xg_never_trains_on_outer_heldout_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for match_id in range(12):
        row = {
            "shot_id": f"shot-{match_id}",
            "match_id": match_id,
            "model_eligible": True,
            "goal": match_id % 2,
        }
        row.update({column: 1 for column in nested_xg_labels.XG_MODEL_FEATURE_COLUMNS})
        rows.append(row)
    shots = pd.DataFrame(rows)
    heldout_ids = {10, 11}
    observed_training_sets: list[set[int]] = []

    def fake_predict(
        train: pd.DataFrame,
        prediction: pd.DataFrame,
        spec: FrozenXGSpec,
    ) -> np.ndarray:
        del spec
        train_ids = set(train["match_id"].astype(int))
        observed_training_sets.append(train_ids)
        assert not train_ids.intersection(prediction["match_id"].astype(int))
        assert not train_ids.intersection(heldout_ids)
        return np.full(len(prediction), 0.1)

    monkeypatch.setattr(nested_xg_labels, "_predict", fake_predict)
    spec = FrozenXGSpec(
        model_name="xgboost",
        config=XGBoostConfig(2, 0.1, 10, 1, 1, 1, 0, 1),
        rounds=5,
        retained_calibration=False,
        temperature=1.0,
    )
    cache = tmp_path / "nested.parquet"
    result = nested_xg_predictions(
        shots,
        set(range(10)),
        heldout_ids,
        spec,
        outer_name="test",
        cache_path=cache,
    )

    assert len(observed_training_sets) == 6
    assert len(result) == 12
    assert not result["shot_id"].duplicated().any()
    assert result.loc[result["match_id"].isin(heldout_ids), "outer_role"].eq(
        "outer_heldout"
    ).all()


def test_preprocessor_handles_missing_and_unseen_categories() -> None:
    row = {column: 1 for column in ACTION_VALUE_FEATURE_COLUMNS}
    row.update({"previous_action_type": "Pass", "current_play_pattern": "Regular Play"})
    frame = pd.DataFrame([row, row])
    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(frame)
    unseen = frame.copy()
    unseen["previous_action_type"] = "Never Seen"
    assert preprocessor.transform(unseen).shape == matrix.shape
    assert nonnegative(np.array([-1.0, 0.3])).tolist() == [0.0, 0.3]


def test_action_transitions_never_use_opponent_value() -> None:
    states = pd.DataFrame(
        [
            {"event_id": "success", "match_id": 1, "possession_id": 1, "event_index": 1, "event_type": "Pass", "event_success": True, "start_x": 50.0, "start_y": 40.0, "end_x": 60.0, "end_y": 40.0, "is_product_cohort": True, "player_id": 1, "team_id": 10, "under_pressure": False},
            {"event_id": "carry", "match_id": 1, "possession_id": 1, "event_index": 2, "event_type": "Carry", "event_success": True, "start_x": 60.0, "start_y": 40.0, "end_x": 70.0, "end_y": 40.0, "is_product_cohort": True, "player_id": 1, "team_id": 10, "under_pressure": True},
            {"event_id": "failed", "match_id": 1, "possession_id": 1, "event_index": 3, "event_type": "Pass", "event_success": False, "start_x": 70.0, "start_y": 40.0, "end_x": 80.0, "end_y": 40.0, "is_product_cohort": True, "player_id": 1, "team_id": 10, "under_pressure": False},
        ]
    )
    oof = pd.DataFrame({"event_id": ["success", "carry", "failed"], "predicted_state_value": [0.1, 0.2, 0.3], "fold": [1, 1, 1]})
    xpass = pd.DataFrame({"event_id": ["success", "failed"], "pass_index": [11, 12], "expected_completion": [0.8, 0.4]})
    actions = build_attacking_actions(states, oof, xpass)
    assert actions.loc[actions.action_id.eq("success"), "state_value_after"].item() == 0.2
    assert actions.loc[actions.action_id.eq("carry"), "state_value_after"].item() == 0.3
    assert actions.loc[actions.action_id.eq("failed"), "state_value_after"].item() == 0
    assert actions.loc[actions.action_id.eq("failed"), "attacking_value"].item() == -0.3


def test_player_profiles_keep_low_samples_with_derived_thresholds() -> None:
    actions = pd.DataFrame(
        {
            "player_id": [1, 1, 2], "match_id": [1, 2, 1], "action_type": ["Pass", "Carry", "Pass"],
            "attacking_value": [0.1, -0.02, 0.03], "progressive": [True, False, False], "under_pressure": [False, True, False],
        }
    )
    profiles, audit = build_player_attacking_profiles(actions)
    assert len(profiles) == 2
    assert audit["thresholds"]["actions"] >= 1
    assert set(profiles.player_id) == {1, 2}
