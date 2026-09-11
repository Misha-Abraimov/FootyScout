import numpy as np
import pandas as pd

from analytics.shot_features import XG_MODEL_FEATURE_COLUMNS
from analytics.xg_model import (
    assess_calibration,
    build_xg_preprocessor,
    grouped_folds,
    probabilities_from_margins,
    split_by_match,
)


def model_frame(matches: int = 20) -> pd.DataFrame:
    rows = []
    for match_id in range(matches):
        for shot in range(6):
            rows.append(
                {
                    "match_id": match_id,
                    "goal": int(shot == 0),
                    "shot_x": 100.0 + shot,
                    "shot_y": 30.0 + shot,
                    "distance_to_goal": 20.0 - shot,
                    "angle_to_goal": 0.2 + shot / 100,
                    "under_pressure": shot % 2 == 0,
                    "first_time": False,
                    "one_on_one": False,
                    "open_goal": False,
                    "body_part": "Right Foot" if shot else None,
                    "shot_type": "Open Play",
                    "technique": "Normal",
                    "play_pattern": "Regular Play",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_splits_never_overlap_matches() -> None:
    splits = split_by_match(model_frame())
    assert not set(splits.train_match_ids) & set(splits.validation_match_ids)
    assert not set(splits.train_match_ids) & set(splits.test_match_ids)
    assert not set(splits.validation_match_ids) & set(splits.test_match_ids)


def test_grouped_folds_hold_out_every_row_once() -> None:
    frame = model_frame()
    folds = grouped_folds(frame)
    heldout = np.concatenate([test for _, test in folds])
    assert sorted(heldout.tolist()) == list(range(len(frame)))


def test_preprocessor_handles_missing_and_unseen_categories() -> None:
    frame = model_frame()
    preprocessor = build_xg_preprocessor()
    matrix = preprocessor.fit_transform(frame[XG_MODEL_FEATURE_COLUMNS])
    unseen = frame.iloc[[0]].copy()
    unseen["body_part"] = "Shoulder"
    transformed = preprocessor.transform(unseen[XG_MODEL_FEATURE_COLUMNS])
    assert matrix.shape[0] == len(frame)
    assert transformed.shape[1] == matrix.shape[1]
    assert np.isfinite(transformed).all()


def test_probabilities_are_bounded_and_calibration_is_validation_only() -> None:
    margins = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    target = np.array([0, 0, 0, 1, 1])
    probabilities = probabilities_from_margins(margins)
    result = assess_calibration(margins, target)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert result.temperature > 0

