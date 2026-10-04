"""Offline causal-sequence and action-alignment contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from analytics.action_value_features import ACTION_VALUE_FEATURE_COLUMNS, TARGET_COLUMN
from analytics.possession_gru import CurrentStateMLP, NeuralConfig, PossessionGRU, parameter_count
from analytics.possession_gru_evaluation import action_values_from_state_predictions
from analytics.possession_sequence import (
    StatePreprocessor,
    collate_possessions,
    make_sequences,
    ordered_states,
)
from analytics.train_possession_gru import masked_loss, set_seed


def example_states() -> pd.DataFrame:
    rows = []
    for event_id, match_id, possession_id, event_index, target in [
        ("first", 1, 1, 3, 0.2),
        ("second", 1, 1, 8, 0.2),
        ("third", 1, 1, 10, 0.2),
        ("other", 2, 1, 2, 0.0),
    ]:
        row = {column: 1.0 for column in ACTION_VALUE_FEATURE_COLUMNS}
        row.update({
            "event_id": event_id,
            "match_id": match_id,
            "possession_id": possession_id,
            "event_index": event_index,
            TARGET_COLUMN: target,
            "previous_action_type": "Pass",
            "current_play_pattern": "Regular Play",
            "previous_action_success": True,
            "under_pressure": False,
            "ball_x": float(event_index),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def test_sequences_order_pre_event_states_and_preserve_every_row() -> None:
    states = example_states().iloc[[2, 3, 0, 1]]
    preprocessor = StatePreprocessor().fit(states)
    records = make_sequences(states, preprocessor)
    assert [(record.match_id, record.possession_id) for record in records] == [(1, 1), (2, 1)]
    assert records[0].row_index.tolist() == [0, 1, 2]
    assert records[0].target.tolist() == pytest.approx([0.2, 0.2, 0.2])
    assert sorted(np.concatenate([record.row_index for record in records]).tolist()) == [0, 1, 2, 3]


def test_validation_rejects_duplicate_order_and_negative_targets() -> None:
    states = example_states()
    duplicate = pd.concat([states, states.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique event ID"):
        ordered_states(duplicate)
    negative = states.copy()
    negative.loc[0, TARGET_COLUMN] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        ordered_states(negative)


def test_train_only_preprocessing_and_unknown_categories() -> None:
    training = example_states().iloc[:3].copy()
    training.loc[0, "ball_x"] = np.nan
    preprocessor = StatePreprocessor().fit(training)
    before = preprocessor.means.copy()
    heldout = example_states().iloc[[3]].copy()
    heldout["ball_x"] = 10_000.0
    heldout["previous_action_type"] = "Unseen Action"
    heldout["current_play_pattern"] = None
    continuous, binary, categorical = preprocessor.transform(heldout)
    assert np.array_equal(preprocessor.means, before)
    assert categorical.tolist() == [[0, 0]]
    assert np.isfinite(continuous).all() and np.isfinite(binary).all()
    assert "future_oof_xg_same_possession" not in preprocessor.metadata()["numeric_columns"]


def test_padding_mask_shapes_and_eligible_loss_only() -> None:
    states = example_states()
    records = make_sequences(states, StatePreprocessor().fit(states))
    batch = collate_possessions(records)
    assert batch["continuous"].shape[:2] == (2, 3)
    assert batch["categorical"].shape == (2, 3, 2)
    assert batch["mask"].tolist() == [[True, True, True], [True, False, False]]
    prediction = torch.zeros_like(batch["target"])
    first = masked_loss(prediction, batch["target"], batch["mask"], 0.05)
    prediction[1, 1:] = 1000
    second = masked_loss(prediction, batch["target"], batch["mask"], 0.05)
    assert float(first) == pytest.approx(float(second))


def test_gru_is_causal_nonnegative_and_deterministic() -> None:
    states = example_states()
    preprocessor = StatePreprocessor().fit(states)
    records = make_sequences(states, preprocessor)
    batch = collate_possessions(records)
    config = NeuralConfig(hidden_size=16)
    set_seed(config.seed)
    model = PossessionGRU(preprocessor.vocabulary_sizes, config).eval()
    first = model(batch["continuous"], batch["binary"], batch["categorical"], batch["lengths"])
    changed = batch["continuous"].clone()
    changed[0, 2, :] = 1_000
    later_changed = model(changed, batch["binary"], batch["categorical"], batch["lengths"])
    assert first.shape == (2, 3)
    assert torch.all(first >= 0)
    assert torch.allclose(first[0, :2], later_changed[0, :2])
    set_seed(config.seed)
    duplicate = PossessionGRU(preprocessor.vocabulary_sizes, config).eval()
    repeated = duplicate(batch["continuous"], batch["binary"], batch["categorical"], batch["lengths"])
    assert torch.equal(first, repeated)
    assert parameter_count(model) > parameter_count(CurrentStateMLP(preprocessor.vocabulary_sizes, config))


def test_gru_checkpoint_can_be_loaded(tmp_path: Path) -> None:
    config = NeuralConfig(hidden_size=16)
    vocabulary_sizes = [3, 2]
    set_seed(config.seed)
    original = PossessionGRU(vocabulary_sizes, config)
    checkpoint = tmp_path / "gru.pt"
    torch.save(original.state_dict(), checkpoint)
    restored = PossessionGRU(vocabulary_sizes, config)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_action_alignment_never_uses_next_possession_or_failed_after_state() -> None:
    states = pd.DataFrame([
        {"event_id": "pass", "match_id": 1, "possession_id": 1, "event_index": 1, "event_type": "Pass", "event_success": True, "end_x": 60, "end_y": 40},
        {"event_id": "carry", "match_id": 1, "possession_id": 1, "event_index": 2, "event_type": "Carry", "event_success": True, "end_x": 70, "end_y": 40},
        {"event_id": "failed", "match_id": 1, "possession_id": 1, "event_index": 3, "event_type": "Pass", "event_success": False, "end_x": 80, "end_y": 40},
        {"event_id": "opponent", "match_id": 1, "possession_id": 2, "event_index": 4, "event_type": "Carry", "event_success": True, "end_x": 90, "end_y": 40},
    ])
    predictions = pd.DataFrame({
        "event_id": ["pass", "carry", "failed", "opponent"],
        "value_xgboost": [0.1, 0.2, 0.3, 0.9],
        "value_gru": [0.05, 0.15, 0.25, 0.8],
    })
    actions = action_values_from_state_predictions(states, predictions).set_index("event_id")
    assert actions.loc["pass", "impact_value_xgboost"] == pytest.approx(0.1)
    assert actions.loc["carry", "impact_value_gru"] == pytest.approx(0.1)
    assert actions.loc["failed", "after_value_xgboost"] == 0
    assert actions.loc["failed", "impact_value_xgboost"] == pytest.approx(-0.3)
    assert actions.loc["opponent", "after_value_gru"] == 0
