"""Offline hurdle/causal-transformer leakage and alignment contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from analytics.action_value_features import ACTION_VALUE_FEATURE_COLUMNS, TARGET_COLUMN
from analytics.possession_gru_evaluation import action_values_from_state_predictions
from analytics.possession_hurdle import (
    CurrentStateHurdleMLP,
    HurdleConfig,
    HurdleHeads,
    expected_value,
    hurdle_loss,
    positive_target,
)
from analytics.possession_sequence import StatePreprocessor, collate_possessions, make_sequences
from analytics.possession_transformer import (
    HybridCausalTransformer,
    causal_attention_mask,
    sinusoidal_positions,
)
from analytics.train_possession_gru import set_seed
from analytics.train_possession_transformer import LengthBucketSampler, predict_records


def states() -> pd.DataFrame:
    rows = []
    for event_id, match, possession, index, target in [
        ("late", 1, 1, 3, 0.2), ("first", 1, 1, 1, 0.2),
        ("middle", 1, 1, 2, 0.2), ("other", 2, 1, 1, 0.0),
    ]:
        row = {column: 1.0 for column in ACTION_VALUE_FEATURE_COLUMNS}
        row.update({
            "event_id": event_id, "match_id": match, "possession_id": possession,
            "event_index": index, TARGET_COLUMN: target,
            "previous_action_type": "Pass", "current_play_pattern": "Regular Play",
            "previous_action_success": True, "under_pressure": False,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def fixture_batch() -> tuple[StatePreprocessor, dict[str, torch.Tensor]]:
    frame = states()
    preprocessing = StatePreprocessor().fit(frame)
    records = make_sequences(frame, preprocessing)
    return preprocessing, collate_possessions(records)


def test_causal_mask_forbids_only_future_and_positions_are_distinct() -> None:
    assert causal_attention_mask(3).tolist() == [
        [False, True, True], [False, False, True], [False, False, False]
    ]
    positions = sinusoidal_positions(3, 8, torch.device("cpu"))
    assert positions.shape == (3, 8)
    assert not torch.equal(positions[0], positions[1])


def test_hurdle_target_product_and_positive_only_loss() -> None:
    target = torch.tensor([[0.0, 0.2, 0.0], [0.3, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    logit = torch.zeros_like(target, requires_grad=True)
    magnitude = torch.full_like(target, 0.1, requires_grad=True)
    assert positive_target(target).tolist() == [[0, 1, 0], [1, 0, 0]]
    assert torch.allclose(expected_value(logit, magnitude), torch.full_like(target, 0.05))
    first = hurdle_loss(logit, magnitude, target, mask, HurdleConfig())
    changed = magnitude.detach().clone()
    changed[0, 0] = 500  # eligible zero target: no magnitude supervision
    changed[0, 2] = 500  # padding: no supervision at all
    second = hurdle_loss(logit, changed, target, mask, HurdleConfig())
    assert float(first[1].detach()) == pytest.approx(float(second[1].detach()))
    assert float(first[2].detach()) == pytest.approx(float(second[2].detach()))
    assert float(first[0].detach()) == pytest.approx(float(second[0].detach()))
    altered_positive = changed.clone()
    altered_positive[0, 1] = 2.0
    assert float(hurdle_loss(logit, altered_positive, target, mask, HurdleConfig())[2].detach()) > float(first[2].detach())
    with pytest.raises(ValueError, match="non-negative"):
        expected_value(logit, -magnitude)


def test_log1p_conditional_magnitude_has_correct_inverse_and_mask() -> None:
    set_seed(42)
    plain = HurdleHeads(4, 8, 0.0).eval()
    transformed = HurdleHeads(4, 8, 0.0, log1p_magnitude=True).eval()
    transformed.load_state_dict(plain.state_dict())
    inputs = torch.ones(1, 3, 4)
    with torch.no_grad():
        plain_logit, plain_magnitude = plain(inputs)
        transformed_logit, transformed_magnitude = transformed(inputs)
    assert torch.equal(plain_logit, transformed_logit)
    assert torch.allclose(transformed_magnitude, torch.expm1(plain_magnitude))
    target = torch.tensor([[0.0, 0.3, 0.0]])
    mask = torch.tensor([[True, True, False]])
    config = HurdleConfig(log1p_magnitude=True)
    first = hurdle_loss(transformed_logit, transformed_magnitude, target, mask, config)
    altered = transformed_magnitude.clone()
    altered[0, 0] = 100
    altered[0, 2] = 100
    second = hurdle_loss(transformed_logit, altered, target, mask, config)
    assert float(first[2]) == pytest.approx(float(second[2]))


@pytest.mark.parametrize("kind", ["mlp", "transformer", "transformer_no_fusion"])
def test_forward_shapes_nonnegative_and_padding_is_ignored(kind: str) -> None:
    preprocessing, batch = fixture_batch()
    config = HurdleConfig(d_model=16, num_heads=4, num_layers=1, dropout=0.0)
    set_seed(config.seed)
    model = (
        CurrentStateHurdleMLP(preprocessing.vocabulary_sizes, config)
        if kind == "mlp" else HybridCausalTransformer(
            preprocessing.vocabulary_sizes,
            HurdleConfig(d_model=16, num_heads=4, num_layers=1, dropout=0.0, use_fusion=kind == "transformer"),
        )
    ).eval()
    arguments = (batch["continuous"], batch["binary"], batch["categorical"], batch["mask"])
    with torch.no_grad():
        logit, magnitude = model(*arguments)
        assert logit.shape == (2, 3)
        assert magnitude.shape == (2, 3)
        assert torch.all(magnitude >= 0)
        assert torch.all(expected_value(logit, magnitude) >= 0)
        changed = batch["continuous"].clone()
        changed[1, 1:] = 1000  # padded states may not affect the first real state
        second = model(changed, batch["binary"], batch["categorical"], batch["mask"])
        assert torch.allclose(logit[1, 0], second[0][1, 0], atol=1e-6)
        assert torch.allclose(magnitude[1, 0], second[1][1, 0], atol=1e-6)
    set_seed(config.seed)
    duplicate = (
        CurrentStateHurdleMLP(preprocessing.vocabulary_sizes, config)
        if kind == "mlp" else HybridCausalTransformer(
            preprocessing.vocabulary_sizes,
            HurdleConfig(d_model=16, num_heads=4, num_layers=1, dropout=0.0, use_fusion=kind == "transformer"),
        )
    ).eval()
    with torch.no_grad():
        assert torch.equal(logit, duplicate(*arguments)[0])


def test_transformer_has_no_future_timestep_influence() -> None:
    preprocessing, batch = fixture_batch()
    config = HurdleConfig(d_model=16, num_heads=4, num_layers=1, dropout=0.0)
    set_seed(config.seed)
    model = HybridCausalTransformer(preprocessing.vocabulary_sizes, config).eval()
    with torch.no_grad():
        original = model(batch["continuous"], batch["binary"], batch["categorical"], batch["mask"])
        changed = batch["continuous"].clone()
        changed[0, 2] = 1000
        altered = model(changed, batch["binary"], batch["categorical"], batch["mask"])
    assert torch.allclose(original[0][0, :2], altered[0][0, :2], atol=1e-6)
    assert torch.allclose(original[1][0, :2], altered[1][0, :2], atol=1e-6)


def test_order_pre_event_alignment_train_only_vocabulary_and_one_prediction() -> None:
    frame = states()
    train = frame.iloc[:3]
    preprocessing = StatePreprocessor().fit(train)
    heldout = frame.iloc[[3]].copy()
    heldout["previous_action_type"] = "Future-only category"
    assert preprocessing.transform(heldout)[2][0, 0] == 0
    assert "Future-only category" not in preprocessing.vocabularies["previous_action_type"]
    records = make_sequences(frame, preprocessing)
    assert records[0].row_index.tolist() == [1, 2, 0]
    assert records[0].target.tolist() == pytest.approx([0.2, 0.2, 0.2])
    config = HurdleConfig(d_model=16, num_heads=4, num_layers=1, dropout=0.0)
    model = CurrentStateHurdleMLP(preprocessing.vocabulary_sizes, config)
    prediction = predict_records(model, records, config, torch.device("cpu"))
    assert len(prediction) == len(frame)
    assert set(prediction.index) == set(frame.index)
    assert prediction.index.is_unique
    assert np.allclose(prediction.prediction, prediction.probability * prediction.positive_magnitude)
    assert prediction.prediction.ge(0).all()
    sampler = LengthBucketSampler(records, 1, 42, False)
    assert sorted(index for batch in sampler for index in batch) == [0, 1]


def test_action_before_after_same_possession_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    events = pd.DataFrame([
        {"event_id": "first", "match_id": 1, "possession_id": 1, "event_index": 1, "event_type": "Pass", "event_success": True, "end_x": 60, "end_y": 40},
        {"event_id": "last", "match_id": 1, "possession_id": 1, "event_index": 2, "event_type": "Carry", "event_success": True, "end_x": 70, "end_y": 40},
        {"event_id": "other", "match_id": 1, "possession_id": 2, "event_index": 3, "event_type": "Pass", "event_success": True, "end_x": 80, "end_y": 40},
    ])
    predictions = pd.DataFrame({"event_id": ["first", "last", "other"], "value_hybrid_transformer": [0.1, 0.2, 0.9]})
    impacts = action_values_from_state_predictions(events, predictions).set_index("event_id")
    assert impacts.loc["first", "impact_value_hybrid_transformer"] == pytest.approx(0.1)
    assert impacts.loc["last", "after_value_hybrid_transformer"] == 0
    config = HurdleConfig(d_model=16, num_heads=4, num_layers=1)
    model = HybridCausalTransformer([3, 2], config)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = HybridCausalTransformer([3, 2], config)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())


def test_fit_selection_interface_has_no_test_data() -> None:
    from inspect import signature

    from analytics.train_possession_transformer import fit_with_validation

    assert "test" not in signature(fit_with_validation).parameters
