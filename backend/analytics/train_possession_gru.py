"""Offline GRU-versus-frozen-XGBoost experiment for pre-event possession value.

Run from the repository root with the analytics extra installed:
    python -m analytics.train_possession_gru

This module never writes production XGBoost artifacts, database rows or API data.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from analytics.action_value_features import ACTION_VALUE_FEATURE_COLUMNS, TARGET_COLUMN
from analytics.action_value_model import build_preprocessor, feature_frame, nonnegative
from analytics.nested_xg_labels import (
    apply_nested_state_targets,
    load_frozen_xg_spec,
    nested_xg_predictions,
)
from analytics.possession_gru import CurrentStateMLP, NeuralConfig, PossessionGRU, parameter_count
from analytics.possession_gru_evaluation import (
    action_comparison,
    action_values_from_state_predictions,
    regression_metrics,
)
from analytics.possession_sequence import (
    PossessionDataset,
    SequenceRecord,
    StatePreprocessor,
    collate_possessions,
    make_sequences,
)
from analytics.train_action_value_model import (
    NESTED_XG_CACHE_DIR,
    SHOTS_PATH,
    STATES_PATH,
    XG_METADATA_PATH,
    _fit_selected,
    fixed_split,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "models" / "experiments" / "possession_gru"
PREDICTIONS_DIR = ROOT / "data" / "processed" / "experiments" / "possession_gru"
XGB_METADATA_PATH = ROOT / "models" / "action_value_model_metadata.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def loader(records: list[SequenceRecord], config: NeuralConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        PossessionDataset(records),
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_possessions,
        num_workers=0,
        generator=generator,
    )


def masked_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float) -> torch.Tensor:
    error = nn.functional.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    return error[mask].mean()


def run_epoch(
    model: nn.Module,
    batches: DataLoader,
    config: NeuralConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    absolute = squared = huber = 0.0
    count = 0
    for batch in batches:
        continuous = batch["continuous"].to(device)
        binary = batch["binary"].to(device)
        categorical = batch["categorical"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction = model(continuous, binary, categorical, batch["lengths"])
            loss = masked_loss(prediction, target, mask, config.huber_beta)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
        difference = prediction.detach()[mask] - target[mask]
        size = int(mask.sum())
        count += size
        huber += float(loss.detach()) * size
        absolute += float(difference.abs().sum())
        squared += float(difference.square().sum())
    return {"huber": huber / count, "mae": absolute / count, "rmse": (squared / count) ** 0.5}


def model_of_kind(kind: str, vocabulary_sizes: list[int], config: NeuralConfig) -> nn.Module:
    if kind == "gru":
        return PossessionGRU(vocabulary_sizes, config)
    if kind == "mlp":
        return CurrentStateMLP(vocabulary_sizes, config)
    raise ValueError(f"Unknown model kind: {kind}")


def fit_with_validation(
    kind: str,
    train_records: list[SequenceRecord],
    validation_records: list[SequenceRecord],
    vocabulary_sizes: list[int],
    config: NeuralConfig,
    device: torch.device,
) -> tuple[nn.Module, int, list[dict[str, float | int]], float]:
    set_seed(config.seed)
    model = model_of_kind(kind, vocabulary_sizes, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_batches = loader(train_records, config, shuffle=True)
    validation_batches = loader(validation_records, config, shuffle=False)
    best_rmse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_batches, config, device, optimizer)
        validation_metrics = run_epoch(model, validation_batches, config, device, None)
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        })
        print(
            f"{kind} epoch {epoch:02d}: train RMSE {train_metrics['rmse']:.6f}, "
            f"validation RMSE {validation_metrics['rmse']:.6f}",
            flush=True,
        )
        if validation_metrics["rmse"] < best_rmse - 1e-6:
            best_rmse = validation_metrics["rmse"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    return model, best_epoch, history, time.perf_counter() - started


def refit_epochs(
    kind: str,
    records: list[SequenceRecord],
    vocabulary_sizes: list[int],
    config: NeuralConfig,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, float]:
    set_seed(config.seed)
    model = model_of_kind(kind, vocabulary_sizes, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    batches = loader(records, config, shuffle=True)
    started = time.perf_counter()
    for epoch in range(epochs):
        metrics = run_epoch(model, batches, config, device, optimizer)
        print(f"{kind} non-test refit {epoch + 1:02d}/{epochs}: RMSE {metrics['rmse']:.6f}", flush=True)
    return model, time.perf_counter() - started


@torch.inference_mode()
def predict_records(
    model: nn.Module, records: list[SequenceRecord], config: NeuralConfig, device: torch.device
) -> pd.DataFrame:
    model.eval()
    rows: list[pd.DataFrame] = []
    for batch in loader(records, config, shuffle=False):
        prediction = model(
            batch["continuous"].to(device),
            batch["binary"].to(device),
            batch["categorical"].to(device),
            batch["lengths"],
        ).cpu()
        mask = batch["mask"]
        rows.append(pd.DataFrame({
            "row_index": batch["row_index"][mask].numpy(),
            "prediction": prediction[mask].numpy(),
        }))
    output = pd.concat(rows, ignore_index=True).set_index("row_index")
    if output.index.has_duplicates or len(output) != sum(len(record.target) for record in records):
        raise AssertionError("Every state must receive exactly one neural prediction")
    return output


def nested_labels(
    states: pd.DataFrame, shots: pd.DataFrame, split: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = load_frozen_xg_spec(XG_METADATA_PATH)
    train_ids = set(split["train_match_ids"])
    validation_ids = set(split["validation_match_ids"])
    test_ids = set(split["test_match_ids"])
    validation_xg = nested_xg_predictions(
        shots, train_ids, validation_ids, spec,
        outer_name="fixed_validation",
        cache_path=NESTED_XG_CACHE_DIR / "fixed_validation.parquet",
    )
    non_test_ids = train_ids | validation_ids
    test_xg = nested_xg_predictions(
        shots, non_test_ids, test_ids, spec,
        outer_name="fixed_untouched_test",
        cache_path=NESTED_XG_CACHE_DIR / "fixed_untouched_test.parquet",
    )
    train = apply_nested_state_targets(
        states.loc[states.match_id.isin(train_ids)],
        validation_xg.loc[validation_xg.match_id.isin(train_ids)],
    )
    validation = apply_nested_state_targets(
        states.loc[states.match_id.isin(validation_ids)],
        validation_xg.loc[validation_xg.match_id.isin(validation_ids)],
    )
    non_test = apply_nested_state_targets(
        states.loc[states.match_id.isin(non_test_ids)],
        test_xg.loc[test_xg.match_id.isin(non_test_ids)],
    )
    test = apply_nested_state_targets(
        states.loc[states.match_id.isin(test_ids)],
        test_xg.loc[test_xg.match_id.isin(test_ids)],
    )
    return train, validation, non_test, test


def frozen_xgboost_predictions(non_test: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Reconstruct only the archived fixed-test baseline, never a production model."""
    archived = json.loads(XGB_METADATA_PATH.read_text(encoding="utf-8"))
    preprocessor = build_preprocessor()
    train_matrix = preprocessor.fit_transform(feature_frame(non_test))
    test_matrix = preprocessor.transform(feature_frame(test))
    selected = archived["selection"]
    index = next(
        (number for number, candidate in enumerate(archived["candidates"])
         if candidate["name"] == selected["model"]),
        0,
    )
    model = _fit_selected(
        selected["model"], index, selected["rounds"],
        train_matrix, non_test[TARGET_COLUMN].to_numpy(dtype=float),
    )
    if selected["model"] == "hurdle":
        from analytics.action_value_model import predict_hurdle

        prediction = predict_hurdle(model[0], model[1], test_matrix)
    else:
        prediction = nonnegative(model.predict(test_matrix))
    reported = archived["untouched_test_metrics"]["selected_model"]["rmse"]
    observed = regression_metrics(test[TARGET_COLUMN].to_numpy(), prediction)["rmse"]
    if abs(observed - reported) > 1e-6:
        raise AssertionError(f"Frozen XGBoost fixed-test RMSE changed: {observed} vs {reported}")
    return prediction


def save_plots(
    history: dict[str, list[dict[str, float | int]]],
    predictions: pd.DataFrame,
    actions: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in history.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = [int(row["epoch"]) for row in rows]
        ax.plot(epochs, [row["train_rmse"] for row in rows], label="Training")
        ax.plot(epochs, [row["validation_rmse"] for row in rows], label="Validation")
        ax.set(xlabel="Epoch", ylabel="RMSE", title=f"{name.upper()} learning curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{name}_learning_curve.png", dpi=140)
        plt.close(fig)
    sample = predictions.sample(n=min(5000, len(predictions)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(sample[TARGET_COLUMN], sample["value_gru"], s=4, alpha=0.15)
    axes[0].set(xlabel="Observed future OOF xG", ylabel="GRU prediction", title="Held-out states")
    axes[1].hist(
        [sample["value_xgboost"], sample["value_gru"]],
        bins=60, label=["XGBoost", "GRU"], density=True,
    )
    axes[1].set(xlabel="Predicted possession value", ylabel="Density", title="Prediction distribution")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "test_state_predictions.png", dpi=140)
    plt.close(fig)
    sample_actions = actions.sample(n=min(5000, len(actions)), random_state=42)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sample_actions.impact_value_xgboost, sample_actions.impact_value_gru, s=4, alpha=0.15)
    ax.set(xlabel="XGBoost action value", ylabel="GRU action value", title="Held-out action alignment")
    fig.tight_layout()
    fig.savefig(output / "test_action_value_comparison.png", dpi=140)
    plt.close(fig)


def run(config: NeuralConfig, *, mlp_ablation: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = pd.read_parquet(STATES_PATH)
    shots = pd.read_parquet(SHOTS_PATH)
    train_base, validation_base, test_base, split = fixed_split(states)
    if set(train_base.match_id) & set(validation_base.match_id) or set(train_base.match_id) & set(test_base.match_id) or set(validation_base.match_id) & set(test_base.match_id):
        raise AssertionError("Frozen match-grouped splits overlap")
    train, validation, non_test, test = nested_labels(states, shots, split)
    if any(frame[TARGET_COLUMN].lt(0).any() for frame in (train, validation, non_test, test)):
        raise ValueError("Softplus requires a non-negative target")

    preprocessing = StatePreprocessor().fit(train)
    train_records = make_sequences(train, preprocessing)
    validation_records = make_sequences(validation, preprocessing)
    history: dict[str, list[dict[str, float | int]]] = {}
    best_epochs: dict[str, int] = {}
    parameters: dict[str, int] = {}
    timings: dict[str, float] = {}
    validation_metrics: dict[str, dict[str, float | int]] = {}
    kinds = ["gru", "mlp"] if mlp_ablation else ["gru"]
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "validation_preprocessor.json").write_text(
        json.dumps(preprocessing.metadata(), indent=2), encoding="utf-8"
    )
    for kind in kinds:
        model, best_epoch, model_history, seconds = fit_with_validation(
            kind, train_records, validation_records, preprocessing.vocabulary_sizes, config, device
        )
        predictions = predict_records(model, validation_records, config, device).loc[validation.index]
        validation_metrics[kind] = regression_metrics(
            validation[TARGET_COLUMN].to_numpy(), predictions.prediction.to_numpy()
        )
        torch.save(model.cpu().state_dict(), ARTIFACT_DIR / f"{kind}_validation_best.pt")
        parameters[kind] = parameter_count(model)
        history[kind] = model_history
        best_epochs[kind] = best_epoch
        timings[f"{kind}_validation_seconds"] = seconds

    # The fixed test set is opened only after validation epoch selection is complete.
    final_preprocessing = StatePreprocessor().fit(non_test)
    non_test_records = make_sequences(non_test, final_preprocessing)
    test_records = make_sequences(test, final_preprocessing)
    (ARTIFACT_DIR / "test_preprocessor.json").write_text(
        json.dumps(final_preprocessing.metadata(), indent=2), encoding="utf-8"
    )
    scored = test[["event_id", "match_id", "possession_id", "event_index", TARGET_COLUMN]].copy()
    baseline_started = time.perf_counter()
    scored["value_xgboost"] = frozen_xgboost_predictions(non_test, test)
    timings["xgboost_refit_seconds"] = time.perf_counter() - baseline_started
    for kind in kinds:
        model, seconds = refit_epochs(
            kind, non_test_records, final_preprocessing.vocabulary_sizes,
            config, best_epochs[kind], device,
        )
        timings[f"{kind}_non_test_refit_seconds"] = seconds
        torch.save(model.cpu().state_dict(), ARTIFACT_DIR / f"{kind}_non_test_test.pt")
        scored[f"value_{kind}"] = predict_records(model, test_records, config, device).loc[
            test.index, "prediction"
        ].to_numpy()
    if len(scored) != len(test) or scored.event_id.duplicated().any() or scored.isna().any().any():
        raise AssertionError("Every fixed-test state needs one complete prediction per model")
    scored.to_parquet(PREDICTIONS_DIR / "fixed_test_state_predictions.parquet", index=False)
    target = scored[TARGET_COLUMN].to_numpy()
    test_metrics = {
        name: regression_metrics(target, scored[f"value_{name}"].to_numpy())
        for name in ["xgboost", *kinds]
    }
    test_actions = action_values_from_state_predictions(test, scored)
    test_actions.to_parquet(PREDICTIONS_DIR / "fixed_test_action_values.parquet", index=False)
    action_metrics = action_comparison(test_actions)
    save_plots(history, scored, test_actions, ARTIFACT_DIR / "figures")
    timings["total_seconds"] = time.perf_counter() - started
    result: dict[str, Any] = {
        "status": "offline_experiment_only",
        "target": TARGET_COLUMN,
        "state_alignment": "pre-event; current event is in the future target but not in features",
        "feature_columns": ACTION_VALUE_FEATURE_COLUMNS,
        "split": {
            "method": split["method"],
            "train_matches": len(split["train_match_ids"]),
            "validation_matches": len(split["validation_match_ids"]),
            "test_matches": len(split["test_match_ids"]),
            "train_states": len(train),
            "validation_states": len(validation),
            "test_states": len(test),
        },
        "target_distribution": {
            "minimum": float(states[TARGET_COLUMN].min()),
            "maximum": float(states[TARGET_COLUMN].max()),
            "zero_fraction": float(states[TARGET_COLUMN].eq(0).mean()),
        },
        "config": config.metadata(),
        "device": str(device),
        "trainable_parameters": parameters,
        "best_validation_epochs": best_epochs,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "action_comparison": action_metrics,
        "timings": timings,
        "artifacts": {
            "model_directory": str(ARTIFACT_DIR.relative_to(ROOT)),
            "prediction_directory": str(PREDICTIONS_DIR.relative_to(ROOT)),
        },
    }
    (ARTIFACT_DIR / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "validation": validation_metrics,
        "test": test_metrics,
        "action_comparison": action_metrics,
        "timings": timings,
    }, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-mlp", action="store_true")
    arguments = parser.parse_args()
    config = NeuralConfig(epochs=arguments.epochs, batch_size=arguments.batch_size)
    run(config, mlp_ablation=not arguments.skip_mlp)


if __name__ == "__main__":
    main()
