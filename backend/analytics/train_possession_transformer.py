"""Offline hurdle MLP / causal Transformer experiment on the frozen possession split.

Run with the analytics extra installed: python -m analytics.train_possession_transformer
No production artifact, service, or database is written by this module.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from analytics.action_value_features import ACTION_VALUE_FEATURE_COLUMNS, TARGET_COLUMN
from analytics.possession_gru import parameter_count
from analytics.possession_gru_evaluation import (
    action_values_from_state_predictions,
    regression_metrics,
)
from analytics.possession_hurdle import (
    CurrentStateHurdleMLP,
    HurdleConfig,
    expected_value,
    hurdle_loss,
)
from analytics.possession_sequence import (
    PossessionDataset,
    SequenceRecord,
    StatePreprocessor,
    collate_possessions,
    make_sequences,
)
from analytics.possession_transformer import HybridCausalTransformer
from analytics.possession_transformer_evaluation import head_metrics, impact_comparison
from analytics.train_action_value_model import SHOTS_PATH, STATES_PATH, fixed_split
from analytics.train_possession_gru import nested_labels, set_seed

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "models" / "experiments" / "possession_transformer"
PREDICTION_DIR = ROOT / "data" / "processed" / "experiments" / "possession_transformer"
GRU_ARTIFACT_DIR = ROOT / "models" / "experiments" / "possession_gru"
GRU_PREDICTION_PATH = ROOT / "data" / "processed" / "experiments" / "possession_gru" / "fixed_test_state_predictions.parquet"
KINDS = ("hurdle_mlp", "hybrid_transformer")


class LengthBucketSampler(Sampler[list[int]]):
    """Reduce attention over padding without duplicating possession prefixes."""

    def __init__(
        self, records: list[SequenceRecord], batch_size: int, seed: int, shuffle: bool
    ) -> None:
        self.records = records
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def __iter__(self):  # type: ignore[override]
        rng = random.Random(self.seed + self.epoch)
        indices = sorted(range(len(self.records)), key=lambda index: len(self.records[index].target))
        batches = [indices[index:index + self.batch_size] for index in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches


def batches(
    records: list[SequenceRecord], config: HurdleConfig, shuffle: bool
) -> DataLoader:
    return DataLoader(
        PossessionDataset(records),
        batch_sampler=LengthBucketSampler(records, config.batch_size, config.seed, shuffle),
        collate_fn=collate_possessions,
        num_workers=0,
    )


def model_of_kind(kind: str, vocabulary_sizes: list[int], config: HurdleConfig) -> nn.Module:
    if kind == "hurdle_mlp":
        return CurrentStateHurdleMLP(vocabulary_sizes, config)
    if kind in {"hybrid_transformer", "transformer_no_fusion"}:
        return HybridCausalTransformer(
            vocabulary_sizes,
            replace(config, use_fusion=kind == "hybrid_transformer"),
        )
    raise ValueError(f"Unknown model kind: {kind}")


def run_epoch(
    model: nn.Module,
    data: DataLoader,
    config: HurdleConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    total_loss = total_bce = total_magnitude = squared = absolute = 0.0
    count = 0
    for batch in data:
        continuous = batch["continuous"].to(device)
        binary = batch["binary"].to(device)
        categorical = batch["categorical"].to(device)
        target = batch["target"].to(device)
        eligible = batch["mask"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logit, magnitude = model(continuous, binary, categorical, eligible)
            loss, bce, conditional = hurdle_loss(logit, magnitude, target, eligible, config)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
        prediction = expected_value(logit.detach(), magnitude.detach())
        error = prediction[eligible] - target[eligible]
        size = int(eligible.sum())
        count += size
        total_loss += float(loss.detach()) * size
        total_bce += float(bce.detach()) * size
        total_magnitude += float(conditional.detach()) * size
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
    return {
        "loss": total_loss / count,
        "bce": total_bce / count,
        "magnitude_loss": total_magnitude / count,
        "mae": absolute / count,
        "rmse": (squared / count) ** 0.5,
    }


def fit_with_validation(
    kind: str,
    train_records: list[SequenceRecord],
    validation_records: list[SequenceRecord],
    vocabulary_sizes: list[int],
    config: HurdleConfig,
    device: torch.device,
) -> tuple[nn.Module, int, list[dict[str, float | int]], float]:
    """Checkpoint selection depends on validation RMSE only, never test data."""
    set_seed(config.seed)
    model = model_of_kind(kind, vocabulary_sizes, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training = batches(train_records, config, True)
    validating = batches(validation_records, config, False)
    history: list[dict[str, float | int]] = []
    best_rmse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train = run_epoch(model, training, config, device, optimizer)
        validation = run_epoch(model, validating, config, device, None)
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train.items()},
            **{f"validation_{key}": value for key, value in validation.items()},
        })
        print(
            f"{kind} epoch {epoch:02d}: train RMSE {train['rmse']:.6f}, "
            f"validation RMSE {validation['rmse']:.6f}", flush=True,
        )
        if validation["rmse"] < best_rmse - 1e-6:
            best_rmse = validation["rmse"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        elif epoch - best_epoch >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, history, time.perf_counter() - started


def refit_epochs(
    kind: str,
    records: list[SequenceRecord],
    vocabulary_sizes: list[int],
    config: HurdleConfig,
    epochs: int,
    device: torch.device,
) -> tuple[nn.Module, float]:
    set_seed(config.seed)
    model = model_of_kind(kind, vocabulary_sizes, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    training = batches(records, config, True)
    started = time.perf_counter()
    for epoch in range(epochs):
        metrics = run_epoch(model, training, config, device, optimizer)
        print(f"{kind} non-test refit {epoch + 1:02d}/{epochs}: RMSE {metrics['rmse']:.6f}", flush=True)
    return model, time.perf_counter() - started


@torch.inference_mode()
def predict_records(
    model: nn.Module, records: list[SequenceRecord], config: HurdleConfig, device: torch.device
) -> pd.DataFrame:
    model.eval()
    frames: list[pd.DataFrame] = []
    for batch in batches(records, config, False):
        logit, magnitude = model(
            batch["continuous"].to(device),
            batch["binary"].to(device),
            batch["categorical"].to(device),
            batch["mask"].to(device),
        )
        mask = batch["mask"]
        probability = torch.sigmoid(logit).cpu()
        magnitude = magnitude.cpu()
        frames.append(pd.DataFrame({
            "row_index": batch["row_index"][mask].numpy(),
            "probability": probability[mask].numpy(),
            "positive_magnitude": magnitude[mask].numpy(),
            "prediction": (probability * magnitude)[mask].numpy(),
        }))
    result = pd.concat(frames, ignore_index=True).set_index("row_index")
    if result.index.has_duplicates or len(result) != sum(len(record.target) for record in records):
        raise AssertionError("Every eligible state needs exactly one hurdle prediction")
    return result


def frozen_baselines(test: pd.DataFrame) -> pd.DataFrame:
    """Read prior fixed-test output; never alter the completed GRU experiment."""
    prior = pd.read_parquet(GRU_PREDICTION_PATH)
    columns = ["event_id", "value_xgboost", "value_mlp", "value_gru"]
    if prior.event_id.duplicated().any() or len(prior) != len(test):
        raise AssertionError("Archived baseline does not cover the fixed test set once")
    if set(prior.event_id) != set(test.event_id):
        raise AssertionError("Archived baseline uses different test events")
    merged = test[["event_id", TARGET_COLUMN]].merge(
        prior[columns], on="event_id", validate="one_to_one"
    )
    if not np.allclose(
        merged[TARGET_COLUMN],
        test[["event_id", TARGET_COLUMN]].merge(
            prior[["event_id", TARGET_COLUMN]], on="event_id", validate="one_to_one"
        )[f"{TARGET_COLUMN}_y"],
        atol=1e-7,
    ):
        raise AssertionError("Archived fixed-test labels changed")
    archived = json.loads((GRU_ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    for kind in ("xgboost", "mlp", "gru"):
        measured = regression_metrics(
            merged[TARGET_COLUMN].to_numpy(), merged[f"value_{kind}"].to_numpy()
        )["rmse"]
        if abs(measured - archived["test_metrics"][kind]["rmse"]) > 1e-7:
            raise AssertionError(f"Archived {kind} fixed-test metric changed")
    return prior[columns]


def save_plots(
    history: dict[str, list[dict[str, float | int]]],
    scored: pd.DataFrame,
    actions: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 4))
    for ax, (kind, rows) in zip(np.atleast_1d(axes), history.items(), strict=True):
        epochs = [row["epoch"] for row in rows]
        ax.plot(epochs, [row["train_loss"] for row in rows], label="Train")
        ax.plot(epochs, [row["validation_loss"] for row in rows], label="Validation")
        ax.set(xlabel="Epoch", ylabel="Multi-task loss", title=kind)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "train_validation_loss.png", dpi=140)
    plt.close(fig)
    sample = scored.sample(n=min(5000, len(scored)), random_state=42)
    model = "hybrid_transformer"
    target = sample[TARGET_COLUMN].to_numpy()
    prediction = sample[f"value_{model}"].to_numpy()
    figures: list[tuple[str, np.ndarray, np.ndarray, str, str]] = [
        ("predicted_vs_actual.png", target, prediction, "Target", "Predicted V"),
        ("positive_magnitude.png", target[target > 0], sample.loc[sample[TARGET_COLUMN] > 0, f"positive_magnitude_{model}"].to_numpy(), "Positive target", "Conditional magnitude"),
        ("action_value_scatter.png", actions["impact_value_xgboost"].sample(n=min(5000, len(actions)), random_state=42).to_numpy(), actions["impact_value_hybrid_transformer"].loc[actions["impact_value_xgboost"].sample(n=min(5000, len(actions)), random_state=42).index].to_numpy(), "XGBoost impact", "Transformer impact"),
    ]
    for filename, x, y, xlabel, ylabel in figures:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x, y, s=4, alpha=0.15)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(prediction - target, bins=80)
    axes[0].set(xlabel="Prediction - target", title="Residuals")
    zero = sample[TARGET_COLUMN].eq(0)
    axes[1].hist(
        [sample.loc[zero, f"probability_{model}"], sample.loc[~zero, f"probability_{model}"]],
        bins=40, label=["Zero", "Positive"], density=True,
    )
    axes[1].set(xlabel="P(future xG > 0)", title="Probability distributions")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "residual_and_probability_distributions.png", dpi=140)
    plt.close(fig)
    curves = {
        kind: head_metrics(
            scored[TARGET_COLUMN].to_numpy(),
            scored[f"probability_{kind}"].to_numpy(),
            scored[f"positive_magnitude_{kind}"].to_numpy(),
        )["calibration_bins"] for kind in KINDS
    }
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    for kind, rows in curves.items():
        ax.plot([row["mean_probability"] for row in rows], [row["positive_rate"] for row in rows], "o-", label=kind)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Fixed-test calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "probability_calibration.png", dpi=140)
    plt.close(fig)
    metric_data = {
        kind: regression_metrics(scored[TARGET_COLUMN].to_numpy(), scored[f"value_{kind}"].to_numpy())
        for kind in ("xgboost", "mlp", "gru", *KINDS)
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("rmse", "positive_target_rmse"), strict=True):
        ax.bar(list(metric_data), [value[metric] for value in metric_data.values()])
        ax.tick_params(axis="x", rotation=25)
        ax.set(ylabel=metric, title=metric.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output / "rmse_comparisons.png", dpi=140)
    plt.close(fig)


def run(config: HurdleConfig, *, no_fusion_ablation: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = pd.read_parquet(STATES_PATH)
    shots = pd.read_parquet(SHOTS_PATH)
    train_base, validation_base, test_base, split = fixed_split(states)
    if (
        set(train_base.match_id) & set(validation_base.match_id)
        or set(train_base.match_id) & set(test_base.match_id)
        or set(validation_base.match_id) & set(test_base.match_id)
    ):
        raise AssertionError("Frozen match splits overlap")
    train, validation, non_test, test = nested_labels(states, shots, split)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    preprocessing = StatePreprocessor().fit(train)
    train_records = make_sequences(train, preprocessing)
    validation_records = make_sequences(validation, preprocessing)
    (ARTIFACT_DIR / "validation_preprocessor.json").write_text(
        json.dumps(preprocessing.metadata(), indent=2), encoding="utf-8"
    )
    kinds = [*KINDS, *(["transformer_no_fusion"] if no_fusion_ablation else [])]
    validation_metrics: dict[str, Any] = {}
    validation_heads: dict[str, Any] = {}
    histories: dict[str, list[dict[str, float | int]]] = {}
    epochs: dict[str, int] = {}
    parameters: dict[str, int] = {}
    timings: dict[str, float] = {}
    for kind in kinds:
        model, best_epoch, history, seconds = fit_with_validation(
            kind, train_records, validation_records, preprocessing.vocabulary_sizes, config, device
        )
        prediction = predict_records(model, validation_records, config, device).loc[validation.index]
        actual = validation[TARGET_COLUMN].to_numpy()
        validation_metrics[kind] = regression_metrics(actual, prediction.prediction.to_numpy())
        validation_heads[kind] = head_metrics(
            actual, prediction.probability.to_numpy(), prediction.positive_magnitude.to_numpy()
        )
        torch.save(model.cpu().state_dict(), ARTIFACT_DIR / f"{kind}_validation_best.pt")
        histories[kind] = history
        epochs[kind] = best_epoch
        parameters[kind] = parameter_count(model)
        timings[f"{kind}_validation_seconds"] = seconds
    # No fixed-test data informs configuration, checkpoint, ablation, or epoch selection.
    frozen = frozen_baselines(test)
    final_preprocessing = StatePreprocessor().fit(non_test)
    non_test_records = make_sequences(non_test, final_preprocessing)
    test_records = make_sequences(test, final_preprocessing)
    (ARTIFACT_DIR / "test_preprocessor.json").write_text(
        json.dumps(final_preprocessing.metadata(), indent=2), encoding="utf-8"
    )
    scored = test[["event_id", "match_id", "possession_id", "event_index", TARGET_COLUMN]].copy()
    scored = scored.merge(frozen, on="event_id", validate="one_to_one", sort=False)
    if len(scored) != len(test) or scored.event_id.duplicated().any():
        raise AssertionError("Frozen predictions must align one-to-one")
    for kind in kinds:
        model, seconds = refit_epochs(
            kind, non_test_records, final_preprocessing.vocabulary_sizes,
            config, epochs[kind], device,
        )
        timings[f"{kind}_non_test_refit_seconds"] = seconds
        torch.save(model.cpu().state_dict(), ARTIFACT_DIR / f"{kind}_non_test_test.pt")
        prediction = predict_records(model, test_records, config, device).loc[test.index]
        for source, destination in (
            ("prediction", f"value_{kind}"),
            ("probability", f"probability_{kind}"),
            ("positive_magnitude", f"positive_magnitude_{kind}"),
        ):
            scored[destination] = prediction[source].to_numpy()
    if scored.isna().any().any() or (scored.filter(like="value_") < 0).any().any():
        raise AssertionError("Fixed-test predictions must be finite and non-negative")
    scored.to_parquet(PREDICTION_DIR / "fixed_test_state_predictions.parquet", index=False)
    actual = scored[TARGET_COLUMN].to_numpy()
    test_metrics = {
        kind: regression_metrics(actual, scored[f"value_{kind}"].to_numpy())
        for kind in ("xgboost", "mlp", "gru", *kinds)
    }
    test_heads = {
        kind: head_metrics(
            actual,
            scored[f"probability_{kind}"].to_numpy(),
            scored[f"positive_magnitude_{kind}"].to_numpy(),
        ) for kind in kinds
    }
    actions = action_values_from_state_predictions(test, scored)
    actions.to_parquet(PREDICTION_DIR / "fixed_test_action_values.parquet", index=False)
    impact = impact_comparison(actions)
    save_plots(histories, scored, actions, ARTIFACT_DIR / "figures")
    timings["total_seconds"] = time.perf_counter() - started
    positive = train[TARGET_COLUMN].gt(0)
    result: dict[str, Any] = {
        "status": "offline_experiment_only",
        "target": TARGET_COLUMN,
        "state_alignment": "pre-event current state; no future event inputs",
        "feature_columns": ACTION_VALUE_FEATURE_COLUMNS,
        "split": {
            "method": split["method"],
            "train_matches": len(split["train_match_ids"]),
            "validation_matches": len(split["validation_match_ids"]),
            "test_matches": len(split["test_match_ids"]),
            "train_states": len(train), "validation_states": len(validation), "test_states": len(test),
        },
        "training_positive_fraction": float(positive.mean()),
        "training_bce_pos_weight_if_balanced": float((~positive).sum() / positive.sum()),
        "bce_pos_weight_used": False,
        "bce_weight_reason": "Unweighted BCE retains an interpretable probability; report calibration separately.",
        "config": config.metadata(),
        "device": str(device),
        "trainable_parameters": parameters,
        "best_validation_epochs": epochs,
        "validation_metrics": validation_metrics,
        "validation_head_metrics": validation_heads,
        "test_metrics": test_metrics,
        "test_head_metrics": test_heads,
        "impact_comparison": impact,
        "timings": timings,
        "no_test_tuning": True,
        "ablation_no_fusion_run": no_fusion_ablation,
    }
    (ARTIFACT_DIR / "config.json").write_text(json.dumps(config.metadata(), indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "validation": validation_metrics, "fixed_test": test_metrics,
        "head_metrics": test_heads, "timings": timings,
    }, indent=2), flush=True)
    return result


def run_validation_ablation(kind: str, config: HurdleConfig, stem: str) -> dict[str, Any]:
    """Optional validation-only ablation; never score or select on the fixed test."""
    started = time.perf_counter()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = pd.read_parquet(STATES_PATH)
    shots = pd.read_parquet(SHOTS_PATH)
    _, _, _, split = fixed_split(states)
    train, validation, _, _ = nested_labels(states, shots, split)
    preprocessing = StatePreprocessor().fit(train)
    train_records = make_sequences(train, preprocessing)
    validation_records = make_sequences(validation, preprocessing)
    model, best_epoch, history, seconds = fit_with_validation(
        kind, train_records, validation_records,
        preprocessing.vocabulary_sizes, config, device,
    )
    prediction = predict_records(model, validation_records, config, device).loc[validation.index]
    actual = validation[TARGET_COLUMN].to_numpy()
    result: dict[str, Any] = {
        "status": f"validation_only_{stem}_ablation",
        "test_data_used": False,
        "config": config.metadata(),
        "device": str(device),
        "trainable_parameters": parameter_count(model),
        "best_validation_epoch": best_epoch,
        "validation_metrics": regression_metrics(actual, prediction.prediction.to_numpy()),
        "validation_head_metrics": head_metrics(
            actual, prediction.probability.to_numpy(), prediction.positive_magnitude.to_numpy()
        ),
        "training_seconds": seconds,
        "total_seconds": time.perf_counter() - started,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), ARTIFACT_DIR / f"{stem}_validation_best.pt")
    (ARTIFACT_DIR / f"{stem}_validation_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / f"{stem}_validation_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-fusion-ablation", action="store_true")
    parser.add_argument("--no-fusion-only", action="store_true")
    parser.add_argument("--log1p-magnitude-only", action="store_true")
    arguments = parser.parse_args()
    config = HurdleConfig(epochs=arguments.epochs, batch_size=arguments.batch_size)
    if arguments.no_fusion_only:
        run_validation_ablation(
            "transformer_no_fusion", replace(config, use_fusion=False), "no_fusion"
        )
    elif arguments.log1p_magnitude_only:
        run_validation_ablation(
            "hurdle_mlp", replace(config, log1p_magnitude=True), "log1p_magnitude_mlp"
        )
    else:
        run(config, no_fusion_ablation=arguments.no_fusion_ablation)


if __name__ == "__main__":
    main()
