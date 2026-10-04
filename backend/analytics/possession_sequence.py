"""Causal, pre-event possession sequences for offline state-value experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from analytics.action_value_features import (
    ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS,
    ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS,
    ACTION_VALUE_FEATURE_COLUMNS,
    ACTION_VALUE_NUMERIC_FEATURE_COLUMNS,
    KNOWN_ACTION_VALUE_LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)

UNKNOWN = "Unknown"
GROUP_COLUMNS = ["match_id", "possession_id"]
ORDER_COLUMNS = ["match_id", "possession_id", "event_index"]


@dataclass(frozen=True)
class SequenceRecord:
    continuous: np.ndarray
    binary: np.ndarray
    categorical: np.ndarray
    target: np.ndarray
    row_index: np.ndarray
    match_id: int
    possession_id: int


class StatePreprocessor:
    """Fit medians/scales and categorical vocabularies on training states only."""

    def __init__(self) -> None:
        self.medians: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.binary_modes: np.ndarray | None = None
        self.vocabularies: dict[str, dict[str, int]] = {}

    @property
    def vocabulary_sizes(self) -> list[int]:
        return [len(self.vocabularies[column]) for column in ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS]

    def fit(self, train: pd.DataFrame) -> StatePreprocessor:
        if KNOWN_ACTION_VALUE_LEAKAGE_COLUMNS.intersection(ACTION_VALUE_FEATURE_COLUMNS):
            raise ValueError("Sequence feature contract includes a future or identity field")
        numeric = train[ACTION_VALUE_NUMERIC_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        self.medians = np.nanmedian(numeric, axis=0)
        if not np.isfinite(self.medians).all():
            raise ValueError("Every numeric feature needs an observed training value")
        filled = np.where(np.isfinite(numeric), numeric, self.medians)
        self.means = filled.mean(axis=0)
        self.scales = filled.std(axis=0)
        self.scales[self.scales < 1e-8] = 1.0
        binary = train[ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        self.binary_modes = np.array(
            [1.0 if np.nansum(binary[:, i]) > np.isfinite(binary[:, i]).sum() / 2 else 0.0
             for i in range(binary.shape[1])],
            dtype=np.float32,
        )
        for column in ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS:
            observed = train[column].dropna().astype(str)
            vocabulary = {UNKNOWN: 0}
            for value in sorted(set(observed) - {UNKNOWN}):
                vocabulary[value] = len(vocabulary)
            self.vocabularies[column] = vocabulary
        return self

    def transform(self, states: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.medians is None or self.means is None or self.scales is None or self.binary_modes is None:
            raise RuntimeError("Preprocessor must be fitted on training states")
        numeric = states[ACTION_VALUE_NUMERIC_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        numeric = np.where(np.isfinite(numeric), numeric, self.medians)
        continuous = ((numeric - self.means) / self.scales).astype(np.float32)
        binary = states[ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        binary = np.where(np.isfinite(binary), binary, self.binary_modes).astype(np.float32)
        categories = np.column_stack([
            states[column].fillna(UNKNOWN).astype(str).map(
                self.vocabularies[column]
            ).fillna(0).to_numpy(dtype=np.int64)
            for column in ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS
        ])
        if not np.isfinite(continuous).all() or not np.isfinite(binary).all():
            raise AssertionError("Transformed state features must be finite")
        return continuous, binary, categories

    def metadata(self) -> dict[str, object]:
        if self.medians is None or self.means is None or self.scales is None or self.binary_modes is None:
            raise RuntimeError("Preprocessor has not been fitted")
        return {
            "numeric_columns": ACTION_VALUE_NUMERIC_FEATURE_COLUMNS,
            "binary_columns": ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS,
            "categorical_columns": ACTION_VALUE_CATEGORICAL_FEATURE_COLUMNS,
            "medians": self.medians.tolist(),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "binary_modes": self.binary_modes.tolist(),
            "vocabularies": self.vocabularies,
            "unknown_index": 0,
        }


def ordered_states(states: pd.DataFrame) -> pd.DataFrame:
    """Sort eligible pre-event states without changing their original row identity."""
    required = set(ORDER_COLUMNS + ["event_id", TARGET_COLUMN, *ACTION_VALUE_FEATURE_COLUMNS])
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"States are missing required columns: {sorted(missing)}")
    if states["event_id"].isna().any() or states["event_id"].duplicated().any():
        raise ValueError("Every state needs a unique event ID")
    if states.duplicated(ORDER_COLUMNS).any():
        raise ValueError("State indices must be unique within a possession")
    target = pd.to_numeric(states[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or target.lt(0).any() or not np.isfinite(target).all():
        raise ValueError("Future-OOF-xG target must be finite and non-negative")
    return states.sort_values(ORDER_COLUMNS, kind="stable")


def make_sequences(states: pd.DataFrame, preprocessor: StatePreprocessor) -> list[SequenceRecord]:
    """One row per pre-event state; timestep t sees only features through that state.

    The target at t includes an eligible shot at event t because the current event
    has not happened yet. A unidirectional GRU prevents later timesteps from
    changing the hidden state used for this pre-event prediction.
    """
    ordered = ordered_states(states)
    continuous, binary, categorical = preprocessor.transform(ordered)
    target = ordered[TARGET_COLUMN].to_numpy(dtype=np.float32)
    indices = ordered.index.to_numpy(dtype=np.int64)
    records: list[SequenceRecord] = []
    positions = ordered.reset_index(drop=True).groupby(GROUP_COLUMNS, sort=False).indices
    for (match_id, possession_id), position in positions.items():
        records.append(SequenceRecord(
            continuous=continuous[position],
            binary=binary[position],
            categorical=categorical[position],
            target=target[position],
            row_index=indices[position],
            match_id=int(match_id),
            possession_id=int(possession_id),
        ))
    if sum(len(record.target) for record in records) != len(states):
        raise AssertionError("Sequence construction must preserve every state once")
    return records


class PossessionDataset(Dataset[SequenceRecord]):
    def __init__(self, records: list[SequenceRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceRecord:
        return self.records[index]


def collate_possessions(records: list[SequenceRecord]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(record.target) for record in records], dtype=torch.long)
    maximum = int(lengths.max())
    mask = torch.arange(maximum)[None, :] < lengths[:, None]

    def padded(field: str, dtype: torch.dtype) -> torch.Tensor:
        tensors = [torch.as_tensor(getattr(record, field), dtype=dtype) for record in records]
        return pad_sequence(tensors, batch_first=True)

    return {
        "continuous": padded("continuous", torch.float32),
        "binary": padded("binary", torch.float32),
        "categorical": padded("categorical", torch.long),
        "target": padded("target", torch.float32),
        "row_index": padded("row_index", torch.long),
        "lengths": lengths,
        "mask": mask,
    }
