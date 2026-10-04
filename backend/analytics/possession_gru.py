"""Small causal GRU and current-state neural ablation for possession value."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from analytics.action_value_features import (
    ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS,
    ACTION_VALUE_NUMERIC_FEATURE_COLUMNS,
)


@dataclass(frozen=True)
class NeuralConfig:
    hidden_size: int = 64
    gru_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 18
    patience: int = 4
    seed: int = 42
    huber_beta: float = 0.05
    gradient_clip: float = 1.0

    def metadata(self) -> dict[str, int | float]:
        return asdict(self)


def embedding_sizes(vocabulary_sizes: list[int]) -> list[int]:
    return [min(8, max(2, (size + 1) // 2)) for size in vocabulary_sizes]


class StateEncoder(nn.Module):
    def __init__(self, vocabulary_sizes: list[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, width, padding_idx=0)
            for size, width in zip(vocabulary_sizes, embedding_sizes(vocabulary_sizes), strict=True)
        )
        self.output_size = (
            len(ACTION_VALUE_NUMERIC_FEATURE_COLUMNS)
            + len(ACTION_VALUE_BOOLEAN_FEATURE_COLUMNS)
            + sum(embedding_sizes(vocabulary_sizes))
        )

    def forward(
        self, continuous: torch.Tensor, binary: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        encoded = [continuous, binary]
        encoded.extend(
            embedding(categorical[:, :, index])
            for index, embedding in enumerate(self.embeddings)
        )
        return torch.cat(encoded, dim=-1)


class PossessionGRU(nn.Module):
    """A unidirectional recurrent state estimate at every pre-event timestep."""

    def __init__(self, vocabulary_sizes: list[int], config: NeuralConfig) -> None:
        super().__init__()
        self.encoder = StateEncoder(vocabulary_sizes)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.output_size, config.hidden_size), nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus()
        )
        nn.init.constant_(self.head[2].bias, -4.0)

    def forward(
        self,
        continuous: torch.Tensor,
        binary: torch.Tensor,
        categorical: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        current = self.projection(self.encoder(continuous, binary, categorical))
        packed = pack_padded_sequence(current, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        padded, _ = pad_packed_sequence(
            hidden, batch_first=True, total_length=continuous.shape[1]
        )
        return self.head(padded).squeeze(-1)


class CurrentStateMLP(nn.Module):
    """Same inputs and target, but no history or recurrence."""

    def __init__(self, vocabulary_sizes: list[int], config: NeuralConfig) -> None:
        super().__init__()
        self.encoder = StateEncoder(vocabulary_sizes)
        self.network = nn.Sequential(
            nn.Linear(self.encoder.output_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        nn.init.constant_(self.network[-2].bias, -4.0)

    def forward(
        self,
        continuous: torch.Tensor,
        binary: torch.Tensor,
        categorical: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        del lengths
        return self.network(self.encoder(continuous, binary, categorical)).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
