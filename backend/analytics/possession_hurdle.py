"""Two-head, non-negative offline possession-value models and masked objective."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional

from analytics.possession_gru import StateEncoder


@dataclass(frozen=True)
class HurdleConfig:
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ffn_size: int = 128
    dropout: float = 0.1
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 12
    patience: int = 3
    classification_weight: float = 1.0
    magnitude_weight: float = 1.0
    huber_beta: float = 0.05
    gradient_clip: float = 1.0
    seed: int = 42
    use_fusion: bool = True
    log1p_magnitude: bool = False

    def metadata(self) -> dict[str, int | float | bool]:
        return asdict(self)


def positive_target(target: torch.Tensor) -> torch.Tensor:
    """The binary label is strictly positive remaining future OOF xG."""
    return (target > 0).to(target.dtype)


def expected_value(logit: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    if torch.any(magnitude < 0):
        raise ValueError("Conditional magnitude must be non-negative")
    return torch.sigmoid(logit) * magnitude


def hurdle_loss(
    logit: torch.Tensor,
    magnitude: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    config: HurdleConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exclude padding and regress magnitude on positive eligible states only.

    Unweighted BCE preserves the probability interpretation. The training-set
    class balance is reported, but no class weighting is applied by default.
    """
    if not torch.any(eligible):
        raise ValueError("At least one eligible state is required")
    binary = positive_target(target)
    classification = functional.binary_cross_entropy_with_logits(
        logit[eligible], binary[eligible]
    )
    positive = eligible & (target > 0)
    magnitude_prediction = torch.log1p(magnitude[positive]) if config.log1p_magnitude else magnitude[positive]
    magnitude_target = torch.log1p(target[positive]) if config.log1p_magnitude else target[positive]
    magnitude_loss = (
        functional.smooth_l1_loss(
            magnitude_prediction, magnitude_target, beta=config.huber_beta
        )
        if torch.any(positive)
        else magnitude.sum() * 0.0
    )
    total = (
        config.classification_weight * classification
        + config.magnitude_weight * magnitude_loss
    )
    return total, classification, magnitude_loss


class HurdleHeads(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float,
        log1p_magnitude: bool = False,
    ) -> None:
        super().__init__()
        self.log1p_magnitude = log1p_magnitude
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classification = nn.Linear(hidden_size, 1)
        self.magnitude = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.classification.bias, -1.7)
        nn.init.constant_(self.magnitude.bias, -2.5)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(state)
        transformed_magnitude = functional.softplus(self.magnitude(shared).squeeze(-1))
        magnitude = (
            torch.expm1(transformed_magnitude)
            if self.log1p_magnitude else transformed_magnitude
        )
        return self.classification(shared).squeeze(-1), magnitude


class CurrentStateHurdleMLP(nn.Module):
    """Current pre-event features only; sequence length is immaterial."""

    def __init__(self, vocabulary_sizes: list[int], config: HurdleConfig) -> None:
        super().__init__()
        self.encoder = StateEncoder(vocabulary_sizes)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.output_size, config.d_model),
            nn.ReLU(),
        )
        self.heads = HurdleHeads(
            config.d_model, config.d_model, config.dropout, config.log1p_magnitude
        )

    def forward(
        self,
        continuous: torch.Tensor,
        binary: torch.Tensor,
        categorical: torch.Tensor,
        eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del eligible
        state = self.projection(self.encoder(continuous, binary, categorical))
        return self.heads(state)
