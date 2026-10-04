"""Small causal Transformer with explicit current pre-event state fusion."""

from __future__ import annotations

import math

import torch
from torch import nn

from analytics.possession_gru import StateEncoder
from analytics.possession_hurdle import HurdleConfig, HurdleHeads


def causal_attention_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
    """True entries forbid attention from timestep t to any later timestep."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def sinusoidal_positions(length: int, width: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros(length, width, device=device)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: width // 2])
    return encoding


class HybridCausalTransformer(nn.Module):
    """One pass over each possession yields all causally valid state predictions."""

    def __init__(self, vocabulary_sizes: list[int], config: HurdleConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config
        self.encoder = StateEncoder(vocabulary_sizes)
        self.history_projection = nn.Linear(self.encoder.output_size, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_size,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        if config.use_fusion:
            self.current_projection = nn.Sequential(
                nn.Linear(self.encoder.output_size, config.d_model), nn.ReLU()
            )
        self.heads = HurdleHeads(
            config.d_model * (2 if config.use_fusion else 1),
            config.d_model,
            config.dropout,
            config.log1p_magnitude,
        )

    def forward(
        self,
        continuous: torch.Tensor,
        binary: torch.Tensor,
        categorical: torch.Tensor,
        eligible: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.encoder(continuous, binary, categorical)
        history = self.history_projection(current)
        history = history + sinusoidal_positions(
            history.shape[1], history.shape[2], history.device
        )[None, :, :]
        history = self.transformer(
            history,
            mask=causal_attention_mask(history.shape[1], history.device),
            src_key_padding_mask=~eligible,
        )
        representation = (
            torch.cat([history, self.current_projection(current)], dim=-1)
            if self.config.use_fusion
            else history
        )
        return self.heads(representation)
