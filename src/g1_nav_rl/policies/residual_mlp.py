"""Small MLP used by the residual local-command policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualMLPCfg:
    """Architecture config for compact-feature residual control."""

    input_dim: int
    hidden_sizes: tuple[int, ...] = (128, 128)
    output_dim: int = 3
    activation: str = "elu"
    zero_init_output: bool = True

    def __post_init__(self) -> None:
        if int(self.input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(self.output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must be non-empty")
        if str(self.activation).lower() not in {"elu", "relu", "tanh", "gelu"}:
            raise ValueError("activation must be one of: elu, relu, tanh, gelu")


def _activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "elu":
        return nn.ELU()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name!r}")


class ResidualMLP(nn.Module):
    """Feed-forward network producing bounded-command residual logits."""

    def __init__(self, cfg: ResidualMLPCfg) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = int(cfg.input_dim)
        for hidden in cfg.hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden)))
            layers.append(_activation(cfg.activation))
            in_dim = int(hidden)
        final = nn.Linear(in_dim, int(cfg.output_dim))
        if bool(cfg.zero_init_output):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != int(self.cfg.input_dim):
            raise ValueError(
                f"features must have shape [N,{self.cfg.input_dim}], got {tuple(features.shape)}"
            )
        return self.net(features.to(dtype=torch.float32))


__all__ = ["ResidualMLP", "ResidualMLPCfg"]
