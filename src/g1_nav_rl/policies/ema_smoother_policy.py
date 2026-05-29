"""Deterministic EMA command smoother using the local-policy interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .local_command_policy import LocalPolicyInput, LocalPolicyOutput


@dataclass(frozen=True)
class EMASmootherPolicyCfg:
    """Configuration for a first-order command smoother.

    ``alpha`` is the weight on the new nominal command.  Acceleration limits are
    per policy step, not per second; the caller may derive them from its control
    period before constructing the config.
    """

    alpha: float = 0.35
    max_delta_vx: float = 0.08
    max_delta_vy: float = 0.06
    max_delta_wz: float = 0.15
    source: str = "ema_smoother_policy"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("EMASmootherPolicyCfg.alpha must be in [0,1]")
        for name in ("max_delta_vx", "max_delta_vy", "max_delta_wz"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")


class EMASmootherPolicy:
    """Smooth the deterministic command without learning."""

    def __init__(self, cfg: EMASmootherPolicyCfg | None = None) -> None:
        self.cfg = cfg or EMASmootherPolicyCfg()
        self._last_output: torch.Tensor | None = None

    def reset(self, *, num_envs: int | None = None, device: torch.device | str | None = None) -> None:
        if num_envs is None:
            self._last_output = None
        else:
            self._last_output = torch.zeros((int(num_envs), 3), dtype=torch.float32, device=device or "cpu")

    def step(self, policy_input: LocalPolicyInput) -> LocalPolicyOutput:
        nominal = policy_input.nominal_command.to(dtype=torch.float32)
        previous = policy_input.last_command.to(device=nominal.device, dtype=torch.float32)
        if self._last_output is not None and self._last_output.shape == nominal.shape:
            previous = self._last_output.to(device=nominal.device, dtype=torch.float32)

        raw = float(self.cfg.alpha) * nominal + (1.0 - float(self.cfg.alpha)) * previous
        delta = raw - previous
        limits = torch.tensor(
            [float(self.cfg.max_delta_vx), float(self.cfg.max_delta_vy), float(self.cfg.max_delta_wz)],
            dtype=torch.float32,
            device=nominal.device,
        )
        limited_delta = torch.maximum(torch.minimum(delta, limits), -limits)
        command = previous + limited_delta
        self._last_output = command.detach().clone()
        metadata: dict[str, Any] = {
            "policy": "ema_smoother",
            "alpha": float(self.cfg.alpha),
            "max_delta": limits.detach().cpu(),
        }
        return LocalPolicyOutput(command=command.contiguous(), residual=command - nominal, metadata=metadata)


__all__ = ["EMASmootherPolicy", "EMASmootherPolicyCfg"]
