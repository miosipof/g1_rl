from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NavigationRewardCfg:
    """Simple command-level navigation reward configuration.

    This reward is intentionally minimal. It is sufficient for an initial
    environment API and deterministic-policy evaluation, but should be revisited
    before serious RL training.
    """

    progress_weight: float = 1.0
    command_penalty_weight: float = 0.01
    success_bonus: float = 1.0


def compute_navigation_reward(
    progress: torch.Tensor,
    command: torch.Tensor,
    success: torch.Tensor,
    cfg: NavigationRewardCfg,
) -> torch.Tensor:
    """Compute a batched scalar reward.

    Args:
        progress: Positive when distance to target decreases, shape ``[N]``.
        command: Final command sent to the proxy, shape ``[N, 3]``.
        success: Target-success flags, shape ``[N]``.
        cfg: Reward weights.

    Returns:
        Reward tensor with shape ``[N]``.
    """
    if progress.ndim != 1:
        raise ValueError(f"progress must have shape [N], got {tuple(progress.shape)}")
    if command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError(f"command must have shape [N, 3], got {tuple(command.shape)}")
    if success.ndim != 1:
        raise ValueError(f"success must have shape [N], got {tuple(success.shape)}")
    if progress.shape[0] != command.shape[0] or progress.shape[0] != success.shape[0]:
        raise ValueError("progress, command, and success batch sizes must match")

    command_norm = torch.linalg.norm(command, dim=-1)
    reward = float(cfg.progress_weight) * progress
    reward = reward - float(cfg.command_penalty_weight) * command_norm
    reward = reward + success.to(dtype=reward.dtype) * float(cfg.success_bonus)
    return reward
