"""Policy wrapper around the existing DWA/DWB-style local velocity rollout."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.planning.local_velocity_rollout import LocalVelocityRolloutCfg, plan_local_velocity_command

from .local_command_policy import LocalPolicyInput, LocalPolicyOutput


@dataclass(frozen=True)
class LocalRolloutPolicyCfg:
    """Configuration for using local velocity rollout as a local policy."""

    grid_cfg: OccupancyGridCfg = field(default_factory=OccupancyGridCfg)
    rollout_cfg: LocalVelocityRolloutCfg = field(default_factory=LocalVelocityRolloutCfg)
    path_target_index: int = 2
    source: str = "local_rollout_policy"

    def __post_init__(self) -> None:
        if int(self.path_target_index) < 0:
            raise ValueError("path_target_index must be non-negative")


class LocalRolloutPolicy:
    """Select a smooth local command using sampled command rollouts."""

    def __init__(self, cfg: LocalRolloutPolicyCfg | None = None) -> None:
        self.cfg = cfg or LocalRolloutPolicyCfg()
        self.last_info: dict[str, torch.Tensor] | None = None

    def reset(self, *, num_envs: int | None = None, device: torch.device | str | None = None) -> None:
        self.last_info = None

    def _select_local_target(self, policy_input: LocalPolicyInput) -> torch.Tensor:
        path = policy_input.path_xy_b
        if path.shape[1] == 0:
            return policy_input.target_xy_b
        idx = min(int(self.cfg.path_target_index), path.shape[1] - 1)
        candidate = path[:, idx, :]
        nonzero = torch.linalg.norm(candidate, dim=-1) > 1.0e-6
        return torch.where(nonzero[:, None], candidate, policy_input.target_xy_b)

    def step(self, policy_input: LocalPolicyInput) -> LocalPolicyOutput:
        if policy_input.occupancy_grid is None:
            raise ValueError("LocalRolloutPolicy requires occupancy_grid")
        local_target = self._select_local_target(policy_input).to(dtype=torch.float32)
        command, info = plan_local_velocity_command(
            occupancy_grid=policy_input.occupancy_grid.to(dtype=torch.float32),
            target_xy_b=local_target,
            last_command=policy_input.last_command,
            grid_cfg=self.cfg.grid_cfg,
            cfg=self.cfg.rollout_cfg,
        )
        self.last_info = info
        metadata: dict[str, Any] = {
            "policy": "local_rollout",
            "path_target_index": int(self.cfg.path_target_index),
            "best_score": info.get("best_score"),
            "best_clearance": info.get("best_clearance"),
            "any_collision_free": info.get("any_collision_free"),
        }
        return LocalPolicyOutput(command=command, residual=command - policy_input.nominal_command, metadata=metadata)


__all__ = ["LocalRolloutPolicy", "LocalRolloutPolicyCfg"]
