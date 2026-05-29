"""Isaac-independent core for A*-conditioned residual navigation.

This module is deliberately not an Isaac Lab ``DirectRLEnv``.  It packages the
common logic needed by: unit tests, offline log replay, BC data collection, PPO
wrappers, and real dry-runs.  Isaac-specific vectorized stepping can wrap this
core later without duplicating feature construction or local-policy logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from g1_nav_rl.features.path_features import fixed_body_path_points
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.runtime import NavRuntime, NavRuntimeConfig, NavRuntimeStep
from g1_nav_rl.navigation.simple_nav import (
    Pose2D,
    SimpleNavConfig,
    SimpleNavigator,
    Target2D,
    world_to_body_xy,
)
from g1_nav_rl.planning.grid_astar import GridAStarCfg
from g1_nav_rl.policies.local_command_policy import LocalCommandPolicy, LocalPolicyInput, TrackerState
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicy, RLResidualPolicyCfg


@dataclass(frozen=True)
class NavResidualCoreCfg:
    """Configuration for the hardware-neutral residual policy core."""

    grid_cfg: OccupancyGridCfg = field(default_factory=OccupancyGridCfg)
    astar_cfg: GridAStarCfg = field(default_factory=lambda: GridAStarCfg(inflation_radius=0.35))
    nav_cfg: SimpleNavConfig = field(default_factory=SimpleNavConfig)
    runtime_cfg: NavRuntimeConfig = field(default_factory=NavRuntimeConfig)
    residual_policy_cfg: RLResidualPolicyCfg = field(default_factory=RLResidualPolicyCfg)
    path_points: int = 8
    path_spacing_m: float = 0.25
    default_tracker_state: TrackerState = TrackerState.TRACKING


@dataclass(frozen=True)
class NavResidualCoreStep:
    """One output of :class:`NavResidualCore`."""

    command: torch.Tensor
    nominal_command: torch.Tensor
    residual: torch.Tensor | None
    policy_input: LocalPolicyInput
    runtime_step: NavRuntimeStep
    metadata: Mapping[str, Any]


class NavResidualCore:
    """Run A* runtime + local residual policy for one navigation stream."""

    def __init__(
        self,
        cfg: NavResidualCoreCfg | None = None,
        *,
        policy: LocalCommandPolicy | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or NavResidualCoreCfg()
        self.device = torch.device(device)
        navigator = SimpleNavigator(
            grid_cfg=self.cfg.grid_cfg,
            astar_cfg=self.cfg.astar_cfg,
            config=self.cfg.nav_cfg,
        )
        self.runtime = NavRuntime(navigator=navigator, config=self.cfg.runtime_cfg)
        self.policy = policy or RLResidualPolicy(self.cfg.residual_policy_cfg, device=self.device)
        self.step_index = 0
        self.last_command = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        self.runtime.reset()
        self.step_index = 0
        self.last_command.zero_()
        self.policy.reset(num_envs=1, device=self.device)

    def build_policy_input(
        self,
        *,
        pose: Pose2D,
        target: Target2D,
        occupancy_grid: torch.Tensor,
        nominal_command: torch.Tensor,
        target_visible: bool = True,
        target_age_s: float = 0.0,
        tracker_state: TrackerState | int = TrackerState.TRACKING,
    ) -> LocalPolicyInput:
        """Construct a batched ``LocalPolicyInput`` from current runtime state."""

        if occupancy_grid.ndim == 2:
            grid = occupancy_grid.unsqueeze(0)
        elif occupancy_grid.ndim == 3 and occupancy_grid.shape[0] == 1:
            grid = occupancy_grid
        else:
            raise ValueError(f"occupancy_grid must have shape [H,W] or [1,H,W], got {tuple(occupancy_grid.shape)}")
        grid = grid.to(device=self.device, dtype=torch.float32)
        target_xy_b = torch.tensor([world_to_body_xy(target.xy, pose)], dtype=torch.float32, device=self.device)
        remaining_body = self.runtime.navigator.remaining_body_path(pose)
        path_xy_b = fixed_body_path_points(
            remaining_body,
            num_points=int(self.cfg.path_points),
            spacing_m=float(self.cfg.path_spacing_m),
            device=self.device,
        ).unsqueeze(0)
        return LocalPolicyInput(
            occupancy_grid=grid,
            target_xy_b=target_xy_b,
            path_xy_b=path_xy_b,
            nominal_command=nominal_command.to(device=self.device, dtype=torch.float32),
            last_command=self.last_command.to(device=self.device, dtype=torch.float32),
            target_visible=torch.tensor([bool(target_visible)], dtype=torch.float32, device=self.device),
            target_age_s=torch.tensor([float(target_age_s)], dtype=torch.float32, device=self.device),
            tracker_state=torch.tensor([int(tracker_state)], dtype=torch.long, device=self.device),
        )

    def step(
        self,
        *,
        pose: Pose2D,
        target: Target2D,
        occupancy_grid: torch.Tensor,
        target_visible: bool = True,
        target_age_s: float = 0.0,
        tracker_state: TrackerState | int | None = None,
    ) -> NavResidualCoreStep:
        runtime_step = self.runtime.step(pose, target, occupancy_grid, step_index=self.step_index)
        self.step_index += 1
        nav_command = runtime_step.command.command
        nominal = torch.tensor([nav_command.tuple], dtype=torch.float32, device=self.device)
        policy_input = self.build_policy_input(
            pose=pose,
            target=target,
            occupancy_grid=occupancy_grid,
            nominal_command=nominal,
            target_visible=target_visible,
            target_age_s=target_age_s,
            tracker_state=self.cfg.default_tracker_state if tracker_state is None else tracker_state,
        )
        policy_output = self.policy.step(policy_input)
        self.last_command = policy_output.command.detach().clone()
        metadata = {
            "policy": "nav_residual_core",
            "runtime": runtime_step,
            "local_policy": None if policy_output.metadata is None else dict(policy_output.metadata),
        }
        return NavResidualCoreStep(
            command=policy_output.command,
            nominal_command=nominal,
            residual=policy_output.residual,
            policy_input=policy_input,
            runtime_step=runtime_step,
            metadata=metadata,
        )


__all__ = ["NavResidualCore", "NavResidualCoreCfg", "NavResidualCoreStep"]
