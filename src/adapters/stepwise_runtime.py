"""Adapter-layer runtime for the input-agnostic stepwise navigator.

This module wires existing normalized adapter contracts into
``g1_nav_rl.navigation.stepwise``.  It stays hardware-neutral: callers can feed
observations from Isaac, replay, or real sensors as long as they are represented
as ``RobotObservation`` with optional ``target_hint`` and ``lidar`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from g1_nav_rl.navigation.stepwise import (
    StepwiseDecision,
    StepwiseNavConfig,
    StepwiseNavigator,
    front_clearance_from_points,
)

from .g1_policy_runtime import occupancy_grid_from_observation
from .types import Command2D, RobotObservation


@dataclass(frozen=True)
class StepwiseTargetNavPolicyRuntimeConfig:
    """Configuration for adapter-level stepwise target navigation."""

    stepwise_cfg: StepwiseNavConfig = field(default_factory=StepwiseNavConfig)
    require_lidar: bool = True
    front_x_min_m: float = 0.0
    front_x_max_m: float = 1.0
    front_y_abs_max_m: float = 0.35
    front_z_min_m: float | None = None
    front_z_max_m: float | None = None
    source: str = "stepwise_target_nav_policy"


class StepwiseTargetNavPolicyRuntime:
    """Policy runtime that returns one stepwise command per observation.

    Inputs:
    - ``RobotObservation.target_hint``: body-frame stand-off goal from YOLO/RGB-D,
      a streaming detection client, or a simulator oracle.
    - ``RobotObservation.lidar``: body-frame points used to build the occupancy
      grid and compute a hard front-forward guard.

    Output:
    - ``Command2D`` with an optional positive ``duration_s`` for executable
      actions.  Stop/observe decisions return a zero command without duration.
    """

    def __init__(self, config: StepwiseTargetNavPolicyRuntimeConfig | None = None) -> None:
        self.config = config or StepwiseTargetNavPolicyRuntimeConfig()
        self.navigator = StepwiseNavigator(self.config.stepwise_cfg)
        self.last_decision: StepwiseDecision | None = None

    def reset(self) -> None:
        self.navigator.reset()
        self.last_decision = None

    def step(self, observation: RobotObservation) -> Command2D:
        target_xy_b = None if observation.target_hint is None else observation.target_hint.xy
        occupancy_grid = None
        lidar_point_count = 0
        front_min_distance_m = None

        # In Isaac smoke tests the simulator already builds the planning grid
        # according to --grid_source (lidar / world / hybrid / memory).  Do not
        # silently rebuild a different LiDAR-only grid here; otherwise A* plans
        # on one map while visualizations and legacy checks describe another.
        grid_override = None
        if observation.metadata is not None:
            grid_override = observation.metadata.get("stepwise_occupancy_grid")
        if grid_override is not None:
            grid = torch.as_tensor(grid_override, dtype=torch.float32).detach().cpu().contiguous()
            if grid.ndim == 3:
                if int(grid.shape[0]) != 1:
                    raise ValueError(
                        "stepwise_occupancy_grid override expects [H,W] or [1,H,W], "
                        f"got {tuple(grid.shape)}"
                    )
                grid = grid[0]
            if grid.ndim != 2:
                raise ValueError(
                    "stepwise_occupancy_grid override expects [H,W] or [1,H,W], "
                    f"got {tuple(grid.shape)}"
                )
            expected_shape = (self.config.stepwise_cfg.grid_cfg.height, self.config.stepwise_cfg.grid_cfg.width)
            if tuple(grid.shape) != expected_shape:
                raise ValueError(
                    "stepwise_occupancy_grid shape does not match stepwise grid_cfg: "
                    f"got {tuple(grid.shape)}, expected {expected_shape}"
                )
            occupancy_grid = grid

        if observation.lidar is not None:
            lidar_point_count = int(observation.lidar.point_count)
            if occupancy_grid is None:
                occupancy_grid, lidar_point_count = occupancy_grid_from_observation(
                    observation,
                    grid_cfg=self.config.stepwise_cfg.grid_cfg,
                    require_lidar=True,
                )
            front_min_distance_m = front_clearance_from_points(
                observation.lidar.points_body_xyz,
                x_min_m=float(self.config.front_x_min_m),
                x_max_m=float(self.config.front_x_max_m),
                y_abs_max_m=float(self.config.front_y_abs_max_m),
                z_min_m=self.config.front_z_min_m,
                z_max_m=self.config.front_z_max_m,
            )
        elif self.config.require_lidar and occupancy_grid is None:
            occupancy_grid = None
        elif occupancy_grid is None:
            occupancy_grid, lidar_point_count = occupancy_grid_from_observation(
                observation,
                grid_cfg=self.config.stepwise_cfg.grid_cfg,
                require_lidar=False,
            )

        decision = self.navigator.decide(
            target_xy_b=target_xy_b,
            occupancy_grid=occupancy_grid,
            front_min_distance_m=front_min_distance_m,
            metadata={
                "observation_timestamp_s": float(observation.timestamp_s),
                "observation_has_lidar": bool(observation.has_lidar),
                "observation_has_target_hint": bool(observation.target_hint is not None),
                "observation_has_rgbd_frame": bool(observation.has_rgbd),
                # Kept for compatibility with older trace consumers.  Prefer
                # observation_has_rgbd_frame in new logs.
                "observation_has_rgbd": bool(observation.has_rgbd),
                "lidar_point_count": int(lidar_point_count),
                "sensor_ages_s": observation.sensor_ages_s(),
            },
        )
        self.last_decision = decision
        action = decision.action
        duration_s = float(action.duration_s) if action.duration_s > 0.0 and decision.execute else None
        return Command2D(
            vx=float(action.command.vx) if decision.execute else 0.0,
            vy=float(action.command.vy) if decision.execute else 0.0,
            wz=float(action.command.wz) if decision.execute else 0.0,
            timestamp_s=float(observation.timestamp_s),
            duration_s=duration_s,
            source=str(self.config.source),
            metadata={
                "policy": "stepwise-target-nav",
                "stepwise_decision": decision.to_dict(),
            },
        )


def stepwise_decision_to_dict(decision: StepwiseDecision | None) -> dict[str, Any] | None:
    """Small helper for scripts that want JSON-friendly debug output."""

    if decision is None:
        return None
    return decision.to_dict()


__all__ = [
    "StepwiseTargetNavPolicyRuntime",
    "StepwiseTargetNavPolicyRuntimeConfig",
    "stepwise_decision_to_dict",
]
