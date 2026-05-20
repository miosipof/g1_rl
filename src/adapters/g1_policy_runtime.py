"""g1_rl policy/runtime adapters for dry-run and staged sim-to-real tests.

This module is the seam between the hardware-neutral adapter layer and the
existing deterministic ``g1_nav_rl`` navigation stack.  It deliberately avoids
Isaac, Unitree, RealSense, Livox SDK, ROS, and OpenCV imports.

The initial runtime implemented here is intentionally conservative:

- ``ConstantVelocityPolicy`` reproduces the previous dry-run dummy policy.
- ``LidarTargetNavPolicyRuntime`` consumes ``RobotObservation.lidar`` and a
  manually supplied ``Target2D``; it rasterizes the body-frame LiDAR cloud into
  a local occupancy grid and calls ``g1_nav_rl.navigation.NavRuntime`` to emit a
  body-frame ``Command2D``.

This is not full target-following yet.  It is the first meaningful bridge from
real/replay LiDAR observations into the same navigation logic validated in sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg, rasterize_points_to_occupancy_grid
from g1_nav_rl.navigation.runtime import NavRuntime, NavRuntimeConfig, NavRuntimeStep
from g1_nav_rl.navigation.simple_nav import Pose2D, SimpleNavConfig, SimpleNavigator, Target2D
from g1_nav_rl.planning.grid_astar import GridAStarCfg

from .config import AdapterRuntimeConfig
from .types import Command2D, LidarFrame, RobotObservation


@dataclass(frozen=True)
class ConstantVelocityPolicy:
    """Minimal policy used for bridge smoke tests.

    It is retained as a baseline because it proves the bridge/safety/sink path
    without depending on any navigation or perception behavior.
    """

    vx: float = 0.05
    vy: float = 0.0
    wz: float = 0.0
    source: str = "constant_velocity_policy"

    def step(self, observation: RobotObservation) -> Command2D:
        return Command2D(
            self.vx,
            self.vy,
            self.wz,
            timestamp_s=observation.timestamp_s,
            source=self.source,
            metadata={"policy": "constant", "observation_has_lidar": observation.has_lidar},
        )


@dataclass(frozen=True)
class LidarTargetNavPolicyRuntimeConfig:
    """Configuration for LiDAR-only manual-target navigation dry-runs."""

    target: Target2D
    grid_cfg: OccupancyGridCfg = field(default_factory=OccupancyGridCfg)
    astar_cfg: GridAStarCfg = field(default_factory=lambda: GridAStarCfg(inflation_radius=0.35))
    nav_cfg: SimpleNavConfig = field(default_factory=SimpleNavConfig)
    runtime_cfg: NavRuntimeConfig = field(default_factory=NavRuntimeConfig)
    require_lidar: bool = True
    require_pose: bool = False
    default_pose: Pose2D = field(default_factory=lambda: Pose2D(0.0, 0.0, 0.0))
    source: str = "lidar_target_nav_policy"

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: AdapterRuntimeConfig,
        *,
        target: Target2D,
        enable_replanning: bool = False,
        replan_every: int = 0,
        inflation_radius_m: float | None = None,
        max_vx: float | None = None,
        max_vy: float | None = None,
        max_wz: float | None = None,
    ) -> "LidarTargetNavPolicyRuntimeConfig":
        """Build a conservative navigation config from the real adapter config.

        Command maxima default to the real command limits, so proposed commands
        are already close to what the real safety supervisor will allow.  The
        safety supervisor still remains authoritative and may further clip/stop.
        """

        limits = runtime_config.command_limits
        nav_cfg = SimpleNavConfig(
            max_vx=float(max_vx if max_vx is not None else max(abs(limits.vx_min), abs(limits.vx_max))),
            max_vy=float(max_vy if max_vy is not None else max(abs(limits.vy_min), abs(limits.vy_max))),
            max_wz=float(max_wz if max_wz is not None else max(abs(limits.wz_min), abs(limits.wz_max))),
            allow_backward=False,
        )
        astar_cfg = GridAStarCfg(
            inflation_radius=float(inflation_radius_m if inflation_radius_m is not None else 0.35),
            allow_diagonal=True,
            heuristic_weight=1.0,
            max_expansions=0,
        )
        runtime_cfg = NavRuntimeConfig(enable_replanning=bool(enable_replanning), replan_every=int(replan_every))
        return cls(target=target, nav_cfg=nav_cfg, astar_cfg=astar_cfg, runtime_cfg=runtime_cfg)


@dataclass(frozen=True)
class LidarTargetNavDebug:
    """Debug information from the most recent navigation-policy step."""

    target_xy: tuple[float, float]
    pose_xy_yaw: tuple[float, float, float]
    lidar_points: int
    occupied_cells: int
    plan_found: bool
    plan_reason: str | None
    waypoint_index: int
    waypoint_count: int
    target_distance: float
    target_reached: bool
    replan_count: int
    path_blocked: bool
    command: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_xy": list(self.target_xy),
            "pose_xy_yaw": list(self.pose_xy_yaw),
            "lidar_points": self.lidar_points,
            "occupied_cells": self.occupied_cells,
            "plan_found": self.plan_found,
            "plan_reason": self.plan_reason,
            "waypoint_index": self.waypoint_index,
            "waypoint_count": self.waypoint_count,
            "target_distance": self.target_distance,
            "target_reached": self.target_reached,
            "replan_count": self.replan_count,
            "path_blocked": self.path_blocked,
            "command": list(self.command),
        }


class LidarTargetNavPolicyRuntime:
    """Manual-target LiDAR navigation policy for dry-run sim-to-real bridging.

    Inputs:
      - ``RobotObservation.pose`` or ``RobotObservation.robot_state.pose``;
      - ``RobotObservation.lidar.points_body_xyz``;
      - a configured world-frame ``Target2D`` or ``observation.target_hint``.

    Output:
      - ``Command2D`` proposed by the deterministic ``g1_nav_rl`` NavRuntime.
    """

    def __init__(self, config: LidarTargetNavPolicyRuntimeConfig) -> None:
        self.config = config
        navigator = SimpleNavigator(
            grid_cfg=config.grid_cfg,
            astar_cfg=config.astar_cfg,
            config=config.nav_cfg,
        )
        self.runtime = NavRuntime(navigator=navigator, config=config.runtime_cfg)
        self.step_index = 0
        self._last_target: Target2D = config.target
        self.last_debug: LidarTargetNavDebug | None = None

    def reset(self) -> None:
        self.runtime.reset()
        self.step_index = 0
        self.last_debug = None
        self._last_target = self.config.target

    def step(self, observation: RobotObservation) -> Command2D:
        pose = pose_from_observation(
            observation,
            default_pose=self.config.default_pose,
            require_pose=self.config.require_pose,
        )
        target = observation.target_hint or self.config.target
        if target.xy != self._last_target.xy:
            self.runtime.reset()
            self._last_target = target

        occupancy_grid, point_count = occupancy_grid_from_observation(
            observation,
            grid_cfg=self.config.grid_cfg,
            require_lidar=self.config.require_lidar,
        )
        runtime_step = self.runtime.step(
            pose,
            target,
            occupancy_grid,
            step_index=self.step_index,
        )
        self.step_index += 1

        nav_command = runtime_step.command.command
        debug = make_lidar_nav_debug(
            runtime_step=runtime_step,
            pose=pose,
            target=target,
            occupancy_grid=occupancy_grid,
            lidar_point_count=point_count,
            command_tuple=nav_command.tuple,
        )
        self.last_debug = debug
        return Command2D(
            vx=float(nav_command.vx),
            vy=float(nav_command.vy),
            wz=float(nav_command.wz),
            timestamp_s=observation.timestamp_s,
            source=self.config.source,
            metadata={"policy": "target-nav", **debug.to_dict()},
        )


# Backwards-compatible alias for shorter imports in scripts/tests.
TargetNavPolicyRuntime = LidarTargetNavPolicyRuntime
TargetNavPolicyRuntimeConfig = LidarTargetNavPolicyRuntimeConfig


def pose_from_observation(
    observation: RobotObservation,
    *,
    default_pose: Pose2D | None = None,
    require_pose: bool = False,
) -> Pose2D:
    """Extract the best available ``Pose2D`` from a robot observation."""

    if observation.pose is not None:
        return observation.pose
    if observation.robot_state is not None and observation.robot_state.pose is not None:
        return observation.robot_state.pose
    if require_pose:
        raise ValueError("RobotObservation does not contain pose or robot_state.pose")
    if default_pose is None:
        return Pose2D(0.0, 0.0, 0.0)
    return default_pose


def occupancy_grid_from_lidar_frame(lidar: LidarFrame, *, grid_cfg: OccupancyGridCfg) -> torch.Tensor:
    """Rasterize one normalized body-frame LiDAR frame into ``[H, W]`` grid."""

    points = torch.as_tensor(lidar.points_body_xy, dtype=torch.float32).unsqueeze(0)
    grid = rasterize_points_to_occupancy_grid(points, valid=None, cfg=grid_cfg)
    return grid[0].contiguous()


def occupancy_grid_from_observation(
    observation: RobotObservation,
    *,
    grid_cfg: OccupancyGridCfg,
    require_lidar: bool = True,
) -> tuple[torch.Tensor, int]:
    """Build a local occupancy grid from ``RobotObservation.lidar``."""

    if observation.lidar is None:
        if require_lidar:
            raise ValueError("RobotObservation.lidar is required for target-nav policy")
        return torch.zeros((grid_cfg.height, grid_cfg.width), dtype=torch.float32), 0
    return occupancy_grid_from_lidar_frame(observation.lidar, grid_cfg=grid_cfg), observation.lidar.point_count


def make_lidar_nav_debug(
    *,
    runtime_step: NavRuntimeStep,
    pose: Pose2D,
    target: Target2D,
    occupancy_grid: torch.Tensor,
    lidar_point_count: int,
    command_tuple: tuple[float, float, float],
) -> LidarTargetNavDebug:
    plan = runtime_step.plan
    nav_cmd = runtime_step.command
    return LidarTargetNavDebug(
        target_xy=target.xy,
        pose_xy_yaw=(pose.x, pose.y, pose.yaw),
        lidar_points=int(lidar_point_count),
        occupied_cells=int((occupancy_grid > 0.0).sum().item()),
        plan_found=False if plan is None else bool(plan.found),
        plan_reason=None if plan is None else str(plan.reason),
        waypoint_index=int(nav_cmd.waypoint_index),
        waypoint_count=0 if plan is None else int(plan.waypoint_count),
        target_distance=float(nav_cmd.target_distance),
        target_reached=bool(nav_cmd.target_reached),
        replan_count=int(runtime_step.replan_count),
        path_blocked=bool(runtime_step.path_conflict.blocked),
        command=tuple(float(v) for v in command_tuple),
    )


__all__ = [
    "ConstantVelocityPolicy",
    "LidarTargetNavDebug",
    "LidarTargetNavPolicyRuntime",
    "LidarTargetNavPolicyRuntimeConfig",
    "TargetNavPolicyRuntime",
    "TargetNavPolicyRuntimeConfig",
    "make_lidar_nav_debug",
    "occupancy_grid_from_lidar_frame",
    "occupancy_grid_from_observation",
    "pose_from_observation",
]
