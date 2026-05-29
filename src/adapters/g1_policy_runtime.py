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
from g1_nav_rl.envs.nav_residual_core import NavResidualCore, NavResidualCoreCfg, NavResidualCoreStep
from g1_nav_rl.navigation.runtime import NavRuntime, NavRuntimeConfig, NavRuntimeStep
from g1_nav_rl.navigation.simple_nav import Pose2D, SimpleNavConfig, SimpleNavigator, Target2D
from g1_nav_rl.planning.grid_astar import GridAStarCfg
from g1_nav_rl.policies.local_command_policy import TrackerState, tracker_state_from_value
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicy, RLResidualPolicyCfg

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
        command_mode: str = "holonomic",
        heading_lateral_scale: float = 0.20,
        heading_min_vx_scale: float = 0.25,
        heading_slow_angle_rad: float = 0.35,
        heading_rotate_in_place_angle_rad: float = 0.80,
        heading_rotate_vx_scale: float = 0.0,
        heading_rotate_lateral_scale: float = 0.0,
        path_lookahead_m: float = 0.0,
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
            command_mode=str(command_mode).replace("-", "_"),
            heading_lateral_scale=float(heading_lateral_scale),
            heading_min_vx_scale=float(heading_min_vx_scale),
            heading_slow_angle_rad=float(heading_slow_angle_rad),
            heading_rotate_in_place_angle_rad=float(heading_rotate_in_place_angle_rad),
            heading_rotate_vx_scale=float(heading_rotate_vx_scale),
            heading_rotate_lateral_scale=float(heading_rotate_lateral_scale),
            path_lookahead_m=float(path_lookahead_m),
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
    command_mode: str = "holonomic"
    waypoint_xy_b: tuple[float, float] | None = None
    waypoint_xy_w: tuple[float, float] | None = None
    path_yaw: float | None = None
    bearing_error: float | None = None
    heading_alignment_scale: float | None = None
    heading_turn_in_place: bool = False
    raw_command: tuple[float, float, float] | None = None

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
            "command_mode": self.command_mode,
            "waypoint_xy_b": None if self.waypoint_xy_b is None else list(self.waypoint_xy_b),
            "waypoint_xy_w": None if self.waypoint_xy_w is None else list(self.waypoint_xy_w),
            "path_yaw": self.path_yaw,
            "bearing_error": self.bearing_error,
            "heading_alignment_scale": self.heading_alignment_scale,
            "heading_turn_in_place": bool(self.heading_turn_in_place),
            "raw_command": None if self.raw_command is None else list(self.raw_command),
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




@dataclass(frozen=True)
class RLResidualTargetNavPolicyRuntimeConfig(LidarTargetNavPolicyRuntimeConfig):
    """Configuration for A* + learned bounded-residual local command runtime."""

    residual_policy_cfg: RLResidualPolicyCfg = field(default_factory=RLResidualPolicyCfg)
    checkpoint_path: str | None = None
    device: str = "cpu"
    source: str = "rl_residual_target_nav_policy"

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: AdapterRuntimeConfig,
        *,
        target: Target2D,
        checkpoint_path: str | None = None,
        residual_policy_cfg: RLResidualPolicyCfg | None = None,
        enable_replanning: bool = False,
        replan_every: int = 0,
        inflation_radius_m: float | None = None,
        max_vx: float | None = None,
        max_vy: float | None = None,
        max_wz: float | None = None,
        command_mode: str = "heading_aligned",
        device: str = "cpu",
    ) -> "RLResidualTargetNavPolicyRuntimeConfig":
        base = LidarTargetNavPolicyRuntimeConfig.from_runtime_config(
            runtime_config,
            target=target,
            enable_replanning=enable_replanning,
            replan_every=replan_every,
            inflation_radius_m=inflation_radius_m,
            max_vx=max_vx,
            max_vy=max_vy,
            max_wz=max_wz,
            command_mode=command_mode,
        )
        residual_cfg = residual_policy_cfg or RLResidualPolicyCfg(grid_cfg=base.grid_cfg)
        residual_cfg = replace(residual_cfg, grid_cfg=base.grid_cfg)
        return cls(
            target=base.target,
            grid_cfg=base.grid_cfg,
            astar_cfg=base.astar_cfg,
            nav_cfg=base.nav_cfg,
            runtime_cfg=base.runtime_cfg,
            require_lidar=base.require_lidar,
            require_pose=base.require_pose,
            default_pose=base.default_pose,
            residual_policy_cfg=residual_cfg,
            checkpoint_path=checkpoint_path,
            device=str(device),
        )


class RLResidualTargetNavPolicyRuntime:
    """Real/replay runtime for ``A* nominal command + bounded RL residual``.

    This class keeps the same adapter inputs/outputs as
    :class:`LidarTargetNavPolicyRuntime`, but routes the deterministic command
    through :class:`g1_nav_rl.policies.RLResidualPolicy` before the external
    safety supervisor sees it.
    """

    def __init__(self, config: RLResidualTargetNavPolicyRuntimeConfig) -> None:
        self.config = config
        residual_cfg = replace(config.residual_policy_cfg, grid_cfg=config.grid_cfg)
        policy = RLResidualPolicy(
            residual_cfg,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
        )
        core_cfg = NavResidualCoreCfg(
            grid_cfg=config.grid_cfg,
            astar_cfg=config.astar_cfg,
            nav_cfg=config.nav_cfg,
            runtime_cfg=config.runtime_cfg,
            residual_policy_cfg=residual_cfg,
            path_points=int(residual_cfg.path_points),
        )
        self.core = NavResidualCore(core_cfg, policy=policy, device=config.device)
        self._last_target: Target2D = config.target
        self.last_debug: dict[str, Any] | None = None

    def reset(self) -> None:
        self.core.reset()
        self._last_target = self.config.target
        self.last_debug = None

    def step(self, observation: RobotObservation) -> Command2D:
        pose = pose_from_observation(
            observation,
            default_pose=self.config.default_pose,
            require_pose=self.config.require_pose,
        )
        target = observation.target_hint or self.config.target
        if target.xy != self._last_target.xy:
            self.core.reset()
            self._last_target = target

        occupancy_grid, point_count = occupancy_grid_from_observation(
            observation,
            grid_cfg=self.config.grid_cfg,
            require_lidar=self.config.require_lidar,
        )
        target_visible = _metadata_bool(observation.metadata, "target_visible", default=True)
        target_age_s = _metadata_float(observation.metadata, "target_age_s", default=0.0)
        tracker_state = _metadata_tracker_state(observation.metadata, default=TrackerState.TRACKING)

        core_step = self.core.step(
            pose=pose,
            target=target,
            occupancy_grid=occupancy_grid,
            target_visible=target_visible,
            target_age_s=target_age_s,
            tracker_state=tracker_state,
        )
        command_tensor = core_step.command.detach().cpu()[0]
        nominal_tensor = core_step.nominal_command.detach().cpu()[0]
        residual_tensor = None if core_step.residual is None else core_step.residual.detach().cpu()[0]
        command_tuple = tuple(float(v) for v in command_tensor.tolist())

        nav_debug = make_lidar_nav_debug(
            runtime_step=core_step.runtime_step,
            pose=pose,
            target=target,
            occupancy_grid=occupancy_grid,
            lidar_point_count=point_count,
            command_tuple=command_tuple,
        ).to_dict()
        debug = {
            **nav_debug,
            "policy": "rl-residual-target-nav",
            "nominal_command": [float(v) for v in nominal_tensor.tolist()],
            "residual": None if residual_tensor is None else [float(v) for v in residual_tensor.tolist()],
            "target_visible": bool(target_visible),
            "target_age_s": float(target_age_s),
            "tracker_state": int(tracker_state),
        }
        self.last_debug = debug
        return Command2D(
            vx=command_tuple[0],
            vy=command_tuple[1],
            wz=command_tuple[2],
            timestamp_s=observation.timestamp_s,
            source=self.config.source,
            metadata=debug,
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
        command_mode=str(nav_cmd.command_mode),
        waypoint_xy_b=None if nav_cmd.waypoint_xy_b is None else tuple(float(v) for v in nav_cmd.waypoint_xy_b),
        waypoint_xy_w=None if nav_cmd.waypoint_xy_w is None else tuple(float(v) for v in nav_cmd.waypoint_xy_w),
        path_yaw=float(nav_cmd.path_yaw),
        bearing_error=None if nav_cmd.bearing_error is None else float(nav_cmd.bearing_error),
        heading_alignment_scale=None if nav_cmd.heading_alignment_scale is None else float(nav_cmd.heading_alignment_scale),
        heading_turn_in_place=bool(getattr(nav_cmd, "heading_turn_in_place", False)),
        raw_command=None if nav_cmd.raw_command is None else tuple(float(v) for v in nav_cmd.raw_command.tuple),
    )


def _metadata_bool(metadata: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "visible", "tracking"}
    return bool(value)


def _metadata_float(metadata: Mapping[str, Any], key: str, *, default: float) -> float:
    value = metadata.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metadata_tracker_state(metadata: Mapping[str, Any], *, default: TrackerState) -> TrackerState:
    for key in ("tracker_state", "target_tracker_state", "target_state"):
        if key in metadata:
            try:
                return tracker_state_from_value(metadata[key])
            except ValueError:
                return default
    if not _metadata_bool(metadata, "target_visible", default=True):
        return TrackerState.LOST
    return default


__all__ = [
    "ConstantVelocityPolicy",
    "LidarTargetNavDebug",
    "LidarTargetNavPolicyRuntime",
    "LidarTargetNavPolicyRuntimeConfig",
    "RLResidualTargetNavPolicyRuntime",
    "RLResidualTargetNavPolicyRuntimeConfig",
    "TargetNavPolicyRuntime",
    "TargetNavPolicyRuntimeConfig",
    "make_lidar_nav_debug",
    "occupancy_grid_from_lidar_frame",
    "occupancy_grid_from_observation",
    "pose_from_observation",
]
