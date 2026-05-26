"""Simple deterministic 2D navigation policy.

This module is the reusable version of the minimal navigation logic proven in
``scripts/smoke_test_nav_lidar.py``:

- a target is given as world-frame ``Target2D(x, y)``;
- the current robot pose is given as world-frame ``Pose2D(x, y, yaw)``;
- an occupancy grid is supplied in the robot/body frame used by
  ``g1_nav_rl.mapping.occupancy_grid``;
- A* plans once in body frame, then the path is converted to world frame;
- the controller advances monotonically through waypoints and maps the current
  waypoint to ``(vx, vy, wz)``;
- there is no shortcutting, smoothing, local escape, or safety override here.

The class is deliberately small. Isaac, LiDAR, visualization, RGB-D, and real
G1 command transport stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.planning.grid_astar import GridAStarCfg, GridAStarResult, plan_grid_path_xy

PointXY = tuple[float, float]


@dataclass(frozen=True)
class Target2D:
    """World-frame navigation target."""

    x: float
    y: float

    @property
    def xy(self) -> PointXY:
        return float(self.x), float(self.y)


@dataclass(frozen=True)
class Pose2D:
    """World-frame planar robot pose."""

    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> PointXY:
        return float(self.x), float(self.y)

    @classmethod
    def from_sequence(cls, value: Sequence[float]) -> "Pose2D":
        if len(value) != 3:
            raise ValueError(f"Pose2D requires 3 values (x, y, yaw), got {value!r}")
        return cls(float(value[0]), float(value[1]), float(value[2]))


@dataclass(frozen=True)
class MotionCommand:
    """Body-frame velocity command compatible with the G1 proxy command format."""

    vx: float
    vy: float
    wz: float

    @property
    def tuple(self) -> tuple[float, float, float]:
        return float(self.vx), float(self.vy), float(self.wz)

    def to_tensor(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor([[self.vx, self.vy, self.wz]], dtype=torch.float32, device=device or "cpu")


@dataclass(frozen=True)
class SimpleNavConfig:
    """Controller parameters matching ``smoke_test_nav_lidar.py`` defaults.

    ``command_mode`` controls how a selected A* waypoint becomes a body-frame
    velocity command:

    - ``"holonomic"`` keeps the original controller: x error drives ``vx``,
      y error drives ``vy``, and path tangent drives ``wz``.
    - ``"heading_aligned"`` is intended for real G1 tests: the robot yaw is
      driven toward the waypoint/velocity bearing, forward velocity is softly
      reduced when the bearing is large, and lateral velocity is damped.
    """

    waypoint_reached_radius: float = 0.12
    k_xy: float = 1.5
    k_yaw: float = 2.0
    max_vx: float = 0.30
    max_vy: float = 0.25
    max_wz: float = 0.60
    allow_backward: bool = False
    success_radius: float = 0.75
    command_mode: str = "holonomic"
    heading_lateral_scale: float = 0.20
    heading_min_vx_scale: float = 0.25
    heading_slow_angle_rad: float = 0.35
    heading_rotate_in_place_angle_rad: float = 0.80
    heading_rotate_vx_scale: float = 0.0
    heading_rotate_lateral_scale: float = 0.0
    path_lookahead_m: float = 0.0

    def __post_init__(self) -> None:
        command_mode = str(self.command_mode).replace("-", "_")
        if command_mode not in {"holonomic", "heading_aligned"}:
            raise ValueError("SimpleNavConfig.command_mode must be 'holonomic' or 'heading_aligned'")
        for name in (
            "heading_lateral_scale",
            "heading_min_vx_scale",
            "heading_rotate_vx_scale",
            "heading_rotate_lateral_scale",
            "path_lookahead_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"SimpleNavConfig.{name} must be non-negative and finite")
        if float(self.heading_slow_angle_rad) < 0.0:
            raise ValueError("SimpleNavConfig.heading_slow_angle_rad must be non-negative")
        if float(self.heading_rotate_in_place_angle_rad) < float(self.heading_slow_angle_rad):
            raise ValueError("heading_rotate_in_place_angle_rad must be >= heading_slow_angle_rad")


@dataclass(frozen=True)
class NavigationPlan:
    """A* path result represented both in body frame and world frame."""

    found: bool
    reason: str
    astar_result: GridAStarResult
    path_xy_b: list[PointXY]
    path_xy_w: list[PointXY]

    @property
    def waypoint_count(self) -> int:
        return len(self.path_xy_w)


@dataclass(frozen=True)
class NavigationCommand:
    """One output of the simple navigator."""

    command: MotionCommand
    target_reached: bool
    waypoint_index: int
    waypoint_xy_w: PointXY | None
    waypoint_xy_b: PointXY | None
    target_distance: float
    path_yaw: float
    command_mode: str = "holonomic"
    bearing_error: float | None = None
    heading_alignment_scale: float | None = None
    heading_turn_in_place: bool = False
    raw_command: MotionCommand | None = None


@dataclass(frozen=True)
class CommandSequenceResult:
    """Open-loop kinematic rollout helper for non-Isaac tests."""

    plan: NavigationPlan
    commands: list[NavigationCommand]
    poses: list[Pose2D]
    target_reached: bool


def angle_wrap(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def clip(v: float, lo: float, hi: float) -> float:
    return min(max(float(v), float(lo)), float(hi))


def world_to_body_xy(point_xy_w: PointXY, pose: Pose2D) -> PointXY:
    """Transform a world-frame point into the robot/body frame."""
    dx = float(point_xy_w[0]) - pose.x
    dy = float(point_xy_w[1]) - pose.y
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def body_to_world_xy(point_xy_b: PointXY, pose: Pose2D) -> PointXY:
    """Transform a body-frame point into world frame."""
    x_b, y_b = float(point_xy_b[0]), float(point_xy_b[1])
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    return pose.x + c * x_b - s * y_b, pose.y + s * x_b + c * y_b


def body_path_to_world_path(path_xy_b: Sequence[PointXY], pose: Pose2D) -> list[PointXY]:
    return [body_to_world_xy(xy, pose) for xy in path_xy_b]


def world_path_to_body_path(path_xy_w: Sequence[PointXY], pose: Pose2D) -> list[PointXY]:
    return [world_to_body_xy(xy, pose) for xy in path_xy_w]


def path_length_xy(path: Sequence[PointXY]) -> float:
    if len(path) <= 1:
        return 0.0
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]))


def select_waypoint(
    path_xy: Sequence[PointXY],
    current_xy: PointXY,
    waypoint_index: int,
    waypoint_reached_radius: float,
) -> int:
    """Advance monotonically through the existing path; never shortcut."""
    i = int(waypoint_index)
    while i < len(path_xy) - 1:
        wx, wy = path_xy[i]
        if math.hypot(wx - current_xy[0], wy - current_xy[1]) > float(waypoint_reached_radius):
            break
        i += 1
    return i


def select_lookahead_point(
    path_xy: Sequence[PointXY],
    current_xy: PointXY,
    waypoint_index: int,
    lookahead_m: float,
) -> PointXY:
    """Select a smoother control point along the remaining path.

    ``waypoint_index`` still controls progress/reached logic. This helper only
    chooses the point used for command generation. A positive lookahead reduces
    grid-cell-level heading jitter and makes real-G1 yaw commands align with the
    path corridor rather than the nearest cell.
    """

    if not path_xy:
        return float(current_xy[0]), float(current_xy[1])
    idx = max(0, min(int(waypoint_index), len(path_xy) - 1))
    lookahead = max(0.0, float(lookahead_m))
    if lookahead <= 1.0e-9:
        x, y = path_xy[idx]
        return float(x), float(y)

    prev = (float(current_xy[0]), float(current_xy[1]))
    accumulated = 0.0
    for j in range(idx, len(path_xy)):
        nxt = (float(path_xy[j][0]), float(path_xy[j][1]))
        seg_len = math.hypot(nxt[0] - prev[0], nxt[1] - prev[1])
        if seg_len <= 1.0e-9:
            prev = nxt
            continue
        if accumulated + seg_len >= lookahead:
            t = (lookahead - accumulated) / seg_len
            return (prev[0] + t * (nxt[0] - prev[0]), prev[1] + t * (nxt[1] - prev[1]))
        accumulated += seg_len
        prev = nxt
    x, y = path_xy[-1]
    return float(x), float(y)


def path_tangent_yaw(path_xy: Sequence[PointXY], waypoint_index: int, fallback_yaw: float) -> float:
    if waypoint_index <= 0 or waypoint_index >= len(path_xy):
        return float(fallback_yaw)
    prev_xy = path_xy[max(0, waypoint_index - 1)]
    next_xy = path_xy[waypoint_index]
    dx = next_xy[0] - prev_xy[0]
    dy = next_xy[1] - prev_xy[1]
    if abs(dx) + abs(dy) < 1e-9:
        return float(fallback_yaw)
    return math.atan2(dy, dx)


def command_to_waypoint(
    pose: Pose2D,
    waypoint_xy_w: PointXY,
    path_yaw: float,
    cfg: SimpleNavConfig,
) -> tuple[MotionCommand, PointXY]:
    """Map a world-frame waypoint to the original holonomic controller.

    This is equivalent to ``command_to_waypoint`` in ``smoke_test_nav_lidar.py``.
    """
    x_b, y_b = world_to_body_xy(waypoint_xy_w, pose)
    min_vx = -float(cfg.max_vx) if bool(cfg.allow_backward) else 0.0
    vx = clip(float(cfg.k_xy) * x_b, min_vx, float(cfg.max_vx))
    vy = clip(float(cfg.k_xy) * y_b, -float(cfg.max_vy), float(cfg.max_vy))
    yaw_error = angle_wrap(float(path_yaw) - pose.yaw)
    wz = clip(float(cfg.k_yaw) * yaw_error, -float(cfg.max_wz), float(cfg.max_wz))
    return MotionCommand(vx, vy, wz), (x_b, y_b)


def command_to_waypoint_heading_aligned(
    pose: Pose2D,
    waypoint_xy_w: PointXY,
    cfg: SimpleNavConfig,
) -> tuple[MotionCommand, PointXY, float, float, bool, MotionCommand]:
    """Map a waypoint to a real-G1-friendly heading-aligned command.

    The controller prefers ``vx + wz`` over lateral strafing.  The desired yaw
    rate is based on the bearing to the current waypoint in body frame, not only
    the A* path tangent.  Forward speed is softly reduced as the bearing grows,
    while lateral velocity remains available but damped by
    ``heading_lateral_scale``.
    """
    x_b, y_b = world_to_body_xy(waypoint_xy_w, pose)
    distance = math.hypot(x_b, y_b)
    bearing = angle_wrap(math.atan2(y_b, x_b)) if distance > 1.0e-9 else 0.0
    abs_bearing = abs(bearing)
    cos_scale = max(0.0, math.cos(bearing))
    base_alignment_scale = max(float(cfg.heading_min_vx_scale), cos_scale)
    lateral_scale = float(cfg.heading_lateral_scale)
    slow_angle = max(0.0, float(cfg.heading_slow_angle_rad))
    rotate_angle = max(slow_angle, float(cfg.heading_rotate_in_place_angle_rad))

    turn_in_place = rotate_angle > 0.0 and abs_bearing >= rotate_angle
    if rotate_angle > slow_angle and abs_bearing > slow_angle:
        blend = min(1.0, max(0.0, (abs_bearing - slow_angle) / (rotate_angle - slow_angle)))
    elif turn_in_place:
        blend = 1.0
    else:
        blend = 0.0

    alignment_scale = (1.0 - blend) * base_alignment_scale + blend * float(cfg.heading_rotate_vx_scale)
    lateral_scale = (1.0 - blend) * lateral_scale + blend * float(cfg.heading_rotate_lateral_scale)
    if turn_in_place:
        alignment_scale = float(cfg.heading_rotate_vx_scale)
        lateral_scale = float(cfg.heading_rotate_lateral_scale)

    min_vx = -float(cfg.max_vx) if bool(cfg.allow_backward) else 0.0
    raw_vx = float(cfg.k_xy) * distance * alignment_scale
    raw_vy = float(cfg.k_xy) * y_b * lateral_scale
    raw_wz = float(cfg.k_yaw) * bearing
    raw = MotionCommand(raw_vx, raw_vy, raw_wz)
    cmd = MotionCommand(
        clip(raw_vx, min_vx, float(cfg.max_vx)),
        clip(raw_vy, -float(cfg.max_vy), float(cfg.max_vy)),
        clip(raw_wz, -float(cfg.max_wz), float(cfg.max_wz)),
    )
    return cmd, (x_b, y_b), bearing, alignment_scale, turn_in_place, raw


def integrate_body_command(pose: Pose2D, command: MotionCommand, dt: float) -> Pose2D:
    """Simple planar kinematic integration used by non-Isaac tests."""
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    x_next = pose.x + (c * command.vx - s * command.vy) * float(dt)
    y_next = pose.y + (s * command.vx + c * command.vy) * float(dt)
    yaw_next = angle_wrap(pose.yaw + command.wz * float(dt))
    return Pose2D(x_next, y_next, yaw_next)


class SimpleNavigator:
    """Stateful target-to-command navigator.

    Typical online use:

    .. code-block:: python

        nav = SimpleNavigator(grid_cfg, astar_cfg)
        plan = nav.plan(pose, target, occupancy_grid)
        while running:
            cmd = nav.command(pose)

    Replanning is intentionally explicit: call :meth:`plan` again with the new
    pose, target, and occupancy grid. The class does not own sensors or mapping.
    """

    def __init__(
        self,
        grid_cfg: OccupancyGridCfg,
        astar_cfg: GridAStarCfg | None = None,
        config: SimpleNavConfig | None = None,
    ) -> None:
        self.grid_cfg = grid_cfg
        self.astar_cfg = astar_cfg or GridAStarCfg()
        self.config = config or SimpleNavConfig()
        self.plan_result: NavigationPlan | None = None
        self.waypoint_index: int = 0

    def reset(self) -> None:
        self.plan_result = None
        self.waypoint_index = 0

    def plan(
        self,
        pose: Pose2D,
        target: Target2D,
        occupancy_grid: torch.Tensor,
        *,
        allow_goal_relocation: bool = True,
    ) -> NavigationPlan:
        """Plan from the current pose to ``target`` using a body-frame grid."""
        goal_xy_b = world_to_body_xy(target.xy, pose)
        astar = plan_grid_path_xy(
            occupancy_grid=occupancy_grid,
            start_xy_b=(0.0, 0.0),
            goal_xy_b=goal_xy_b,
            grid_cfg=self.grid_cfg,
            planner_cfg=self.astar_cfg,
            allow_goal_relocation=allow_goal_relocation,
        )
        path_xy_b = list(astar.path_xy_b)
        path_xy_w = body_path_to_world_path(path_xy_b, pose) if astar.found else []
        if path_xy_w:
            path_xy_w[0] = pose.xy
            path_xy_w[-1] = target.xy

        plan = NavigationPlan(
            found=bool(astar.found),
            reason=str(astar.reason),
            astar_result=astar,
            path_xy_b=path_xy_b,
            path_xy_w=path_xy_w,
        )
        self.plan_result = plan
        self.waypoint_index = 1 if len(path_xy_w) > 1 else 0
        return plan

    def set_world_path(self, path_xy_w: Sequence[PointXY]) -> None:
        """Install an externally planned world-frame path.

        This is useful for tests that still call A* outside this class but want
        the same smoke-test command generator.
        """
        dummy = GridAStarResult(
            found=bool(path_xy_w),
            start_cell=(0, 0),
            goal_cell=(0, 0),
            path_cells=[],
            path_xy_b=[],
            cost=math.inf,
            expansions=0,
            reason="external_path",
        )
        self.plan_result = NavigationPlan(
            found=bool(path_xy_w),
            reason="external_path",
            astar_result=dummy,
            path_xy_b=[],
            path_xy_w=[(float(x), float(y)) for x, y in path_xy_w],
        )
        self.waypoint_index = 1 if len(path_xy_w) > 1 else 0

    def remaining_world_path(self) -> list[PointXY]:
        if self.plan_result is None:
            return []
        if self.waypoint_index >= len(self.plan_result.path_xy_w):
            return []
        return list(self.plan_result.path_xy_w[self.waypoint_index :])

    def remaining_body_path(self, pose: Pose2D) -> list[PointXY]:
        return world_path_to_body_path(self.remaining_world_path(), pose)

    def command(self, pose: Pose2D, target: Target2D | None = None) -> NavigationCommand:
        """Return one ``(vx, vy, wz)`` command for the current pose.

        The internal waypoint index is advanced monotonically, exactly like the
        smoke-test controller.
        """
        if self.plan_result is None or not self.plan_result.path_xy_w:
            dist = math.inf if target is None else math.hypot(target.x - pose.x, target.y - pose.y)
            return NavigationCommand(
                command=MotionCommand(0.0, 0.0, 0.0),
                target_reached=False,
                waypoint_index=0,
                waypoint_xy_w=None,
                waypoint_xy_b=None,
                target_distance=dist,
                path_yaw=pose.yaw,
            )

        path = self.plan_result.path_xy_w
        if target is None:
            target_xy = path[-1]
        else:
            target_xy = target.xy
        target_distance = math.hypot(target_xy[0] - pose.x, target_xy[1] - pose.y)
        target_reached = target_distance <= float(self.config.success_radius)

        self.waypoint_index = select_waypoint(
            path_xy=path,
            current_xy=pose.xy,
            waypoint_index=self.waypoint_index,
            waypoint_reached_radius=self.config.waypoint_reached_radius,
        )
        progress_waypoint_xy_w = path[self.waypoint_index]
        waypoint_xy_w = select_lookahead_point(
            path_xy=path,
            current_xy=pose.xy,
            waypoint_index=self.waypoint_index,
            lookahead_m=float(self.config.path_lookahead_m),
        )
        tangent_yaw = path_tangent_yaw(path, self.waypoint_index, pose.yaw)
        command_mode = str(self.config.command_mode).replace("-", "_")
        bearing_error: float | None = None
        heading_alignment_scale: float | None = None
        heading_turn_in_place = False
        raw_command: MotionCommand | None = None
        if command_mode == "heading_aligned":
            command, waypoint_xy_b, bearing_error, heading_alignment_scale, heading_turn_in_place, raw_command = command_to_waypoint_heading_aligned(
                pose, waypoint_xy_w, self.config
            )
        elif command_mode == "holonomic":
            command, waypoint_xy_b = command_to_waypoint(pose, waypoint_xy_w, tangent_yaw, self.config)
            yaw_error = angle_wrap(float(tangent_yaw) - pose.yaw)
            bearing_error = yaw_error
            heading_alignment_scale = 1.0
            raw_command = command
        else:
            raise ValueError(f"Unsupported SimpleNavConfig.command_mode: {self.config.command_mode!r}")

        if target_reached:
            command = MotionCommand(0.0, 0.0, 0.0)

        return NavigationCommand(
            command=command,
            target_reached=target_reached,
            waypoint_index=self.waypoint_index,
            waypoint_xy_w=waypoint_xy_w,
            waypoint_xy_b=waypoint_xy_b,
            target_distance=target_distance,
            path_yaw=tangent_yaw,
            command_mode=command_mode,
            bearing_error=bearing_error,
            heading_alignment_scale=heading_alignment_scale,
            heading_turn_in_place=heading_turn_in_place,
            raw_command=raw_command,
        )

    def step(
        self,
        pose: Pose2D,
        target: Target2D,
        occupancy_grid: torch.Tensor | None = None,
        *,
        allow_goal_relocation: bool = True,
    ) -> NavigationCommand:
        """Plan if needed, then emit one command.

        This keeps the minimal default behavior: one plan, then direct waypoint
        following. To replan, call :meth:`plan` explicitly before ``step`` or
        reset the navigator.
        """
        if self.plan_result is None:
            if occupancy_grid is None:
                raise ValueError("occupancy_grid is required for the first SimpleNavigator.step() call")
            self.plan(pose, target, occupancy_grid, allow_goal_relocation=allow_goal_relocation)
        return self.command(pose, target)

    def rollout_commands(
        self,
        initial_pose: Pose2D,
        target: Target2D,
        occupancy_grid: torch.Tensor,
        *,
        dt: float,
        max_steps: int,
        allow_goal_relocation: bool = True,
    ) -> CommandSequenceResult:
        """Generate a kinematic command sequence without Isaac.

        This is a convenience helper for simple tests. Real Isaac/G1 execution
        should call :meth:`command` online using measured pose feedback.
        """
        pose = initial_pose
        plan = self.plan(pose, target, occupancy_grid, allow_goal_relocation=allow_goal_relocation)
        poses = [pose]
        commands: list[NavigationCommand] = []
        reached = False
        if not plan.found:
            return CommandSequenceResult(plan, commands, poses, False)

        for _ in range(int(max_steps)):
            nav_cmd = self.command(pose, target)
            commands.append(nav_cmd)
            reached = bool(nav_cmd.target_reached)
            if reached:
                break
            pose = integrate_body_command(pose, nav_cmd.command, dt)
            poses.append(pose)
        return CommandSequenceResult(plan, commands, poses, reached)
