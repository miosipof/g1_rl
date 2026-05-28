"""Synchronized plan-execute-replan navigation core.

The controller in this module is deliberately input-agnostic.  It does not know
about Isaac Sim, RealSense, Livox, YOLO, Unitree SDK, or the real/sim command
sink.  Callers provide normalized observations:

- a current local body-frame target point from RGB-D/YOLO or a stream;
- a current local body-frame occupancy grid built from LiDAR/depth;
- an optional raw frontal clearance scalar for a last-resort forward stop.

The controller keeps exactly one piece of state: the currently committed A*
plan and its command queue.  Replanning is a synchronized procedure:

1. stop/observe is requested when a trigger fires;
2. on the next call, the latest target + LiDAR grid are used to run A*;
3. the new path is converted to a plain sequence of body-frame velocity
   commands ``[vx, vy, wz]_i``;
4. commands are executed sequentially until another trigger invalidates the
   plan.

The controller does not smooth commands, does not run yaw escape, does not
propagate targets, and does not modify the planned path with shortcut/coalescing
heuristics.  A* remains responsible for obstacle inflation and avoidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.simple_nav import MotionCommand, angle_wrap, path_length_xy
from g1_nav_rl.planning.grid_astar import (
    GridAStarCfg,
    GridAStarResult,
    bresenham_cells,
    grid_cell_to_xy,
    inflate_occupancy_grid,
    is_cell_in_bounds,
    plan_grid_path_xy,
    xy_to_grid_cell,
)

PointXY = tuple[float, float]
PoseXYYaw = tuple[float, float, float]


class StepwiseActionKind(str, Enum):
    """Discrete command kinds emitted by the synchronized controller."""

    TURN = "turn"
    FORWARD = "forward"
    STOP_OBSERVE = "stop_observe"


@dataclass(frozen=True)
class StepwiseAction:
    """One executable body-frame velocity command.

    ``duration_s`` is exact for the command and is never clipped inside this
    module.  If a runner wants to impose a hardware timeout, it should stop and
    request a synchronized replan rather than pretending the full action was
    executed.
    """

    kind: StepwiseActionKind | str
    command: MotionCommand
    duration_s: float
    angle_rad: float | None = None
    distance_m: float | None = None
    source: str = "stepwise_nav"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, StepwiseActionKind) else StepwiseActionKind(str(self.kind))
        duration_s = float(self.duration_s)
        if not math.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError(f"StepwiseAction.duration_s must be finite and non-negative, got {duration_s}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "duration_s", duration_s)
        if self.angle_rad is not None:
            angle = float(self.angle_rad)
            if not math.isfinite(angle):
                raise ValueError("StepwiseAction.angle_rad must be finite when provided")
            object.__setattr__(self, "angle_rad", angle)
        if self.distance_m is not None:
            distance = float(self.distance_m)
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError("StepwiseAction.distance_m must be finite and non-negative when provided")
            object.__setattr__(self, "distance_m", distance)

    @property
    def is_stop(self) -> bool:
        return self.kind == StepwiseActionKind.STOP_OBSERVE or (
            abs(self.command.vx) < 1.0e-9 and abs(self.command.vy) < 1.0e-9 and abs(self.command.wz) < 1.0e-9
        )

    @classmethod
    def stop_observe(cls, *, source: str = "stepwise_stop_observe", reason: str = "stop_observe") -> "StepwiseAction":
        return cls(
            kind=StepwiseActionKind.STOP_OBSERVE,
            command=MotionCommand(0.0, 0.0, 0.0),
            duration_s=0.0,
            source=source,
            metadata={"reason": reason},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "command": [float(v) for v in self.command.tuple],
            "duration_s": float(self.duration_s),
            "angle_rad": None if self.angle_rad is None else float(self.angle_rad),
            "distance_m": None if self.distance_m is None else float(self.distance_m),
            "source": str(self.source),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StepwiseNavConfig:
    """Configuration for synchronized A* plan execution.

    The names keep compatibility with earlier prototypes and CLI flags, but the
    behavior is now simpler: A* is planned once per sync and converted into a
    queue of exact velocity commands.  Replanning is triggered by path blockage,
    target/pose mismatch, explicit periodic checkpoints, exhausted queues, or
    caller-forced replans.
    """

    grid_cfg: OccupancyGridCfg = field(default_factory=OccupancyGridCfg)
    astar_cfg: GridAStarCfg = field(default_factory=lambda: GridAStarCfg(inflation_radius=0.35))

    # Motion conversion.  Speeds are command magnitudes, not smoothing gains.
    forward_segment_m: float = 0.35  # maximum committed translation segment emitted from the A* path
    forward_speed_mps: float = 0.35
    turn_speed_radps: float = 0.50
    turn_angle_max_rad: float = math.radians(35.0)  # retained for compatibility; not used as a cap
    min_turn_angle_rad: float = math.radians(3.0)
    min_forward_m: float = 0.02
    action_duration_max_s: float = 0.0  # retained for compatibility; not used as a clip

    # Replan triggers.  Action-count replanning is disabled by default because
    # A* grid-cell actions can be very short.  Prefer distance-based sync
    # replanning or explicit invalidation triggers.
    replan_every_actions: int = 0
    replan_every_forward_m: float = 0.0
    target_change_replan_m: float = 0.50
    path_block_check: bool = True
    path_block_lookahead_m: float = 1.50  # retained for old configs; immediate guard does not use it
    # Immediate-sweep guard inflation is separate from A* planning inflation.
    # A* may use robot_radius + safety_margin, but the current-frame execution
    # guard should detect imminent collision, not merely that the robot is inside
    # the conservative planning margin.  None preserves legacy behavior.
    path_block_inflation_radius_m: float | None = None
    sweep_start_ignore_m: float = 0.05  # ignore origin-cell inflation when checking the next commanded sweep

    # Kept for compatibility. Missing target stops by default. For real
    # hardware tests that intentionally lock the first detection as a fixed
    # destination, the runner may allow the already-committed plan queue to
    # continue after target loss. This does not create new plans without a
    # target; it only executes the active queue.
    blind_plan_actions: int = 0
    continue_active_plan_without_target: bool = False

    # Last-resort raw front guard before executing a forward/translation command.
    emergency_front_stop_m: float = 0.35
    success_radius_m: float = 0.75
    start_xy_b: PointXY = (0.0, 0.0)
    allow_goal_relocation: bool = True
    allow_start_relocation: bool = True
    source: str = "stepwise_nav"

    def __post_init__(self) -> None:
        nonnegative = (
            "forward_segment_m",
            "forward_speed_mps",
            "turn_speed_radps",
            "turn_angle_max_rad",
            "min_turn_angle_rad",
            "min_forward_m",
            "action_duration_max_s",
            "replan_every_forward_m",
            "target_change_replan_m",
            "path_block_lookahead_m",
            "sweep_start_ignore_m",
            "emergency_front_stop_m",
            "success_radius_m",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"StepwiseNavConfig.{name} must be finite and non-negative")
        if self.path_block_inflation_radius_m is not None:
            value = float(self.path_block_inflation_radius_m)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("StepwiseNavConfig.path_block_inflation_radius_m must be finite and non-negative or None")
        if self.forward_speed_mps <= 0.0:
            raise ValueError("forward_speed_mps must be positive")
        if self.turn_speed_radps <= 0.0:
            raise ValueError("turn_speed_radps must be positive")
        if int(self.replan_every_actions) < 0:
            raise ValueError("replan_every_actions must be non-negative")
        if int(self.blind_plan_actions) < 0:
            raise ValueError("blind_plan_actions must be non-negative")
        if len(self.start_xy_b) != 2:
            raise ValueError("start_xy_b must be a 2-tuple")


@dataclass(frozen=True)
class StepwisePlanResult:
    """A synchronized A* plan and the command queue derived from it."""

    astar_result: GridAStarResult
    actions: list[StepwiseAction]
    target_xy_b: PointXY

    @property
    def found(self) -> bool:
        return bool(self.astar_result.found)

    @property
    def path_xy_b(self) -> list[PointXY]:
        return list(self.astar_result.path_xy_b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": bool(self.found),
            "reason": str(self.astar_result.reason),
            # Target in the body frame at the synchronized replan that created
            # this active plan.  It is not necessarily the current observed
            # target on later execution steps.
            "target_xy_b_at_replan": [float(v) for v in self.target_xy_b],
            "target_xy_b": [float(v) for v in self.target_xy_b],  # compatibility alias
            "path_len": int(len(self.astar_result.path_xy_b)),
            "path_length_m": float(path_length_xy(self.astar_result.path_xy_b)),
            "action_count": int(len(self.actions)),
            "start_cell": [int(v) for v in self.astar_result.start_cell],
            "goal_cell": [int(v) for v in self.astar_result.goal_cell],
        }


@dataclass(frozen=True)
class StepwiseDecision:
    """One controller decision returned to a sim or real runner."""

    action: StepwiseAction
    execute: bool
    reason: str
    target_visible: bool
    target_reached: bool = False
    plan: StepwisePlanResult | None = None
    blocked: bool = False
    front_min_distance_m: float | None = None
    blind_actions_used: int = 0
    queue_len_after_pop: int = 0
    replan_attempted: bool = False
    replan_reason: str | None = None
    cycle_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "execute": bool(self.execute),
            "reason": str(self.reason),
            "target_visible": bool(self.target_visible),
            "target_reached": bool(self.target_reached),
            "blocked": bool(self.blocked),
            "front_min_distance_m": self.front_min_distance_m,
            "blind_actions_used": int(self.blind_actions_used),
            "queue_len_after_pop": int(self.queue_len_after_pop),
            "replan_attempted": bool(self.replan_attempted),
            "replan_reason": self.replan_reason,
            "action": self.action.to_dict(),
            "plan": None if self.plan is None else self.plan.to_dict(),
            "metadata": dict(self.metadata),
        }


def _as_xy(value: Sequence[float], *, name: str) -> PointXY:
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {value!r}")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x, y


def _duration_for_turn(angle_rad: float, cfg: StepwiseNavConfig) -> float:
    return abs(float(angle_rad)) / float(cfg.turn_speed_radps)


def _duration_for_translation(distance_m: float, cfg: StepwiseNavConfig) -> float:
    return abs(float(distance_m)) / float(cfg.forward_speed_mps)


def _plan_to_body_xy(point_xy: PointXY, expected_pose: PoseXYYaw) -> PointXY:
    px, py = float(point_xy[0]), float(point_xy[1])
    x, y, yaw = expected_pose
    dx = px - x
    dy = py - y
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _apply_action_to_expected_pose(pose: PoseXYYaw, action: StepwiseAction) -> PoseXYYaw:
    x, y, yaw = pose
    dt = float(action.duration_s)
    dx_b = float(action.command.vx) * dt
    dy_b = float(action.command.vy) * dt
    dyaw = float(action.command.wz) * dt
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (
        x + c * dx_b - s * dy_b,
        y + s * dx_b + c * dy_b,
        angle_wrap(yaw + dyaw),
    )


def _make_turn_action(angle_rad: float, *, cfg: StepwiseNavConfig, path_index: int) -> StepwiseAction:
    wz = math.copysign(float(cfg.turn_speed_radps), float(angle_rad))
    return StepwiseAction(
        kind=StepwiseActionKind.TURN,
        command=MotionCommand(0.0, 0.0, wz),
        duration_s=_duration_for_turn(angle_rad, cfg),
        angle_rad=float(angle_rad),
        source=cfg.source,
        metadata={"path_index": int(path_index), "mode": "sync_plan_command_queue"},
    )


def _make_translation_action(local_xy: PointXY, *, cfg: StepwiseNavConfig, path_index: int) -> StepwiseAction:
    lx, ly = float(local_xy[0]), float(local_xy[1])
    distance = math.hypot(lx, ly)
    if distance <= 1.0e-9:
        raise ValueError("cannot create translation action for zero distance")
    duration = _duration_for_translation(distance, cfg)
    return StepwiseAction(
        kind=StepwiseActionKind.FORWARD,
        command=MotionCommand(lx / duration, ly / duration, 0.0),
        duration_s=duration,
        distance_m=distance,
        source=cfg.source,
        metadata={"path_index": int(path_index), "mode": "sync_plan_command_queue"},
    )


def _sample_polyline_by_distance(points: Sequence[PointXY], *, step_m: float, min_tail_m: float) -> list[tuple[PointXY, int]]:
    """Sample a polyline every ``step_m`` meters, preserving the final point.

    Returns ``(point, source_path_index)`` pairs.  The source index is only
    metadata for debugging; sampled points can lie inside a segment.
    """

    if len(points) < 2:
        return []
    step = max(float(step_m), float(min_tail_m), 1.0e-6)
    samples: list[tuple[PointXY, int]] = []
    prev = points[0]
    remaining_to_next_sample = step

    for idx in range(1, len(points)):
        curr = points[idx]
        seg_dx = curr[0] - prev[0]
        seg_dy = curr[1] - prev[1]
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len <= 1.0e-9:
            prev = curr
            continue
        ux = seg_dx / seg_len
        uy = seg_dy / seg_len
        travelled = 0.0
        while travelled + remaining_to_next_sample <= seg_len + 1.0e-9:
            travelled += remaining_to_next_sample
            samples.append(((prev[0] + ux * travelled, prev[1] + uy * travelled), idx))
            remaining_to_next_sample = step
        remaining_to_next_sample -= max(0.0, seg_len - travelled)
        prev = curr

    end = points[-1]
    if not samples:
        if math.hypot(end[0] - points[0][0], end[1] - points[0][1]) >= min_tail_m:
            samples.append((end, len(points) - 1))
    else:
        last = samples[-1][0]
        if math.hypot(end[0] - last[0], end[1] - last[1]) >= min_tail_m:
            samples.append((end, len(points) - 1))
    return samples


def _make_forward_only_action(distance_m: float, *, cfg: StepwiseNavConfig, path_index: int) -> StepwiseAction:
    distance = float(distance_m)
    if distance <= 0.0:
        raise ValueError("forward-only action distance must be positive")
    duration = _duration_for_translation(distance, cfg)
    return StepwiseAction(
        kind=StepwiseActionKind.FORWARD,
        command=MotionCommand(float(cfg.forward_speed_mps), 0.0, 0.0),
        duration_s=duration,
        distance_m=distance,
        source=cfg.source,
        metadata={
            "path_index": int(path_index),
            "mode": "sync_plan_lookahead_queue",
            "forward_only": True,
        },
    )


def path_to_stepwise_actions(
    path_xy_b: Sequence[Sequence[float]],
    *,
    cfg: StepwiseNavConfig | None = None,
) -> list[StepwiseAction]:
    """Convert an A* body-frame path into a simple command queue.

    The real G1 does not reliably execute many 5 cm grid-cell commands.  This
    converter therefore samples the A* polyline into short committed lookahead
    segments, then emits ``Turn`` followed by pure ``Forward`` commands.  It does
    not shortcut the path through obstacles; it follows the original A* polyline
    at a coarser command resolution.

    The command invariant is intentionally simple:

    ``yaw toward the next sampled path point -> drive forward -> stop/observe``.

    Lateral velocity is never emitted here.  If the A* start was relocated
    because the inflated start cell was occupied, command generation still starts
    from the real body origin; the immediate swept-footprint guard remains
    responsible for rejecting unsafe first translations.
    """

    cfg = cfg or StepwiseNavConfig()
    raw_points = [_as_xy(p, name="path_xy_b point") for p in path_xy_b]
    if len(raw_points) < 2:
        return []

    start = _as_xy(cfg.start_xy_b, name="start_xy_b")
    # Always generate commands from the actual robot body origin, not from a
    # potentially relocated A* start cell.  This prevents the real runner from
    # executing actions whose frame starts at a fictitious free cell.
    points: list[PointXY] = [start]
    points.extend(raw_points[1:] if math.hypot(raw_points[0][0] - start[0], raw_points[0][1] - start[1]) > 1.0e-6 else raw_points[1:])

    sampled = _sample_polyline_by_distance(
        points,
        step_m=float(cfg.forward_segment_m),
        min_tail_m=float(cfg.min_forward_m),
    )
    if not sampled:
        return []

    actions: list[StepwiseAction] = []
    expected_pose: PoseXYYaw = (float(start[0]), float(start[1]), 0.0)

    for waypoint, path_index in sampled:
        local = _plan_to_body_xy(waypoint, expected_pose)
        distance = math.hypot(local[0], local[1])
        if distance < float(cfg.min_forward_m):
            continue

        bearing = angle_wrap(math.atan2(local[1], local[0]))
        if abs(bearing) >= float(cfg.min_turn_angle_rad):
            turn = _make_turn_action(bearing, cfg=cfg, path_index=path_index)
            actions.append(turn)
            expected_pose = _apply_action_to_expected_pose(expected_pose, turn)
            local = _plan_to_body_xy(waypoint, expected_pose)
            distance = math.hypot(local[0], local[1])
            if distance < float(cfg.min_forward_m):
                continue

        translate = _make_forward_only_action(distance, cfg=cfg, path_index=path_index)
        actions.append(translate)
        expected_pose = _apply_action_to_expected_pose(expected_pose, translate)

    return actions


def plan_stepwise_actions(
    occupancy_grid: torch.Tensor,
    target_xy_b: Sequence[float],
    *,
    cfg: StepwiseNavConfig | None = None,
) -> StepwisePlanResult:
    """Plan from current body origin to a local target and build commands."""

    cfg = cfg or StepwiseNavConfig()
    target = _as_xy(target_xy_b, name="target_xy_b")
    astar = plan_grid_path_xy(
        occupancy_grid,
        cfg.start_xy_b,
        target,
        cfg.grid_cfg,
        cfg.astar_cfg,
        allow_goal_relocation=bool(cfg.allow_goal_relocation),
        allow_start_relocation=bool(cfg.allow_start_relocation),
    )
    actions = path_to_stepwise_actions(astar.path_xy_b, cfg=cfg) if astar.found else []
    return StepwisePlanResult(astar_result=astar, actions=actions, target_xy_b=target)


def target_reached(target_xy_b: Sequence[float] | None, *, cfg: StepwiseNavConfig | None = None) -> bool:
    """Return true if the local target/stand-off goal is already reached."""

    if target_xy_b is None:
        return False
    cfg = cfg or StepwiseNavConfig()
    x, y = _as_xy(target_xy_b, name="target_xy_b")
    return math.hypot(x, y) <= float(cfg.success_radius_m)


def front_clearance_from_points(
    points_body_xyz: Any,
    *,
    x_min_m: float = 0.0,
    x_max_m: float = 1.0,
    y_abs_max_m: float = 0.35,
    z_min_m: float | None = None,
    z_max_m: float | None = None,
) -> float | None:
    """Compute minimum frontal obstacle distance from body-frame points."""

    pts = torch.as_tensor(points_body_xyz, dtype=torch.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_body_xyz must have shape Nx2 or Nx3, got {tuple(pts.shape)}")
    if pts.numel() == 0:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    mask = torch.isfinite(pts[:, :2]).all(dim=1)
    mask = mask & (x >= float(x_min_m)) & (x <= float(x_max_m)) & (torch.abs(y) <= float(y_abs_max_m))
    if pts.shape[1] >= 3:
        z = pts[:, 2]
        mask = mask & torch.isfinite(z)
        if z_min_m is not None:
            mask = mask & (z >= float(z_min_m))
        if z_max_m is not None:
            mask = mask & (z <= float(z_max_m))
    if not bool(mask.any()):
        return None
    return float(x[mask].min().item())


def forward_action_blocked(
    action: StepwiseAction,
    *,
    front_min_distance_m: float | None,
    cfg: StepwiseNavConfig | None = None,
) -> bool:
    """Apply the last-resort raw forward guard."""

    cfg = cfg or StepwiseNavConfig()
    if action.kind != StepwiseActionKind.FORWARD:
        return False
    if front_min_distance_m is None:
        return False
    # Translation commands may include a lateral component, but they still move
    # the robot footprint.  The raw corridor guard is intentionally conservative:
    # if something is too close in front, do not translate.
    return float(front_min_distance_m) < float(cfg.emergency_front_stop_m)


def _target_mismatch_m(*, observed_target_xy_b: PointXY, plan_target_xy_b: PointXY, expected_pose: PoseXYYaw) -> float:
    expected_target_now_b = _plan_to_body_xy(plan_target_xy_b, expected_pose)
    return math.hypot(
        float(observed_target_xy_b[0]) - expected_target_now_b[0],
        float(observed_target_xy_b[1]) - expected_target_now_b[1],
    )


def _inflated_grid(
    occupancy_grid: torch.Tensor,
    cfg: StepwiseNavConfig,
    *,
    inflation_radius: float | None = None,
) -> torch.Tensor:
    grid = occupancy_grid
    if grid.ndim == 3:
        if grid.shape[0] != 1:
            raise ValueError("path-block check expects a single grid or batch size 1")
        grid = grid[0]
    if grid.ndim != 2:
        raise ValueError(f"occupancy_grid must have shape [H,W] or [1,H,W], got {tuple(occupancy_grid.shape)}")
    return inflate_occupancy_grid(
        grid,
        cfg=cfg.grid_cfg,
        inflation_radius=float(cfg.astar_cfg.inflation_radius if inflation_radius is None else inflation_radius),
        occupancy_threshold=cfg.astar_cfg.occupancy_threshold,
    ).to(dtype=torch.bool, device="cpu")


def translation_action_blocked_by_grid(
    action: StepwiseAction,
    occupancy_grid: torch.Tensor,
    *,
    cfg: StepwiseNavConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Check the next translation command against the current inflated grid.

    This guard is deliberately current-frame only.  It checks the immediate
    commanded translation primitive, not the whole old A* path.  Cells at the
    robot origin are ignored because obstacle inflation can mark the current
    start cell occupied when the robot is exactly on the discretized inflation
    boundary; that condition is diagnostic, but it should not be interpreted as
    the *next* command entering an obstacle.
    """

    cfg = cfg or StepwiseNavConfig()
    if not cfg.path_block_check:
        return False, {"checked": False, "reason": "disabled"}
    if action.kind != StepwiseActionKind.FORWARD:
        return False, {"checked": False, "reason": "non_translation_action", "action_kind": action.kind.value}

    dx = float(action.command.vx) * float(action.duration_s)
    dy = float(action.command.vy) * float(action.duration_s)
    distance = math.hypot(dx, dy)
    if distance < float(cfg.min_forward_m):
        return False, {"checked": False, "reason": "short_translation", "distance_m": float(distance)}

    grid = occupancy_grid
    if grid.ndim == 3:
        if grid.shape[0] != 1:
            raise ValueError("translation sweep check expects a single grid or batch size 1")
        grid = grid[0]
    if grid.ndim != 2:
        raise ValueError(f"occupancy_grid must have shape [H,W] or [1,H,W], got {tuple(occupancy_grid.shape)}")
    raw_occupied = (grid.detach().cpu() > float(cfg.astar_cfg.occupancy_threshold)).to(dtype=torch.bool)
    guard_inflation_radius = (
        float(cfg.astar_cfg.inflation_radius)
        if cfg.path_block_inflation_radius_m is None
        else float(cfg.path_block_inflation_radius_m)
    )
    inflated = _inflated_grid(grid, cfg, inflation_radius=guard_inflation_radius)
    height, width = int(inflated.shape[0]), int(inflated.shape[1])
    start_cell = xy_to_grid_cell((0.0, 0.0), cfg.grid_cfg)
    end_cell = xy_to_grid_cell((dx, dy), cfg.grid_cfg)
    effective_start_ignore_m = max(float(cfg.sweep_start_ignore_m), 2.0 * float(cfg.grid_cfg.resolution))
    base_meta: dict[str, Any] = {
        "grid_resolution_m": float(cfg.grid_cfg.resolution),
        "planning_inflation_radius_m": float(cfg.astar_cfg.inflation_radius),
        "guard_inflation_radius_m": float(guard_inflation_radius),
        # Compatibility alias; new readers should prefer guard_inflation_radius_m.
        "inflation_radius_m": float(guard_inflation_radius),
        "occupancy_threshold": float(cfg.astar_cfg.occupancy_threshold),
        "start_ignore_m": float(effective_start_ignore_m),
        "start_cell": list(start_cell),
        "end_cell": list(end_cell),
        "end_xy_b": [float(dx), float(dy)],
        "distance_m": float(distance),
    }
    if not is_cell_in_bounds(start_cell, height=height, width=width):
        return True, {**base_meta, "checked": True, "reason": "sweep_start_out_of_bounds"}
    if not is_cell_in_bounds(end_cell, height=height, width=width):
        return True, {**base_meta, "checked": True, "reason": "sweep_end_out_of_bounds"}

    if distance <= effective_start_ignore_m:
        return False, {
            **base_meta,
            "checked": True,
            "reason": "translation_within_start_ignore_radius",
            "checked_cells": 0,
            "ignored_start_cells": 0,
            "start_raw_occupied": bool(raw_occupied[start_cell[0], start_cell[1]].item()) if is_cell_in_bounds(start_cell, height=height, width=width) else None,
            "start_inflated_occupied": bool(inflated[start_cell[0], start_cell[1]].item()) if is_cell_in_bounds(start_cell, height=height, width=width) else None,
        }

    ux = dx / distance
    uy = dy / distance
    checked_cells = 0
    ignored_start_cells = 0
    for cell in bresenham_cells(start_cell, end_cell):
        if not is_cell_in_bounds(cell, height=height, width=width):
            return True, {
                **base_meta,
                "checked": True,
                "reason": "sweep_cell_out_of_bounds",
                "cell": list(cell),
                "checked_cells": int(checked_cells),
                "ignored_start_cells": int(ignored_start_cells),
            }
        cell_xy = grid_cell_to_xy(cell, cfg.grid_cfg)
        cell_distance = math.hypot(float(cell_xy[0]), float(cell_xy[1]))
        cell_along = float(cell_xy[0]) * ux + float(cell_xy[1]) * uy
        # Ignore the current origin area, not just the literal start cell.  A
        # discretized inflated obstacle can occupy cells a few centimeters from
        # the origin when the robot is already near the inflation boundary.
        # That is useful diagnostics, but it should not be reported as the next
        # commanded translation entering an obstacle.
        if cell_along <= effective_start_ignore_m or cell_distance <= effective_start_ignore_m:
            ignored_start_cells += 1
            continue
        checked_cells += 1
        if bool(inflated[cell[0], cell[1]].item()):
            return True, {
                **base_meta,
                "checked": True,
                "reason": "immediate_sweep_inflated_occupancy_intersection",
                "cell": list(cell),
                "cell_xy_b": [float(cell_xy[0]), float(cell_xy[1])],
                "distance_from_origin_m": float(cell_distance),
                "distance_along_command_m": float(cell_along),
                "raw_occupied_at_blocked_cell": bool(raw_occupied[cell[0], cell[1]].item()),
                "inflated_occupied_at_blocked_cell": True,
                "checked_cells": int(checked_cells),
                "ignored_start_cells": int(ignored_start_cells),
            }

    return False, {
        **base_meta,
        "checked": True,
        "reason": "clear",
        "checked_cells": int(checked_cells),
        "ignored_start_cells": int(ignored_start_cells),
        "start_raw_occupied": bool(raw_occupied[start_cell[0], start_cell[1]].item()) if is_cell_in_bounds(start_cell, height=height, width=width) else None,
        "start_inflated_occupied": bool(inflated[start_cell[0], start_cell[1]].item()) if is_cell_in_bounds(start_cell, height=height, width=width) else None,
    }


def remaining_path_blocked(
    occupancy_grid: torch.Tensor,
    plan: StepwisePlanResult,
    *,
    cfg: StepwiseNavConfig | None = None,
    expected_pose: PoseXYYaw = (0.0, 0.0, 0.0),
    current_action_index: int = 0,
) -> tuple[bool, dict[str, Any]]:
    """Compatibility wrapper for old callers.

    The previous implementation checked the full transformed remaining path
    against the current inflated grid.  That was frame-sensitive and caused
    false ``path_blocked`` events.  The cleaned-up controller only checks the
    next queued translation primitive.
    """

    cfg = cfg or StepwiseNavConfig()
    if plan is None or not plan.found:
        return False, {"checked": False, "reason": "no_plan"}
    if not (0 <= int(current_action_index) < len(plan.actions)):
        return False, {"checked": False, "reason": "no_next_action"}
    return translation_action_blocked_by_grid(plan.actions[int(current_action_index)], occupancy_grid, cfg=cfg)


class StepwiseNavigator:
    """Synchronized A* plan executor.

    Public behavior:
    - no target -> stop;
    - no plan / pending sync -> plan from latest target + grid;
    - active plan -> execute next queued command;
    - next translation blocked / target mismatch / periodic trigger -> stop
      now, then replan from fresh data on the next call.
    """

    def __init__(self, cfg: StepwiseNavConfig | None = None) -> None:
        self.cfg = cfg or StepwiseNavConfig()
        self.action_queue: list[StepwiseAction] = []
        self.current_action_index = 0
        self.last_plan: StepwisePlanResult | None = None
        self.cycle_index = 0
        self.executed_actions_since_replan = 0
        self.executed_forward_m_since_replan = 0.0
        self.blind_actions_used = 0
        self.pending_replan_reason: str | None = None
        self.expected_pose_in_plan: PoseXYYaw = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self.action_queue.clear()
        self.current_action_index = 0
        self.last_plan = None
        self.cycle_index = 0
        self.executed_actions_since_replan = 0
        self.executed_forward_m_since_replan = 0.0
        self.blind_actions_used = 0
        self.pending_replan_reason = None
        self.expected_pose_in_plan = (0.0, 0.0, 0.0)

    def _stop_decision(
        self,
        *,
        reason: str,
        target_visible: bool,
        target_reached_value: bool = False,
        blocked: bool = False,
        front_min_distance_m: float | None = None,
        replan_reason: str | None = None,
        replan_attempted: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepwiseDecision:
        action = StepwiseAction.stop_observe(reason=reason)
        decision = StepwiseDecision(
            action=action,
            execute=False,
            reason=reason,
            target_visible=bool(target_visible),
            target_reached=bool(target_reached_value),
            plan=self.last_plan,
            blocked=bool(blocked),
            front_min_distance_m=front_min_distance_m,
            blind_actions_used=0,
            queue_len_after_pop=max(0, len(self.action_queue) - self.current_action_index),
            replan_attempted=bool(replan_attempted),
            replan_reason=replan_reason,
            cycle_index=self.cycle_index,
            metadata=dict(metadata or {}),
        )
        self.cycle_index += 1
        return decision

    def _request_sync_stop(self, reason: str, *, front_min_distance_m: float | None, metadata: Mapping[str, Any]) -> StepwiseDecision:
        self.pending_replan_reason = reason
        self.action_queue.clear()
        self.current_action_index = 0
        return self._stop_decision(
            reason=reason,
            target_visible=True,
            blocked=reason in {"path_blocked", "forward_blocked"},
            front_min_distance_m=front_min_distance_m,
            replan_reason=reason,
            metadata={**metadata, "sync_replan_reason": reason},
        )

    def _sync_replan(
        self,
        *,
        target_xy_b: Sequence[float],
        occupancy_grid: torch.Tensor,
        replan_reason: str,
    ) -> StepwisePlanResult:
        plan = plan_stepwise_actions(occupancy_grid, target_xy_b, cfg=self.cfg)
        self.last_plan = plan
        self.action_queue = list(plan.actions)
        self.current_action_index = 0
        self.executed_actions_since_replan = 0
        self.executed_forward_m_since_replan = 0.0
        self.blind_actions_used = 0
        self.pending_replan_reason = None
        self.expected_pose_in_plan = (0.0, 0.0, 0.0)
        return plan

    def _execute_next_action(
        self,
        *,
        target_visible: bool,
        front_min_distance_m: float | None,
        replan_reason: str | None,
        replan_attempted: bool,
        metadata: Mapping[str, Any],
    ) -> StepwiseDecision:
        assert self.last_plan is not None
        action = self.action_queue[self.current_action_index]
        blocked = forward_action_blocked(action, front_min_distance_m=front_min_distance_m, cfg=self.cfg)
        if blocked:
            return self._request_sync_stop("forward_blocked", front_min_distance_m=front_min_distance_m, metadata=metadata)

        self.current_action_index += 1
        self.executed_actions_since_replan += 1
        if action.kind == StepwiseActionKind.FORWARD:
            self.executed_forward_m_since_replan += float(action.distance_m or 0.0)
        self.expected_pose_in_plan = _apply_action_to_expected_pose(self.expected_pose_in_plan, action)

        decision = StepwiseDecision(
            action=action,
            execute=not action.is_stop,
            reason="execute_action" if not action.is_stop else "stop_observe",
            target_visible=bool(target_visible),
            plan=self.last_plan,
            front_min_distance_m=front_min_distance_m,
            blind_actions_used=0,
            queue_len_after_pop=max(0, len(self.action_queue) - self.current_action_index),
            replan_attempted=bool(replan_attempted),
            replan_reason=replan_reason,
            cycle_index=self.cycle_index,
            metadata={**dict(metadata), "expected_pose_in_plan_after_action": list(self.expected_pose_in_plan)},
        )
        self.cycle_index += 1
        return decision

    def decide(
        self,
        *,
        target_xy_b: Sequence[float] | None,
        occupancy_grid: torch.Tensor | None,
        front_min_distance_m: float | None = None,
        force_replan: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepwiseDecision:
        """Return the next synchronized plan-execution command."""

        meta = dict(metadata or {})
        target_visible = target_xy_b is not None
        active_plan = self.last_plan is not None and self.current_action_index < len(self.action_queue)
        continuing_without_target = False

        if target_visible:
            assert target_xy_b is not None
            target_xy = _as_xy(target_xy_b, name="target_xy_b")
            if target_reached(target_xy, cfg=self.cfg):
                self.action_queue.clear()
                self.current_action_index = 0
                self.pending_replan_reason = None
                return self._stop_decision(
                    reason="target_reached",
                    target_visible=True,
                    target_reached_value=True,
                    front_min_distance_m=front_min_distance_m,
                    metadata=meta,
                )
        else:
            target_xy = None
            can_continue_active = (
                bool(self.cfg.continue_active_plan_without_target)
                and active_plan
                and self.pending_replan_reason is None
            )
            if not can_continue_active:
                self.action_queue.clear()
                self.current_action_index = 0
                self.pending_replan_reason = None
                return self._stop_decision(
                    reason="target_missing_stop_observe",
                    target_visible=False,
                    front_min_distance_m=front_min_distance_m,
                    metadata=meta,
                )
            continuing_without_target = True
            meta["continuing_active_plan_without_target"] = True

        if occupancy_grid is None:
            return self._stop_decision(
                reason="missing_occupancy_grid",
                target_visible=bool(target_visible),
                front_min_distance_m=front_min_distance_m,
                replan_reason=self.pending_replan_reason,
                metadata=meta,
            )

        if force_replan:
            if not target_visible:
                return self._stop_decision(
                    reason="target_missing_stop_observe",
                    target_visible=False,
                    front_min_distance_m=front_min_distance_m,
                    replan_reason="force_replan",
                    metadata={**meta, "force_replan_without_target": True},
                )
            return self._request_sync_stop("force_replan", front_min_distance_m=front_min_distance_m, metadata=meta)

        need_immediate_replan_reason: str | None = None
        if self.pending_replan_reason is not None:
            need_immediate_replan_reason = self.pending_replan_reason
        elif not active_plan:
            need_immediate_replan_reason = "initial_sync" if self.last_plan is None else "queue_exhausted"

        if need_immediate_replan_reason is not None:
            if target_xy is None:
                self.action_queue.clear()
                self.current_action_index = 0
                self.pending_replan_reason = None
                return self._stop_decision(
                    reason="target_missing_stop_observe",
                    target_visible=False,
                    front_min_distance_m=front_min_distance_m,
                    replan_reason=need_immediate_replan_reason,
                    metadata={**meta, "replan_requires_target": True},
                )
            plan = self._sync_replan(target_xy_b=target_xy, occupancy_grid=occupancy_grid, replan_reason=need_immediate_replan_reason)
            if not plan.found:
                return self._stop_decision(
                    reason="plan_failed",
                    target_visible=True,
                    front_min_distance_m=front_min_distance_m,
                    replan_attempted=True,
                    replan_reason=need_immediate_replan_reason,
                    metadata={**meta, "plan_failure_reason": plan.astar_result.reason},
                )
            if not self.action_queue:
                return self._stop_decision(
                    reason="empty_action_queue",
                    target_visible=True,
                    front_min_distance_m=front_min_distance_m,
                    replan_attempted=True,
                    replan_reason=need_immediate_replan_reason,
                    metadata=meta,
                )
            return self._execute_next_action(
                target_visible=True,
                front_min_distance_m=front_min_distance_m,
                replan_reason=need_immediate_replan_reason,
                replan_attempted=True,
                metadata=meta,
            )

        assert self.last_plan is not None

        if target_xy is not None and float(self.cfg.target_change_replan_m) > 0.0:
            mismatch = _target_mismatch_m(
                observed_target_xy_b=target_xy,
                plan_target_xy_b=self.last_plan.target_xy_b,
                expected_pose=self.expected_pose_in_plan,
            )
            meta["target_expected_mismatch_m"] = float(mismatch)
            if mismatch >= float(self.cfg.target_change_replan_m):
                return self._request_sync_stop("target_changed", front_min_distance_m=front_min_distance_m, metadata=meta)

        # Periodic replans require a current target.  When the real runner is
        # intentionally executing the first-detection plan through occlusion,
        # continue the active queue instead of stopping to request an impossible
        # target-less replan.
        if not continuing_without_target:
            if int(self.cfg.replan_every_actions) > 0 and self.executed_actions_since_replan >= int(self.cfg.replan_every_actions):
                return self._request_sync_stop("periodic_replan_actions", front_min_distance_m=front_min_distance_m, metadata=meta)

            if float(self.cfg.replan_every_forward_m) > 0.0 and self.executed_forward_m_since_replan >= float(self.cfg.replan_every_forward_m):
                return self._request_sync_stop("periodic_replan_distance", front_min_distance_m=front_min_distance_m, metadata=meta)

        if 0 <= self.current_action_index < len(self.action_queue):
            next_action = self.action_queue[self.current_action_index]
            blocked, block_info = translation_action_blocked_by_grid(next_action, occupancy_grid, cfg=self.cfg)
        else:
            blocked, block_info = False, {"checked": False, "reason": "no_next_action"}
        meta["path_block_check"] = block_info
        if blocked:
            return self._request_sync_stop("path_blocked", front_min_distance_m=front_min_distance_m, metadata=meta)

        return self._execute_next_action(
            target_visible=bool(target_visible),
            front_min_distance_m=front_min_distance_m,
            replan_reason=None,
            replan_attempted=False,
            metadata=meta,
        )


__all__ = [
    "PointXY",
    "StepwiseActionKind",
    "StepwiseAction",
    "StepwiseNavConfig",
    "StepwisePlanResult",
    "StepwiseDecision",
    "StepwiseNavigator",
    "front_clearance_from_points",
    "forward_action_blocked",
    "remaining_path_blocked",
    "translation_action_blocked_by_grid",
    "path_to_stepwise_actions",
    "plan_stepwise_actions",
    "target_reached",
]
