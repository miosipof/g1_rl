"""Minimal runtime wrapper around :mod:`g1_nav_rl.navigation.simple_nav`.

This module extracts the online part of ``scripts/smoke_test_nav_lidar.py``
without adding a new architecture:

- keep the current A* plan;
- check whether the remaining path still intersects the current inflated grid;
- optionally replan when the path is blocked or at a periodic interval;
- return the next ``MotionCommand(vx, vy, wz)``.

Isaac, LiDAR access, occupancy-grid construction, visualization, and command
transport stay outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.planning.grid_astar import GridAStarCfg, inflate_occupancy_grid, xy_to_grid_cell
from g1_nav_rl.navigation.simple_nav import (
    MotionCommand,
    NavigationCommand,
    NavigationPlan,
    Pose2D,
    SimpleNavigator,
    Target2D,
    world_path_to_body_path,
)

PointXY = tuple[float, float]


@dataclass(frozen=True)
class NavRuntimeConfig:
    """Small runtime/replanning configuration.

    Defaults keep the Phase-1 behavior: one initial plan and no automatic
    replanning unless explicitly enabled.
    """

    enable_replanning: bool = False
    replan_every: int = 0
    path_check_ignore_distance: float = 0.20
    path_check_lookahead: float = 2.00
    allow_goal_relocation: bool = True
    stop_if_replan_fails: bool = False


@dataclass(frozen=True)
class PathConflictResult:
    """Result of checking a remaining path against the current occupancy grid."""

    blocked: bool
    blocked_xy_b: PointXY | None


@dataclass(frozen=True)
class NavRuntimeStep:
    """One runtime step result."""

    command: NavigationCommand
    plan: NavigationPlan | None
    path_conflict: PathConflictResult
    replan_attempted: bool
    replan_accepted: bool
    replan_failed: bool
    replan_reason: str | None
    should_stop: bool
    replan_count: int


@dataclass(frozen=True)
class NavRuntimePlanResult:
    """Explicit planning result returned by :meth:`NavRuntime.plan`."""

    plan: NavigationPlan
    replan_count: int


def iter_sampled_path_points(
    path_xy: Sequence[PointXY],
    *,
    spacing: float,
    ignore_distance: float,
    max_distance: float,
) -> list[PointXY]:
    """Sample a polyline after an initial ignore band and up to a lookahead distance."""
    if len(path_xy) < 2:
        return []
    spacing = max(1.0e-3, float(spacing))
    ignore_distance = max(0.0, float(ignore_distance))
    max_distance = max(0.0, float(max_distance))
    out: list[PointXY] = []
    accumulated = 0.0
    for (x0, y0), (x1, y1) in zip(path_xy[:-1], path_xy[1:]):
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len <= 1.0e-9:
            continue
        n = max(1, int(math.ceil(seg_len / spacing)))
        for i in range(1, n + 1):
            t = float(i) / float(n)
            d = accumulated + t * seg_len
            if d < ignore_distance:
                continue
            if max_distance > 0.0 and d > max_distance:
                return out
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        accumulated += seg_len
        if max_distance > 0.0 and accumulated > max_distance:
            break
    return out


def path_conflicts_with_inflated_grid(
    path_xy_b: Sequence[PointXY],
    occupancy_grid: torch.Tensor,
    *,
    grid_cfg: OccupancyGridCfg,
    astar_cfg: GridAStarCfg,
    ignore_distance: float,
    lookahead_distance: float,
) -> PathConflictResult:
    """Check whether a body-frame path intersects current inflated occupancy."""
    inflated = inflate_occupancy_grid(
        occupancy_grid,
        cfg=grid_cfg,
        inflation_radius=float(astar_cfg.inflation_radius),
        occupancy_threshold=float(astar_cfg.occupancy_threshold),
    )
    samples = iter_sampled_path_points(
        path_xy_b,
        spacing=0.5 * float(grid_cfg.resolution),
        ignore_distance=float(ignore_distance),
        max_distance=float(lookahead_distance),
    )
    for xy in samples:
        row, col = xy_to_grid_cell(xy, grid_cfg)
        if 0 <= row < grid_cfg.height and 0 <= col < grid_cfg.width and bool(inflated[row, col]):
            return PathConflictResult(True, xy)
    return PathConflictResult(False, None)


class NavRuntime:
    """Minimal online navigation runtime.

    ``NavRuntime`` deliberately owns only planner state and replanning decisions.
    The caller is responsible for providing the latest pose, target, and current
    occupancy grid.
    """

    def __init__(
        self,
        navigator: SimpleNavigator,
        config: NavRuntimeConfig | None = None,
    ) -> None:
        self.navigator = navigator
        self.config = config or NavRuntimeConfig()
        self.replan_count = 0

    @property
    def plan_result(self) -> NavigationPlan | None:
        return self.navigator.plan_result

    @property
    def waypoint_index(self) -> int:
        return self.navigator.waypoint_index

    def reset(self) -> None:
        self.navigator.reset()
        self.replan_count = 0

    def current_path_xy_w(self) -> list[PointXY]:
        plan = self.navigator.plan_result
        if plan is None:
            return []
        return list(plan.path_xy_w)

    def remaining_path_xy_w(self) -> list[PointXY]:
        return self.navigator.remaining_world_path()

    def plan(
        self,
        pose: Pose2D,
        target: Target2D,
        occupancy_grid: torch.Tensor,
        *,
        allow_goal_relocation: bool | None = None,
    ) -> NavRuntimePlanResult:
        plan = self.navigator.plan(
            pose,
            target,
            occupancy_grid,
            allow_goal_relocation=self.config.allow_goal_relocation if allow_goal_relocation is None else bool(allow_goal_relocation),
        )
        return NavRuntimePlanResult(plan=plan, replan_count=self.replan_count)

    def check_current_path(
        self,
        pose: Pose2D,
        occupancy_grid: torch.Tensor,
    ) -> PathConflictResult:
        remaining_w = self.navigator.remaining_world_path()
        if len(remaining_w) < 2:
            return PathConflictResult(False, None)
        remaining_b = world_path_to_body_path(remaining_w, pose)
        return path_conflicts_with_inflated_grid(
            remaining_b,
            occupancy_grid,
            grid_cfg=self.navigator.grid_cfg,
            astar_cfg=self.navigator.astar_cfg,
            ignore_distance=float(self.config.path_check_ignore_distance),
            lookahead_distance=float(self.config.path_check_lookahead),
        )

    def _restore_plan(self, plan: NavigationPlan | None, waypoint_index: int) -> None:
        self.navigator.plan_result = plan
        self.navigator.waypoint_index = int(waypoint_index)

    def step(
        self,
        pose: Pose2D,
        target: Target2D,
        occupancy_grid: torch.Tensor,
        *,
        step_index: int = 0,
    ) -> NavRuntimeStep:
        """Update runtime state and return the next command.

        The first call plans if needed. If replanning is enabled, the current
        remaining path is checked against ``occupancy_grid`` and replanned only
        when it is blocked or when ``replan_every`` triggers.
        """
        if self.navigator.plan_result is None:
            self.navigator.plan(
                pose,
                target,
                occupancy_grid,
                allow_goal_relocation=bool(self.config.allow_goal_relocation),
            )

        conflict = PathConflictResult(False, None)
        attempted = False
        accepted = False
        failed = False
        reason: str | None = None
        should_stop = False

        if bool(self.config.enable_replanning) and self.navigator.plan_result is not None:
            conflict = self.check_current_path(pose, occupancy_grid)
            periodic = (
                int(self.config.replan_every) > 0
                and int(step_index) > 0
                and int(step_index) % int(self.config.replan_every) == 0
            )
            if conflict.blocked or periodic:
                attempted = True
                reason = "path_blocked" if conflict.blocked else "periodic"
                old_plan = self.navigator.plan_result
                old_waypoint_index = self.navigator.waypoint_index
                new_plan = self.navigator.plan(
                    pose,
                    target,
                    occupancy_grid,
                    allow_goal_relocation=bool(self.config.allow_goal_relocation),
                )
                if new_plan.found and len(new_plan.path_xy_w) >= 2:
                    accepted = True
                    self.replan_count += 1
                else:
                    failed = True
                    if bool(self.config.stop_if_replan_fails):
                        should_stop = True
                    else:
                        self._restore_plan(old_plan, old_waypoint_index)

        if should_stop:
            nav_cmd = NavigationCommand(
                command=MotionCommand(0.0, 0.0, 0.0),
                target_reached=False,
                waypoint_index=self.navigator.waypoint_index,
                waypoint_xy_w=None,
                waypoint_xy_b=None,
                target_distance=math.hypot(target.x - pose.x, target.y - pose.y),
                path_yaw=pose.yaw,
            )
        else:
            nav_cmd = self.navigator.command(pose, target)

        return NavRuntimeStep(
            command=nav_cmd,
            plan=self.navigator.plan_result,
            path_conflict=conflict,
            replan_attempted=attempted,
            replan_accepted=accepted,
            replan_failed=failed,
            replan_reason=reason,
            should_stop=should_stop,
            replan_count=self.replan_count,
        )
