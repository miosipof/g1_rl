#!/usr/bin/env python3
"""
Minimal deterministic 2D navigation smoke test.

This is intentionally simpler than smoke_test_lidar_astar_planner.py:
- known scenario geometry, no LiDAR/RayCaster dependency
- obstacle inflation = robot_radius + safety_margin
- target success radius defaults to hand_length
- one A* plan at reset
- direct waypoint-to-(vx, vy, wz) mapping
- no replanning
- no command smoothing
- no local escape / emergency override

The goal is to isolate whether the planned discrete path plus the simple follower
keeps the robot center outside the inflated obstacle region.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


XY = Tuple[float, float]
Pose2D = Tuple[float, float, float]


@dataclass(frozen=True)
class RectObstacle:
    """Axis-aligned rectangular obstacle in world XY."""

    name: str
    center: XY
    size: XY

    @property
    def x_min(self) -> float:
        return self.center[0] - 0.5 * self.size[0]

    @property
    def x_max(self) -> float:
        return self.center[0] + 0.5 * self.size[0]

    @property
    def y_min(self) -> float:
        return self.center[1] - 0.5 * self.size[1]

    @property
    def y_max(self) -> float:
        return self.center[1] + 0.5 * self.size[1]

    def contains_inflated(self, x: np.ndarray, y: np.ndarray, inflation: float) -> np.ndarray:
        return (
            (x >= self.x_min - inflation)
            & (x <= self.x_max + inflation)
            & (y >= self.y_min - inflation)
            & (y <= self.y_max + inflation)
        )

    def signed_distance_to_rect(self, x: float, y: float) -> float:
        """Signed distance from point to the non-inflated rectangle boundary.

        Positive outside, negative inside.
        """
        dx = abs(x - self.center[0]) - 0.5 * self.size[0]
        dy = abs(y - self.center[1]) - 0.5 * self.size[1]
        outside_dx = max(dx, 0.0)
        outside_dy = max(dy, 0.0)
        outside = math.hypot(outside_dx, outside_dy)
        inside = min(max(dx, dy), 0.0)
        return outside + inside


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    start: Pose2D
    target: XY
    obstacles: Tuple[RectObstacle, ...]


@dataclass
class AStarResult:
    found: bool
    reason: str
    path_xy: List[XY]
    expansions: int
    cost: float


@dataclass
class StepRecord:
    step: int
    x: float
    y: float
    yaw: float
    waypoint_index: int
    waypoint_x: float
    waypoint_y: float
    target_distance: float
    min_obstacle_clearance: float
    vx: float
    vy: float
    wz: float


def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def scenario_library() -> Dict[str, Scenario]:
    """Known deterministic scenarios.

    front_box matches the earlier front-box smoke-test layout:
    start=(0, 0), target=(4, 0), one 0.8 x 0.8 m box centered at x=2.2.
    """
    return {
        "front_box": Scenario(
            name="front_box",
            description="Single rectangular box centered on the direct start-to-target line.",
            start=(0.0, 0.0, 0.0),
            target=(4.0, 0.0),
            obstacles=(
                RectObstacle(
                    name="FrontBox",
                    center=(2.2, 0.0),
                    size=(0.8, 0.8),
                ),
            ),
        ),
        "empty": Scenario(
            name="empty",
            description="No obstacles; should drive nearly straight to the target.",
            start=(0.0, 0.0, 0.0),
            target=(4.0, 0.0),
            obstacles=(),
        ),
        "offset_box": Scenario(
            name="offset_box",
            description="Box offset from the direct line; verifies no unnecessary detour.",
            start=(0.0, 0.0, 0.0),
            target=(4.0, 0.0),
            obstacles=(
                RectObstacle(
                    name="OffsetBox",
                    center=(2.2, 0.75),
                    size=(0.8, 0.8),
                ),
            ),
        ),
        "narrow_gap": Scenario(
            name="narrow_gap",
            description="Two boxes with a central gap; useful for inflation sanity checks.",
            start=(0.0, 0.0, 0.0),
            target=(4.0, 0.0),
            obstacles=(
                RectObstacle("UpperBox", center=(2.2, 0.85), size=(0.8, 0.8)),
                RectObstacle("LowerBox", center=(2.2, -0.85), size=(0.8, 0.8)),
            ),
        ),
        "wide_gap": Scenario(
            name="wide_gap",
            description="Two boxes with a central gap; useful for inflation sanity checks.",
            start=(0.0, 0.0, 0.0),
            target=(4.0, 0.0),
            obstacles=(
                RectObstacle("UpperBox", center=(2.2, 0.85), size=(0.8, 0.8)),
                RectObstacle("LowerBox", center=(2.2, -1.85), size=(0.8, 0.8)),
            ),
        ),        
    }


class OccupancyGrid:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
        occupied: np.ndarray,
    ) -> None:
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.resolution = float(resolution)
        self.occupied = occupied.astype(bool)
        self.height, self.width = self.occupied.shape

    def world_to_grid(self, xy: XY) -> Tuple[int, int]:
        i = int(round((xy[0] - self.x_min) / self.resolution))
        j = int(round((xy[1] - self.y_min) / self.resolution))
        return i, j

    def grid_to_world(self, ij: Tuple[int, int]) -> XY:
        i, j = ij
        return (
            self.x_min + i * self.resolution,
            self.y_min + j * self.resolution,
        )

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.width and 0 <= j < self.height

    def is_free(self, i: int, j: int) -> bool:
        return self.in_bounds(i, j) and not bool(self.occupied[j, i])


def build_occupancy_grid(
    scenario: Scenario,
    *,
    resolution: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    inflation: float,
) -> OccupancyGrid:
    xs = np.linspace(x_min, x_max, int(round((x_max - x_min) / resolution)) + 1)
    ys = np.linspace(y_min, y_max, int(round((y_max - y_min) / resolution)) + 1)
    xx, yy = np.meshgrid(xs, ys)

    occupied = np.zeros_like(xx, dtype=bool)
    for obstacle in scenario.obstacles:
        occupied |= obstacle.contains_inflated(xx, yy, inflation)

    return OccupancyGrid(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        resolution=resolution,
        occupied=occupied,
    )


def astar_plan(
    grid: OccupancyGrid,
    start_xy: XY,
    target_xy: XY,
    *,
    diagonal: bool = True,
) -> AStarResult:
    start = grid.world_to_grid(start_xy)
    goal = grid.world_to_grid(target_xy)

    if not grid.is_free(*start):
        return AStarResult(False, "start_cell_occupied", [], 0, math.inf)
    if not grid.is_free(*goal):
        return AStarResult(False, "goal_cell_occupied", [], 0, math.inf)

    if diagonal:
        moves = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        ]
    else:
        moves = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
        ]

    def heuristic(cell: Tuple[int, int]) -> float:
        return math.hypot(
            (cell[0] - goal[0]) * grid.resolution,
            (cell[1] - goal[1]) * grid.resolution,
        )

    open_heap: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), start))

    g_score: Dict[Tuple[int, int], float] = {start: 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed = set()
    expansions = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue

        closed.add(current)
        expansions += 1

        if current == goal:
            path_cells = [current]
            while path_cells[-1] != start:
                path_cells.append(came_from[path_cells[-1]])
            path_cells.reverse()
            path_xy = [grid.grid_to_world(cell) for cell in path_cells]
            path_xy[0] = start_xy
            path_xy[-1] = target_xy
            return AStarResult(True, "goal_reached", path_xy, expansions, g_score[current])

        ci, cj = current
        for di, dj, move_cost in moves:
            ni, nj = ci + di, cj + dj
            if not grid.is_free(ni, nj):
                continue

            # Prevent diagonal corner cutting through inflated obstacle corners.
            if di != 0 and dj != 0:
                if not grid.is_free(ci + di, cj) or not grid.is_free(ci, cj + dj):
                    continue

            neighbor = (ni, nj)
            tentative = g_score[current] + move_cost * grid.resolution
            if tentative >= g_score.get(neighbor, math.inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(neighbor), neighbor))

    return AStarResult(False, "open_set_exhausted", [], expansions, math.inf)


def min_robot_clearance_to_obstacles(
    xy: XY,
    obstacles: Sequence[RectObstacle],
    robot_radius: float,
) -> float:
    if not obstacles:
        return math.inf
    x, y = xy
    return min(ob.signed_distance_to_rect(x, y) - robot_radius for ob in obstacles)


def select_waypoint(
    path_xy: Sequence[XY],
    current_xy: XY,
    waypoint_index: int,
    waypoint_reached_radius: float,
) -> int:
    """Advance monotonically through the already-planned path.

    This intentionally does not shortcut to a line-of-sight target. It only skips
    waypoints already inside the reached radius.
    """
    i = waypoint_index
    while i < len(path_xy) - 1:
        wx, wy = path_xy[i]
        if math.hypot(wx - current_xy[0], wy - current_xy[1]) > waypoint_reached_radius:
            break
        i += 1
    return i


def path_tangent_yaw(path_xy: Sequence[XY], waypoint_index: int, fallback_yaw: float) -> float:
    if waypoint_index <= 0 or waypoint_index >= len(path_xy):
        return fallback_yaw

    prev_xy = path_xy[max(0, waypoint_index - 1)]
    next_xy = path_xy[waypoint_index]
    dx = next_xy[0] - prev_xy[0]
    dy = next_xy[1] - prev_xy[1]
    if abs(dx) + abs(dy) < 1e-9:
        return fallback_yaw
    return math.atan2(dy, dx)


def command_to_waypoint(
    pose: Pose2D,
    waypoint_xy: XY,
    path_yaw: float,
    *,
    k_xy: float,
    k_yaw: float,
    max_vx: float,
    max_vy: float,
    max_wz: float,
    allow_backward: bool,
) -> Tuple[float, float, float, Tuple[float, float]]:
    x, y, yaw = pose
    dx_w = waypoint_xy[0] - x
    dy_w = waypoint_xy[1] - y

    c = math.cos(yaw)
    s = math.sin(yaw)

    # World -> body transform.
    x_b = c * dx_w + s * dy_w
    y_b = -s * dx_w + c * dy_w

    min_vx = -max_vx if allow_backward else 0.0
    vx = clip(k_xy * x_b, min_vx, max_vx)
    vy = clip(k_xy * y_b, -max_vy, max_vy)

    yaw_error = angle_wrap(path_yaw - yaw)
    wz = clip(k_yaw * yaw_error, -max_wz, max_wz)

    return vx, vy, wz, (x_b, y_b)


def integrate_body_command(pose: Pose2D, command: Tuple[float, float, float], dt: float) -> Pose2D:
    x, y, yaw = pose
    vx, vy, wz = command

    c = math.cos(yaw)
    s = math.sin(yaw)

    x_next = x + (c * vx - s * vy) * dt
    y_next = y + (s * vx + c * vy) * dt
    yaw_next = angle_wrap(yaw + wz * dt)

    return x_next, y_next, yaw_next


def simulate_navigation(
    scenario: Scenario,
    path_xy: Sequence[XY],
    *,
    max_steps: int,
    dt: float,
    success_radius: float,
    robot_radius: float,
    safety_margin: float,
    waypoint_reached_radius: float,
    k_xy: float,
    k_yaw: float,
    max_vx: float,
    max_vy: float,
    max_wz: float,
    allow_backward: bool,
    print_every: int,
    save_every: int,
    output_dir: Path,
    save_grid: bool,
    grid: OccupancyGrid,
) -> Tuple[bool, List[StepRecord], float, float]:
    pose: Pose2D = scenario.start
    waypoint_index = 1 if len(path_xy) > 1 else 0
    records: List[StepRecord] = []
    min_clearance = math.inf
    min_margin_slack = math.inf

    if save_grid:
        draw_frame(
            output_dir=output_dir,
            filename="nav_grid_000000.png",
            scenario=scenario,
            grid=grid,
            path_xy=path_xy,
            records=[],
            pose=pose,
            waypoint_xy=path_xy[waypoint_index],
            robot_radius=robot_radius,
            inflation=robot_radius + safety_margin,
            success_radius=success_radius,
        )

    reached = False

    for step in range(max_steps + 1):
        x, y, yaw = pose
        target_distance = math.hypot(scenario.target[0] - x, scenario.target[1] - y)

        waypoint_index = select_waypoint(
            path_xy=path_xy,
            current_xy=(x, y),
            waypoint_index=waypoint_index,
            waypoint_reached_radius=waypoint_reached_radius,
        )
        waypoint_xy = path_xy[waypoint_index]
        tangent_yaw = path_tangent_yaw(path_xy, waypoint_index, yaw)

        vx, vy, wz, waypoint_xy_b = command_to_waypoint(
            pose,
            waypoint_xy,
            tangent_yaw,
            k_xy=k_xy,
            k_yaw=k_yaw,
            max_vx=max_vx,
            max_vy=max_vy,
            max_wz=max_wz,
            allow_backward=allow_backward,
        )

        clearance = min_robot_clearance_to_obstacles((x, y), scenario.obstacles, robot_radius)
        min_clearance = min(min_clearance, clearance)
        min_margin_slack = min(min_margin_slack, clearance - safety_margin)

        record = StepRecord(
            step=step,
            x=x,
            y=y,
            yaw=yaw,
            waypoint_index=waypoint_index,
            waypoint_x=waypoint_xy[0],
            waypoint_y=waypoint_xy[1],
            target_distance=target_distance,
            min_obstacle_clearance=clearance,
            vx=vx,
            vy=vy,
            wz=wz,
        )
        records.append(record)

        should_print = step == 0 or step % print_every == 0 or target_distance <= success_radius
        if should_print:
            print_step(
                record=record,
                total_waypoints=len(path_xy),
                waypoint_xy_b=waypoint_xy_b,
                success_radius=success_radius,
                robot_radius=robot_radius,
                safety_margin=safety_margin,
            )

        if save_grid and (step > 0) and (step % save_every == 0):
            draw_frame(
                output_dir=output_dir,
                filename=f"nav_grid_{step:06d}.png",
                scenario=scenario,
                grid=grid,
                path_xy=path_xy,
                records=records,
                pose=pose,
                waypoint_xy=waypoint_xy,
                robot_radius=robot_radius,
                inflation=robot_radius + safety_margin,
                success_radius=success_radius,
            )

        if target_distance <= success_radius:
            reached = True
            break

        if clearance < 0.0:
            print(f"[WARN]: collision detected at step={step}: clearance={clearance:.3f} m")
            break

        pose = integrate_body_command(pose, (vx, vy, wz), dt)

    if save_grid:
        draw_frame(
            output_dir=output_dir,
            filename="nav_grid_final.png",
            scenario=scenario,
            grid=grid,
            path_xy=path_xy,
            records=records,
            pose=(records[-1].x, records[-1].y, records[-1].yaw),
            waypoint_xy=(records[-1].waypoint_x, records[-1].waypoint_y),
            robot_radius=robot_radius,
            inflation=robot_radius + safety_margin,
            success_radius=success_radius,
        )

    return reached, records, min_clearance, min_margin_slack


def print_step(
    *,
    record: StepRecord,
    total_waypoints: int,
    waypoint_xy_b: XY,
    success_radius: float,
    robot_radius: float,
    safety_margin: float,
) -> None:
    print("-" * 80)
    print(f"step                    : {record.step}")
    print(f"target distance         : {record.target_distance:.3f}")
    print(f"target reached radius   : {success_radius:.3f}")
    print(f"waypoint index          : {record.waypoint_index}/{total_waypoints - 1}")
    print(f"waypoint xy world       : ({record.waypoint_x:.3f}, {record.waypoint_y:.3f})")
    print(f"waypoint xy body        : ({waypoint_xy_b[0]:.3f}, {waypoint_xy_b[1]:.3f})")
    print(f"command                 : [{record.vx:.3f}, {record.vy:.3f}, {record.wz:.3f}]")
    print(f"pose                    : [{record.x:.3f}, {record.y:.3f}, yaw={record.yaw:.3f}]")
    if math.isfinite(record.min_obstacle_clearance):
        print(f"obstacle edge clearance : {record.min_obstacle_clearance:.3f}")
        print(f"planning safety margin  : {safety_margin:.3f}")
        print(f"robot radius            : {robot_radius:.3f}")
    else:
        print("obstacle edge clearance : inf")
    print("replanning              : disabled")
    print("command smoothing       : disabled")
    print("local escape            : disabled")


def write_records_csv(path: Path, records: Sequence[StepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "x",
                "y",
                "yaw",
                "waypoint_index",
                "waypoint_x",
                "waypoint_y",
                "target_distance",
                "min_obstacle_clearance",
                "vx",
                "vy",
                "wz",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.step,
                    f"{r.x:.9f}",
                    f"{r.y:.9f}",
                    f"{r.yaw:.9f}",
                    r.waypoint_index,
                    f"{r.waypoint_x:.9f}",
                    f"{r.waypoint_y:.9f}",
                    f"{r.target_distance:.9f}",
                    f"{r.min_obstacle_clearance:.9f}",
                    f"{r.vx:.9f}",
                    f"{r.vy:.9f}",
                    f"{r.wz:.9f}",
                ]
            )


def draw_frame(
    *,
    output_dir: Path,
    filename: str,
    scenario: Scenario,
    grid: OccupancyGrid,
    path_xy: Sequence[XY],
    records: Sequence[StepRecord],
    pose: Pose2D,
    waypoint_xy: XY,
    robot_radius: float,
    inflation: float,
    success_radius: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except Exception as exc:
        print(f"[WARN]: matplotlib unavailable; skipping frame save ({exc})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Occupancy grid as a faint background.
    ax.imshow(
        grid.occupied.astype(float),
        origin="lower",
        extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
        alpha=0.18,
        interpolation="nearest",
    )

    # Obstacles: original and inflated rectangles.
    for obstacle in scenario.obstacles:
        inflated = Rectangle(
            (obstacle.x_min - inflation, obstacle.y_min - inflation),
            obstacle.size[0] + 2.0 * inflation,
            obstacle.size[1] + 2.0 * inflation,
            fill=False,
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(inflated)

        original = Rectangle(
            (obstacle.x_min, obstacle.y_min),
            obstacle.size[0],
            obstacle.size[1],
            fill=True,
            alpha=0.45,
        )
        ax.add_patch(original)
        ax.text(obstacle.center[0], obstacle.center[1], obstacle.name, ha="center", va="center")

    if path_xy:
        px = [p[0] for p in path_xy]
        py = [p[1] for p in path_xy]
        ax.plot(px, py, linewidth=2.0, label="A* path")

    if records:
        tx = [r.x for r in records]
        ty = [r.y for r in records]
        ax.plot(tx, ty, linewidth=2.0, label="executed trajectory")

    sx, sy, _ = scenario.start
    ax.scatter([sx], [sy], s=80, marker="o", label="start")
    ax.scatter([scenario.target[0]], [scenario.target[1]], s=90, marker="*", label="target")
    ax.add_patch(Circle(scenario.target, success_radius, fill=False, linestyle=":", linewidth=1.5))

    x, y, yaw = pose
    ax.scatter([x], [y], s=80, marker="o", label="robot")
    ax.add_patch(Circle((x, y), robot_radius, fill=False, linewidth=1.5))
    ax.arrow(
        x,
        y,
        0.35 * math.cos(yaw),
        0.35 * math.sin(yaw),
        head_width=0.06,
        length_includes_head=True,
    )

    ax.scatter([waypoint_xy[0]], [waypoint_xy[1]], s=55, marker="x", label="current waypoint")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(grid.x_min, grid.x_max)
    ax.set_ylim(grid.y_min, grid.y_max)
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="upper right")
    ax.set_title(f"{scenario.name}: A* path + simple waypoint follower")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[INFO]: Saved nav debug image: {out}")


def parse_args() -> argparse.Namespace:
    scenarios = sorted(scenario_library().keys())

    parser = argparse.ArgumentParser(
        description="Minimal deterministic 2D nav smoke test: one A* plan, simple command mapping, no escape/replan/smoothing."
    )
    parser.add_argument("--scenario", choices=scenarios, default="front_box")
    parser.add_argument("--output_dir", default="outputs/nav/front_box")

    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--no_save_grid", action="store_true")

    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--x_min", type=float, default=-1.0)
    parser.add_argument("--x_max", type=float, default=5.0)
    parser.add_argument("--y_min", type=float, default=-3.0)
    parser.add_argument("--y_max", type=float, default=3.0)
    parser.add_argument("--disable_diagonal", action="store_true")

    parser.add_argument("--robot_radius", type=float, default=0.34)
    parser.add_argument("--safety_margin", type=float, default=0.35)
    parser.add_argument(
        "--hand_length",
        type=float,
        default=0.75,
        help="Default target reached radius. This models the target as reachable by hand/arm without requiring the base center to hit it exactly.",
    )
    parser.add_argument(
        "--success_radius",
        type=float,
        default=None,
        help="Override target reached radius. Defaults to --hand_length.",
    )

    parser.add_argument("--waypoint_reached_radius", type=float, default=0.12)
    parser.add_argument("--k_xy", type=float, default=1.5)
    parser.add_argument("--k_yaw", type=float, default=2.0)
    parser.add_argument("--max_vx", type=float, default=0.30)
    parser.add_argument("--max_vy", type=float, default=0.25)
    parser.add_argument("--max_wz", type=float, default=0.60)
    parser.add_argument("--allow_backward", action="store_true")

    parser.add_argument("--assert_planner", action="store_true")
    parser.add_argument("--assert_success", action="store_true")
    parser.add_argument("--assert_no_collision", action="store_true")
    parser.add_argument(
        "--assert_margin",
        action="store_true",
        help="Assert the executed robot disk stayed outside the planned safety margin, not merely outside collision.",
    )
    parser.add_argument(
        "--margin_tolerance",
        type=float,
        default=0.03,
        help="Tolerance for --assert_margin, in meters.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = scenario_library()
    scenario = scenarios[args.scenario]

    output_dir = Path(args.output_dir)
    if args.output_dir == "outputs/nav/front_box" and scenario.name != "front_box":
        output_dir = Path("outputs/nav") / scenario.name

    output_dir.mkdir(parents=True, exist_ok=True)

    success_radius = args.success_radius if args.success_radius is not None else args.hand_length
    inflation = args.robot_radius + args.safety_margin

    print("[INFO]: Starting minimal nav smoke test...")
    print(f"[INFO]: scenario: {scenario.name} — {scenario.description}")
    print(f"[INFO]: start pose: {scenario.start}")
    print(f"[INFO]: target_xy_w: {scenario.target}")
    print(f"[INFO]: target reached radius: {success_radius:.3f} (hand_length={args.hand_length:.3f})")
    print(f"[INFO]: robot_radius={args.robot_radius:.3f}, safety_margin={args.safety_margin:.3f}, inflation={inflation:.3f}")
    print(f"[INFO]: A*: one-shot plan, diagonal={not args.disable_diagonal}, resolution={args.resolution:.3f}")
    print("[INFO]: controller: direct waypoint mapping to [vx, vy, wz]")
    print("[INFO]: replanning=disabled, command_smoothing=disabled, local_escape=disabled")
    print(f"[INFO]: max_steps={args.max_steps}, print_every={args.print_every}, save_every={args.save_every}")

    if scenario.obstacles:
        for idx, obstacle in enumerate(scenario.obstacles):
            print(
                f"  obstacle[{idx}]: name={obstacle.name} center={list(obstacle.center)} size={list(obstacle.size)}"
            )
    else:
        print("  obstacles: none")

    grid = build_occupancy_grid(
        scenario,
        resolution=args.resolution,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        inflation=inflation,
    )

    plan = astar_plan(
        grid,
        start_xy=(scenario.start[0], scenario.start[1]),
        target_xy=scenario.target,
        diagonal=not args.disable_diagonal,
    )

    print(f"[INFO]: occupancy grid: H={grid.height}, W={grid.width}, occupied_cells={int(grid.occupied.sum())}")
    print(f"[INFO]: A* found={plan.found}, reason={plan.reason}, expansions={plan.expansions}, cost={plan.cost:.3f}")
    print(f"[INFO]: A* path length cells={len(plan.path_xy)}")

    if args.assert_planner and not plan.found:
        raise AssertionError(f"A* planner failed: {plan.reason}")

    if not plan.found:
        return 2

    reached, records, min_clearance, min_margin_slack = simulate_navigation(
        scenario,
        plan.path_xy,
        max_steps=args.max_steps,
        dt=args.dt,
        success_radius=success_radius,
        robot_radius=args.robot_radius,
        safety_margin=args.safety_margin,
        waypoint_reached_radius=args.waypoint_reached_radius,
        k_xy=args.k_xy,
        k_yaw=args.k_yaw,
        max_vx=args.max_vx,
        max_vy=args.max_vy,
        max_wz=args.max_wz,
        allow_backward=args.allow_backward,
        print_every=max(1, args.print_every),
        save_every=max(1, args.save_every),
        output_dir=output_dir,
        save_grid=not args.no_save_grid,
        grid=grid,
    )

    csv_path = output_dir / "nav_trace.csv"
    write_records_csv(csv_path, records)

    first_dist = records[0].target_distance if records else math.inf
    last_dist = records[-1].target_distance if records else math.inf
    min_dist = min((r.target_distance for r in records), default=math.inf)

    print("=" * 80)
    print(f"[INFO]: target distance first/last/min: {first_dist:.3f} / {last_dist:.3f} / {min_dist:.3f}")
    if math.isfinite(min_clearance):
        print(f"[INFO]: min obstacle edge clearance: {min_clearance:.3f}")
        print(f"[INFO]: min safety-margin slack: {min_margin_slack:.3f}")
    else:
        print("[INFO]: min obstacle edge clearance: inf")
        print("[INFO]: min safety-margin slack: inf")
    print(f"[INFO]: target reached: {reached} step={records[-1].step if reached and records else None}")
    print(f"[INFO]: wrote trace: {csv_path}")
    print("[INFO]: minimal nav smoke test complete.")

    if args.assert_no_collision and math.isfinite(min_clearance) and min_clearance < 0.0:
        raise AssertionError(f"collision: min_clearance={min_clearance:.3f}")

    if args.assert_margin and math.isfinite(min_margin_slack) and min_margin_slack < -args.margin_tolerance:
        raise AssertionError(
            f"executed trajectory cut into planning margin: "
            f"min_slack={min_margin_slack:.3f}, tolerance={args.margin_tolerance:.3f}"
        )

    if args.assert_success and not reached:
        raise AssertionError(f"target not reached: last_distance={last_dist:.3f}, success_radius={success_radius:.3f}")

    return 0 if reached else 1


if __name__ == "__main__":
    raise SystemExit(main())
