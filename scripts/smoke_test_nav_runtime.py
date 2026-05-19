#!/usr/bin/env python3
"""Non-Isaac smoke test for the Phase-2 navigation runtime.

It verifies the minimal online wrapper behavior:

1. plan once on an empty grid;
2. emit a command;
3. introduce a body-frame obstacle on the remaining path;
4. detect path conflict and replan;
5. keep emitting ``(vx, vy, wz)`` commands.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg  # noqa: E402
from g1_nav_rl.planning.grid_astar import GridAStarCfg, xy_to_grid_cell  # noqa: E402
from g1_nav_rl.navigation.simple_nav import (  # noqa: E402
    Pose2D,
    SimpleNavConfig,
    SimpleNavigator,
    Target2D,
    integrate_body_command,
)
from g1_nav_rl.navigation.runtime import NavRuntime, NavRuntimeConfig  # noqa: E402


def make_grid(cfg: OccupancyGridCfg) -> torch.Tensor:
    return torch.zeros((cfg.height, cfg.width), dtype=torch.float32)


def add_box_b(grid: torch.Tensor, cfg: OccupancyGridCfg, *, x0: float, x1: float, y0: float, y1: float) -> None:
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    row_a, col_a = xy_to_grid_cell((xmax, ymin), cfg)
    row_b, col_b = xy_to_grid_cell((xmin, ymax), cfg)
    r0, r1 = sorted((max(0, row_a), min(cfg.height - 1, row_b)))
    c0, c1 = sorted((max(0, col_a), min(cfg.width - 1, col_b)))
    grid[r0 : r1 + 1, c0 : c1 + 1] = 1.0


def main() -> None:
    grid_cfg = OccupancyGridCfg(x_min=-2.0, x_max=6.0, y_min=-4.0, y_max=4.0, resolution=0.05)
    astar_cfg = GridAStarCfg(inflation_radius=0.35, allow_diagonal=True)
    nav_cfg = SimpleNavConfig(success_radius=0.75)
    runtime = NavRuntime(
        SimpleNavigator(grid_cfg, astar_cfg, nav_cfg),
        NavRuntimeConfig(
            enable_replanning=True,
            replan_every=0,
            path_check_ignore_distance=0.20,
            path_check_lookahead=2.50,
        ),
    )

    pose = Pose2D(0.0, 0.0, 0.0)
    target = Target2D(4.0, 0.0)
    empty_grid = make_grid(grid_cfg)

    first = runtime.step(pose, target, empty_grid, step_index=0)
    if runtime.plan_result is None or not runtime.plan_result.found:
        raise RuntimeError("initial plan failed")
    print(
        f"[INFO]: initial plan found={runtime.plan_result.found} "
        f"waypoints={len(runtime.plan_result.path_xy_w)} command={first.command.command.tuple}"
    )

    # Move a little, then introduce a blocking local obstacle in front of the robot.
    for i in range(10):
        step = runtime.step(pose, target, empty_grid, step_index=i + 1)
        pose = integrate_body_command(pose, step.command.command, dt=0.02)

    blocked_grid = make_grid(grid_cfg)
    add_box_b(blocked_grid, grid_cfg, x0=0.80, x1=1.50, y0=-0.55, y1=0.55)
    blocked_step = runtime.step(pose, target, blocked_grid, step_index=11)
    if not blocked_step.replan_attempted:
        raise RuntimeError("expected path-blocked replanning attempt")
    if not blocked_step.replan_accepted:
        raise RuntimeError("expected replanning to find an alternate path")
    if runtime.replan_count <= 0:
        raise RuntimeError("expected runtime replan_count > 0")

    vx, vy, wz = blocked_step.command.command.tuple
    print(
        f"[INFO]: replan accepted count={runtime.replan_count} "
        f"reason={blocked_step.replan_reason} command=({vx:.3f}, {vy:.3f}, {wz:.3f})"
    )
    if abs(vy) < 1.0e-4 and abs(wz) < 1.0e-4:
        print("[WARN]: alternate path command is nearly straight at first step; this can be acceptable after replanning")

    # Continue for a short rollout on the replanned grid to ensure commands remain finite.
    for i in range(12, 40):
        step = runtime.step(pose, target, blocked_grid, step_index=i)
        cmd = step.command.command
        vals = cmd.tuple
        if not all(math.isfinite(v) for v in vals):
            raise RuntimeError(f"non-finite command at step {i}: {vals}")
        pose = integrate_body_command(pose, cmd, dt=0.02)

    print("[INFO]: runtime smoke test passed")


if __name__ == "__main__":
    main()
