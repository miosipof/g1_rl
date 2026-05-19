#!/usr/bin/env python3
"""Small non-Isaac smoke test for the Phase-1 simple navigation module.

This validates the same target -> A* plan -> monotonic waypoint commands path
without launching Isaac Sim.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg  # noqa: E402
from g1_nav_rl.planning.grid_astar import GridAStarCfg  # noqa: E402
from g1_nav_rl.navigation.simple_nav import Pose2D, SimpleNavConfig, SimpleNavigator, Target2D  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_x", type=float, default=4.0)
    parser.add_argument("--target_y", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--safety_margin", type=float, default=0.35)
    parser.add_argument("--body_radius", type=float, default=0.34)
    args = parser.parse_args()

    grid_cfg = OccupancyGridCfg(
        x_min=-2.0,
        x_max=6.0,
        y_min=-4.0,
        y_max=4.0,
        resolution=args.resolution,
    )
    astar_cfg = GridAStarCfg(
        occupancy_threshold=0.5,
        inflation_radius=float(args.body_radius + args.safety_margin),
        allow_diagonal=True,
        heuristic_weight=1.0,
        max_expansions=0,
    )
    nav_cfg = SimpleNavConfig()

    # Empty grid smoke test. Obstacle-bearing tests are handled by smoke_test_nav.py
    # and smoke_test_nav_lidar.py; this script only checks the extracted module.
    occupancy = torch.zeros((grid_cfg.height, grid_cfg.width), dtype=torch.float32)
    nav = SimpleNavigator(grid_cfg=grid_cfg, astar_cfg=astar_cfg, config=nav_cfg)
    result = nav.rollout_commands(
        initial_pose=Pose2D(0.0, 0.0, 0.0),
        target=Target2D(args.target_x, args.target_y),
        occupancy_grid=occupancy,
        dt=args.dt,
        max_steps=args.max_steps,
    )

    print(f"[INFO]: plan found={result.plan.found} reason={result.plan.reason} waypoints={result.plan.waypoint_count}")
    for i, cmd in enumerate(result.commands):
        if i == 0 or i % max(1, args.print_every) == 0 or cmd.target_reached:
            pose = result.poses[min(i, len(result.poses) - 1)]
            print(
                f"step={i:04d} pose=({pose.x:.3f},{pose.y:.3f},{pose.yaw:.3f}) "
                f"dist={cmd.target_distance:.3f} waypoint={cmd.waypoint_index} "
                f"cmd=({cmd.command.vx:.3f},{cmd.command.vy:.3f},{cmd.command.wz:.3f})"
            )
    print(f"[INFO]: target_reached={result.target_reached} commands={len(result.commands)}")
    if not result.plan.found or not result.target_reached:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
