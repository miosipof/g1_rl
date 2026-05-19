#!/usr/bin/env python3
"""Offline smoke test for the real-robot SLAM snapshot adapter.

This script accepts the JSON metadata format produced by the G1 SLAM snapshot
probe and converts the referenced point cloud into the local body-frame
occupancy grid expected by ``SimpleNavigator`` / ``NavRuntime``.

It can also run without real files using ``--synthetic``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.mapping.slam_cloud import (
    SlamCloudGridConfig,
    load_slam_info_json,
    pose4x4_to_pose2d,
    slam_cloud_to_occupancy_grid,
    slam_info_json_to_occupancy_grid,
)
from g1_nav_rl.navigation.simple_nav import Pose2D, SimpleNavConfig, SimpleNavigator, Target2D
from g1_nav_rl.navigation.runtime import NavRuntime, NavRuntimeConfig
from g1_nav_rl.planning.grid_astar import GridAStarCfg, inflate_occupancy_grid


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--info_json", type=str, default=None, help="Path to SLAM snapshot JSON metadata.")
    p.add_argument("--synthetic", action="store_true", help="Use a synthetic SLAM snapshot if no real files are available.")
    p.add_argument("--pose_convention", choices=("map_from_body", "body_from_map"), default="map_from_body")
    p.add_argument("--height_filter_frame", choices=("body", "map"), default="body")
    p.add_argument("--z_min", type=float, default=-0.8)
    p.add_argument("--z_max", type=float, default=1.2)
    p.add_argument("--x_min", type=float, default=-2.0)
    p.add_argument("--x_max", type=float, default=6.0)
    p.add_argument("--y_min", type=float, default=-4.0)
    p.add_argument("--y_max", type=float, default=4.0)
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--min_points_per_cell", type=int, default=1)
    p.add_argument("--self_filter_radius", type=float, default=None)
    p.add_argument("--self_filter_z", type=float, default=None)
    p.add_argument("--inflation_radius", type=float, default=0.69, help="A* inflation radius, usually body_radius+safety_margin.")
    p.add_argument("--target_x", type=float, default=4.0, help="Target x in map/world frame.")
    p.add_argument("--target_y", type=float, default=0.0, help="Target y in map/world frame.")
    p.add_argument("--max_steps", type=int, default=20, help="Number of runtime command steps to print.")
    p.add_argument("--save_grid", type=str, default=None, help="Optional PNG path for the inflated planning grid.")
    return p


def make_synthetic_snapshot() -> tuple[dict, np.ndarray, np.ndarray]:
    """Create a small synthetic map with a wall-like point cloud."""
    # Robot at map origin, body aligned with map.
    T_map_body = np.eye(4, dtype=np.float32)
    # Vertical wall in front of robot: x=2.0, y=-1.0..1.0, z=-0.2..0.8.
    ys = np.linspace(-1.0, 1.0, 81)
    zs = np.linspace(-0.2, 0.8, 21)
    yy, zz = np.meshgrid(ys, zs)
    xx = np.full_like(yy, 2.0)
    wall = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)
    # Add some floor points that should be removable by z_min if desired.
    fx = np.linspace(0.0, 5.0, 60)
    fy = np.linspace(-2.0, 2.0, 40)
    fxx, fyy = np.meshgrid(fx, fy)
    floor = np.stack([fxx.ravel(), fyy.ravel(), np.full(fxx.size, -1.2)], axis=-1).astype(np.float32)
    points = np.concatenate([wall, floor], axis=0)
    info = {
        "status": "synthetic",
        "env": {"LIDAR_SELF_FILTER_RADIUS": "0.3", "LIDAR_SELF_FILTER_Z": "0.24"},
        "files": {
            "slam_map_npy_xyz_m": "synthetic_slam_map.npy",
            "slam_pose_4x4_npy": "synthetic_slam_pose.npy",
        },
    }
    return info, points, T_map_body


def save_grid_png(grid: torch.Tensor, path: str | Path, grid_cfg: OccupancyGridCfg, target_xy_b: tuple[float, float] | None = None) -> None:
    from PIL import Image, ImageDraw

    if grid.ndim == 3:
        grid2 = grid[0]
    else:
        grid2 = grid
    occ = grid2.detach().cpu().numpy() > 0.0
    img = np.full(occ.shape, 255, dtype=np.uint8)
    img[occ] = 35
    rgb = np.stack([img, img, img], axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    def xy_to_rc(x, y):
        r = int(math.floor((grid_cfg.x_max - x) / grid_cfg.resolution))
        c = int(math.floor((y - grid_cfg.y_min) / grid_cfg.resolution))
        return r, c

    # robot origin
    rr, cc = xy_to_rc(0.0, 0.0)
    if 0 <= rr < grid_cfg.height and 0 <= cc < grid_cfg.width:
        draw.ellipse((cc - 4, rr - 4, cc + 4, rr + 4), fill=(255, 220, 0), outline=(0, 0, 0))
    if target_xy_b is not None:
        tr, tc = xy_to_rc(float(target_xy_b[0]), float(target_xy_b[1]))
        if 0 <= tr < grid_cfg.height and 0 <= tc < grid_cfg.width:
            draw.ellipse((tc - 4, tr - 4, tc + 4, tr + 4), fill=(0, 90, 255), outline=(0, 0, 0))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def main() -> int:
    args = build_argparser().parse_args()
    grid_cfg = OccupancyGridCfg(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        resolution=args.resolution,
    )
    slam_cfg = SlamCloudGridConfig(
        grid_cfg=grid_cfg,
        z_min=args.z_min,
        z_max=args.z_max,
        height_filter_frame=args.height_filter_frame,
        min_points_per_cell=args.min_points_per_cell,
        self_filter_radius=args.self_filter_radius,
        self_filter_z=args.self_filter_z,
        pose_convention=args.pose_convention,
    )

    if args.synthetic or args.info_json is None:
        info, points, pose_4x4 = make_synthetic_snapshot()
        result = slam_cloud_to_occupancy_grid(points, pose_4x4, config=slam_cfg, info=info)
        print("[INFO]: using synthetic SLAM snapshot")
    else:
        info = load_slam_info_json(args.info_json)
        result = slam_info_json_to_occupancy_grid(args.info_json, config=slam_cfg)
        print(f"[INFO]: loaded SLAM metadata: {args.info_json}")

    print("[INFO]: SLAM occupancy adapter stats:")
    print(json.dumps(result.stats, indent=2))
    print(f"[INFO]: robot pose from SLAM: x={result.pose.x:.3f}, y={result.pose.y:.3f}, yaw={result.pose.yaw:.3f}")

    astar_cfg = GridAStarCfg(inflation_radius=args.inflation_radius)
    nav = SimpleNavigator(grid_cfg, astar_cfg, SimpleNavConfig())
    runtime = NavRuntime(nav, NavRuntimeConfig(enable_replanning=True, replan_every=0))
    target = Target2D(args.target_x, args.target_y)

    planner_grid = result.planner_grid
    step = runtime.step(result.pose, target, planner_grid, step_index=0)
    plan = step.plan
    if plan is None:
        print("[INFO]: no plan object")
    else:
        print(
            f"[INFO]: A* found={plan.found} reason={plan.reason} "
            f"waypoints={plan.waypoint_count} expansions={plan.astar_result.expansions}"
        )
    print(
        f"[INFO]: command[0]=({step.command.command.vx:.3f}, "
        f"{step.command.command.vy:.3f}, {step.command.command.wz:.3f}) "
        f"target_distance={step.command.target_distance:.3f}"
    )

    pose = result.pose
    for i in range(1, max(1, int(args.max_steps))):
        # This smoke test does not simulate real motion; it verifies repeated calls are finite.
        step = runtime.step(pose, target, planner_grid, step_index=i)
        cmd = step.command.command
        print(f"[INFO]: command[{i}]=({cmd.vx:.3f}, {cmd.vy:.3f}, {cmd.wz:.3f}) replan_count={step.replan_count}")

    if args.save_grid:
        inflated = inflate_occupancy_grid(
            planner_grid,
            cfg=grid_cfg,
            inflation_radius=args.inflation_radius,
        )
        # target for image must be local/body-frame; use the planner helper indirectly.
        from g1_nav_rl.navigation.simple_nav import world_to_body_xy

        target_b = world_to_body_xy(target.xy, result.pose)
        save_grid_png(inflated.to(dtype=torch.float32), args.save_grid, grid_cfg, target_b)
        print(f"[INFO]: saved inflated planning grid: {args.save_grid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
