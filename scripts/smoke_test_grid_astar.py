#!/usr/bin/env python3
"""Smoke-test occupancy-grid A* planning without Isaac Sim."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg, occupancy_grid_to_uint8_image
from g1_nav_rl.planning.grid_astar import (
    GridAStarCfg,
    PathFollowerCfg,
    follow_body_frame_path,
    grid_cell_to_xy,
    inflate_occupancy_grid,
    plan_grid_path_xy,
    shortcut_grid_path_result,
    xy_to_grid_cell,
)


def fill_rect_xy(grid: torch.Tensor, cfg: OccupancyGridCfg, *, x0: float, x1: float, y0: float, y1: float) -> None:
    """Mark a body-frame rectangle as occupied in a single [H, W] grid."""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    for row in range(cfg.height):
        x, _ = grid_cell_to_xy((row, 0), cfg)
        if not (xa <= x <= xb):
            continue
        for col in range(cfg.width):
            _, y = grid_cell_to_xy((row, col), cfg)
            if ya <= y <= yb:
                grid[row, col] = cfg.occupied_value


def make_case(case: str, cfg: OccupancyGridCfg) -> tuple[torch.Tensor, tuple[float, float], tuple[float, float], str]:
    grid = torch.zeros((cfg.height, cfg.width), dtype=torch.float32)
    start = (0.0, 0.0)
    goal = (4.5, 0.0)

    if case == "clear":
        desc = "No obstacle; A* should produce a near-straight path."
    elif case == "front":
        desc = "A rectangle blocks the direct A->B line; A* should go around it."
        fill_rect_xy(grid, cfg, x0=1.2, x1=2.4, y0=-0.45, y1=0.45)
    elif case == "gap":
        desc = "A wall with an upper/lower split and a gap; A* should pass through the gap."
        fill_rect_xy(grid, cfg, x0=1.8, x1=2.0, y0=-2.0, y1=-0.80)
        fill_rect_xy(grid, cfg, x0=1.8, x1=2.0, y0=0.80, y1=2.0)
    elif case == "blocked":
        desc = "A full wall blocks the local map; A* should report no path."
        fill_rect_xy(grid, cfg, x0=1.8, x1=2.0, y0=-3.0, y1=3.0)
    else:
        raise ValueError(f"Unknown case: {case}")
    return grid, start, goal, desc


def draw_grid_with_path(
    grid: torch.Tensor,
    cfg: OccupancyGridCfg,
    start: tuple[float, float],
    goal: tuple[float, float],
    path_cells: list[tuple[int, int]],
    shortcut_cells: list[tuple[int, int]],
    path: Path,
) -> None:
    img_tensor = occupancy_grid_to_uint8_image(grid[None], env_id=0)
    # Use a light background with black obstacles.
    arr = (255 - img_tensor.numpy()).astype("uint8")
    image = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(image)

    def cell_center_px(cell: tuple[int, int]) -> tuple[int, int]:
        row, col = cell
        return int(col), int(row)

    if len(path_cells) >= 2:
        draw.line([cell_center_px(c) for c in path_cells], fill=(30, 144, 255), width=2)
    if len(shortcut_cells) >= 2:
        draw.line([cell_center_px(c) for c in shortcut_cells], fill=(220, 20, 60), width=3)

    for xy, color in ((start, (0, 180, 0)), (goal, (180, 0, 180))):
        row, col = xy_to_grid_cell(xy, cfg)
        r = 4
        draw.ellipse((col - r, row - r, col + r, row + r), fill=color)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_case(case: str, args: argparse.Namespace) -> None:
    grid_cfg = OccupancyGridCfg(
        x_min=-1.0,
        x_max=5.0,
        y_min=-3.0,
        y_max=3.0,
        resolution=args.resolution,
    )
    planner_cfg = GridAStarCfg(
        inflation_radius=args.inflation_radius,
        heuristic_weight=args.heuristic_weight,
        allow_diagonal=not args.no_diagonal,
    )
    follower_cfg = PathFollowerCfg()

    grid, start, goal, desc = make_case(case, grid_cfg)
    result = plan_grid_path_xy(grid, start, goal, grid_cfg, planner_cfg)
    shortcut = shortcut_grid_path_result(result, grid, grid_cfg, planner_cfg)
    follower = follow_body_frame_path(shortcut.path_xy_b, follower_cfg)

    inflated = inflate_occupancy_grid(
        grid,
        cfg=grid_cfg,
        inflation_radius=planner_cfg.inflation_radius,
        occupancy_threshold=planner_cfg.occupancy_threshold,
    ).to(dtype=torch.float32)

    out_dir = Path(args.output_dir) / case
    draw_grid_with_path(
        inflated,
        grid_cfg,
        start,
        goal,
        result.path_cells,
        shortcut.path_cells,
        out_dir / "astar_path.png",
    )

    print("-" * 80)
    print(f"case               : {case}")
    print(f"description        : {desc}")
    print(f"start_xy_b         : {[round(v, 3) for v in start]}")
    print(f"goal_xy_b          : {[round(v, 3) for v in goal]}")
    print(f"grid H,W,res       : {grid_cfg.height}, {grid_cfg.width}, {grid_cfg.resolution}")
    print(f"inflation radius   : {planner_cfg.inflation_radius:.3f}")
    print(f"found              : {result.found}")
    print(f"reason             : {result.reason}")
    print(f"expansions         : {result.expansions}")
    print(f"raw path cells     : {len(result.path_cells)}")
    print(f"shortcut cells     : {len(shortcut.path_cells)}")
    print(f"first waypoints    : {[[round(a, 3), round(b, 3)] for a, b in shortcut.path_xy_b[:6]]}")
    print(f"follow command     : {[round(float(v), 3) for v in follower.command[0].tolist()]}  # [vx, vy, wz]")
    print(f"lookahead_xy_b     : {None if follower.lookahead_xy_b is None else [round(v, 3) for v in follower.lookahead_xy_b]}")
    print(f"saved path image   : {out_dir / 'astar_path.png'}")

    if args.assert_planner:
        if case == "blocked":
            if result.found:
                raise AssertionError("blocked case unexpectedly found a path")
            return
        if not result.found:
            raise AssertionError(f"{case} case did not find a path: {result.reason}")
        if len(shortcut.path_cells) < 2:
            raise AssertionError(f"{case} shortcut path too short")
        cmd = follower.command[0]
        if float(cmd[0]) <= 0.0:
            raise AssertionError(f"{case} follower did not command forward progress: {cmd.tolist()}")
        if case == "front":
            max_abs_y = max(abs(y) for _, y in shortcut.path_xy_b)
            if max_abs_y <= 0.45:
                raise AssertionError(f"front path did not route around obstacle enough: max_abs_y={max_abs_y}")
        if case == "gap":
            # The central gap is open after inflation, so the shortcut path should remain near the centerline.
            max_abs_y = max(abs(y) for _, y in shortcut.path_xy_b)
            if max_abs_y >= 0.80:
                raise AssertionError(f"gap path appears to route around the wall instead of through the gap: {shortcut.path_xy_b}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["clear", "front", "gap", "blocked", "all"], default="all")
    parser.add_argument("--output_dir", type=str, default="outputs/grid_astar")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--inflation_radius", type=float, default=0.35)
    parser.add_argument("--heuristic_weight", type=float, default=1.0)
    parser.add_argument("--no_diagonal", action="store_true")
    parser.add_argument("--assert_planner", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = ["clear", "front", "gap", "blocked"] if args.case == "all" else [args.case]
    for case in cases:
        run_case(case, args)
    print("[INFO]: Grid A* smoke test complete.")


if __name__ == "__main__":
    main()
