#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg, occupancy_grid_to_uint8_image
from g1_nav_rl.planning.local_velocity_rollout import LocalVelocityRolloutCfg, plan_local_velocity_command


def _mark_rect(grid: torch.Tensor, grid_cfg: OccupancyGridCfg, *, x0: float, x1: float, y0: float, y1: float) -> None:
    """Mark an axis-aligned obstacle rectangle in body-frame coordinates."""
    if grid.ndim != 3 or grid.shape[0] != 1:
        raise ValueError("This smoke helper expects grid shape [1, H, W].")
    xs = torch.linspace(grid_cfg.x_max - 0.5 * grid_cfg.resolution, grid_cfg.x_min + 0.5 * grid_cfg.resolution, grid_cfg.height)
    ys = torch.linspace(grid_cfg.y_min + 0.5 * grid_cfg.resolution, grid_cfg.y_max - 0.5 * grid_cfg.resolution, grid_cfg.width)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    grid[0, mask] = grid_cfg.occupied_value


def build_case(case: str, grid_cfg: OccupancyGridCfg) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return occupancy grid, target_xy_b and description for a synthetic case."""
    grid = torch.zeros((1, grid_cfg.height, grid_cfg.width), dtype=torch.float32)
    target_xy_b = torch.tensor([[3.0, 0.0]], dtype=torch.float32)

    if case == "clear":
        desc = "No obstacle; target straight ahead. Planner should move forward."
    elif case == "front":
        _mark_rect(grid, grid_cfg, x0=0.45, x1=0.95, y0=-0.25, y1=0.25)
        desc = "Obstacle in front; planner should avoid driving straight into it."
    elif case == "left":
        _mark_rect(grid, grid_cfg, x0=0.45, x1=1.10, y0=0.15, y1=0.70)
        desc = "Obstacle front-left; planner should prefer neutral/right lateral motion."
    elif case == "right":
        _mark_rect(grid, grid_cfg, x0=0.45, x1=1.10, y0=-0.70, y1=-0.15)
        desc = "Obstacle front-right; planner should prefer neutral/left lateral motion."
    elif case == "blocked":
        _mark_rect(grid, grid_cfg, x0=0.25, x1=1.25, y0=-1.00, y1=1.00)
        desc = "Wide obstacle in front; planner should slow or stop."
    else:
        raise ValueError(f"Unknown case: {case}")

    return grid, target_xy_b, desc


def save_grid_png(grid: torch.Tensor, path: Path) -> None:
    """Save occupancy grid as a grayscale PNG when PIL is available."""
    try:
        from PIL import Image
    except Exception:
        print("[WARN]: PIL not available; skipping grid PNG save")
        return
    image = occupancy_grid_to_uint8_image(grid, env_id=0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path)


def assert_case(case: str, command: torch.Tensor, info: dict[str, torch.Tensor]) -> None:
    vx, vy, wz = [float(x) for x in command[0].detach().cpu()]
    best_clearance = float(info["best_clearance"][0].detach().cpu())
    any_free = bool(info["any_collision_free"][0].detach().cpu().item())

    if case == "clear":
        if vx <= 0.0:
            raise AssertionError(f"clear case expected positive vx, got command={command.tolist()}")
    elif case == "front":
        if vx > 0.15 and abs(vy) < 0.05 and abs(wz) < 0.05:
            raise AssertionError(f"front obstacle expected non-straight or slow command, got {command.tolist()}")
    elif case == "left":
        if vy > 0.05:
            raise AssertionError(f"left obstacle expected non-left lateral command, got {command.tolist()}")
    elif case == "right":
        if vy < -0.05:
            raise AssertionError(f"right obstacle expected non-right lateral command, got {command.tolist()}")
    elif case == "blocked":
        if any_free and vx > 0.2 and best_clearance < 0.5:
            raise AssertionError(f"blocked case expected cautious command, got {command.tolist()}, clearance={best_clearance}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-Python local velocity rollout planner smoke test.")
    parser.add_argument("--case", choices=("clear", "front", "left", "right", "blocked", "all"), default="all")
    parser.add_argument("--assert_planner", action="store_true", help="Enable simple case assertions.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/local_planner"))
    args = parser.parse_args()

    grid_cfg = OccupancyGridCfg(x_min=-1.5, x_max=4.0, y_min=-2.5, y_max=2.5, resolution=0.05)
    planner_cfg = LocalVelocityRolloutCfg(
        horizon_s=2.5,
        dt=0.1,
        robot_radius=0.32,
        vx_samples=(0.0, 0.1, 0.2, 0.3),
        vy_samples=(-0.25, -0.15, 0.0, 0.15, 0.25),
        wz_samples=(-0.6, -0.3, 0.0, 0.3, 0.6),
        progress_weight=6.0,
        clearance_weight=1.5,
        command_smoothness_weight=0.1,
    )
    cases = ["clear", "front", "left", "right", "blocked"] if args.case == "all" else [args.case]

    last_command = torch.zeros((1, 3), dtype=torch.float32)
    for case in cases:
        grid, target_xy_b, desc = build_case(case, grid_cfg)
        command, info = plan_local_velocity_command(
            occupancy_grid=grid,
            target_xy_b=target_xy_b,
            last_command=last_command,
            grid_cfg=grid_cfg,
            cfg=planner_cfg,
        )

        save_grid_png(grid, args.output_dir / case / "occupancy_grid.png")
        selected_traj = info["selected_trajectory_xy_b"][0].detach().cpu()
        final_xy = selected_traj[-1].tolist()
        occupied_cells = int((grid[0] > 0.5).sum().item())

        print("-" * 80)
        print(f"case               : {case}")
        print(f"description        : {desc}")
        print(f"target_xy_b        : {target_xy_b[0].tolist()}")
        print(f"occupied cells     : {occupied_cells}")
        print(f"selected command   : {command[0].tolist()}  # [vx, vy, wz]")
        print(f"best score         : {float(info['best_score'][0].item()):.3f}")
        print(f"best progress      : {float(info['best_progress'][0].item()):.3f}")
        print(f"best clearance     : {float(info['best_clearance'][0].item()):.3f}")
        print(f"any collision-free : {bool(info['any_collision_free'][0].item())}")
        print(f"selected final xy  : {[round(float(x), 3) for x in final_xy]}")
        print(f"saved grid         : {args.output_dir / case / 'occupancy_grid.png'}")

        if args.assert_planner:
            assert_case(case, command, info)

        last_command = command.detach()

    print("[INFO]: Local planner smoke test complete.")


if __name__ == "__main__":
    main()
