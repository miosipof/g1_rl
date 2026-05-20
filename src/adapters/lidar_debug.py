"""LiDAR/target-navigation debug outputs for dry-run bridge tests.

The helpers in this module are intentionally dependency-light. They do not
import Isaac, Unitree SDK, RealSense, Livox SDK, ROS, or OpenCV.  They save
artifacts that let us inspect whether real/replay LiDAR points are being turned
into the occupancy grid and plan that the dry-run policy uses.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.simple_nav import Pose2D, world_to_body_xy
from g1_nav_rl.planning.grid_astar import xy_to_grid_cell

from .g1_policy_runtime import occupancy_grid_from_lidar_frame
from .types import RobotObservation


PointXY = tuple[float, float]


def _json_safe(value: Any) -> Any:
    """Convert nested runtime objects into JSON-serializable values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _grid_to_uint8_rgb(occupancy_grid: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a 2D occupancy grid to an RGB image array."""

    if torch.is_tensor(occupancy_grid):
        grid_np = occupancy_grid.detach().cpu().numpy()
    else:
        grid_np = np.asarray(occupancy_grid)
    if grid_np.ndim != 2:
        raise ValueError(f"occupancy_grid must be 2D, got shape={grid_np.shape}")
    grid_np = np.nan_to_num(grid_np.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    occupied = grid_np > 0.0
    image = np.full((grid_np.shape[0], grid_np.shape[1], 3), 245, dtype=np.uint8)
    image[occupied] = np.array([25, 25, 25], dtype=np.uint8)
    return image


def _draw_cell(image: np.ndarray, row: int, col: int, color: tuple[int, int, int], radius: int = 1) -> None:
    h, w = image.shape[:2]
    for rr in range(max(0, row - radius), min(h, row + radius + 1)):
        for cc in range(max(0, col - radius), min(w, col + radius + 1)):
            image[rr, cc] = np.array(color, dtype=np.uint8)


def _draw_polyline_cells(
    image: np.ndarray,
    points_xy_b: Sequence[PointXY],
    cfg: OccupancyGridCfg,
    *,
    color: tuple[int, int, int],
) -> None:
    if not points_xy_b:
        return
    last: tuple[int, int] | None = None
    for xy in points_xy_b:
        row, col = xy_to_grid_cell(xy, cfg)
        if not (0 <= row < cfg.height and 0 <= col < cfg.width):
            last = None
            continue
        _draw_cell(image, row, col, color, radius=1)
        if last is not None:
            # Sparse Bresenham-like interpolation in image space.
            r0, c0 = last
            steps = max(abs(row - r0), abs(col - c0), 1)
            for i in range(steps + 1):
                t = float(i) / float(steps)
                rr = int(round(r0 + t * (row - r0)))
                cc = int(round(c0 + t * (col - c0)))
                if 0 <= rr < cfg.height and 0 <= cc < cfg.width:
                    _draw_cell(image, rr, cc, color, radius=0)
        last = (row, col)


def save_occupancy_debug_image(
    path: str | Path,
    occupancy_grid: torch.Tensor | np.ndarray,
    *,
    grid_cfg: OccupancyGridCfg,
    plan_xy_b: Sequence[PointXY] | None = None,
    target_xy_b: PointXY | None = None,
) -> None:
    """Save a PNG visualizing occupancy plus optional plan/target overlays.

    Color convention:
      - light background: free/unknown cells;
      - dark cells: occupied cells;
      - blue polyline: current planned path in body frame;
      - green square: target in body frame;
      - orange square: robot origin.
    """

    image = _grid_to_uint8_rgb(occupancy_grid)
    if plan_xy_b:
        _draw_polyline_cells(image, plan_xy_b, grid_cfg, color=(45, 95, 225))
    if target_xy_b is not None:
        row, col = xy_to_grid_cell(target_xy_b, grid_cfg)
        if 0 <= row < grid_cfg.height and 0 <= col < grid_cfg.width:
            _draw_cell(image, row, col, (30, 170, 80), radius=3)
    origin = xy_to_grid_cell((0.0, 0.0), grid_cfg)
    if 0 <= origin[0] < grid_cfg.height and 0 <= origin[1] < grid_cfg.width:
        _draw_cell(image, origin[0], origin[1], (245, 160, 30), radius=2)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(image, mode="RGB").save(path)
    except Exception as exc:  # pragma: no cover - exercised only without Pillow/PNG support
        raise RuntimeError("Pillow is required to save occupancy debug PNG images") from exc


class LidarDebugWriter:
    """Write per-step LiDAR/target-navigation artifacts.

    The writer emits both step-indexed files and convenient latest-file aliases:

    - ``lidar_points_body_step_000000.npy`` and ``lidar_points_body.npy``;
    - ``occupancy_grid_step_000000.png`` and ``occupancy_grid.png``;
    - ``occupancy_grid_meta_step_000000.json`` and ``occupancy_grid_meta.json``;
    - ``target_nav_plan_body_step_000000.npy`` and ``target_nav_plan.npy``;
    - ``target_nav_command_log.csv``.
    """

    def __init__(self, output_dir: str | Path, *, grid_cfg: OccupancyGridCfg) -> None:
        self.output_dir = Path(output_dir)
        self.grid_cfg = grid_cfg
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.command_log_path = self.output_dir / "target_nav_command_log.csv"
        self._command_log_initialized = False

    def write_step(self, step: int, bridge_result: Any, *, policy: Any | None = None) -> dict[str, str]:
        obs: RobotObservation | None = bridge_result.observation
        if obs is None or obs.lidar is None:
            return {}

        paths: dict[str, str] = {}
        step_tag = f"{int(step):06d}"
        lidar_points = obs.lidar.points_body_xyz.astype(np.float32, copy=False)

        lidar_step_path = self.output_dir / f"lidar_points_body_step_{step_tag}.npy"
        np.save(lidar_step_path, lidar_points)
        np.save(self.output_dir / "lidar_points_body.npy", lidar_points)
        paths["lidar_points_body_step"] = str(lidar_step_path)

        occupancy_grid = occupancy_grid_from_lidar_frame(obs.lidar, grid_cfg=self.grid_cfg)
        plan_xy_b, plan_xy_w = self._plan_paths(policy)
        target_xy_b = self._target_body_xy(obs, policy)

        image_step_path = self.output_dir / f"occupancy_grid_step_{step_tag}.png"
        save_occupancy_debug_image(
            image_step_path,
            occupancy_grid,
            grid_cfg=self.grid_cfg,
            plan_xy_b=plan_xy_b,
            target_xy_b=target_xy_b,
        )
        save_occupancy_debug_image(
            self.output_dir / "occupancy_grid.png",
            occupancy_grid,
            grid_cfg=self.grid_cfg,
            plan_xy_b=plan_xy_b,
            target_xy_b=target_xy_b,
        )
        paths["occupancy_grid_step"] = str(image_step_path)

        if plan_xy_b:
            plan_body = np.asarray(plan_xy_b, dtype=np.float32)
            plan_body_step = self.output_dir / f"target_nav_plan_body_step_{step_tag}.npy"
            np.save(plan_body_step, plan_body)
            np.save(self.output_dir / "target_nav_plan.npy", plan_body)
            paths["target_nav_plan_body_step"] = str(plan_body_step)
        if plan_xy_w:
            plan_world = np.asarray(plan_xy_w, dtype=np.float32)
            plan_world_step = self.output_dir / f"target_nav_plan_world_step_{step_tag}.npy"
            np.save(plan_world_step, plan_world)
            paths["target_nav_plan_world_step"] = str(plan_world_step)

        meta = self._metadata(step, bridge_result, occupancy_grid, policy=policy, output_paths=paths)
        meta_step_path = self.output_dir / f"occupancy_grid_meta_step_{step_tag}.json"
        self._write_json(meta_step_path, meta)
        self._write_json(self.output_dir / "occupancy_grid_meta.json", meta)
        paths["occupancy_grid_meta_step"] = str(meta_step_path)

        self._append_command_log(step, bridge_result, occupancy_grid, policy=policy)
        paths["target_nav_command_log"] = str(self.command_log_path)
        return paths

    def _plan_paths(self, policy: Any | None) -> tuple[list[PointXY], list[PointXY]]:
        if policy is None or not hasattr(policy, "runtime"):
            return [], []
        plan = getattr(policy.runtime, "plan_result", None)
        if plan is None:
            return [], []
        return list(getattr(plan, "path_xy_b", []) or []), list(getattr(plan, "path_xy_w", []) or [])

    def _target_body_xy(self, obs: RobotObservation, policy: Any | None) -> PointXY | None:
        target = obs.target_hint
        if target is None and policy is not None and hasattr(policy, "config"):
            target = getattr(policy.config, "target", None)
        pose = obs.pose or (obs.robot_state.pose if obs.robot_state is not None else None)
        if target is None or pose is None:
            return None
        if not isinstance(pose, Pose2D):
            return None
        return world_to_body_xy(target.xy, pose)

    def _metadata(
        self,
        step: int,
        bridge_result: Any,
        occupancy_grid: torch.Tensor,
        *,
        policy: Any | None,
        output_paths: Mapping[str, str],
    ) -> dict[str, Any]:
        obs: RobotObservation | None = bridge_result.observation
        lidar = None if obs is None else obs.lidar
        decision = bridge_result.safety_decision
        proposed = bridge_result.proposed_command
        safe = None if decision is None else decision.command
        policy_debug = None
        if policy is not None and hasattr(policy, "last_debug"):
            debug = getattr(policy, "last_debug")
            if debug is not None and hasattr(debug, "to_dict"):
                policy_debug = debug.to_dict()

        return _json_safe(
            {
                "step": int(step),
                "timestamp_s": None if obs is None else obs.timestamp_s,
                "observation_source": None if obs is None else obs.metadata.get("source"),
                "lidar": None
                if lidar is None
                else {
                    "timestamp_s": lidar.timestamp_s,
                    "point_count": lidar.point_count,
                    "frame_id": lidar.frame_id,
                    "source_frame_id": lidar.source_frame_id,
                    "metadata": dict(lidar.metadata),
                },
                "grid": {
                    "height": int(occupancy_grid.shape[0]),
                    "width": int(occupancy_grid.shape[1]),
                    "x_min": self.grid_cfg.x_min,
                    "x_max": self.grid_cfg.x_max,
                    "y_min": self.grid_cfg.y_min,
                    "y_max": self.grid_cfg.y_max,
                    "resolution": self.grid_cfg.resolution,
                    "occupied_cells": int((occupancy_grid > 0.0).sum().item()),
                },
                "policy_debug": policy_debug,
                "proposed_command": None if proposed is None else proposed.tuple,
                "safe_command": None if safe is None else safe.tuple,
                "safety_reasons": [] if decision is None else list(decision.reasons),
                "safety_warnings": [] if decision is None else list(decision.warnings),
                "output_paths": dict(output_paths),
            }
        )

    def _append_command_log(self, step: int, bridge_result: Any, occupancy_grid: torch.Tensor, *, policy: Any | None) -> None:
        obs = bridge_result.observation
        decision = bridge_result.safety_decision
        proposed = bridge_result.proposed_command
        safe = None if decision is None else decision.command
        policy_debug = getattr(policy, "last_debug", None) if policy is not None else None
        row = {
            "step": int(step),
            "timestamp_s": "" if obs is None else f"{float(obs.timestamp_s):.6f}",
            "status_health": bridge_result.status.health.value,
            "mode": bridge_result.mode.value,
            "action": bridge_result.action,
            "proposed_vx": "" if proposed is None else f"{float(proposed.vx):.6f}",
            "proposed_vy": "" if proposed is None else f"{float(proposed.vy):.6f}",
            "proposed_wz": "" if proposed is None else f"{float(proposed.wz):.6f}",
            "safe_vx": "" if safe is None else f"{float(safe.vx):.6f}",
            "safe_vy": "" if safe is None else f"{float(safe.vy):.6f}",
            "safe_wz": "" if safe is None else f"{float(safe.wz):.6f}",
            "lidar_points": 0 if obs is None or obs.lidar is None else obs.lidar.point_count,
            "occupied_cells": int((occupancy_grid > 0.0).sum().item()),
            "plan_found": "" if policy_debug is None else bool(policy_debug.plan_found),
            "waypoint_index": "" if policy_debug is None else int(policy_debug.waypoint_index),
            "waypoint_count": "" if policy_debug is None else int(policy_debug.waypoint_count),
            "target_distance": "" if policy_debug is None else f"{float(policy_debug.target_distance):.6f}",
            "safety_reasons": "" if decision is None else ";".join(decision.reasons),
            "safety_warnings": "" if decision is None else ";".join(decision.warnings),
        }
        write_header = not self._command_log_initialized or not self.command_log_path.exists()
        self.command_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.command_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._command_log_initialized = True

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "LidarDebugWriter",
    "save_occupancy_debug_image",
]
