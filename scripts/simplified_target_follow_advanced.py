#!/usr/bin/env python3
"""Minimal real G1 target-follow script.

This script is intentionally NOT a wrapper around the stepwise runtime.  It is a
clean hardware test for the simplest contract:

  first/latest valid target from a YOLO stream
  + current Livox occupancy grid
  -> A* path in the robot body frame
  -> one yaw-first / forward command burst
  -> stop, observe again after 2--3 seconds

Deliberately omitted in this script:
  - PrimaryTargetLock / track-ID persistence
  - target-change replanning during a burst
  - path_block inflated-grid execution guard
  - command odometry
  - continuous SafetySupervisor
  - lateral velocity commands
  - start relocation by default; goal-only relocation is allowed for occupied target stand-off cells
  - stop after every primitive

The only runtime guards kept are raw LiDAR obstacle checks.  Target loss is not
a replan trigger.  If the target is temporarily missing, the script either
continues the current queue or replans only when current LiDAR says that the
active path is blocked.
Real motion still requires --yes-i-understand-this-moves-the-robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from adapters import (
    Command2D,
    DetectionStreamClient,
    Target2D,
    UnitreeLocoAdapter,
    UnitreeLocoAdapterConfig,
    load_adapter_config,
    make_run_id,
    timestamped_output_dir,
    write_run_info,
)
from adapters.config import CommandLimitsConfig
from adapters.lidar_debug import save_occupancy_debug_image
from adapters.real_livox import LivoxAdapterConfig, UnitreeVibesLivox2Adapter, UnitreeVibesLivox2AdapterConfig
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg, rasterize_points_to_occupancy_grid
from g1_nav_rl.planning.grid_astar import GridAStarCfg, GridAStarResult, grid_cell_to_xy, plan_grid_path_xy, xy_to_grid_cell, inflate_occupancy_grid


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _default_config_path() -> Path:
    return ROOT / "configs" / "real_g1.yaml"


def _make_grid_cfg(args: argparse.Namespace) -> OccupancyGridCfg:
    return OccupancyGridCfg(
        x_min=float(args.grid_x_min),
        x_max=float(args.grid_x_max),
        y_min=float(args.grid_y_min),
        y_max=float(args.grid_y_max),
        resolution=float(args.grid_resolution),
    )


def _read_best_stream_target(
    *,
    stream_client: DetectionStreamClient,
    target_class: str,
    min_confidence: float,
    min_depth_valid_ratio: float,
    require_fresh: bool,
) -> tuple[Target2D | None, dict[str, Any]]:
    """Return the highest-confidence current target without target-lock state."""

    result = stream_client.read()
    meta: dict[str, Any] = {
        "stream_status": result.status.health.value,
        "stream_message": result.status.message,
        "stream_details": dict(result.status.details),
        "candidate_count": 0,
        "selected_track": None,
    }
    if result.frame is None:
        return None, meta
    if require_fresh and result.status.health.value != "ok":
        return None, meta

    candidates = []
    for track in result.frame.tracks:
        if track.class_name != str(target_class):
            continue
        if float(track.confidence) < float(min_confidence):
            continue
        if track.target_body_xy is None:
            continue
        if track.depth_valid_ratio is not None and float(track.depth_valid_ratio) < float(min_depth_valid_ratio):
            continue
        x, y = track.target_body_xy
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            continue
        candidates.append(track)

    meta["candidate_count"] = len(candidates)
    if not candidates:
        return None, meta

    best = max(candidates, key=lambda t: float(t.confidence))
    x, y = best.target_body_xy  # type: ignore[misc]

    # Target2D in this repo baseline is intentionally minimal: only x/y.
    # Keep confidence/source/track details in stream metadata, not in the
    # navigation target object.
    target = Target2D(float(x), float(y))
    meta["selected_track"] = best.to_dict()
    meta["selected_target_xy_b"] = [float(x), float(y)]
    meta["selected_confidence"] = float(best.confidence)
    meta["selected_source"] = "stream_best_detection"
    return target, meta


def _wait_for_initial_target(
    *,
    stream_client: DetectionStreamClient,
    target_class: str,
    min_confidence: float,
    min_depth_valid_ratio: float,
    timeout_s: float,
    poll_s: float,
    json_lines: bool,
) -> tuple[Target2D, dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_s)
    last_meta: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        target, meta = _read_best_stream_target(
            stream_client=stream_client,
            target_class=target_class,
            min_confidence=min_confidence,
            min_depth_valid_ratio=min_depth_valid_ratio,
            require_fresh=True,
        )
        last_meta = meta
        if target is not None:
            if json_lines:
                print(json.dumps({"event": "initial_target_acquired", "target_xy_b": [target.x, target.y], "stream": meta}, sort_keys=True))
            else:
                print(f"[INFO]: initial target acquired: xy_b=({target.x:.3f}, {target.y:.3f})")
            return target, meta
        if json_lines:
            print(json.dumps({"event": "waiting_for_initial_target", "stream": meta}, sort_keys=True))
        else:
            print(f"[INFO]: waiting for first target: {meta.get('stream_message')} candidates={meta.get('candidate_count')}")
        time.sleep(float(poll_s))
    raise RuntimeError(f"timed out waiting for initial {target_class!r} target; last_stream={last_meta}")


def _occupancy_from_lidar(points_body_xyz: np.ndarray, grid_cfg: OccupancyGridCfg) -> torch.Tensor:
    points = np.asarray(points_body_xyz, dtype=np.float32)
    if points.size == 0:
        pts = torch.zeros((1, 0, 2), dtype=torch.float32)
    else:
        pts = torch.as_tensor(points[:, :2], dtype=torch.float32).unsqueeze(0)
    return rasterize_points_to_occupancy_grid(pts, valid=None, cfg=grid_cfg)[0].detach().cpu()


def _plan_with_goal_relocation(
    *,
    occupancy: torch.Tensor,
    target: Target2D,
    grid_cfg: OccupancyGridCfg,
    astar_cfg: GridAStarCfg,
    allow_goal_relocation: bool,
    allow_start_relocation: bool,
    max_goal_relocation_m: float,
) -> tuple[GridAStarResult, dict[str, Any]]:
    """Plan from the real body origin, relocating only the goal if needed.

    For object following, the exact RGB-D stand-off point can still fall inside
    inflated LiDAR occupancy: the target object itself, its support surface, or
    an inflated obstacle blob can cover that cell.  Treat that as a goal-side
    issue only.  The start remains strict by default so we never silently plan
    from a fictitious robot pose.
    """

    requested_goal_xy = (float(target.x), float(target.y))
    requested_goal_cell = xy_to_grid_cell(requested_goal_xy, grid_cfg)
    meta: dict[str, Any] = {
        "requested_goal_xy_b": [requested_goal_xy[0], requested_goal_xy[1]],
        "requested_goal_cell": list(requested_goal_cell),
        "goal_relocated": False,
        "goal_relocation_distance_m": 0.0,
        "goal_relocation_max_m": float(max_goal_relocation_m),
    }

    strict = plan_grid_path_xy(
        occupancy,
        start_xy_b=(0.0, 0.0),
        goal_xy_b=requested_goal_xy,
        grid_cfg=grid_cfg,
        planner_cfg=astar_cfg,
        allow_start_relocation=bool(allow_start_relocation),
        allow_goal_relocation=False,
    )
    if strict.found or strict.reason != "goal_occupied" or not bool(allow_goal_relocation):
        return strict, meta

    relocated = plan_grid_path_xy(
        occupancy,
        start_xy_b=(0.0, 0.0),
        goal_xy_b=requested_goal_xy,
        grid_cfg=grid_cfg,
        planner_cfg=astar_cfg,
        allow_start_relocation=bool(allow_start_relocation),
        allow_goal_relocation=True,
    )
    if not relocated.found:
        meta["goal_relocation_attempted"] = True
        meta["goal_relocation_failed_reason"] = relocated.reason
        return strict, meta

    relocated_goal_xy = grid_cell_to_xy(relocated.goal_cell, grid_cfg)
    relocation_distance = math.hypot(relocated_goal_xy[0] - requested_goal_xy[0], relocated_goal_xy[1] - requested_goal_xy[1])
    meta.update({
        "goal_relocation_attempted": True,
        "goal_relocated": True,
        "relocated_goal_cell": list(relocated.goal_cell),
        "relocated_goal_xy_b": [float(relocated_goal_xy[0]), float(relocated_goal_xy[1])],
        "goal_relocation_distance_m": float(relocation_distance),
    })

    if relocation_distance > float(max_goal_relocation_m):
        meta["goal_relocation_rejected_reason"] = "relocated_goal_too_far"
        meta["relocated_plan_reason"] = relocated.reason
        return strict, meta

    return GridAStarResult(
        found=relocated.found,
        start_cell=relocated.start_cell,
        goal_cell=relocated.goal_cell,
        path_cells=relocated.path_cells,
        path_xy_b=relocated.path_xy_b,
        cost=relocated.cost,
        expansions=relocated.expansions,
        reason="goal_relocated_from_occupied+" + str(relocated.reason),
    ), meta


def _path_length(path_xy_b: Sequence[tuple[float, float]]) -> float:
    if len(path_xy_b) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path_xy_b[:-1], path_xy_b[1:]):
        total += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    return total


def _sample_path(path_xy_b: Sequence[tuple[float, float]], distance_m: float) -> tuple[float, float] | None:
    if not path_xy_b:
        return None
    if len(path_xy_b) == 1 or distance_m <= 0.0:
        return path_xy_b[0]
    remaining = float(distance_m)
    prev = path_xy_b[0]
    for curr in path_xy_b[1:]:
        seg = math.hypot(float(curr[0]) - float(prev[0]), float(curr[1]) - float(prev[1]))
        if seg <= 1e-9:
            prev = curr
            continue
        if remaining <= seg:
            t = remaining / seg
            return (float(prev[0]) + t * (float(curr[0]) - float(prev[0])), float(prev[1]) + t * (float(curr[1]) - float(prev[1])))
        remaining -= seg
        prev = curr
    return path_xy_b[-1]


def _raw_front_min(points_body_xyz: np.ndarray, *, corridor_half_width_m: float, x_min_m: float, x_max_m: float) -> float | None:
    points = np.asarray(points_body_xyz, dtype=np.float32)
    if points.size == 0:
        return None
    x = points[:, 0]
    y = points[:, 1]
    keep = (x >= float(x_min_m)) & (x <= float(x_max_m)) & (np.abs(y) <= float(corridor_half_width_m))
    if not np.any(keep):
        return None
    return float(np.min(x[keep]))


def _preview_command_burst(queue: Sequence[Command2D], max_duration_s: float) -> list[Command2D]:
    """Return the next checkpoint-sized burst without mutating ``queue``."""

    remaining = float(max_duration_s)
    burst: list[Command2D] = []
    for cmd in queue:
        if remaining <= 1e-9:
            break
        duration = float(cmd.duration_s or 0.0)
        if duration <= 1e-9:
            continue
        used = min(duration, remaining)
        burst.append(_copy_command_with_duration(cmd, used))
        remaining -= used
        if used < duration:
            break
    return burst


def _validate_burst_against_occupancy(
    *,
    occupancy: torch.Tensor,
    grid_cfg: OccupancyGridCfg,
    commands: Sequence[Command2D],
    guard_inflation_m: float,
    occupancy_threshold: float,
    start_ignore_m: float,
    sample_step_m: float,
) -> dict[str, Any]:
    """Validate the next open-loop command burst against the current LiDAR grid.

    This is intentionally narrower than the old path_block guard: it does not
    trigger from target loss, it does not replan by itself, and it only checks
    the commands that are already about to be executed.  A blocked result means
    the *current obstacle map* invalidated the active queue.
    """

    meta: dict[str, Any] = {
        "checked": True,
        "clear": True,
        "reason": "clear",
        "command_count": len(commands),
        "guard_inflation_m": float(guard_inflation_m),
        "start_ignore_m": float(start_ignore_m),
        "sample_step_m": float(sample_step_m),
    }
    if not commands:
        meta["reason"] = "empty_burst"
        return meta

    inflated = inflate_occupancy_grid(
        occupancy,
        cfg=grid_cfg,
        inflation_radius=max(0.0, float(guard_inflation_m)),
        occupancy_threshold=float(occupancy_threshold),
    )
    height, width = int(inflated.shape[0]), int(inflated.shape[1])

    x = 0.0
    y = 0.0
    theta = 0.0
    total_forward_m = 0.0
    checked_samples = 0

    for command_index, cmd in enumerate(commands):
        duration = max(0.0, float(cmd.duration_s or 0.0))
        vx = float(cmd.vx)
        wz = float(cmd.wz)
        if duration <= 1e-9:
            continue
        if abs(vx) > 1e-9:
            distance = vx * duration
            # This simplified runner should only use forward commands; still
            # handle negative distances conservatively by sampling the segment.
            steps = max(1, int(math.ceil(abs(distance) / max(1e-3, float(sample_step_m)))))
            x0, y0 = x, y
            for i in range(1, steps + 1):
                frac = i / steps
                s = distance * frac
                px = x0 + s * math.cos(theta)
                py = y0 + s * math.sin(theta)
                if abs(s) <= float(start_ignore_m) and total_forward_m <= 1e-9:
                    continue
                cell = xy_to_grid_cell((px, py), grid_cfg)
                row, col = cell
                checked_samples += 1
                if row < 0 or row >= height or col < 0 or col >= width:
                    meta.update({
                        "clear": False,
                        "reason": "sample_out_of_grid",
                        "blocked_command_index": command_index,
                        "blocked_xy_b": [float(px), float(py)],
                        "blocked_cell": [int(row), int(col)],
                        "checked_samples": checked_samples,
                    })
                    return meta
                if bool(inflated[row, col].item()):
                    meta.update({
                        "clear": False,
                        "reason": "inflated_occupancy_on_queued_burst",
                        "blocked_command_index": command_index,
                        "blocked_xy_b": [float(px), float(py)],
                        "blocked_cell": [int(row), int(col)],
                        "checked_samples": checked_samples,
                    })
                    return meta
            x = x0 + distance * math.cos(theta)
            y = y0 + distance * math.sin(theta)
            total_forward_m += abs(distance)
        if abs(wz) > 1e-9:
            theta = _angle_normalize(theta + wz * duration)

    meta.update({
        "checked_samples": checked_samples,
        "preview_end_xyyaw_b": [float(x), float(y), float(theta)],
        "preview_forward_m": float(total_forward_m),
    })
    return meta


def _integrate_open_loop_pose(
    pose_xyyaw: tuple[float, float, float],
    commands: Sequence[Command2D],
) -> tuple[float, float, float]:
    """Integrate the relative command burst in the current local plan frame.

    This is deliberately not used as global odometry.  It only estimates where
    the original planned goal lies in the robot's current body frame when the
    target is temporarily occluded.  Target loss itself does not cause a replan;
    this estimate is consumed only by obstacle-triggered replanning.
    """

    x, y, yaw = (float(pose_xyyaw[0]), float(pose_xyyaw[1]), float(pose_xyyaw[2]))
    for cmd in commands:
        dur = max(0.0, float(cmd.duration_s or 0.0))
        if dur <= 1e-9:
            continue
        vx = float(cmd.vx)
        vy = float(cmd.vy)
        wz = float(cmd.wz)

        # All commands generated by this script are either pure yaw or pure
        # forward motion.  Keep the generic projection for robustness.
        if abs(vx) > 1e-9 or abs(vy) > 1e-9:
            dx_body = vx * dur
            dy_body = vy * dur
            c = math.cos(yaw)
            s = math.sin(yaw)
            x += c * dx_body - s * dy_body
            y += s * dx_body + c * dy_body
        if abs(wz) > 1e-9:
            yaw = _angle_normalize(yaw + wz * dur)
    return (float(x), float(y), float(yaw))


def _goal_in_current_body(
    goal_xy_at_plan_b: tuple[float, float] | None,
    pose_since_plan_xyyaw: tuple[float, float, float],
) -> Target2D | None:
    """Transform the active plan goal into the current estimated body frame."""

    if goal_xy_at_plan_b is None:
        return None
    gx, gy = float(goal_xy_at_plan_b[0]), float(goal_xy_at_plan_b[1])
    x, y, yaw = (float(pose_since_plan_xyyaw[0]), float(pose_since_plan_xyyaw[1]), float(pose_since_plan_xyyaw[2]))
    dx = gx - x
    dy = gy - y
    c = math.cos(yaw)
    s = math.sin(yaw)
    # R(-yaw) @ ([gx, gy] - [x, y])
    return Target2D(c * dx + s * dy, -s * dx + c * dy)


def _make_burst_commands(
    *,
    plan: GridAStarResult,
    target: Target2D,
    forward_speed_mps: float,
    checkpoint_period_s: float,
    lookahead_m: float,
    min_turn_angle_rad: float,
    turn_speed_radps: float,
    turn_angle_max_rad: float,
    target_reached_radius_m: float,
) -> tuple[list[Command2D], dict[str, Any]]:
    dist_to_target = math.hypot(float(target.x), float(target.y))
    if dist_to_target <= float(target_reached_radius_m):
        return [], {"reason": "target_reached", "target_distance_m": dist_to_target}
    if not plan.found or len(plan.path_xy_b) < 2:
        return [], {"reason": "no_plan", "plan_reason": plan.reason}

    sample = _sample_path(plan.path_xy_b, float(lookahead_m))
    if sample is None:
        return [], {"reason": "empty_path"}
    sx, sy = float(sample[0]), float(sample[1])
    heading = math.atan2(sy, max(1e-6, sx))
    commands: list[Command2D] = []
    remaining = float(checkpoint_period_s)

    turn_angle = 0.0
    if abs(heading) >= float(min_turn_angle_rad):
        turn_angle = max(-float(turn_angle_max_rad), min(float(turn_angle_max_rad), heading))
        turn_duration = abs(turn_angle) / max(1e-6, float(turn_speed_radps))
        turn_duration = min(turn_duration, remaining)
        wz = math.copysign(float(turn_speed_radps), turn_angle)
        commands.append(Command2D(0.0, 0.0, wz, duration_s=max(0.05, turn_duration), source="simplified_target_follow_turn", metadata={"turn_angle_rad": turn_angle, "lookahead_xy_b": [sx, sy]}))
        remaining -= turn_duration

    forward_duration = max(0.0, remaining)
    # Commit at least one useful forward interval if the path is roughly ahead.
    if forward_duration >= 0.20:
        commands.append(Command2D(float(forward_speed_mps), 0.0, 0.0, duration_s=forward_duration, source="simplified_target_follow_forward", metadata={"lookahead_xy_b": [sx, sy]}))

    return commands, {
        "reason": "commands_ready" if commands else "no_time_for_forward",
        "target_distance_m": dist_to_target,
        "lookahead_xy_b": [sx, sy],
        "lookahead_heading_rad": heading,
        "turn_angle_rad": turn_angle,
        "checkpoint_period_s": checkpoint_period_s,
        "path_length_m": _path_length(plan.path_xy_b),
        "command_count": len(commands),
    }




def _angle_normalize(rad: float) -> float:
    return math.atan2(math.sin(float(rad)), math.cos(float(rad)))


def _cumulative_path_lengths(path_xy_b: Sequence[Sequence[float]]) -> list[float]:
    lengths = [0.0]
    for prev, curr in zip(path_xy_b[:-1], path_xy_b[1:]):
        lengths.append(lengths[-1] + math.hypot(float(curr[0]) - float(prev[0]), float(curr[1]) - float(prev[1])))
    return lengths


def _sample_path_at_distance(path_xy_b: Sequence[Sequence[float]], distance_m: float) -> tuple[float, float] | None:
    if not path_xy_b:
        return None
    if len(path_xy_b) == 1 or float(distance_m) <= 0.0:
        return (float(path_xy_b[0][0]), float(path_xy_b[0][1]))
    remaining = float(distance_m)
    prev = path_xy_b[0]
    for curr in path_xy_b[1:]:
        seg = math.hypot(float(curr[0]) - float(prev[0]), float(curr[1]) - float(prev[1]))
        if seg <= 1e-9:
            prev = curr
            continue
        if remaining <= seg:
            t = remaining / seg
            return (
                float(prev[0]) + t * (float(curr[0]) - float(prev[0])),
                float(prev[1]) + t * (float(curr[1]) - float(prev[1])),
            )
        remaining -= seg
        prev = curr
    return (float(path_xy_b[-1][0]), float(path_xy_b[-1][1]))


def _make_open_loop_command_queue(
    *,
    plan: GridAStarResult,
    target: Target2D,
    forward_speed_mps: float,
    checkpoint_period_s: float,
    waypoint_spacing_m: float,
    min_turn_angle_rad: float,
    turn_speed_radps: float,
    turn_angle_max_rad: float,
    target_reached_radius_m: float,
) -> tuple[list[Command2D], dict[str, Any]]:
    """Convert a fresh body-frame A* path into a relative yaw/forward queue.

    This queue is only generated from a fresh target observation.  If the target
    is later temporarily missing, the runner may continue this already-committed
    queue, but it must not run a new A* plan to an old body-frame target.  This
    is the key distinction that keeps occlusion handling geometrically sane
    without relying on command odometry.
    """
    dist_to_target = math.hypot(float(target.x), float(target.y))
    if dist_to_target <= float(target_reached_radius_m):
        return [], {"reason": "target_reached", "target_distance_m": dist_to_target}
    if not plan.found or len(plan.path_xy_b) < 2:
        return [], {"reason": "no_plan", "plan_reason": plan.reason}

    path_len = _path_length(plan.path_xy_b)
    if path_len <= 1e-6:
        return [], {"reason": "empty_path", "path_length_m": path_len}

    spacing = max(0.15, float(waypoint_spacing_m))
    sample_distances = []
    d = spacing
    while d < path_len:
        sample_distances.append(d)
        d += spacing
    sample_distances.append(path_len)

    x = 0.0
    y = 0.0
    theta = 0.0
    commands: list[Command2D] = []
    waypoints: list[list[float]] = []

    for d in sample_distances:
        sample = _sample_path_at_distance(plan.path_xy_b, d)
        if sample is None:
            continue
        wx, wy = float(sample[0]), float(sample[1])
        dx = wx - x
        dy = wy - y
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            continue
        desired = math.atan2(dy, dx)
        turn_remaining = _angle_normalize(desired - theta)
        while abs(turn_remaining) >= float(min_turn_angle_rad):
            step = max(-float(turn_angle_max_rad), min(float(turn_angle_max_rad), turn_remaining))
            duration = abs(step) / max(1e-6, float(turn_speed_radps))
            wz = math.copysign(float(turn_speed_radps), step)
            commands.append(Command2D(0.0, 0.0, wz, duration_s=max(0.05, duration), source="simplified_target_follow_turn", metadata={"turn_angle_rad": step, "waypoint_xy_b": [wx, wy]}))
            theta = _angle_normalize(theta + step)
            turn_remaining = _angle_normalize(desired - theta)
            # Avoid pathological command queues if turn_angle_max is tiny.
            if len(commands) > 512:
                break
        forward_duration = dist / max(1e-6, float(forward_speed_mps))
        if forward_duration >= 0.10:
            commands.append(Command2D(float(forward_speed_mps), 0.0, 0.0, duration_s=forward_duration, source="simplified_target_follow_forward", metadata={"waypoint_xy_b": [wx, wy]}))
        # The command queue is an open-loop approximation of the planned path.
        # Advance the virtual pose to the sampled path waypoint.
        x, y = wx, wy
        waypoints.append([wx, wy])

    return commands, {
        "reason": "queue_ready" if commands else "empty_command_queue",
        "target_distance_m": dist_to_target,
        "path_length_m": path_len,
        "queue_command_count": len(commands),
        "waypoint_count": len(waypoints),
        "waypoint_spacing_m": spacing,
        "checkpoint_period_s": float(checkpoint_period_s),
        "first_waypoint_xy_b": None if not waypoints else waypoints[0],
        "last_waypoint_xy_b": None if not waypoints else waypoints[-1],
    }


def _copy_command_with_duration(command: Command2D, duration_s: float) -> Command2D:
    return Command2D(
        float(command.vx),
        float(command.vy),
        float(command.wz),
        timestamp_s=command.timestamp_s,
        duration_s=float(duration_s),
        source=command.source,
        metadata=dict(command.metadata),
    )


def _pop_command_burst(queue: list[Command2D], max_duration_s: float) -> tuple[list[Command2D], dict[str, Any]]:
    """Pop commands from a persistent queue, splitting long commands if needed."""
    remaining = max(0.0, float(max_duration_s))
    burst: list[Command2D] = []
    consumed = 0.0
    while queue and remaining > 1e-6:
        cmd = queue[0]
        dur = float(cmd.duration_s or 0.0)
        if dur <= 1e-6:
            queue.pop(0)
            continue
        if dur <= remaining + 1e-6:
            burst.append(queue.pop(0))
            remaining -= dur
            consumed += dur
        else:
            burst.append(_copy_command_with_duration(cmd, remaining))
            queue[0] = _copy_command_with_duration(cmd, dur - remaining)
            consumed += remaining
            remaining = 0.0
    return burst, {
        "burst_duration_s": consumed,
        "queue_remaining_commands": len(queue),
        "queue_empty_after_pop": not bool(queue),
    }

def _send_command_for_duration(
    *,
    sink: UnitreeLocoAdapter,
    command: Command2D,
    command_period_s: float,
    real_motion: bool,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if command.is_stop:
        result = sink.stop(timestamp_s=time.monotonic())
        return [{"phase": "stop", "accepted": result.accepted, "status": _json_safe(result.status)}]

    duration = float(command.duration_s or 0.0)
    if duration <= 0.0:
        return []

    if not real_motion:
        result = sink.send_command(command, require_state_gate=False)
        return [{"phase": "dry_send", "accepted": result.accepted, "sent_command": _json_safe(result.command), "status": _json_safe(result.status)}]

    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        now = time.monotonic()
        cmd = Command2D(command.vx, command.vy, command.wz, timestamp_s=now, duration_s=command.duration_s, source=command.source, metadata=dict(command.metadata))
        result = sink.send_command(cmd, require_state_gate=True)
        events.append({"phase": "send", "accepted": result.accepted, "sent_command": _json_safe(result.command), "status": _json_safe(result.status)})
        if not result.accepted:
            break
        sleep_s = min(float(command_period_s), max(0.0, deadline - time.monotonic()))
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    return events


def _execute_burst(*, sink: UnitreeLocoAdapter, commands: Sequence[Command2D], command_period_s: float, real_motion: bool) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, command in enumerate(commands):
        command_events = _send_command_for_duration(sink=sink, command=command, command_period_s=command_period_s, real_motion=real_motion)
        events.append({"burst_index": idx, "command": _json_safe(command), "events": command_events})
        if any(not bool(e.get("accepted", True)) for e in command_events):
            break
    stop = sink.stop(timestamp_s=time.monotonic())
    events.append({"phase": "burst_stop", "accepted": stop.accepted, "status": _json_safe(stop.status)})
    return events


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal real G1 target follower: stream target + LiDAR A* + rare command bursts.")
    p.add_argument("--config", default=str(_default_config_path()))
    p.add_argument("--unitree-vibes-dir", required=True)
    p.add_argument("--livox-config", required=True)
    p.add_argument("--host-ip", required=True)
    p.add_argument("--iface", default="eth0")
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-root", default="outputs/real_runs")
    p.add_argument("--no-timestamp-debug-dirs", action="store_true")
    p.add_argument("--json-lines", action="store_true")

    p.add_argument("--detection-stream-path", required=True)
    p.add_argument("--detection-stream-max-age-s", type=float, default=5.0)
    p.add_argument("--target-class", default="bottle")
    p.add_argument("--target-min-confidence", type=float, default=0.10)
    p.add_argument("--target-min-depth-valid-ratio", type=float, default=0.0)
    p.add_argument("--initial-target-timeout-s", type=float, default=20.0)
    p.add_argument(
        "--target-missing-policy",
        choices=["continue_plan", "stop", "reuse_last"],
        default="continue_plan",
        help="When the stream has no current target at a checkpoint: continue_plan executes the previously generated open-loop queue; stop stops. reuse_last is accepted as a deprecated alias for continue_plan and does not replan to stale body-frame coordinates.",
    )

    p.add_argument("--max-cycles", type=int, default=10)
    p.add_argument("--checkpoint-period-s", type=float, default=2.5)
    p.add_argument(
        "--max-burst-duration-s",
        type=float,
        default=2.5,
        help=(
            "Hard cap on a single open-loop motion burst before the next stop/observe checkpoint. "
            "This prevents a long checkpoint-period from driving the robot deep into an obstacle before LiDAR can revalidate the path. "
            "Set <=0 to disable the cap."
        ),
    )
    p.add_argument("--settle-s", type=float, default=0.50)
    p.add_argument("--command-period-s", type=float, default=0.10)
    p.add_argument("--stop-after-burst-s", type=float, default=0.20)

    p.add_argument("--forward-speed-mps", type=float, default=0.20)
    p.add_argument("--turn-speed-radps", type=float, default=0.25)
    p.add_argument("--turn-angle-max-deg", type=float, default=30.0)
    p.add_argument("--min-turn-angle-deg", type=float, default=10.0)
    p.add_argument("--lookahead-m", type=float, default=0.80)
    p.add_argument("--target-reached-radius-m", type=float, default=0.75)

    p.add_argument("--grid-x-min", type=float, default=-2.0)
    p.add_argument("--grid-x-max", type=float, default=6.0)
    p.add_argument("--grid-y-min", type=float, default=-4.0)
    p.add_argument("--grid-y-max", type=float, default=4.0)
    p.add_argument("--grid-resolution", type=float, default=0.05)
    p.add_argument("--astar-inflation-m", type=float, default=0.35)
    p.add_argument("--occupancy-threshold", type=float, default=0.5)
    p.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    p.add_argument("--astar-max-expansions", type=int, default=0)
    p.add_argument("--no-diagonal-astar", action="store_true")
    p.add_argument("--allow-start-relocation", action="store_true")
    p.add_argument("--allow-goal-relocation", dest="allow_goal_relocation", action="store_true", default=True)
    p.add_argument("--disable-goal-relocation", dest="allow_goal_relocation", action="store_false")
    p.add_argument("--goal-relocation-radius-m", type=float, default=0.80)

    p.add_argument("--emergency-front-stop-m", type=float, default=0.45)
    p.add_argument("--front-corridor-half-width-m", type=float, default=0.25)
    p.add_argument("--disable-emergency-front-stop", action="store_true")
    p.add_argument(
        "--obstacle-replan-front-m",
        type=float,
        default=0.70,
        help="Raw front LiDAR distance that triggers an obstacle-driven A* replan before the emergency stop threshold is reached. Set <=0 to disable obstacle-driven replanning.",
    )
    p.add_argument(
        "--clear-queue-on-emergency-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear the active open-loop queue when the raw emergency front stop fires, so old commands cannot resume after an obstacle stop.",
    )
    p.add_argument(
        "--obstacle-replan-with-estimated-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the target is temporarily missing, allow obstacle-triggered replanning toward the active goal estimated from executed command bursts. Target loss alone still does not trigger a replan.",
    )

    p.add_argument(
        "--validate-active-queue-with-lidar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before continuing an active open-loop queue while the target is missing, validate the next effective burst against the current LiDAR occupancy grid. If blocked, the replan reason is obstacle/path blockage, not target missing.",
    )
    p.add_argument(
        "--validate-executed-burst-with-lidar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate every command burst selected for execution against the current LiDAR occupancy grid before sending it. If blocked, clear the queue and stop instead of executing a geometrically stale or unsafe burst.",
    )
    p.add_argument("--queued-segment-guard-inflation-m", type=float, default=0.25)
    p.add_argument("--queued-segment-start-ignore-m", type=float, default=0.15)
    p.add_argument("--queued-segment-sample-step-m", type=float, default=0.05)

    p.add_argument("--livox-frame-time", type=float, default=0.20)
    p.add_argument("--livox-frame-packets", type=int, default=120)
    p.add_argument("--livox-timeout", type=float, default=2.0)

    p.add_argument("--command-vx-scale", type=float, default=1.0)
    p.add_argument("--command-vy-scale", type=float, default=0.0)
    p.add_argument("--command-wz-scale", type=float, default=1.0)
    p.add_argument("--allow-non-200", action="store_true")
    p.add_argument("--expected-fsm-id", type=int, default=200)
    p.add_argument("--state-samples", type=int, default=1)
    p.add_argument("--state-interval-s", type=float, default=0.05)
    p.add_argument("--yes-i-understand-this-moves-the-robot", action="store_true")

    p.add_argument("--save-lidar-debug", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_adapter_config(args.config)
    run_id = args.run_id or make_run_id(prefix="simplified_target_follow")
    debug_base = None
    if args.save_lidar_debug is not None:
        debug_base = timestamped_output_dir(args.save_lidar_debug, run_id=run_id, enabled=not bool(args.no_timestamp_debug_dirs))
        assert debug_base is not None
        debug_base.mkdir(parents=True, exist_ok=True)
    else:
        debug_base = Path(args.output_root) / run_id / "simplified_target_follow"
        debug_base.mkdir(parents=True, exist_ok=True)

    real_motion = bool(args.yes_i_understand_this_moves_the_robot)
    grid_cfg = _make_grid_cfg(args)
    astar_cfg = GridAStarCfg(
        occupancy_threshold=float(args.occupancy_threshold),
        inflation_radius=float(args.astar_inflation_m),
        allow_diagonal=not bool(args.no_diagonal_astar),
        heuristic_weight=float(args.astar_heuristic_weight),
        max_expansions=int(args.astar_max_expansions),
    )

    stream_client = DetectionStreamClient(
        path=args.detection_stream_path,
        max_age_s=float(args.detection_stream_max_age_s),
        missing_ok=True,
    )

    livox_adapter = UnitreeVibesLivox2Adapter(
        UnitreeVibesLivox2AdapterConfig(
            livox_config_path=str(args.livox_config),
            unitree_vibes_dir=str(args.unitree_vibes_dir),
            host_ip=str(args.host_ip),
            frame_time_s=float(args.livox_frame_time),
            frame_packets=int(args.livox_frame_packets),
            read_timeout_s=float(args.livox_timeout),
            adapter_config=LivoxAdapterConfig.from_lidar_config(cfg.lidar),
        )
    )

    require_fsm_id = None if args.allow_non_200 else int(args.expected_fsm_id)
    sink = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(
            iface=str(args.iface),
            dry_run=not real_motion,
            require_arm=True,
            require_fsm_id=require_fsm_id,
            command_period_s=float(args.command_period_s),
            command_vx_scale=float(args.command_vx_scale),
            command_vy_scale=float(args.command_vy_scale),
            command_wz_scale=float(args.command_wz_scale),
            state_samples=int(args.state_samples),
            state_interval_s=float(args.state_interval_s),
        ),
        command_limits=cfg.command_limits,
    )

    run_info = {
        "script": "scripts/simplified_target_follow.py",
        "run_id": run_id,
        "real_motion": real_motion,
        "simple_contract": {
            "target_selection": "best current stream detection, no target lock, no track-id persistence",
            "planning": "A* on current Livox occupancy grid from body origin; goal-only relocation around occupied target stand-off is allowed; stale body-frame targets are never replanned",
            "execution": "yaw-first/forward-only burst, stop only at effective checkpoint; if target is temporarily missing, optionally continue the previously generated open-loop queue",
            "disabled": [
                "StepwiseTargetNavPolicyRuntime",
                "PrimaryTargetLock",
                "SafetySupervisor",
                "command_odometry",
                "target_change_replanning",
                "path_block_guard",
                "lateral_commands",
            ],
        },
        "args": vars(args),
        "grid_cfg": grid_cfg,
        "astar_cfg": astar_cfg,
    }
    write_run_info(debug_base, run_info)
    trace_path = debug_base / "simplified_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    requested_checkpoint_period_s = float(args.checkpoint_period_s)
    if float(args.max_burst_duration_s) > 0.0:
        effective_burst_period_s = min(requested_checkpoint_period_s, float(args.max_burst_duration_s))
    else:
        effective_burst_period_s = requested_checkpoint_period_s

    print("[INFO]: simplified_target_follow starting")
    print(f"[INFO]: run_id={run_id} real_motion={real_motion}")
    print(f"[INFO]: stream={args.detection_stream_path}")
    print(f"[INFO]: no target lock, no stepwise runtime, no path_block guard")
    if effective_burst_period_s < requested_checkpoint_period_s:
        print(f"[INFO]: effective motion burst capped at {effective_burst_period_s:.3f}s from requested checkpoint_period_s={requested_checkpoint_period_s:.3f}s")
    if real_motion:
        print("[WARN]: REAL MOTION ENABLED; operator must be ready to stop the robot")
    else:
        print("[INFO]: dry-run mode: no real Move commands will be sent")

    exit_code = 0
    last_target: Target2D | None = None
    active_queue: list[Command2D] = []
    active_goal_xy_at_plan_b: tuple[float, float] | None = None
    pose_since_plan_xyyaw: tuple[float, float, float] = (0.0, 0.0, 0.0)
    livox_adapter.start()
    sink.connect()
    sink.arm()
    try:
        # Initial target acquisition is explicit.  This is the only required
        # target precondition for the simple test.
        last_target, target_meta = _wait_for_initial_target(
            stream_client=stream_client,
            target_class=str(args.target_class),
            min_confidence=float(args.target_min_confidence),
            min_depth_valid_ratio=float(args.target_min_depth_valid_ratio),
            timeout_s=float(args.initial_target_timeout_s),
            poll_s=0.25,
            json_lines=bool(args.json_lines),
        )

        for cycle in range(int(args.max_cycles)):
            stop_before = sink.stop(timestamp_s=time.monotonic())
            if float(args.settle_s) > 0.0:
                time.sleep(float(args.settle_s))

            now = time.monotonic()
            current_target, target_meta = _read_best_stream_target(
                stream_client=stream_client,
                target_class=str(args.target_class),
                min_confidence=float(args.target_min_confidence),
                min_depth_valid_ratio=float(args.target_min_depth_valid_ratio),
                require_fresh=True,
            )
            target_source = "current_stream"
            target_missing_policy = str(args.target_missing_policy)
            if target_missing_policy == "reuse_last":
                # Backward-compatible CLI spelling.  The old behavior replanned
                # to a stale body-frame target after robot motion, which is
                # geometrically invalid and caused obstacle runs to turn/replan
                # erratically.  Treat it as continue_plan instead.
                target_missing_policy = "continue_plan"
            if current_target is not None:
                last_target = current_target
            elif target_missing_policy == "continue_plan" and active_queue:
                target_source = "active_plan_continued_target_missing"
            else:
                current_target = None
                target_source = "missing_stop"

            lidar = livox_adapter.read_lidar_frame(timeout_s=float(args.livox_timeout))
            occupancy = _occupancy_from_lidar(lidar.points_body_xyz, grid_cfg)
            front_check_x_max = max(
                float(args.emergency_front_stop_m),
                float(args.obstacle_replan_front_m) if float(args.obstacle_replan_front_m) > 0.0 else 0.0,
            )
            front_min = _raw_front_min(
                lidar.points_body_xyz,
                corridor_half_width_m=float(args.front_corridor_half_width_m),
                x_min_m=0.0,
                x_max_m=front_check_x_max,
            )

            plan = None
            plan_meta: dict[str, Any] = {}
            commands: list[Command2D] = []
            command_meta: dict[str, Any] = {"reason": "no_target"}
            emergency_stop = False
            obstacle_replan_requested = False
            obstacle_replan_trigger: dict[str, Any] = {"requested": False}
            obstacle_replan_front_m = float(args.obstacle_replan_front_m)
            if obstacle_replan_front_m > 0.0 and front_min is not None and front_min <= obstacle_replan_front_m:
                obstacle_replan_requested = True
                obstacle_replan_trigger = {
                    "requested": True,
                    "reason": "front_obstacle_distance",
                    "front_min_m": front_min,
                    "threshold_m": obstacle_replan_front_m,
                }

            queue_validation: dict[str, Any] = {"checked": False, "reason": "not_applicable"}
            if (
                bool(args.validate_active_queue_with_lidar)
                and target_source == "active_plan_continued_target_missing"
                and active_queue
                and not obstacle_replan_requested
            ):
                preview = _preview_command_burst(active_queue, effective_burst_period_s)
                queue_validation = _validate_burst_against_occupancy(
                    occupancy=occupancy,
                    grid_cfg=grid_cfg,
                    commands=preview,
                    guard_inflation_m=float(args.queued_segment_guard_inflation_m),
                    occupancy_threshold=float(args.occupancy_threshold),
                    start_ignore_m=float(args.queued_segment_start_ignore_m),
                    sample_step_m=float(args.queued_segment_sample_step_m),
                )
                if not bool(queue_validation.get("clear", True)):
                    obstacle_replan_requested = True
                    obstacle_replan_trigger = {
                        "requested": True,
                        "reason": "active_queue_blocked_by_current_lidar",
                        "queue_validation": queue_validation,
                    }

            if (not args.disable_emergency_front_stop) and front_min is not None and front_min <= float(args.emergency_front_stop_m):
                emergency_stop = True

            if current_target is not None or (obstacle_replan_requested and bool(args.obstacle_replan_with_estimated_goal) and active_goal_xy_at_plan_b is not None):
                # Fresh target: replan in the current body frame.
                # Missing target: replan only if current LiDAR reports an
                # obstacle on/near the active path.  The target-missing event
                # itself is not a replan trigger; it only forces us to estimate
                # the active fixed goal from the executed open-loop burst.
                planning_target = current_target
                planning_source = "current_stream"
                if planning_target is None:
                    planning_target = _goal_in_current_body(active_goal_xy_at_plan_b, pose_since_plan_xyyaw)
                    planning_source = "obstacle_replan_estimated_goal_target_missing"

                if planning_target is None:
                    active_queue.clear()
                    command_meta = {
                        "reason": "obstacle_replan_missing_goal",
                        "stale_body_target_replan_blocked": True,
                        "obstacle_replan_trigger": obstacle_replan_trigger,
                        "queue_validation": queue_validation,
                    }
                else:
                    if current_target is None:
                        target_source = planning_source
                    # A* consumes either a fresh body-frame target or an
                    # obstacle-triggered estimate of the active fixed goal.
                    # It is never called merely because the target is missing.
                    plan, plan_meta = _plan_with_goal_relocation(
                        occupancy=occupancy,
                        target=planning_target,
                        grid_cfg=grid_cfg,
                        astar_cfg=astar_cfg,
                        allow_start_relocation=bool(args.allow_start_relocation),
                        allow_goal_relocation=bool(args.allow_goal_relocation),
                        max_goal_relocation_m=float(args.goal_relocation_radius_m),
                    )
                    new_queue, queue_meta = _make_open_loop_command_queue(
                        plan=plan,
                        target=planning_target,
                        forward_speed_mps=float(args.forward_speed_mps),
                        checkpoint_period_s=effective_burst_period_s,
                        waypoint_spacing_m=float(args.lookahead_m),
                        min_turn_angle_rad=math.radians(float(args.min_turn_angle_deg)),
                        turn_speed_radps=float(args.turn_speed_radps),
                        turn_angle_max_rad=math.radians(float(args.turn_angle_max_deg)),
                        target_reached_radius_m=float(args.target_reached_radius_m),
                    )
                    active_queue = list(new_queue)
                    active_goal_xy_at_plan_b = (float(planning_target.x), float(planning_target.y))
                    pose_since_plan_xyyaw = (0.0, 0.0, 0.0)
                    commands, pop_meta = _pop_command_burst(active_queue, effective_burst_period_s)
                    command_meta = dict(queue_meta) | {
                        "plan_goal": plan_meta,
                        "queue_pop": pop_meta,
                        "planning_source": planning_source,
                        "obstacle_replan_trigger": obstacle_replan_trigger,
                        "queue_validation": queue_validation,
                        "reason": "obstacle_replan_commands_ready" if (obstacle_replan_requested and current_target is None and commands) else ("commands_ready" if commands else queue_meta.get("reason", "empty_command_queue")),
                    }
                    if emergency_stop:
                        if bool(args.clear_queue_on_emergency_stop):
                            active_queue.clear()
                        commands = []
                        command_meta = dict(command_meta) | {
                            "reason": "emergency_front_stop",
                            "front_min_m": front_min,
                            "active_queue_cleared": bool(args.clear_queue_on_emergency_stop),
                            "motion_suppressed_after_plan": True,
                        }
            elif target_source == "active_plan_continued_target_missing" and active_queue and not emergency_stop:
                # No fresh target and no LiDAR obstacle replan request.  Continue
                # the previously generated relative command queue, but do NOT
                # replan using last_target because it is expressed in an old body
                # frame.  Replanning under target loss is allowed only above when
                # the current LiDAR obstacle condition requested it.
                commands, pop_meta = _pop_command_burst(active_queue, effective_burst_period_s)
                estimated_goal = _goal_in_current_body(active_goal_xy_at_plan_b, pose_since_plan_xyyaw)
                command_meta = {
                    "reason": "active_plan_continued_target_missing" if commands else "active_plan_empty_target_missing",
                    "queue_pop": pop_meta,
                    "stale_body_target_replan_blocked": True,
                    "obstacle_replan_trigger": obstacle_replan_trigger,
                    "queue_validation": queue_validation,
                    "last_target_xy_b_for_logging_only": None if last_target is None else [float(last_target.x), float(last_target.y)],
                    "active_goal_estimated_xy_b": None if estimated_goal is None else [float(estimated_goal.x), float(estimated_goal.y)],
                }
            else:
                if emergency_stop:
                    if bool(args.clear_queue_on_emergency_stop):
                        active_queue.clear()
                    commands = []
                    command_meta = {
                        "reason": "emergency_front_stop",
                        "front_min_m": front_min,
                        "active_queue_cleared": bool(args.clear_queue_on_emergency_stop),
                        "obstacle_replan_trigger": obstacle_replan_trigger,
                        "queue_validation": queue_validation,
                        "active_queue_len": len(active_queue),
                    }
                else:
                    command_meta = {
                        "reason": "target_missing",
                        "stale_body_target_replan_blocked": True,
                        "obstacle_replan_trigger": obstacle_replan_trigger,
                        "queue_validation": queue_validation,
                        "active_queue_len": len(active_queue),
                    }

            if commands and bool(args.validate_executed_burst_with_lidar):
                execution_validation = _validate_burst_against_occupancy(
                    occupancy=occupancy,
                    grid_cfg=grid_cfg,
                    commands=commands,
                    guard_inflation_m=float(args.queued_segment_guard_inflation_m),
                    occupancy_threshold=float(args.occupancy_threshold),
                    start_ignore_m=float(args.queued_segment_start_ignore_m),
                    sample_step_m=float(args.queued_segment_sample_step_m),
                )
                command_meta = dict(command_meta) | {"execution_validation": execution_validation}
                if not bool(execution_validation.get("clear", True)):
                    if bool(args.clear_queue_on_emergency_stop):
                        active_queue.clear()
                    commands = []
                    command_meta = dict(command_meta) | {
                        "reason": "selected_burst_blocked_by_current_lidar",
                        "active_queue_cleared": bool(args.clear_queue_on_emergency_stop),
                    }

            # Debug image before motion.
            image_path = debug_base / f"occupancy_grid_cycle_{cycle:06d}.png"
            try:
                save_occupancy_debug_image(
                    image_path,
                    occupancy,
                    grid_cfg=grid_cfg,
                    plan_xy_b=None if plan is None else plan.path_xy_b,
                    target_xy_b=None if current_target is None else (float(current_target.x), float(current_target.y)),
                )
            except Exception as exc:
                image_path = None  # type: ignore[assignment]
                command_meta = dict(command_meta) | {"debug_image_error": repr(exc)}

            send_events: list[dict[str, Any]] = []
            if commands:
                send_events = _execute_burst(sink=sink, commands=commands, command_period_s=float(args.command_period_s), real_motion=real_motion)
            else:
                stop = sink.stop(timestamp_s=time.monotonic())
                send_events = [{"phase": "no_motion", "accepted": stop.accepted, "status": _json_safe(stop.status), "reason": command_meta.get("reason")}]

            commands_accepted = True
            if commands:
                for burst in send_events:
                    if not isinstance(burst, Mapping):
                        continue
                    for event in burst.get("events", []):
                        if isinstance(event, Mapping) and event.get("accepted") is False:
                            commands_accepted = False
                if commands_accepted:
                    pose_since_plan_xyyaw = _integrate_open_loop_pose(pose_since_plan_xyyaw, commands)

            if float(args.stop_after_burst_s) > 0.0:
                time.sleep(float(args.stop_after_burst_s))

            row = {
                "cycle": int(cycle),
                "timestamp_s": now,
                "target_xy_b": None if current_target is None else [float(current_target.x), float(current_target.y)],
                "target_source": target_source,
                "target_stream": target_meta,
                "active_queue_len_after_decision": len(active_queue),
                "lidar_point_count": int(lidar.point_count),
                "front_min_m": front_min,
                "emergency_stop": emergency_stop,
                "obstacle_replan_requested": obstacle_replan_requested,
                "obstacle_replan_trigger": obstacle_replan_trigger,
                "requested_checkpoint_period_s": requested_checkpoint_period_s,
                "effective_burst_period_s": effective_burst_period_s,
                "active_goal_xy_at_plan_b": None if active_goal_xy_at_plan_b is None else [float(active_goal_xy_at_plan_b[0]), float(active_goal_xy_at_plan_b[1])],
                "pose_since_plan_xyyaw": [float(v) for v in pose_since_plan_xyyaw],
                "occupancy_grid_path": None if image_path is None else str(image_path),
                "plan": None if plan is None else {
                    "found": bool(plan.found),
                    "reason": plan.reason,
                    "start_cell": list(plan.start_cell),
                    "goal_cell": list(plan.goal_cell),
                    "path_len": len(plan.path_xy_b),
                    "path_length_m": _path_length(plan.path_xy_b),
                    "goal_metadata": plan_meta,
                },
                "command_meta": command_meta,
                "commands": [_json_safe(c) for c in commands],
                "stop_before": {"accepted": stop_before.accepted, "status": _json_safe(stop_before.status)},
                "send_events": send_events,
            }
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            if args.json_lines:
                print(json.dumps(_json_safe(row), sort_keys=True))
            else:
                print(
                    f"[INFO]: cycle={cycle} target={row['target_xy_b']} source={target_source} "
                    f"front={front_min} plan={None if plan is None else plan.reason} "
                    f"cmd_reason={command_meta.get('reason')} commands={len(commands)}"
                )

            if command_meta.get("reason") == "target_reached":
                print(f"[INFO]: target reached at cycle={cycle}")
                break
            if real_motion:
                rejected = False
                for burst in send_events:
                    for event in burst.get("events", []) if isinstance(burst, Mapping) else []:
                        if event.get("accepted") is False:
                            rejected = True
                if rejected:
                    print("[ERROR]: command rejected during real motion; stopping")
                    exit_code = 1
                    break
    except KeyboardInterrupt:
        print("[WARN]: interrupted by user; stopping robot")
        exit_code = 130
    except Exception as exc:
        print(f"[ERROR]: simplified_target_follow failed: {exc!r}")
        exit_code = 1
    finally:
        try:
            sink.stop(timestamp_s=time.monotonic())
        finally:
            sink.close()
            livox_adapter.stop()
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
