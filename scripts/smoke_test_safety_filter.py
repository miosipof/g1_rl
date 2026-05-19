#!/usr/bin/env python3
"""Small non-Isaac smoke test for the LiDAR safety filter."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g1_nav_rl.navigation.simple_nav import MotionCommand, Pose2D, Target2D
from g1_nav_rl.navigation.safety import LidarSafetyConfig, LidarSafetyFilter


def make_scan(points: list[tuple[float, float]], *, total_rays: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    pts = torch.zeros((1, total_rays, 2), dtype=torch.float32)
    valid = torch.zeros((1, total_rays), dtype=torch.bool)
    for i, xy in enumerate(points):
        pts[0, i, 0] = float(xy[0])
        pts[0, i, 1] = float(xy[1])
        valid[0, i] = True
    return pts, valid


def apply_case(
    name: str,
    scan_points: list[tuple[float, float]],
    expected: set[str],
    *,
    nominal: MotionCommand | None = None,
) -> None:
    filt = LidarSafetyFilter(
        LidarSafetyConfig(
            front_slow_distance=1.20,
            front_stop_distance=0.70,
            front_backup_distance=0.45,
            side_open_distance=0.90,
            side_slow_distance=0.90,
            side_stop_distance=0.65,
            side_critical_distance=0.45,
            stuck_window_steps=0,
        )
    )
    pts, valid = make_scan(scan_points)
    res = filt.apply(
        nominal or MotionCommand(0.20, 0.0, 0.0),
        lidar_points_b=pts,
        lidar_valid=valid,
        pose=Pose2D(0.0, 0.0, 0.0),
        target=Target2D(4.0, 0.0),
        step_index=0,
    )
    print(
        f"[INFO]: {name:18s} action={res.info.action:14s} "
        f"cmd=({res.command.vx:.3f},{res.command.vy:.3f},{res.command.wz:.3f}) "
        f"clear(front,left,right)=({res.info.front_clearance:.3f},"
        f"{res.info.left_clearance:.3f},{res.info.right_clearance:.3f})"
    )
    if res.info.action not in expected:
        raise RuntimeError(f"{name}: expected one of {sorted(expected)}, got {res.info.action!r}")


def test_stuck() -> None:
    filt = LidarSafetyFilter(
        LidarSafetyConfig(
            stuck_window_steps=5,
            stuck_recovery_steps=3,
            stuck_min_command_speed=0.08,
            stuck_min_pose_progress=0.05,
            stuck_min_target_progress=0.05,
        )
    )
    pts, valid = make_scan([])
    last = None
    for step in range(5):
        last = filt.apply(
            MotionCommand(0.20, 0.0, 0.0),
            lidar_points_b=pts,
            lidar_valid=valid,
            pose=Pose2D(0.0, 0.0, 0.0),
            target=Target2D(4.0, 0.0),
            step_index=step,
        )
    assert last is not None
    print(
        f"[INFO]: {'stuck':18s} action={last.info.action:14s} "
        f"cmd=({last.command.vx:.3f},{last.command.vy:.3f},{last.command.wz:.3f})"
    )
    if last.info.action != "stuck_recovery":
        raise RuntimeError(f"stuck: expected stuck_recovery, got {last.info.action!r}")


def main() -> None:
    # No valid hits: nominal command remains unchanged.
    apply_case("clear", [], {"ok"})

    # Front hit inside slow zone but outside stop zone.
    apply_case("slow", [(0.95, 0.0)], {"slow"})

    # Front blocked, left open, right blocked: sidestep left.
    apply_case("sidestep_left", [(0.60, 0.0), (0.50, -0.50)], {"sidestep_left"})

    # Critical front: backup.
    apply_case("backup", [(0.35, 0.0), (1.5, 1.5)], {"backup"})

    # Side obstacle inside slow distance: reduce forward velocity.
    apply_case("side_slow", [(0.20, -0.70)], {"side_slow"})

    # Side obstacle inside stop distance while nominal command moves into it:
    # suppress into-obstacle lateral velocity.
    apply_case("side_stop", [(0.20, -0.58)], {"side_stop"}, nominal=MotionCommand(0.20, -0.10, 0.0))

    # Side obstacle inside critical distance: sidestep away from it.
    apply_case("side_critical", [(0.10, -0.40)], {"sidestep_left"})

    # Stuck: commanded motion without pose/target progress.
    test_stuck()

    print("[INFO]: safety filter smoke test passed")


if __name__ == "__main__":
    main()
