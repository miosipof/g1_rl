from __future__ import annotations

import math

from g1_nav_rl.navigation.simple_nav import (
    Pose2D,
    SimpleNavConfig,
    command_to_waypoint,
    command_to_waypoint_heading_aligned,
)


def test_heading_aligned_command_prefers_yaw_over_lateral_motion() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    cfg = SimpleNavConfig(
        k_xy=1.0,
        k_yaw=2.0,
        max_vx=1.0,
        max_vy=1.0,
        max_wz=1.0,
        heading_lateral_scale=0.10,
        heading_min_vx_scale=0.25,
    )

    holonomic, _ = command_to_waypoint(pose, (1.0, 1.0), 0.0, cfg)
    heading, waypoint_b, bearing, alignment, raw = command_to_waypoint_heading_aligned(pose, (1.0, 1.0), cfg)

    assert waypoint_b == (1.0, 1.0)
    assert math.isclose(bearing, math.pi / 4.0, rel_tol=1e-6)
    assert 0.25 <= alignment < 1.0
    assert abs(heading.vy) < abs(holonomic.vy)
    assert heading.wz > 0.0
    assert raw.wz > heading.wz  # clipped by max_wz


def test_heading_aligned_command_reduces_forward_speed_for_sideways_waypoint() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    cfg = SimpleNavConfig(
        k_xy=1.0,
        k_yaw=1.0,
        max_vx=1.0,
        max_vy=1.0,
        max_wz=2.0,
        heading_lateral_scale=0.10,
        heading_min_vx_scale=0.20,
    )

    heading, _, bearing, alignment, _ = command_to_waypoint_heading_aligned(pose, (0.0, 1.0), cfg)

    assert math.isclose(abs(bearing), math.pi / 2.0, rel_tol=1e-6)
    assert math.isclose(alignment, 0.20, rel_tol=1e-6)
    assert math.isclose(heading.vx, 0.20, rel_tol=1e-6)
    assert math.isclose(heading.vy, 0.10, rel_tol=1e-6)
    assert heading.wz > 0.0
