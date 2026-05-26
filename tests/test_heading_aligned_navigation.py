from __future__ import annotations

import math

from g1_nav_rl.navigation.simple_nav import (
    Pose2D,
    SimpleNavConfig,
    SimpleNavigator,
    command_to_waypoint,
    command_to_waypoint_heading_aligned,
    select_lookahead_point,
)
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg


def _heading_cmd(pose: Pose2D, waypoint: tuple[float, float], cfg: SimpleNavConfig):
    return command_to_waypoint_heading_aligned(pose, waypoint, cfg)


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
        heading_slow_angle_rad=1.0,
        heading_rotate_in_place_angle_rad=2.0,
    )

    holonomic, _ = command_to_waypoint(pose, (1.0, 1.0), 0.0, cfg)
    heading, waypoint_b, bearing, alignment, turn_in_place, raw = _heading_cmd(pose, (1.0, 1.0), cfg)

    assert waypoint_b == (1.0, 1.0)
    assert math.isclose(bearing, math.pi / 4.0, rel_tol=1e-6)
    assert 0.25 <= alignment < 1.0
    assert turn_in_place is False
    assert abs(heading.vy) < abs(holonomic.vy)
    assert heading.wz > 0.0
    assert raw.wz > heading.wz  # clipped by max_wz


def test_heading_aligned_turn_first_for_sideways_waypoint() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    cfg = SimpleNavConfig(
        k_xy=1.0,
        k_yaw=1.0,
        max_vx=1.0,
        max_vy=1.0,
        max_wz=2.0,
        heading_lateral_scale=0.10,
        heading_min_vx_scale=0.20,
        heading_slow_angle_rad=0.35,
        heading_rotate_in_place_angle_rad=0.80,
        heading_rotate_vx_scale=0.0,
        heading_rotate_lateral_scale=0.0,
    )

    heading, _, bearing, alignment, turn_in_place, _ = _heading_cmd(pose, (0.0, 1.0), cfg)

    assert math.isclose(abs(bearing), math.pi / 2.0, rel_tol=1e-6)
    assert turn_in_place is True
    assert math.isclose(alignment, 0.0, abs_tol=1e-9)
    assert math.isclose(heading.vx, 0.0, abs_tol=1e-9)
    assert math.isclose(heading.vy, 0.0, abs_tol=1e-9)
    assert heading.wz > 0.0


def test_heading_aligned_soft_slowdown_between_angles() -> None:
    pose = Pose2D(0.0, 0.0, 0.0)
    cfg = SimpleNavConfig(
        k_xy=1.0,
        k_yaw=1.0,
        max_vx=10.0,
        max_vy=10.0,
        max_wz=10.0,
        heading_lateral_scale=0.20,
        heading_min_vx_scale=0.20,
        heading_slow_angle_rad=0.20,
        heading_rotate_in_place_angle_rad=1.00,
        heading_rotate_vx_scale=0.0,
        heading_rotate_lateral_scale=0.0,
    )

    heading, _, bearing, alignment, turn_in_place, raw = _heading_cmd(pose, (1.0, 0.6), cfg)

    assert 0.20 < abs(bearing) < 1.00
    assert turn_in_place is False
    assert 0.0 < alignment < math.cos(bearing)
    assert 0.0 < heading.vx < math.hypot(1.0, 0.6)
    assert abs(heading.vy) < abs(0.20 * 0.6)
    assert raw == heading


def test_select_lookahead_point_interpolates_along_remaining_path() -> None:
    path = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.5), (0.2, 1.0)]
    point = select_lookahead_point(path, current_xy=(0.0, 0.0), waypoint_index=1, lookahead_m=0.4)
    assert point == (0.2, 0.2)


def test_heading_aligned_navigator_uses_path_lookahead_for_command_point() -> None:
    nav = SimpleNavigator(
        grid_cfg=OccupancyGridCfg(),
        config=SimpleNavConfig(
            command_mode="heading_aligned",
            path_lookahead_m=0.5,
            k_xy=1.0,
            k_yaw=1.0,
            max_vx=1.0,
            max_vy=1.0,
            max_wz=2.0,
            heading_slow_angle_rad=2.0,
            heading_rotate_in_place_angle_rad=3.0,
        ),
    )
    nav.set_world_path([(0.0, 0.0), (0.1, 0.0), (0.1, 1.0)])
    cmd = nav.command(Pose2D(0.0, 0.0, 0.0))
    assert cmd.waypoint_index == 2  # waypoint 1 is within reached radius
    assert cmd.waypoint_xy_w is not None
    assert 0.04 < cmd.waypoint_xy_w[0] < 0.06
    assert 0.49 < cmd.waypoint_xy_w[1] < 0.51
    assert cmd.bearing_error is not None and cmd.bearing_error > 0.0
