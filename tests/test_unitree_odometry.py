from __future__ import annotations

from types import SimpleNamespace
import math

import pytest

from adapters import AdapterHealth
from adapters.real_unitree_odometry import UnitreeOdometryAdapter, UnitreeOdometryConfig


def _msg(*, x=1.0, y=2.0, z=0.72, yaw=1.2, vx=0.2, vy=-0.1, vz=0.0, yaw_speed=0.3):
    return SimpleNamespace(
        position=[x, y, z],
        velocity=[vx, vy, vz],
        yaw_speed=yaw_speed,
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, yaw]),
    )


def test_odometry_parses_sample_and_builds_atomic_relative_snapshot() -> None:
    now = [10.0]
    odom = UnitreeOdometryAdapter(clock=lambda: now[0])
    odom._on_odom(_msg())
    odom.capture_origin()
    now[0] = 10.01
    odom._on_odom(_msg(x=1.1, y=2.0, yaw=1.3))
    snap = odom.snapshot()
    assert snap.sequence == snap.raw.sequence == 2
    assert snap.epoch == snap.raw.epoch == 0
    assert snap.health == AdapterHealth.OK
    assert snap.raw.vx == pytest.approx(0.2)
    assert math.hypot(snap.relative_pose.x, snap.relative_pose.y) == pytest.approx(0.1)
    assert snap.relative_pose.yaw == pytest.approx(0.1)


def test_invalid_sample_preserves_latest_and_records_rejection() -> None:
    odom = UnitreeOdometryAdapter(clock=lambda: 10.0)
    odom._on_odom(_msg(x=0.0, y=0.0))
    odom._on_odom(_msg(x=float("nan")))
    odom._on_odom(SimpleNamespace(position=[]))
    assert odom.snapshot().sequence == 1
    status = odom.status()
    assert status.details["rejected_count"] == 2
    assert status.details["callback_error_count"] == 1


def test_discontinuity_starts_new_epoch_without_counting_as_same_motion() -> None:
    odom = UnitreeOdometryAdapter(
        UnitreeOdometryConfig(max_position_jump_m=0.5, max_yaw_jump_rad=0.5),
        clock=lambda: 10.0,
    )
    odom._on_odom(_msg(x=0.0, y=0.0, yaw=0.0))
    odom._on_odom(_msg(x=3.0, y=0.0, yaw=0.0))
    snap = odom.snapshot()
    assert snap.sequence == 2
    assert snap.epoch == 1
    assert odom.status().details["discontinuity_count"] == 1


def test_odometry_freshness_uses_receive_time() -> None:
    now = [1.0]
    odom = UnitreeOdometryAdapter(UnitreeOdometryConfig(stale_after_s=0.1), clock=lambda: now[0])
    odom._on_odom(_msg())
    assert odom.status().health == AdapterHealth.OK
    now[0] = 1.2
    assert odom.status().health == AdapterHealth.STALE
