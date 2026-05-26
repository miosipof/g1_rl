from __future__ import annotations

import pytest
import numpy as np

from adapters import (
    AdapterHealth,
    Command2D,
    CommandLimitsConfig,
    LidarFrame,
    RobotObservation,
    RobotState,
    SafetyConfig,
    SafetySupervisor,
    SafetySupervisorConfig,
    WatchdogConfig,
)


def _limits(**overrides):
    data = {
        "vx_min": -0.05,
        "vx_max": 0.10,
        "vy_min": -0.05,
        "vy_max": 0.05,
        "wz_min": -0.20,
        "wz_max": 0.20,
        "max_linear_accel_mps2": None,
        "max_yaw_accel_radps2": None,
        "command_rate_hz": 10.0,
    }
    data.update(overrides)
    return CommandLimitsConfig.from_dict(data)


def _watchdog(**overrides):
    data = {
        "max_rgbd_age_s": 0.50,
        "max_lidar_age_s": 0.30,
        "max_robot_state_age_s": 0.30,
        "max_policy_age_s": 0.20,
        "stop_on_stale_rgbd": False,
        "stop_on_stale_lidar": True,
        "stop_on_stale_robot_state": True,
    }
    data.update(overrides)
    return WatchdogConfig.from_dict(data)


def _safety(**overrides):
    data = {
        "require_operator_enable": True,
        "require_lidar_for_motion": True,
        "require_robot_state_for_motion": True,
        "stop_on_estop": True,
        "stop_on_robot_fault": True,
        "min_battery_percent": 20.0,
        "obstacle_stop_distance_m": 0.55,
        "obstacle_slow_distance_m": 0.90,
    }
    data.update(overrides)
    return SafetyConfig.from_dict(data)


def _lidar(timestamp_s: float, points: np.ndarray | None = None) -> LidarFrame:
    if points is None:
        points = np.array([[2.0, 0.0, 0.5], [3.0, 1.0, 0.5]], dtype=np.float32)
    return LidarFrame(timestamp_s=timestamp_s, points_body_xyz=points)


def _robot_state(timestamp_s: float, **kwargs) -> RobotState:
    data = {"battery_percent": 80.0}
    data.update(kwargs)
    return RobotState(timestamp_s=timestamp_s, **data)


def _obs(now: float = 10.0, **kwargs) -> RobotObservation:
    data = {
        "timestamp_s": now,
        "lidar": _lidar(now),
        "robot_state": _robot_state(now),
    }
    data.update(kwargs)
    return RobotObservation(**data)


def _supervisor(**kwargs) -> SafetySupervisor:
    return SafetySupervisor(
        command_limits=kwargs.pop("command_limits", _limits()),
        watchdog=kwargs.pop("watchdog", _watchdog()),
        safety=kwargs.pop("safety", _safety()),
        clock=kwargs.pop("clock", lambda: 10.0),
        **kwargs,
    )


def test_safety_supervisor_accepts_and_clips_enabled_fresh_command() -> None:
    sup = _supervisor()
    decision = sup.filter_command(
        Command2D(1.0, -1.0, 1.0, timestamp_s=9.95),
        _obs(),
        operator_enabled=True,
        now_s=10.0,
    )

    assert decision.ok or decision.status.health == AdapterHealth.DEGRADED
    assert decision.command.tuple == pytest.approx((0.10, -0.05, 0.20))
    assert "velocity_clipped" in decision.warnings
    assert not decision.command.is_stop


def test_safety_supervisor_stops_without_operator_enable() -> None:
    sup = _supervisor()
    decision = sup.filter_command(Command2D(0.05, 0.0, 0.0, timestamp_s=9.95), _obs(), now_s=10.0)

    assert decision.command.is_stop
    assert decision.forced_stop
    assert "operator_not_enabled" in decision.reasons


def test_safety_supervisor_stops_on_missing_required_lidar() -> None:
    sup = _supervisor(safety=_safety(require_robot_state_for_motion=False))
    observation = RobotObservation(timestamp_s=10.0, robot_state=_robot_state(10.0))

    decision = sup.filter_command(
        Command2D(0.05, 0.0, 0.0, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert decision.command.is_stop
    assert "missing_lidar" in decision.reasons


def test_safety_supervisor_stops_on_stale_lidar_and_robot_state() -> None:
    sup = _supervisor()
    observation = _obs(now=10.0, lidar=_lidar(9.0), robot_state=_robot_state(9.0))

    decision = sup.filter_command(
        Command2D(0.05, 0.0, 0.0, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert decision.command.is_stop
    assert any(r.startswith("stale_lidar") for r in decision.reasons)
    assert any(r.startswith("stale_robot_state") for r in decision.reasons)


def test_safety_supervisor_stops_on_robot_estop_fault_and_low_battery() -> None:
    sup = _supervisor()
    observation = _obs(
        now=10.0,
        robot_state=_robot_state(10.0, is_estopped=True, is_faulted=True, battery_percent=10.0),
    )

    decision = sup.filter_command(
        Command2D(0.05, 0.0, 0.0, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert decision.command.is_stop
    assert "robot_estop" in decision.reasons
    assert "robot_fault" in decision.reasons
    assert any(r.startswith("low_battery") for r in decision.reasons)


def test_safety_supervisor_stops_on_stale_policy_command() -> None:
    sup = _supervisor()
    decision = sup.filter_command(
        Command2D(0.05, 0.0, 0.0, timestamp_s=9.0),
        _obs(),
        operator_enabled=True,
        now_s=10.0,
    )

    assert decision.command.is_stop
    assert any(r.startswith("stale_policy") for r in decision.reasons)


def test_safety_supervisor_limits_acceleration_from_last_command() -> None:
    sup = _supervisor(
        command_limits=_limits(max_linear_accel_mps2=0.10, max_yaw_accel_radps2=0.20),
    )

    first = sup.filter_command(Command2D(0.0, 0.0, 0.0, timestamp_s=10.0), _obs(), operator_enabled=True, now_s=10.0)
    second = sup.filter_command(Command2D(0.10, 0.05, 0.20, timestamp_s=10.1), _obs(now=10.1), operator_enabled=True, now_s=10.1)

    assert first.command.is_stop
    assert second.command.vx == pytest.approx(0.01)
    assert second.command.vy == pytest.approx(0.01)
    assert second.command.wz == pytest.approx(0.02)
    assert "linear_accel_limited" in second.warnings
    assert "yaw_accel_limited" in second.warnings


def test_safety_supervisor_slows_for_front_obstacle() -> None:
    sup = _supervisor()
    points = np.array([[0.725, 0.0, 0.5]], dtype=np.float32)
    observation = _obs(lidar=_lidar(10.0, points))

    decision = sup.filter_command(
        Command2D(0.10, 0.0, 0.0, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert not decision.command.is_stop
    assert 0.0 < decision.command.vx < 0.10
    assert any(w.startswith("front_obstacle_slow") for w in decision.warnings)


def test_safety_supervisor_stops_for_close_front_obstacle() -> None:
    sup = _supervisor()
    points = np.array([[0.40, 0.0, 0.5]], dtype=np.float32)
    observation = _obs(lidar=_lidar(10.0, points))

    decision = sup.filter_command(
        Command2D(0.10, 0.0, 0.0, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert decision.command.is_stop
    assert any(r.startswith("front_obstacle_stop") for r in decision.reasons)


def test_safety_supervisor_does_not_front_stop_for_reverse_command() -> None:
    sup = _supervisor()
    points = np.array([[0.20, 0.0, 0.5]], dtype=np.float32)
    observation = _obs(lidar=_lidar(10.0, points))

    decision = sup.filter_command(
        Command2D(-0.05, 0.0, 0.0, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert not decision.command.is_stop
    assert decision.command.vx == pytest.approx(-0.05)


def test_safety_supervisor_allows_yaw_escape_on_front_stop_when_enabled() -> None:
    sup = _supervisor(
        config=SafetySupervisorConfig(
            allow_yaw_on_front_stop=True,
            front_stop_yaw_escape_min_wz=0.05,
            front_stop_yaw_escape_max_wz=0.60,
            front_stop_yaw_escape_max_linear_mps=0.05,
        )
    )
    points = np.array([[0.20, 0.0, 0.5]], dtype=np.float32)
    observation = _obs(lidar=_lidar(10.0, points))

    decision = sup.filter_command(
        Command2D(0.02, 0.0, 0.40, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert not decision.command.is_stop
    assert decision.command.vx == pytest.approx(0.0)
    assert decision.command.vy == pytest.approx(0.0)
    assert decision.command.wz == pytest.approx(0.20)  # clipped by command limits before safety escape
    assert not decision.reasons
    assert any(w.startswith("front_obstacle_yaw_escape") for w in decision.warnings)
    assert decision.metadata["front_yaw_escape_active"] is True


def test_safety_supervisor_yaw_escape_still_stops_when_linear_too_large() -> None:
    sup = _supervisor(
        config=SafetySupervisorConfig(
            allow_yaw_on_front_stop=True,
            front_stop_yaw_escape_max_linear_mps=0.03,
        )
    )
    points = np.array([[0.20, 0.0, 0.5]], dtype=np.float32)
    observation = _obs(lidar=_lidar(10.0, points))

    decision = sup.filter_command(
        Command2D(0.10, 0.0, 0.20, timestamp_s=9.95),
        observation,
        operator_enabled=True,
        now_s=10.0,
    )

    assert decision.command.is_stop
    assert any(r.startswith("front_obstacle_stop") for r in decision.reasons)
