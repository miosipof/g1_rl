from __future__ import annotations

import math

import pytest

from adapters import (
    AdapterHealth,
    Command2D,
    IsaacCommandSink,
    IsaacOdometryAdapter,
    IsaacOdometryConfig,
    OdometryActionExecutor,
    OdometryActionExecutorConfig,
)
from g1_nav_rl.navigation.stepwise import MotionCommand, StepwiseAction, StepwiseActionKind


def _push(
    odometry: IsaacOdometryAdapter,
    timestamp_s: float,
    *,
    x: float = 0.0,
    yaw: float = 0.0,
    vx: float = 0.0,
    yaw_speed: float = 0.0,
) -> None:
    odometry.push_ground_truth(
        timestamp_s=timestamp_s,
        x=x,
        y=0.0,
        z=0.72,
        yaw=yaw,
        vx=vx,
        vy=0.0,
        yaw_speed=yaw_speed,
    )


def test_isaac_odometry_exposes_relative_unitree_compatible_snapshot() -> None:
    odometry = IsaacOdometryAdapter(IsaacOdometryConfig(sample_period_s=0.0))
    _push(odometry, 0.0, x=1.0, yaw=0.2)
    odometry.capture_origin()
    _push(odometry, 0.02, x=1.1, yaw=0.3, vx=0.2, yaw_speed=0.1)

    snapshot = odometry.snapshot()

    assert snapshot.health == AdapterHealth.OK
    assert snapshot.sequence == 2
    assert snapshot.raw.source == "isaac_odometry"
    assert math.hypot(snapshot.relative_pose.x, snapshot.relative_pose.y) == pytest.approx(0.1)
    assert snapshot.relative_pose.yaw == pytest.approx(0.1)
    assert odometry.status().details["published_count"] == 2


def test_lag_releases_an_older_ground_truth_sample() -> None:
    odometry = IsaacOdometryAdapter(IsaacOdometryConfig(sample_period_s=0.0, lag_s=0.10))
    _push(odometry, 0.00, x=0.0)
    _push(odometry, 0.05, x=0.5)
    assert odometry.status().health == AdapterHealth.UNAVAILABLE

    _push(odometry, 0.10, x=1.0)
    assert odometry.snapshot().raw.x == pytest.approx(0.0)
    assert odometry.status().details["measurement_age_s"] == pytest.approx(0.10)

    _push(odometry, 0.15, x=1.5)
    assert odometry.snapshot().raw.x == pytest.approx(0.5)


def test_noise_is_reproducible_with_seed() -> None:
    config = IsaacOdometryConfig(
        sample_period_s=0.0,
        position_noise_std_m=0.02,
        yaw_noise_std_rad=0.01,
        seed=42,
    )
    first = IsaacOdometryAdapter(config)
    second = IsaacOdometryAdapter(config)
    _push(first, 0.0, x=1.0, yaw=0.3)
    _push(second, 0.0, x=1.0, yaw=0.3)

    assert first.snapshot().raw.x == pytest.approx(second.snapshot().raw.x)
    assert first.snapshot().raw.yaw == pytest.approx(second.snapshot().raw.yaw)
    assert first.snapshot().raw.x != pytest.approx(1.0)


def test_yaw_scale_remains_continuous_across_wrap_boundary() -> None:
    odometry = IsaacOdometryAdapter(IsaacOdometryConfig(sample_period_s=0.0, yaw_scale=1.05))
    _push(odometry, 0.0, yaw=math.pi - 0.02)
    first_yaw = odometry.snapshot().raw.yaw
    _push(odometry, 0.02, yaw=-math.pi + 0.02)
    second_yaw = odometry.snapshot().raw.yaw

    wrapped_delta = math.atan2(math.sin(second_yaw - first_yaw), math.cos(second_yaw - first_yaw))
    assert wrapped_delta == pytest.approx(0.042)


def test_complete_dropout_times_out_without_publishing() -> None:
    odometry = IsaacOdometryAdapter(
        IsaacOdometryConfig(sample_period_s=0.0, dropout_probability=1.0),
    )
    _push(odometry, 0.0)

    with pytest.raises(TimeoutError):
        odometry.wait_for_snapshot(newer_than_sequence=-1, deadline_s=0.01)
    assert odometry.status().details["dropped_count"] == 1


def test_shared_executor_completes_using_pumped_sim_odometry() -> None:
    state = {"time": 0.0, "x": 0.0, "command": Command2D.stop(timestamp_s=0.0)}
    odometry = IsaacOdometryAdapter(IsaacOdometryConfig(sample_period_s=0.02, lag_s=0.04))

    def set_command(command: Command2D) -> None:
        state["command"] = command

    def pump() -> None:
        dt = 0.02
        command = state["command"]
        state["time"] += dt
        state["x"] += command.vx * dt
        _push(
            odometry,
            state["time"],
            x=state["x"],
            vx=command.vx,
            yaw_speed=command.wz,
        )

    odometry.set_pump(pump)
    _push(odometry, 0.0)
    odometry.wait_for_snapshot(newer_than_sequence=-1, deadline_s=0.2)
    odometry.capture_origin()
    sink = IsaacCommandSink(set_command, clock=odometry.clock, command_period_s=0.10)
    executor = OdometryActionExecutor(
        sink=sink,
        odometry=odometry,
        clock=odometry.clock,
        config=OdometryActionExecutorConfig(settle_samples=2, settle_timeout_s=0.5),
    )
    action = StepwiseAction(
        kind=StepwiseActionKind.FORWARD,
        command=MotionCommand(0.1, 0.0, 0.0),
        duration_s=1.0,
        distance_m=0.1,
    )

    result = executor.execute(
        action=action,
        command=Command2D(0.1, 0.0, 0.0, duration_s=1.0),
    )

    assert result.completed
    assert result.reason == "completed"
    assert result.achieved_distance_m is not None
    assert result.achieved_distance_m >= 0.08
    assert result.stop_accepted
    assert state["command"].is_stop
    # Measurement lag lets the simulated robot overshoot the 0.08 m completion
    # threshold before the delayed measurement triggers a stop.
    assert state["x"] > 0.08


def test_isaac_command_sink_reports_callback_failure() -> None:
    def fail(command: Command2D) -> None:
        raise RuntimeError(f"cannot apply {command.tuple}")

    sink = IsaacCommandSink(fail, clock=lambda: 1.0)
    result = sink.send_command(Command2D(0.1, 0.0, 0.0))
    assert not result.accepted
    assert result.status.health == AdapterHealth.FAULT
