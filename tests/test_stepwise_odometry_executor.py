from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapters import AdapterCommandResult, AdapterHealth, AdapterStatus, Command2D, Pose2D
from adapters.real_unitree_odometry import UnitreeOdometrySample, UnitreeOdometrySnapshot
from adapters.stepwise_executor import (
    OdometryActionExecutor,
    OdometryActionExecutorConfig,
    TerminalLocomotionFault,
    real_motion_ready,
    wait_until_stationary,
)
from g1_nav_rl.navigation.stepwise import MotionCommand, StepwiseAction, StepwiseActionKind


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _snap(seq: int, t: float, *, x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, yaw_speed=0.0, epoch=0):
    raw = UnitreeOdometrySample(t, seq, epoch, x, y, 0.72, yaw, vx, vy, 0.0, yaw_speed)
    return UnitreeOdometrySnapshot(raw, Pose2D(x, y, yaw), seq, epoch, AdapterHealth.OK)


class _Odom:
    def __init__(self, clock: _Clock, samples: list[UnitreeOdometrySnapshot]) -> None:
        self.clock = clock
        self.samples = list(samples)
        self.current = self.samples.pop(0)

    def snapshot(self):
        return self.current

    def wait_for_snapshot(self, *, newer_than_sequence: int, deadline_s: float):
        if not self.samples:
            self.clock.now = deadline_s
            raise TimeoutError
        self.current = self.samples.pop(0)
        self.clock.now = self.current.raw.receive_time_s
        return self.current


class _Sink:
    def __init__(self, *, reject_send=False, reject_stop=False) -> None:
        self.config = SimpleNamespace(command_period_s=0.1)
        self.reject_send = reject_send
        self.reject_stop = reject_stop
        self.sends = 0
        self.stops = 0

    def send_command(self, command, *, require_state_gate=True):
        self.sends += 1
        return AdapterCommandResult(
            not self.reject_send,
            command,
            AdapterStatus("sink", AdapterHealth.FAULT if self.reject_send else AdapterHealth.OK),
        )

    def stop(self, *, timestamp_s=None):
        self.stops += 1
        command = Command2D.stop(timestamp_s=timestamp_s)
        return AdapterCommandResult(
            not self.reject_stop,
            command,
            AdapterStatus("sink", AdapterHealth.FAULT if self.reject_stop else AdapterHealth.OK),
        )


def _forward(distance=0.10):
    return StepwiseAction(
        StepwiseActionKind.FORWARD,
        MotionCommand(0.1, 0.0, 0.0),
        duration_s=1.0,
        distance_m=distance,
    )


def _turn(angle=0.30):
    return StepwiseAction(
        StepwiseActionKind.TURN,
        MotionCommand(0.0, 0.0, 0.3 if angle > 0.0 else -0.3),
        duration_s=1.0,
        angle_rad=angle,
    )


def test_forward_completes_from_measured_progress_and_unique_settle_samples() -> None:
    clock = _Clock()
    samples = [
        _snap(1, 0.0),
        _snap(2, 0.1, x=0.04, vx=0.1),
        _snap(3, 0.2, x=0.08, vx=0.1),
        _snap(4, 0.21, x=0.081, vx=0.0),
        _snap(5, 0.22, x=0.081, vx=0.0),
        _snap(6, 0.23, x=0.081, vx=0.0),
    ]
    sink = _Sink()
    executor = OdometryActionExecutor(sink=sink, odometry=_Odom(clock, samples), clock=clock)
    result = executor.execute(action=_forward(), command=Command2D(0.1, 0.0, 0.0, duration_s=1.0))
    assert result.completed
    assert result.reason == "completed"
    assert result.achieved_distance_m == pytest.approx(0.081)
    assert result.odom_sample_count == 5
    assert result.stop_accepted


def test_small_action_cannot_complete_at_zero_displacement() -> None:
    clock = _Clock()
    samples = [_snap(1, 0.0), _snap(2, 0.1), _snap(3, 0.8)]
    cfg = OdometryActionExecutorConfig(no_progress_timeout_s=0.2, settle_samples=1)
    result = OdometryActionExecutor(sink=_Sink(), odometry=_Odom(clock, samples), config=cfg, clock=clock).execute(
        action=_forward(0.02), command=Command2D(0.1, 0.0, 0.0, duration_s=1.0)
    )
    assert not result.completed
    assert result.reason == "no_odometric_progress"


def test_turn_accumulates_wrapped_yaw_from_unique_samples() -> None:
    clock = _Clock()
    samples = [
        _snap(1, 0.0, yaw=3.10),
        _snap(2, 0.1, yaw=-3.08, yaw_speed=0.3),
        _snap(3, 0.2, yaw=-2.93, yaw_speed=0.3),
        _snap(4, 0.21, yaw=-2.91, yaw_speed=0.3),
        _snap(5, 0.22, yaw=-2.90),
        _snap(6, 0.23, yaw=-2.90),
        _snap(7, 0.24, yaw=-2.90),
    ]
    executor = OdometryActionExecutor(sink=_Sink(), odometry=_Odom(clock, samples), clock=clock)
    result = executor.execute(action=_turn(), command=Command2D(0.0, 0.0, 0.3, duration_s=1.0))
    assert result.completed
    assert result.achieved_angle_rad == pytest.approx(0.283185, abs=1e-5)


def test_excess_lateral_motion_fails_and_still_settles() -> None:
    clock = _Clock()
    samples = [
        _snap(1, 0.0),
        _snap(2, 0.1, x=0.01, y=0.13, vx=0.1),
        _snap(3, 0.11, x=0.01, y=0.13),
        _snap(4, 0.12, x=0.01, y=0.13),
        _snap(5, 0.13, x=0.01, y=0.13),
    ]
    result = OdometryActionExecutor(sink=_Sink(), odometry=_Odom(clock, samples), clock=clock).execute(
        action=_forward(), command=Command2D(0.1, 0.0, 0.0, duration_s=1.0)
    )
    assert not result.completed
    assert result.reason == "lateral_deviation"
    assert result.stop_accepted


def test_stop_rejection_is_terminal() -> None:
    clock = _Clock()
    odom = _Odom(clock, [_snap(1, 0.0), _snap(2, 0.1, x=0.09)])
    executor = OdometryActionExecutor(sink=_Sink(reject_stop=True), odometry=odom, clock=clock)
    with pytest.raises(TerminalLocomotionFault):
        executor.execute(action=_forward(), command=Command2D(0.1, 0.0, 0.0, duration_s=1.0))


def test_settle_requires_new_sequences() -> None:
    clock = _Clock()
    odom = _Odom(clock, [_snap(1, 0.0), _snap(2, 0.1)])
    cfg = OdometryActionExecutorConfig(settle_samples=3, settle_timeout_s=0.2)
    result = wait_until_stationary(
        odometry=odom,
        newer_than_sequence=1,
        config=cfg,
        clock=clock,
    )
    assert not result.stationary
    assert result.sample_count == 1


def test_stale_odometry_is_an_explicit_motion_gate() -> None:
    ready, reason = real_motion_ready(
        odom_status=AdapterStatus("odom", AdapterHealth.STALE),
        target_fresh=True,
        stationary_confirmed=True,
    )
    assert not ready
    assert reason == "odometry_not_ok"
