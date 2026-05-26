from __future__ import annotations

import pytest

from adapters import (
    AdapterCommandResult,
    AdapterHealth,
    AdapterStatus,
    Command2D,
    CommandOdometryConfig,
    CommandOdometryEstimator,
    CommandOdometryObservationProvider,
    Pose2D,
    RobotObservation,
    RobotState,
    Target2D,
)


class FakeProvider:
    def __init__(self, *observations: RobotObservation) -> None:
        self.observations = list(observations)
        self.index = 0

    def read_observation(self) -> RobotObservation:
        if self.index >= len(self.observations):
            return self.observations[-1]
        obs = self.observations[self.index]
        self.index += 1
        return obs


class FakeBridgeResult:
    def __init__(self, *, command_result: AdapterCommandResult | None, timestamp_s: float) -> None:
        self.command_result = command_result
        self.status = AdapterStatus(name="bridge", health=AdapterHealth.OK, timestamp_s=timestamp_s)


def test_command_odometry_estimator_integrates_active_command() -> None:
    est = CommandOdometryEstimator(CommandOdometryConfig(max_dt_s=None))

    est.advance_to(10.0)
    est.set_active_command(Command2D(1.0, 0.0, 0.0, timestamp_s=10.0), timestamp_s=10.0)
    pose = est.advance_to(11.5)

    assert pose.x == pytest.approx(1.5)
    assert pose.y == pytest.approx(0.0)
    assert pose.yaw == pytest.approx(0.0)
    assert est.snapshot().integrated_steps == 1
    assert est.snapshot().last_dt_s == pytest.approx(1.5)


def test_command_odometry_estimator_integrates_body_frame_with_yaw() -> None:
    est = CommandOdometryEstimator(CommandOdometryConfig(initial_pose=Pose2D(0.0, 0.0, 1.57079632679), max_dt_s=None))

    est.advance_to(0.0)
    est.set_active_command(Command2D(1.0, 0.0, 0.0), timestamp_s=0.0)
    pose = est.advance_to(1.0)

    assert abs(pose.x) < 1e-6
    assert pose.y == pytest.approx(1.0, abs=1e-6)


def test_command_odometry_caps_dt() -> None:
    est = CommandOdometryEstimator(CommandOdometryConfig(max_dt_s=0.5))

    est.advance_to(0.0)
    est.set_active_command(Command2D(1.0, 0.0, 0.0), timestamp_s=0.0)
    pose = est.advance_to(10.0)

    assert pose.x == pytest.approx(0.5)
    assert est.snapshot().last_dt_s == pytest.approx(0.5)


def test_command_odometry_observation_provider_injects_pose_and_metadata() -> None:
    base = FakeProvider(
        RobotObservation(
            timestamp_s=1.0,
            robot_state=RobotState(timestamp_s=1.0, mode="synthetic", details={"synthetic": True}),
        ),
        RobotObservation(
            timestamp_s=2.0,
            robot_state=RobotState(timestamp_s=2.0, mode="synthetic", details={"synthetic": True}),
        ),
    )
    provider = CommandOdometryObservationProvider(
        base_provider=base,
        estimator=CommandOdometryEstimator(CommandOdometryConfig(max_dt_s=None)),
    )

    obs0 = provider.read_observation()
    provider.set_active_command(Command2D(0.5, 0.0, 0.0), timestamp_s=1.0)
    obs1 = provider.read_observation()

    assert obs0.pose == Pose2D(0.0, 0.0, 0.0)
    assert obs1.pose is not None
    assert obs1.pose.x == pytest.approx(0.5)
    assert obs1.robot_state is not None
    assert obs1.robot_state.pose == obs1.pose
    assert obs1.metadata["command_odometry"]["pose_xy_yaw"][0] == pytest.approx(0.5)
    assert any(status.name == "command_odometry" for status in obs1.statuses)


def test_command_odometry_provider_converts_body_target_hint_to_world() -> None:
    base = FakeProvider(RobotObservation(timestamp_s=0.0, target_hint=Target2D(1.0, 0.0)))
    provider = CommandOdometryObservationProvider(
        base_provider=base,
        estimator=CommandOdometryEstimator(CommandOdometryConfig(initial_pose=Pose2D(2.0, 3.0, 0.0))),
        target_hint_frame="body",
    )

    obs = provider.read_observation()

    assert obs.target_hint is not None
    assert obs.target_hint.xy == pytest.approx((3.0, 3.0))
    assert obs.metadata["command_odometry_target_hint_converted"] is True


def test_command_odometry_update_from_bridge_result_uses_accepted_command() -> None:
    provider = CommandOdometryObservationProvider(
        base_provider=FakeProvider(RobotObservation(timestamp_s=0.0)),
        estimator=CommandOdometryEstimator(CommandOdometryConfig(max_dt_s=None)),
    )
    provider.read_observation()
    cmd = Command2D(0.25, 0.0, 0.0, timestamp_s=0.0)
    result = FakeBridgeResult(
        command_result=AdapterCommandResult(
            accepted=True,
            command=cmd,
            status=AdapterStatus(name="sink", health=AdapterHealth.OK, timestamp_s=0.0),
        ),
        timestamp_s=0.0,
    )

    provider.update_from_bridge_result(result, timestamp_s=0.0)
    provider.estimator.advance_to(2.0)

    assert provider.snapshot().pose.x == pytest.approx(0.5)


def test_command_odometry_update_from_bridge_result_stops_on_rejected_command() -> None:
    provider = CommandOdometryObservationProvider(
        base_provider=FakeProvider(RobotObservation(timestamp_s=0.0)),
        estimator=CommandOdometryEstimator(CommandOdometryConfig(max_dt_s=None)),
    )
    provider.read_observation()
    provider.set_active_command(Command2D(1.0, 0.0, 0.0), timestamp_s=0.0)
    rejected = FakeBridgeResult(
        command_result=AdapterCommandResult(
            accepted=False,
            command=Command2D(1.0, 0.0, 0.0),
            status=AdapterStatus(name="sink", health=AdapterHealth.DEGRADED, timestamp_s=1.0),
        ),
        timestamp_s=1.0,
    )

    provider.update_from_bridge_result(rejected, timestamp_s=1.0)
    provider.estimator.advance_to(2.0)

    # It integrated the previously active command until t=1.0, then stopped.
    assert provider.snapshot().pose.x == pytest.approx(1.0)
    assert provider.snapshot().active_command.is_stop
