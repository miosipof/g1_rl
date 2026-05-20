from __future__ import annotations

import pytest
import numpy as np

from adapters import (
    AdapterHealth,
    BridgeMode,
    Command2D,
    CommandLimitsConfig,
    DryRunCommandSink,
    LidarFrame,
    RealBridge,
    RealBridgeConfig,
    RobotObservation,
    RobotState,
    SafetyConfig,
    SafetySupervisor,
    WatchdogConfig,
    normalize_policy_command,
)


class FakeClock:
    def __init__(self, t: float = 10.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeObservationProvider:
    def __init__(self, observation: RobotObservation) -> None:
        self.observation = observation
        self.read_count = 0

    def read_observation(self) -> RobotObservation:
        self.read_count += 1
        return self.observation


class FailingObservationProvider:
    def read_observation(self) -> RobotObservation:
        raise RuntimeError("sensor failed")


class FakePolicy:
    def __init__(self, command: Command2D) -> None:
        self.command = command
        self.observations: list[RobotObservation] = []

    def step(self, observation: RobotObservation) -> Command2D:
        self.observations.append(observation)
        return self.command


class FailingPolicy:
    def step(self, observation: RobotObservation) -> Command2D:
        raise RuntimeError("policy failed")


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


def _obs(now: float = 10.0, *, lidar_time: float | None = None, state_time: float | None = None) -> RobotObservation:
    lidar = LidarFrame(
        timestamp_s=now if lidar_time is None else lidar_time,
        points_body_xyz=np.array([[2.0, 0.0, 0.5], [2.0, 1.0, 0.5]], dtype=np.float32),
    )
    state = RobotState(
        timestamp_s=now if state_time is None else state_time,
        battery_percent=80.0,
    )
    return RobotObservation(timestamp_s=now, lidar=lidar, robot_state=state)


def _supervisor(clock: FakeClock) -> SafetySupervisor:
    return SafetySupervisor(
        command_limits=_limits(),
        watchdog=_watchdog(),
        safety=_safety(),
        clock=clock,
    )


def _bridge(
    *,
    observation: RobotObservation | None = None,
    policy_command: Command2D | None = None,
    clock: FakeClock | None = None,
    sink: DryRunCommandSink | None = None,
    config: RealBridgeConfig | None = None,
) -> RealBridge:
    clock = clock or FakeClock(10.0)
    sink = sink or DryRunCommandSink()
    return RealBridge(
        observation_provider=FakeObservationProvider(observation or _obs(clock.t)),
        policy_runtime=FakePolicy(policy_command or Command2D(0.05, 0.0, 0.0, timestamp_s=clock.t)),
        safety_supervisor=_supervisor(clock),
        command_sink=sink,
        config=config or RealBridgeConfig(auto_arm_sink_on_follow=False),
        clock=clock,
    )


def test_normalize_policy_command_keeps_command2d_and_fills_timestamp() -> None:
    cmd = normalize_policy_command(Command2D(0.1, 0.0, 0.0), timestamp_s=12.0)
    assert isinstance(cmd, Command2D)
    assert cmd.timestamp_s == pytest.approx(12.0)


def test_real_bridge_follow_step_sends_safety_filtered_command() -> None:
    clock = FakeClock(10.0)
    sink = DryRunCommandSink()
    bridge = _bridge(clock=clock, sink=sink, policy_command=Command2D(1.0, 0.0, 1.0, timestamp_s=10.0))

    status = bridge.follow()
    assert status.ok
    assert bridge.mode == BridgeMode.FOLLOWING

    result = bridge.step(now_s=10.0)

    assert result.mode == BridgeMode.FOLLOWING
    assert result.command_sent
    assert result.proposed_command is not None
    assert result.safety_decision is not None
    assert result.safety_decision.command.tuple == pytest.approx((0.10, 0.0, 0.20))
    assert sink.sent_commands[-1].tuple == pytest.approx((0.10, 0.0, 0.20))


def test_real_bridge_idle_step_sends_stop_when_configured() -> None:
    sink = DryRunCommandSink()
    bridge = _bridge(sink=sink)

    result = bridge.step(now_s=10.0)

    assert result.action == "inactive_stop"
    assert result.command_result is not None
    assert result.command_result.command.is_stop
    assert sink.stop_count == 1


def test_real_bridge_pause_and_stop_send_stop() -> None:
    sink = DryRunCommandSink()
    bridge = _bridge(sink=sink)
    bridge.follow()

    pause_status = bridge.pause()
    assert pause_status.ok
    assert bridge.mode == BridgeMode.PAUSED
    assert sink.stop_count == 1

    stop_status = bridge.stop()
    assert stop_status.ok
    assert bridge.mode == BridgeMode.IDLE
    assert sink.stop_count == 2


def test_real_bridge_stale_observation_is_stopped_by_safety() -> None:
    clock = FakeClock(10.0)
    bridge = _bridge(clock=clock, observation=_obs(10.0, lidar_time=9.0), policy_command=Command2D(0.05, 0.0, 0.0, timestamp_s=10.0))
    bridge.follow()

    result = bridge.step(now_s=10.0)

    assert result.safety_decision is not None
    assert result.safety_decision.command.is_stop
    assert any(r.startswith("stale_lidar") for r in result.safety_decision.reasons)


def test_real_bridge_stop_command_handles_text_command() -> None:
    sink = DryRunCommandSink()
    bridge = _bridge(sink=sink)
    bridge.handle_command("follow")
    status = bridge.handle_command("stop")

    assert status.ok
    assert bridge.mode == BridgeMode.IDLE
    assert sink.stop_count == 1


def test_real_bridge_estop_blocks_follow_until_reset() -> None:
    bridge = _bridge()
    estop = bridge.handle_command("estop")
    assert estop.health == AdapterHealth.FAULT
    assert bridge.mode == BridgeMode.E_STOP

    follow = bridge.follow()
    assert not follow.ok
    assert bridge.mode == BridgeMode.E_STOP

    reset = bridge.reset_estop()
    assert reset.ok
    assert bridge.mode == BridgeMode.IDLE


def test_real_bridge_observation_exception_stops_and_faults() -> None:
    clock = FakeClock(10.0)
    sink = DryRunCommandSink()
    bridge = RealBridge(
        observation_provider=FailingObservationProvider(),
        policy_runtime=FakePolicy(Command2D(0.05, 0.0, 0.0, timestamp_s=10.0)),
        safety_supervisor=_supervisor(clock),
        command_sink=sink,
        config=RealBridgeConfig(auto_arm_sink_on_follow=False),
        clock=clock,
    )
    bridge.follow()

    result = bridge.step(now_s=10.0)

    assert result.mode == BridgeMode.FAULT
    assert result.status.health == AdapterHealth.FAULT
    assert result.command_result is not None
    assert result.command_result.command.is_stop
    assert sink.stop_count == 1


def test_real_bridge_policy_exception_stops_and_faults() -> None:
    clock = FakeClock(10.0)
    sink = DryRunCommandSink()
    bridge = RealBridge(
        observation_provider=FakeObservationProvider(_obs(10.0)),
        policy_runtime=FailingPolicy(),
        safety_supervisor=_supervisor(clock),
        command_sink=sink,
        config=RealBridgeConfig(auto_arm_sink_on_follow=False),
        clock=clock,
    )
    bridge.follow()

    result = bridge.step(now_s=10.0)

    assert result.mode == BridgeMode.FAULT
    assert result.status.health == AdapterHealth.FAULT
    assert result.command_result is not None
    assert result.command_result.command.is_stop
    assert sink.stop_count == 1
