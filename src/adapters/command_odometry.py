"""Command-integrated odometry for staged sim-to-real tests.

This module provides a deliberately simple pose estimator that integrates the
last safety-filtered body-frame command as if the robot executed it exactly.  It
is not a replacement for real odometry/SLAM; it is a short-horizon consistency
helper for local navigation, waypoint progress, and debug metadata while the
real Unitree state/odometry adapter is still pending.

No hardware SDKs are imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

from g1_nav_rl.navigation.simple_nav import (
    Pose2D,
    Target2D,
    angle_wrap,
    body_to_world_xy,
    integrate_body_command,
)

from .types import AdapterHealth, AdapterStatus, BridgeMode, Command2D, RobotObservation, RobotState


Clock = Callable[[], float]
TargetHintFrame = Literal["world", "body"]


@runtime_checkable
class ObservationProviderLike(Protocol):
    """Small protocol for wrapping existing observation providers."""

    def read_observation(self) -> RobotObservation:
        ...


@dataclass(frozen=True)
class CommandOdometryConfig:
    """Configuration for command-integrated odometry.

    ``max_dt_s`` caps one integration step so a stalled process does not create a
    large artificial pose jump when it resumes. Set to ``None`` to disable the
    cap.
    """

    initial_pose: Pose2D = field(default_factory=lambda: Pose2D(0.0, 0.0, 0.0))
    max_dt_s: float | None = 2.0
    source: str = "command_odometry"

    def __post_init__(self) -> None:
        if self.max_dt_s is not None and float(self.max_dt_s) <= 0.0:
            raise ValueError("CommandOdometryConfig.max_dt_s must be positive or None")
        if not self.source:
            raise ValueError("CommandOdometryConfig.source must be non-empty")


@dataclass(frozen=True)
class CommandOdometrySnapshot:
    """Serializable estimator state for debug metadata."""

    pose: Pose2D
    last_update_s: float | None
    active_command: Command2D
    last_dt_s: float | None
    integrated_steps: int
    source: str = "command_odometry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pose_xy_yaw": [float(self.pose.x), float(self.pose.y), float(self.pose.yaw)],
            "last_update_s": None if self.last_update_s is None else float(self.last_update_s),
            "active_command": [
                float(self.active_command.vx),
                float(self.active_command.vy),
                float(self.active_command.wz),
            ],
            "last_dt_s": None if self.last_dt_s is None else float(self.last_dt_s),
            "integrated_steps": int(self.integrated_steps),
        }


class CommandOdometryEstimator:
    """Integrate the currently active body-frame command into a pose estimate.

    Usage pattern in a control loop:

    1. ``advance_to(observation.timestamp_s)`` before policy evaluation;
    2. inject ``estimator.pose`` into the observation;
    3. after command send/log, call ``set_active_command(safe_command, now_s)``.

    The estimator integrates the previously active command over the elapsed time
    since the last update. This approximates "the last command remained active
    until replaced".
    """

    def __init__(self, config: CommandOdometryConfig | None = None) -> None:
        self.config = config or CommandOdometryConfig()
        self.pose: Pose2D = self.config.initial_pose
        self.last_update_s: float | None = None
        self.active_command: Command2D = Command2D.stop(source=f"{self.config.source}_initial_stop")
        self.last_dt_s: float | None = None
        self.integrated_steps: int = 0

    def reset(self, *, pose: Pose2D | None = None, timestamp_s: float | None = None) -> None:
        self.pose = self.config.initial_pose if pose is None else pose
        self.last_update_s = None if timestamp_s is None else float(timestamp_s)
        self.active_command = Command2D.stop(timestamp_s=timestamp_s, source=f"{self.config.source}_reset_stop")
        self.last_dt_s = None
        self.integrated_steps = 0

    def advance_to(self, timestamp_s: float) -> Pose2D:
        """Integrate the active command until ``timestamp_s`` and return pose."""

        now = float(timestamp_s)
        if self.last_update_s is None:
            self.last_update_s = now
            self.last_dt_s = 0.0
            return self.pose

        dt = max(0.0, now - float(self.last_update_s))
        if self.config.max_dt_s is not None:
            dt = min(dt, float(self.config.max_dt_s))
        if dt > 0.0 and not self.active_command.is_stop:
            self.pose = integrate_body_command(self.pose, self.active_command.to_motion_command(), dt)
            # Normalize defensively even though integrate_body_command already wraps.
            self.pose = Pose2D(self.pose.x, self.pose.y, angle_wrap(self.pose.yaw))
            self.integrated_steps += 1
        self.last_update_s = now
        self.last_dt_s = dt
        return self.pose

    def set_active_command(self, command: Command2D, *, timestamp_s: float) -> Pose2D:
        """Advance to ``timestamp_s`` and make ``command`` active for future integration."""

        pose = self.advance_to(timestamp_s)
        self.active_command = command
        return pose

    def set_stop(self, *, timestamp_s: float, source: str | None = None) -> Pose2D:
        return self.set_active_command(
            Command2D.stop(timestamp_s=timestamp_s, source=source or f"{self.config.source}_stop"),
            timestamp_s=timestamp_s,
        )

    def snapshot(self) -> CommandOdometrySnapshot:
        return CommandOdometrySnapshot(
            pose=self.pose,
            last_update_s=self.last_update_s,
            active_command=self.active_command,
            last_dt_s=self.last_dt_s,
            integrated_steps=self.integrated_steps,
            source=self.config.source,
        )


@dataclass
class CommandOdometryObservationProvider:
    """Wrap an observation provider and inject command-integrated pose.

    ``target_hint_frame`` controls how an existing ``RobotObservation.target_hint``
    is interpreted before forwarding to the policy:

    - ``"world"``: keep target hint unchanged;
    - ``"body"``: treat target hint as current body-frame coordinates and convert
      it into world frame using the command-odometry pose.

    The live detect-follow script's current target provider emits local/body-frame
    target hints, so ``"body"`` is useful when command odometry is explicitly
    enabled. Future target-propagation code should replace this simple adapter.
    """

    base_provider: ObservationProviderLike
    estimator: CommandOdometryEstimator = field(default_factory=CommandOdometryEstimator)
    target_hint_frame: TargetHintFrame = "world"
    clock: Clock = time.monotonic
    create_robot_state_if_missing: bool = False
    source_name: str = "CommandOdometryObservationProvider"

    def __post_init__(self) -> None:
        if self.target_hint_frame not in {"world", "body"}:
            raise ValueError("target_hint_frame must be 'world' or 'body'")

    def read_observation(self) -> RobotObservation:
        obs = self.base_provider.read_observation()
        pose = self.estimator.advance_to(float(obs.timestamp_s))
        target_hint = self._convert_target_hint(obs.target_hint, pose)
        robot_state = self._with_pose_robot_state(obs.robot_state, pose, obs.timestamp_s)
        metadata = dict(obs.metadata)
        metadata["command_odometry"] = self.estimator.snapshot().to_dict()
        metadata["command_odometry_target_hint_frame_in"] = self.target_hint_frame
        metadata["command_odometry_target_hint_converted"] = (
            obs.target_hint is not None and self.target_hint_frame == "body"
        )
        statuses = list(obs.statuses)
        statuses.append(
            AdapterStatus(
                name="command_odometry",
                health=AdapterHealth.OK,
                timestamp_s=obs.timestamp_s,
                message="command-integrated pose injected",
                details=self.estimator.snapshot().to_dict(),
            )
        )
        return replace(
            obs,
            pose=pose,
            robot_state=robot_state,
            target_hint=target_hint,
            statuses=tuple(statuses),
            metadata=metadata,
        )

    def set_active_command(self, command: Command2D, *, timestamp_s: float | None = None) -> Pose2D:
        ts = float(self.clock() if timestamp_s is None else timestamp_s)
        return self.estimator.set_active_command(command, timestamp_s=ts)

    def set_stop(self, *, timestamp_s: float | None = None, source: str | None = None) -> Pose2D:
        ts = float(self.clock() if timestamp_s is None else timestamp_s)
        return self.estimator.set_stop(timestamp_s=ts, source=source)

    def update_from_bridge_result(self, result: Any, *, timestamp_s: float | None = None) -> Pose2D:
        """Activate the command that was actually accepted by the command sink.

        If no command was accepted, switch odometry to a stop command. This keeps
        command odometry based on the sent/safe command rather than the proposed
        policy command.
        """

        ts = timestamp_s
        if ts is None:
            status = getattr(result, "status", None)
            ts = None if status is None else getattr(status, "timestamp_s", None)
        if ts is None:
            cmd_result = getattr(result, "command_result", None)
            cmd = None if cmd_result is None else getattr(cmd_result, "command", None)
            ts = None if cmd is None else getattr(cmd, "timestamp_s", None)
        if ts is None:
            ts = self.clock()

        command_result = getattr(result, "command_result", None)
        if command_result is not None and bool(getattr(command_result, "accepted", False)):
            command = command_result.command
        else:
            command = Command2D.stop(timestamp_s=float(ts), source="command_odometry_unaccepted_stop")
        return self.set_active_command(command, timestamp_s=float(ts))

    def snapshot(self) -> CommandOdometrySnapshot:
        return self.estimator.snapshot()

    def _convert_target_hint(self, target: Target2D | None, pose: Pose2D) -> Target2D | None:
        if target is None or self.target_hint_frame == "world":
            return target
        x_w, y_w = body_to_world_xy(target.xy, pose)
        return Target2D(x_w, y_w)

    def _with_pose_robot_state(
        self,
        robot_state: RobotState | None,
        pose: Pose2D,
        timestamp_s: float,
    ) -> RobotState | None:
        details = {"command_odometry": self.estimator.snapshot().to_dict()}
        if robot_state is None:
            if not self.create_robot_state_if_missing:
                return None
            return RobotState(
                timestamp_s=float(timestamp_s),
                pose=pose,
                mode="command_odometry_only",
                details={"synthetic": True, **details},
            )
        merged_details = {**dict(robot_state.details), **details}
        return replace(robot_state, pose=pose, details=merged_details)


__all__ = [
    "CommandOdometryConfig",
    "CommandOdometryEstimator",
    "CommandOdometryObservationProvider",
    "CommandOdometrySnapshot",
]
