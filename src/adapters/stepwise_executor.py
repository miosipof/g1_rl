"""Closed-loop execution of stepwise primitives using measured G1 odometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable

from g1_nav_rl.navigation.stepwise import StepwiseAction, StepwiseActionKind

from .real_unitree_odometry import UnitreeOdometrySnapshot, angle_wrap
from .types import AdapterCommandResult, AdapterHealth, AdapterStatus, Command2D, Pose2D


Clock = Callable[[], float]


@dataclass(frozen=True)
class OdometryActionExecutorConfig:
    distance_tolerance_m: float = 0.02
    distance_tolerance_fraction: float = 0.20
    min_completion_progress_m: float = 0.005
    yaw_tolerance_rad: float = math.radians(2.0)
    yaw_tolerance_fraction: float = 0.20
    min_completion_progress_rad: float = math.radians(0.5)
    max_lateral_error_m: float = 0.12
    wrong_direction_distance_m: float = 0.02
    wrong_direction_yaw_rad: float = math.radians(2.0)
    stale_after_s: float = 0.10
    no_progress_timeout_s: float = 0.75
    min_progress_m: float = 0.01
    min_progress_rad: float = math.radians(1.0)
    timeout_scale: float = 2.0
    timeout_margin_s: float = 0.50
    settle_linear_speed_mps: float = 0.03
    settle_yaw_speed_radps: float = 0.05
    settle_samples: int = 3
    settle_timeout_s: float = 1.0
    command_refresh_period_s: float | None = None

    def __post_init__(self) -> None:
        positive = (
            "distance_tolerance_m", "distance_tolerance_fraction",
            "min_completion_progress_m", "yaw_tolerance_rad",
            "yaw_tolerance_fraction", "min_completion_progress_rad",
            "max_lateral_error_m", "wrong_direction_distance_m",
            "wrong_direction_yaw_rad", "stale_after_s",
            "no_progress_timeout_s", "min_progress_m", "min_progress_rad",
            "timeout_scale", "timeout_margin_s", "settle_linear_speed_mps",
            "settle_yaw_speed_radps", "settle_timeout_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if int(self.settle_samples) < 1:
            raise ValueError("settle_samples must be >= 1")
        object.__setattr__(self, "settle_samples", int(self.settle_samples))
        if self.command_refresh_period_s is not None:
            period = float(self.command_refresh_period_s)
            if not math.isfinite(period) or period <= 0.0:
                raise ValueError("command_refresh_period_s must be positive and finite")
            object.__setattr__(self, "command_refresh_period_s", period)


@dataclass(frozen=True)
class ActionExecutionResult:
    completed: bool
    reason: str
    requested_distance_m: float | None = None
    achieved_distance_m: float | None = None
    requested_angle_rad: float | None = None
    achieved_angle_rad: float | None = None
    lateral_error_m: float | None = None
    elapsed_s: float = 0.0
    settle_time_s: float = 0.0
    start_pose: Pose2D | None = None
    stop_pose: Pose2D | None = None
    settled_pose: Pose2D | None = None
    start_sequence: int | None = None
    stop_sequence: int | None = None
    settled_sequence: int | None = None
    start_epoch: int | None = None
    end_epoch: int | None = None
    send_count: int = 0
    odom_sample_count: int = 0
    stop_accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("start_pose", "stop_pose", "settled_pose"):
            pose = getattr(self, key)
            value[key] = None if pose is None else [pose.x, pose.y, pose.yaw]
        return value


@dataclass(frozen=True)
class StationaryWaitResult:
    stationary: bool
    reason: str
    snapshot: UnitreeOdometrySnapshot | None
    sample_count: int
    elapsed_s: float


class TerminalLocomotionFault(RuntimeError):
    """A stop command was not explicitly accepted; autonomous motion must end."""

    def __init__(self, message: str, *, stop_result: AdapterCommandResult | None = None) -> None:
        super().__init__(message)
        self.stop_result = stop_result


def real_motion_ready(
    *,
    odom_status: AdapterStatus,
    target_fresh: bool,
    stationary_confirmed: bool,
) -> tuple[bool, str]:
    if odom_status.health != AdapterHealth.OK:
        return False, "odometry_not_ok"
    if not target_fresh:
        return False, "target_not_fresh"
    if not stationary_confirmed:
        return False, "robot_not_stationary"
    return True, "ok"


def wait_until_stationary(
    *,
    odometry: Any,
    newer_than_sequence: int,
    config: OdometryActionExecutorConfig,
    clock: Clock = time.monotonic,
    timeout_s: float | None = None,
) -> StationaryWaitResult:
    start_time = float(clock())
    deadline = start_time + float(config.settle_timeout_s if timeout_s is None else timeout_s)
    sequence = int(newer_than_sequence)
    consecutive = 0
    sample_count = 0
    latest: UnitreeOdometrySnapshot | None = None
    while float(clock()) < deadline:
        try:
            snapshot = odometry.wait_for_snapshot(newer_than_sequence=sequence, deadline_s=deadline)
        except TimeoutError:
            break
        if snapshot.sequence <= sequence:
            continue
        sequence = snapshot.sequence
        latest = snapshot
        sample_count += 1
        now = float(clock())
        if snapshot.raw.age_s(now) > config.stale_after_s:
            return StationaryWaitResult(False, "odom_stale", snapshot, sample_count, now - start_time)
        linear_speed = math.hypot(snapshot.raw.vx, snapshot.raw.vy)
        if (
            linear_speed <= config.settle_linear_speed_mps
            and abs(snapshot.raw.yaw_speed) <= config.settle_yaw_speed_radps
        ):
            consecutive += 1
            if consecutive >= config.settle_samples:
                return StationaryWaitResult(True, "stationary", snapshot, sample_count, now - start_time)
        else:
            consecutive = 0
    return StationaryWaitResult(False, "settle_timeout", latest, sample_count, float(clock()) - start_time)


class OdometryActionExecutor:
    def __init__(
        self,
        *,
        sink: Any,
        odometry: Any,
        config: OdometryActionExecutorConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.sink = sink
        self.odometry = odometry
        self.config = config or OdometryActionExecutorConfig()
        self._clock = clock or time.monotonic
        configured_period = self.config.command_refresh_period_s
        sink_period = getattr(getattr(sink, "config", None), "command_period_s", 0.10)
        self.command_refresh_period_s = float(configured_period or sink_period)

    def checked_stop(self) -> AdapterCommandResult:
        result = self.sink.stop(timestamp_s=float(self._clock()))
        if not result.accepted:
            raise TerminalLocomotionFault("stop command was not explicitly accepted", stop_result=result)
        return result

    def wait_until_stationary(self, *, newer_than_sequence: int, timeout_s: float | None = None) -> StationaryWaitResult:
        return wait_until_stationary(
            odometry=self.odometry,
            newer_than_sequence=newer_than_sequence,
            config=self.config,
            clock=self._clock,
            timeout_s=timeout_s,
        )

    def execute(
        self,
        *,
        action: StepwiseAction,
        command: Command2D,
        require_state_gate: bool = True,
    ) -> ActionExecutionResult:
        if action.is_stop or command.is_stop:
            stop = self.checked_stop()
            return ActionExecutionResult(True, "completed", stop_accepted=stop.accepted)

        start = self.odometry.snapshot()
        now = float(self._clock())
        if start.health != AdapterHealth.OK or start.raw.age_s(now) > self.config.stale_after_s:
            stop = self.checked_stop()
            return self._result(start=start, stop=start, settled=None, reason="odom_stale", stop_accepted=stop.accepted)

        required_progress = self._required_progress(action)
        deadline = now + float(action.duration_s) * self.config.timeout_scale + self.config.timeout_margin_s
        next_send_time = now
        last_sequence = start.sequence
        last_progress = 0.0
        last_progress_time = start.raw.receive_time_s
        previous_yaw = start.raw.yaw
        accumulated_yaw = 0.0
        send_count = 0
        sample_count = 0
        reason = "action_timeout"
        geometric_completed = False
        stop_snapshot = start
        pending_exception: BaseException | None = None

        try:
            while True:
                now = float(self._clock())
                if now >= deadline:
                    reason = "action_timeout"
                    break
                if now >= next_send_time:
                    send = self.sink.send_command(command, require_state_gate=require_state_gate)
                    if not send.accepted:
                        reason = "command_rejected"
                        break
                    send_count += 1
                    next_send_time = now + self.command_refresh_period_s
                wait_deadline = min(deadline, now + self.config.stale_after_s)
                try:
                    snapshot = self.odometry.wait_for_snapshot(
                        newer_than_sequence=last_sequence,
                        deadline_s=wait_deadline,
                    )
                except TimeoutError:
                    reason = "odom_stale" if float(self._clock()) < deadline else "action_timeout"
                    break
                if snapshot.sequence <= last_sequence:
                    continue
                last_sequence = snapshot.sequence
                stop_snapshot = snapshot
                sample_count += 1
                now = float(self._clock())
                if snapshot.epoch != start.epoch:
                    reason = "odom_discontinuity"
                    break
                if snapshot.raw.age_s(now) > self.config.stale_after_s:
                    reason = "odom_stale"
                    break

                if action.kind == StepwiseActionKind.FORWARD:
                    along, lateral = self._translation_progress(start, snapshot, action)
                    if along < -self.config.wrong_direction_distance_m:
                        reason = "wrong_direction"
                        break
                    if abs(lateral) > self.config.max_lateral_error_m:
                        reason = "lateral_deviation"
                        break
                    signed_progress = along
                elif action.kind == StepwiseActionKind.TURN:
                    accumulated_yaw += angle_wrap(snapshot.raw.yaw - previous_yaw)
                    previous_yaw = snapshot.raw.yaw
                    direction = 1.0 if float(action.angle_rad or 0.0) >= 0.0 else -1.0
                    signed_progress = accumulated_yaw * direction
                    if signed_progress < -self.config.wrong_direction_yaw_rad:
                        reason = "wrong_direction"
                        break
                else:
                    reason = "invalid_odometry"
                    break

                if signed_progress >= required_progress:
                    geometric_completed = True
                    reason = "completed"
                    break
                increment = self.config.min_progress_m if action.kind == StepwiseActionKind.FORWARD else self.config.min_progress_rad
                if signed_progress >= last_progress + increment:
                    last_progress = signed_progress
                    last_progress_time = snapshot.raw.receive_time_s
                if snapshot.raw.receive_time_s - last_progress_time > self.config.no_progress_timeout_s:
                    reason = "no_odometric_progress"
                    break
        except BaseException as exc:
            pending_exception = exc
            reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else "invalid_odometry"

        stop_result: AdapterCommandResult | None = None
        try:
            # A checked stop is required even when the first command was rejected;
            # the sink remains the authority on whether motion might be active.
            stop_result = self.checked_stop()
        except TerminalLocomotionFault:
            raise

        settled: StationaryWaitResult | None = None
        if send_count > 0:
            settled = self.wait_until_stationary(newer_than_sequence=stop_snapshot.sequence)
            sample_count += settled.sample_count
            if geometric_completed and not settled.stationary:
                reason = settled.reason
                geometric_completed = False

        if pending_exception is not None:
            raise pending_exception

        settled_snapshot = None if settled is None else settled.snapshot
        return self._result(
            start=start,
            stop=stop_snapshot,
            settled=settled_snapshot,
            reason=reason,
            completed=bool(geometric_completed and settled is not None and settled.stationary),
            stop_accepted=bool(stop_result and stop_result.accepted),
            send_count=send_count,
            sample_count=sample_count,
            settle_time_s=0.0 if settled is None else settled.elapsed_s,
            action=action,
            accumulated_yaw=accumulated_yaw,
        )

    def _required_progress(self, action: StepwiseAction) -> float:
        if action.kind == StepwiseActionKind.FORWARD:
            requested = float(action.distance_m or 0.0)
            tolerance = min(
                self.config.distance_tolerance_m,
                self.config.distance_tolerance_fraction * requested,
            )
            required = requested - tolerance
            if requested <= tolerance or required < self.config.min_completion_progress_m:
                raise ValueError("translation primitive is too small for configured completion tolerances")
            return required
        if action.kind == StepwiseActionKind.TURN:
            requested = abs(float(action.angle_rad or 0.0))
            tolerance = min(self.config.yaw_tolerance_rad, self.config.yaw_tolerance_fraction * requested)
            required = requested - tolerance
            if requested <= tolerance or required < self.config.min_completion_progress_rad:
                raise ValueError("turn primitive is too small for configured completion tolerances")
            return required
        raise ValueError(f"unsupported executable action kind: {action.kind.value}")

    @staticmethod
    def _translation_progress(
        start: UnitreeOdometrySnapshot,
        current: UnitreeOdometrySnapshot,
        action: StepwiseAction,
    ) -> tuple[float, float]:
        vx, vy = float(action.command.vx), float(action.command.vy)
        norm = math.hypot(vx, vy)
        if norm <= 1e-9:
            raise ValueError("translation action has no requested direction")
        ux_b, uy_b = vx / norm, vy / norm
        c, s = math.cos(start.raw.yaw), math.sin(start.raw.yaw)
        ux_o = c * ux_b - s * uy_b
        uy_o = s * ux_b + c * uy_b
        dx, dy = current.raw.x - start.raw.x, current.raw.y - start.raw.y
        along = ux_o * dx + uy_o * dy
        lateral = -uy_o * dx + ux_o * dy
        return along, lateral

    def _result(
        self,
        *,
        start: UnitreeOdometrySnapshot,
        stop: UnitreeOdometrySnapshot,
        settled: UnitreeOdometrySnapshot | None,
        reason: str,
        completed: bool = False,
        stop_accepted: bool = False,
        send_count: int = 0,
        sample_count: int = 0,
        settle_time_s: float = 0.0,
        action: StepwiseAction | None = None,
        accumulated_yaw: float = 0.0,
    ) -> ActionExecutionResult:
        final = settled or stop
        requested_distance = None if action is None else action.distance_m
        requested_angle = None if action is None else action.angle_rad
        achieved_distance = lateral = None
        achieved_angle = None
        if action is not None and action.kind == StepwiseActionKind.FORWARD:
            achieved_distance, lateral = self._translation_progress(start, final, action)
        elif action is not None and action.kind == StepwiseActionKind.TURN:
            achieved_angle = accumulated_yaw
            if settled is not None:
                achieved_angle = angle_wrap(settled.raw.yaw - start.raw.yaw)
        return ActionExecutionResult(
            completed=completed,
            reason=reason,
            requested_distance_m=requested_distance,
            achieved_distance_m=achieved_distance,
            requested_angle_rad=requested_angle,
            achieved_angle_rad=achieved_angle,
            lateral_error_m=lateral,
            elapsed_s=max(0.0, stop.raw.receive_time_s - start.raw.receive_time_s),
            settle_time_s=settle_time_s,
            start_pose=start.relative_pose,
            stop_pose=stop.relative_pose,
            settled_pose=None if settled is None else settled.relative_pose,
            start_sequence=start.sequence,
            stop_sequence=stop.sequence,
            settled_sequence=None if settled is None else settled.sequence,
            start_epoch=start.epoch,
            end_epoch=final.epoch,
            send_count=send_count,
            odom_sample_count=sample_count,
            stop_accepted=stop_accepted,
        )


__all__ = [
    "ActionExecutionResult",
    "OdometryActionExecutor",
    "OdometryActionExecutorConfig",
    "StationaryWaitResult",
    "TerminalLocomotionFault",
    "real_motion_ready",
    "wait_until_stationary",
]
