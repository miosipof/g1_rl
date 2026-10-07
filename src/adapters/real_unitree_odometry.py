"""Unitree G1 odometry subscriber with atomic, sequence-driven snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import threading
import time
from typing import Any, Callable, Mapping

from .types import AdapterHealth, AdapterStatus, Pose2D
from .unitree_channel import ensure_unitree_channel_initialized


Clock = Callable[[], float]


def angle_wrap(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


@dataclass(frozen=True)
class UnitreeOdometryConfig:
    iface: str = "eth0"
    topic: str = "rt/odommodestate"
    stale_after_s: float = 0.10
    startup_timeout_s: float = 2.0
    max_position_jump_m: float = 0.50
    max_yaw_jump_rad: float = math.radians(45.0)
    source: str = "unitree_odometry"

    def __post_init__(self) -> None:
        if not str(self.iface).strip():
            raise ValueError("UnitreeOdometryConfig.iface must be non-empty")
        if not str(self.topic).strip():
            raise ValueError("UnitreeOdometryConfig.topic must be non-empty")
        for name in (
            "stale_after_s",
            "startup_timeout_s",
            "max_position_jump_m",
            "max_yaw_jump_rad",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class UnitreeOdometrySample:
    receive_time_s: float
    sequence: int
    epoch: int
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    yaw_speed: float
    source: str = "unitree_odometry"

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - self.receive_time_s)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnitreeOdometrySnapshot:
    raw: UnitreeOdometrySample
    relative_pose: Pose2D
    sequence: int
    epoch: int
    health: AdapterHealth

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw.to_dict(),
            "relative_pose": [self.relative_pose.x, self.relative_pose.y, self.relative_pose.yaw],
            "sequence": self.sequence,
            "epoch": self.epoch,
            "health": self.health.value,
        }


def snapshot_to_dict(snapshot: UnitreeOdometrySnapshot) -> dict[str, Any]:
    return snapshot.to_dict()


class UnitreeOdometryAdapter:
    """Thread-safe subscriber for ``rt/odommodestate``.

    Discontinuous samples are accepted as the first sample of a new epoch.  An
    executor spanning the epoch boundary must reject the action.
    """

    def __init__(
        self,
        config: UnitreeOdometryConfig | None = None,
        *,
        clock: Clock | None = None,
        channel_initializer: Callable[[str], None] | None = None,
        subscriber_factory: Callable[[str, Any], Any] | None = None,
        message_type: Any | None = None,
    ) -> None:
        self.config = config or UnitreeOdometryConfig()
        self._clock = clock or time.monotonic
        self._channel_initializer = channel_initializer or ensure_unitree_channel_initialized
        self._subscriber_factory = subscriber_factory
        self._message_type = message_type
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._latest: UnitreeOdometrySample | None = None
        self._origin: UnitreeOdometrySample | None = None
        self._sequence = 0
        self._epoch = 0
        self._received_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._callback_error_count = 0
        self._discontinuity_count = 0
        self._last_rejection_reason: str | None = None
        self._last_rejection_time_s: float | None = None
        self._subscriber: Any | None = None
        self._connected = False
        self._connection_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> AdapterStatus:
        if self._connected and self._latest is not None:
            return self.status()
        try:
            self._connection_error = None
            self._channel_initializer(self.config.iface)
            factory = self._subscriber_factory
            message_type = self._message_type
            if factory is None or message_type is None:
                from unitree_sdk2py.core.channel import ChannelSubscriber  # type: ignore
                from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_  # type: ignore

                factory = factory or ChannelSubscriber
                message_type = message_type or SportModeState_
            subscriber = factory(self.config.topic, message_type)
            subscriber.Init(self._on_odom, 10)
            self._subscriber = subscriber
            self._connected = True
            deadline = self._clock() + self.config.startup_timeout_s
            try:
                self.wait_for_snapshot(newer_than_sequence=-1, deadline_s=deadline)
            except TimeoutError:
                return self.status(message="no accepted odometry sample before startup timeout")
            return self.status(message="Unitree odometry connected")
        except Exception as exc:
            self._connected = False
            self._connection_error = repr(exc)
            return self.status(message=f"failed to connect Unitree odometry: {exc!r}")

    def close(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        self._connected = False
        if subscriber is None:
            return
        for name in ("Close", "close"):
            closer = getattr(subscriber, name, None)
            if callable(closer):
                closer()
                break

    def _on_odom(self, msg: Any) -> None:
        now = float(self._clock())
        with self._condition:
            self._received_count += 1
        try:
            position = tuple(msg.position)
            velocity = tuple(msg.velocity)
            rpy = tuple(msg.imu_state.rpy)
            if len(position) < 3 or len(velocity) < 3 or len(rpy) < 3:
                raise ValueError("odometry arrays must each contain at least three values")
            values = (
                position[0], position[1], position[2], rpy[2],
                velocity[0], velocity[1], velocity[2], msg.yaw_speed,
            )
            converted = tuple(float(v) for v in values)
            if not all(math.isfinite(v) for v in converted):
                self._reject_sample("non_finite", now)
                return
            x, y, z, yaw, vx, vy, vz, yaw_speed = converted
            candidate = UnitreeOdometrySample(
                receive_time_s=now,
                sequence=0,
                epoch=0,
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                vx=vx,
                vy=vy,
                vz=vz,
                yaw_speed=yaw_speed,
                source=self.config.source,
            )
            with self._condition:
                previous = self._latest
                if previous is not None:
                    distance_jump = math.hypot(candidate.x - previous.x, candidate.y - previous.y)
                    yaw_jump = abs(angle_wrap(candidate.yaw - previous.yaw))
                    if (
                        distance_jump > self.config.max_position_jump_m
                        or yaw_jump > self.config.max_yaw_jump_rad
                    ):
                        self._epoch += 1
                        self._discontinuity_count += 1
                        self._last_rejection_reason = "discontinuity"
                        self._last_rejection_time_s = now
                self._sequence += 1
                accepted = replace(candidate, sequence=self._sequence, epoch=self._epoch)
                self._latest = accepted
                self._accepted_count += 1
                self._condition.notify_all()
        except Exception:
            with self._condition:
                self._callback_error_count += 1
                self._rejected_count += 1
                self._last_rejection_reason = "callback_exception"
                self._last_rejection_time_s = now
                self._condition.notify_all()

    def _reject_sample(self, reason: str, now_s: float) -> None:
        with self._condition:
            self._rejected_count += 1
            self._last_rejection_reason = str(reason)
            self._last_rejection_time_s = float(now_s)
            self._condition.notify_all()

    def latest_raw(self) -> UnitreeOdometrySample | None:
        with self._lock:
            return self._latest

    def require_latest_raw(self) -> UnitreeOdometrySample:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no accepted Unitree odometry sample is available")
            return self._latest

    def capture_origin(self) -> Pose2D:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("cannot capture odometry origin without an accepted sample")
            self._origin = self._latest
            return Pose2D(0.0, 0.0, 0.0)

    def snapshot(self) -> UnitreeOdometrySnapshot:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no accepted Unitree odometry sample is available")
            return self._snapshot_locked(float(self._clock()))

    def _snapshot_locked(self, now_s: float) -> UnitreeOdometrySnapshot:
        assert self._latest is not None
        raw = self._latest
        origin = self._origin
        if origin is None:
            pose = raw.pose
        else:
            dx, dy = raw.x - origin.x, raw.y - origin.y
            c, s = math.cos(origin.yaw), math.sin(origin.yaw)
            pose = Pose2D(
                c * dx + s * dy,
                -s * dx + c * dy,
                angle_wrap(raw.yaw - origin.yaw),
            )
        health = AdapterHealth.OK if raw.age_s(now_s) <= self.config.stale_after_s else AdapterHealth.STALE
        return UnitreeOdometrySnapshot(raw, pose, raw.sequence, raw.epoch, health)

    def relative_pose(self) -> Pose2D:
        return self.snapshot().relative_pose

    def wait_for_snapshot(
        self,
        *,
        newer_than_sequence: int,
        deadline_s: float,
    ) -> UnitreeOdometrySnapshot:
        wall_deadline_s = time.monotonic() + max(0.0, float(deadline_s) - float(self._clock()))
        with self._condition:
            while self._latest is None or self._latest.sequence <= int(newer_than_sequence):
                remaining = min(
                    float(deadline_s) - float(self._clock()),
                    wall_deadline_s - time.monotonic(),
                )
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for a new Unitree odometry sample")
                self._condition.wait(timeout=remaining)
            return self._snapshot_locked(float(self._clock()))

    def status(self, *, message: str | None = None) -> AdapterStatus:
        now = float(self._clock())
        with self._lock:
            latest = self._latest
            if self._connection_error is not None:
                health = AdapterHealth.FAULT
            elif latest is None:
                health = AdapterHealth.UNAVAILABLE
            elif latest.age_s(now) > self.config.stale_after_s:
                health = AdapterHealth.STALE
            else:
                health = AdapterHealth.OK
            details: Mapping[str, Any] = {
                "topic": self.config.topic,
                "sequence": None if latest is None else latest.sequence,
                "epoch": None if latest is None else latest.epoch,
                "age_s": None if latest is None else latest.age_s(now),
                "position": None if latest is None else [latest.x, latest.y, latest.z],
                "yaw": None if latest is None else latest.yaw,
                "velocity": None if latest is None else [latest.vx, latest.vy, latest.vz],
                "yaw_speed": None if latest is None else latest.yaw_speed,
                "received_count": self._received_count,
                "accepted_count": self._accepted_count,
                "rejected_count": self._rejected_count,
                "callback_error_count": self._callback_error_count,
                "discontinuity_count": self._discontinuity_count,
                "last_rejection_reason": self._last_rejection_reason,
                "last_rejection_time_s": self._last_rejection_time_s,
                "connection_error": self._connection_error,
            }
        return AdapterStatus(
            name=self.config.source,
            health=health,
            timestamp_s=None if latest is None else latest.receive_time_s,
            message=message or health.value,
            details=details,
        )


__all__ = [
    "UnitreeOdometryAdapter",
    "UnitreeOdometryConfig",
    "UnitreeOdometrySample",
    "UnitreeOdometrySnapshot",
    "angle_wrap",
    "snapshot_to_dict",
]
