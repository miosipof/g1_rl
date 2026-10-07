"""Dependency-light Isaac Sim odometry and command adapters.

The adapter accepts ground-truth planar state from an Isaac scene and exposes
the same snapshot/wait/status contract as :class:`UnitreeOdometryAdapter`.
Optional lag, scale error, bias, Gaussian noise, and packet loss make it useful
for exercising the real-robot odometry executor without importing Isaac or
Torch in this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .real_unitree_odometry import (
    UnitreeOdometrySample,
    UnitreeOdometrySnapshot,
    angle_wrap,
)
from .types import AdapterCommandResult, AdapterHealth, AdapterStatus, Command2D, Pose2D


Clock = Callable[[], float]
Pump = Callable[[], None]
CommandCallback = Callable[[Command2D], None]


@dataclass(frozen=True)
class IsaacOdometryConfig:
    """Synthetic odometry imperfections applied to Isaac ground truth."""

    sample_period_s: float = 0.02
    lag_s: float = 0.0
    position_noise_std_m: float = 0.0
    yaw_noise_std_rad: float = 0.0
    linear_velocity_noise_std_mps: float = 0.0
    yaw_rate_noise_std_radps: float = 0.0
    position_bias_x_m: float = 0.0
    position_bias_y_m: float = 0.0
    yaw_bias_rad: float = 0.0
    distance_scale: float = 1.0
    yaw_scale: float = 1.0
    dropout_probability: float = 0.0
    stale_after_s: float = 0.10
    seed: int = 0
    source: str = "isaac_odometry"

    def __post_init__(self) -> None:
        nonnegative = (
            "sample_period_s",
            "lag_s",
            "position_noise_std_m",
            "yaw_noise_std_rad",
            "linear_velocity_noise_std_mps",
            "yaw_rate_noise_std_radps",
            "stale_after_s",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")
        for name in ("position_bias_x_m", "position_bias_y_m", "yaw_bias_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("distance_scale", "yaw_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        probability = float(self.dropout_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")
        object.__setattr__(self, "dropout_probability", probability)
        if not str(self.source).strip():
            raise ValueError("source must be non-empty")


@dataclass(frozen=True)
class _GroundTruthSample:
    acquisition_time_s: float
    release_time_s: float
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    yaw_speed: float


class IsaacOdometryAdapter:
    """Turn pushed Isaac ground truth into Unitree-compatible snapshots.

    ``pump`` is optional. In a synchronous Isaac script it should advance one
    physics step and call :meth:`push_ground_truth`. This lets the blocking
    odometry executor drive simulation time while waiting for the next sample.
    """

    def __init__(
        self,
        config: IsaacOdometryConfig | None = None,
        *,
        pump: Pump | None = None,
        initial_time_s: float = 0.0,
    ) -> None:
        self.config = config or IsaacOdometryConfig()
        self._pump = pump
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._rng = np.random.default_rng(int(self.config.seed))
        self._now_s = float(initial_time_s)
        self._pending: deque[_GroundTruthSample] = deque()
        self._latest: UnitreeOdometrySample | None = None
        self._latest_acquisition_time_s: float | None = None
        self._origin: UnitreeOdometrySample | None = None
        self._last_enqueued_time_s: float | None = None
        self._last_truth_yaw: float | None = None
        self._unwrapped_truth_yaw: float | None = None
        self._sequence = 0
        self._submitted_count = 0
        self._published_count = 0
        self._dropped_count = 0

    def set_pump(self, pump: Pump | None) -> None:
        self._pump = pump

    def clock(self) -> float:
        with self._lock:
            return self._now_s

    def push_ground_truth(
        self,
        *,
        timestamp_s: float,
        x: float,
        y: float,
        z: float,
        yaw: float,
        vx: float,
        vy: float,
        vz: float = 0.0,
        yaw_speed: float,
    ) -> None:
        values = tuple(float(v) for v in (timestamp_s, x, y, z, yaw, vx, vy, vz, yaw_speed))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Isaac ground-truth odometry values must be finite")
        timestamp, x, y, z, yaw, vx, vy, vz, yaw_speed = values
        with self._condition:
            if timestamp + 1.0e-12 < self._now_s:
                raise ValueError("Isaac odometry timestamps must be monotonic")
            self._now_s = timestamp
            should_enqueue = (
                self._last_enqueued_time_s is None
                or self.config.sample_period_s == 0.0
                or timestamp - self._last_enqueued_time_s + 1.0e-12 >= self.config.sample_period_s
            )
            if should_enqueue:
                wrapped_yaw = angle_wrap(yaw)
                if self._last_truth_yaw is None:
                    self._unwrapped_truth_yaw = wrapped_yaw
                else:
                    assert self._unwrapped_truth_yaw is not None
                    self._unwrapped_truth_yaw += angle_wrap(wrapped_yaw - self._last_truth_yaw)
                self._last_truth_yaw = wrapped_yaw
                self._pending.append(
                    _GroundTruthSample(
                        acquisition_time_s=timestamp,
                        release_time_s=timestamp + self.config.lag_s,
                        x=x,
                        y=y,
                        z=z,
                        yaw=float(self._unwrapped_truth_yaw),
                        vx=vx,
                        vy=vy,
                        vz=vz,
                        yaw_speed=yaw_speed,
                    )
                )
                self._last_enqueued_time_s = timestamp
                self._submitted_count += 1
            self._release_due_locked(timestamp)

    def _release_due_locked(self, now_s: float) -> None:
        changed = False
        while self._pending and self._pending[0].release_time_s <= now_s + 1.0e-12:
            truth = self._pending.popleft()
            if self._rng.random() < self.config.dropout_probability:
                self._dropped_count += 1
                continue
            position_noise = self._rng.normal(0.0, self.config.position_noise_std_m, size=3)
            velocity_noise = self._rng.normal(0.0, self.config.linear_velocity_noise_std_mps, size=3)
            yaw_noise = float(self._rng.normal(0.0, self.config.yaw_noise_std_rad))
            yaw_rate_noise = float(self._rng.normal(0.0, self.config.yaw_rate_noise_std_radps))
            self._sequence += 1
            self._latest = UnitreeOdometrySample(
                receive_time_s=float(now_s),
                sequence=self._sequence,
                epoch=0,
                x=self.config.distance_scale * truth.x + self.config.position_bias_x_m + float(position_noise[0]),
                y=self.config.distance_scale * truth.y + self.config.position_bias_y_m + float(position_noise[1]),
                z=truth.z + float(position_noise[2]),
                yaw=angle_wrap(self.config.yaw_scale * truth.yaw + self.config.yaw_bias_rad + yaw_noise),
                vx=self.config.distance_scale * truth.vx + float(velocity_noise[0]),
                vy=self.config.distance_scale * truth.vy + float(velocity_noise[1]),
                vz=truth.vz + float(velocity_noise[2]),
                yaw_speed=self.config.yaw_scale * truth.yaw_speed + yaw_rate_noise,
                source=self.config.source,
            )
            self._latest_acquisition_time_s = truth.acquisition_time_s
            self._published_count += 1
            changed = True
        if changed:
            self._condition.notify_all()

    def latest_raw(self) -> UnitreeOdometrySample | None:
        with self._lock:
            return self._latest

    def capture_origin(self) -> Pose2D:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("cannot capture Isaac odometry origin without a sample")
            self._origin = self._latest
        return Pose2D(0.0, 0.0, 0.0)

    def snapshot(self) -> UnitreeOdometrySnapshot:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("no Isaac odometry sample is available")
            return self._snapshot_locked()

    def _snapshot_locked(self) -> UnitreeOdometrySnapshot:
        assert self._latest is not None
        raw = self._latest
        if self._origin is None:
            relative = raw.pose
        else:
            dx, dy = raw.x - self._origin.x, raw.y - self._origin.y
            c, s = math.cos(self._origin.yaw), math.sin(self._origin.yaw)
            relative = Pose2D(
                c * dx + s * dy,
                -s * dx + c * dy,
                angle_wrap(raw.yaw - self._origin.yaw),
            )
        health = AdapterHealth.OK if raw.age_s(self._now_s) <= self.config.stale_after_s else AdapterHealth.STALE
        return UnitreeOdometrySnapshot(raw, relative, raw.sequence, raw.epoch, health)

    def wait_for_snapshot(self, *, newer_than_sequence: int, deadline_s: float) -> UnitreeOdometrySnapshot:
        deadline_s = float(deadline_s)
        wall_deadline_s = time.monotonic() + max(0.05, deadline_s - self.clock() + 0.05)
        while True:
            with self._condition:
                if self._latest is not None and self._latest.sequence > int(newer_than_sequence):
                    return self._snapshot_locked()
                if self._now_s >= deadline_s or time.monotonic() >= wall_deadline_s:
                    raise TimeoutError("timed out waiting for a new Isaac odometry sample")
                pump = self._pump
                if pump is None:
                    remaining = min(deadline_s - self._now_s, wall_deadline_s - time.monotonic(), 0.01)
                    self._condition.wait(timeout=max(0.0, remaining))
                    continue
            # Never call into Isaac while holding the adapter lock.
            pump()

    def status(self, *, message: str | None = None) -> AdapterStatus:
        with self._lock:
            latest = self._latest
            health = (
                AdapterHealth.UNAVAILABLE
                if latest is None
                else AdapterHealth.OK
                if latest.age_s(self._now_s) <= self.config.stale_after_s
                else AdapterHealth.STALE
            )
            details: dict[str, Any] = {
                "sequence": None if latest is None else latest.sequence,
                "age_s": None if latest is None else latest.age_s(self._now_s),
                "measurement_age_s": None
                if self._latest_acquisition_time_s is None
                else max(0.0, self._now_s - self._latest_acquisition_time_s),
                "configured_lag_s": self.config.lag_s,
                "pending_count": len(self._pending),
                "submitted_count": self._submitted_count,
                "published_count": self._published_count,
                "dropped_count": self._dropped_count,
                "noise": {
                    "position_std_m": self.config.position_noise_std_m,
                    "yaw_std_rad": self.config.yaw_noise_std_rad,
                    "linear_velocity_std_mps": self.config.linear_velocity_noise_std_mps,
                    "yaw_rate_std_radps": self.config.yaw_rate_noise_std_radps,
                },
                "distance_scale": self.config.distance_scale,
                "yaw_scale": self.config.yaw_scale,
            }
            timestamp_s = None if latest is None else latest.receive_time_s
        return AdapterStatus(
            name=self.config.source,
            health=health,
            timestamp_s=timestamp_s,
            message=message or health.value,
            details=details,
        )


class IsaacCommandSink:
    """Adapt a simulator command callback to the shared command-sink API."""

    def __init__(
        self,
        set_command: CommandCallback,
        *,
        clock: Clock,
        command_period_s: float = 0.10,
        source: str = "isaac_command_sink",
    ) -> None:
        if command_period_s <= 0.0 or not math.isfinite(float(command_period_s)):
            raise ValueError("command_period_s must be positive and finite")
        self._set_command = set_command
        self._clock = clock
        self.source = str(source)
        self.config = SimpleNamespace(command_period_s=float(command_period_s))

    def send_command(self, command: Command2D, *, require_state_gate: bool = True) -> AdapterCommandResult:
        del require_state_gate
        return self._apply(command, message="Isaac command accepted")

    def stop(self, *, timestamp_s: float | None = None) -> AdapterCommandResult:
        ts = self._clock() if timestamp_s is None else float(timestamp_s)
        return self._apply(Command2D.stop(timestamp_s=ts, source="isaac_stop"), message="Isaac stop accepted")

    def _apply(self, command: Command2D, *, message: str) -> AdapterCommandResult:
        try:
            self._set_command(command)
            status = AdapterStatus(
                name=self.source,
                health=AdapterHealth.OK,
                timestamp_s=self._clock(),
                message=message,
                details={"command": list(command.tuple)},
            )
            return AdapterCommandResult(True, command, status)
        except Exception as exc:
            status = AdapterStatus(
                name=self.source,
                health=AdapterHealth.FAULT,
                timestamp_s=self._clock(),
                message=f"Isaac command failed: {exc!r}",
                details={"exception": repr(exc)},
            )
            return AdapterCommandResult(False, command, status)


__all__ = [
    "IsaacCommandSink",
    "IsaacOdometryAdapter",
    "IsaacOdometryConfig",
]
