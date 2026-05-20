"""Hardware-neutral safety supervisor for sim/real command gating.

The supervisor sits between a policy/runtime that proposes ``Command2D`` values
and a command sink such as the Unitree locomotion adapter.  It deliberately has
no Isaac, Unitree, RealSense, Livox, ROS, or OpenCV imports.

Typical use::

    decision = supervisor.filter_command(
        proposed_command,
        observation,
        operator_enabled=True,
    )
    command_sink.send_command(decision.command)

The output command is always safe to forward to a dry-run sink.  For real robot
motion, callers should additionally require ``decision.allow_motion``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .config import AdapterRuntimeConfig, CommandLimitsConfig, SafetyConfig, WatchdogConfig
from .types import AdapterHealth, AdapterStatus, Command2D, RobotObservation


@dataclass(frozen=True)
class SafetyDecision:
    """Result of safety filtering a proposed command."""

    proposed_command: Command2D
    command: Command2D
    status: AdapterStatus
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status.ok

    @property
    def allow_motion(self) -> bool:
        """Whether a non-zero command may be sent to a real command sink."""

        return self.ok and not self.command.is_stop

    @property
    def forced_stop(self) -> bool:
        return self.command.is_stop and not self.proposed_command.is_stop and bool(self.reasons)

    @property
    def overridden(self) -> bool:
        return self.command.tuple != self.proposed_command.tuple or self.command.source != self.proposed_command.source


@dataclass(frozen=True)
class SafetySupervisorConfig:
    """Local supervisor behavior not already represented in runtime config."""

    front_lateral_half_width_m: float = 0.45
    apply_obstacle_slowdown: bool = True
    apply_accel_limits: bool = True
    stop_source: str = "safety_stop"
    limited_source: str = "safety_limited"
    slowed_source: str = "safety_obstacle_slowed"

    def __post_init__(self) -> None:
        if self.front_lateral_half_width_m <= 0.0 or not math.isfinite(self.front_lateral_half_width_m):
            raise ValueError("front_lateral_half_width_m must be positive finite")


class SafetySupervisor:
    """Filter policy commands before they reach a real or simulated command sink.

    The supervisor applies, in order:
    1. hard-stop gates: operator enable, observation health, robot e-stop/fault,
       missing required sensors, and stale sensor/policy data;
    2. command velocity clipping;
    3. acceleration clipping relative to the last safe command;
    4. optional LiDAR-based front-obstacle slowdown/stop.
    """

    def __init__(
        self,
        *,
        command_limits: CommandLimitsConfig,
        watchdog: WatchdogConfig,
        safety: SafetyConfig,
        config: SafetySupervisorConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.command_limits = command_limits
        self.watchdog = watchdog
        self.safety = safety
        self.config = config or SafetySupervisorConfig()
        self._clock = clock or time.monotonic
        self._last_command: Command2D | None = None
        self._last_decision: SafetyDecision | None = None

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: AdapterRuntimeConfig,
        *,
        config: SafetySupervisorConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> "SafetySupervisor":
        return cls(
            command_limits=runtime_config.command_limits,
            watchdog=runtime_config.watchdog,
            safety=runtime_config.safety,
            config=config,
            clock=clock,
        )

    @property
    def last_command(self) -> Command2D | None:
        return self._last_command

    @property
    def last_decision(self) -> SafetyDecision | None:
        return self._last_decision

    def reset(self, *, last_command: Command2D | None = None) -> None:
        self._last_command = last_command
        self._last_decision = None

    def filter_command(
        self,
        proposed_command: Command2D,
        observation: RobotObservation,
        *,
        operator_enabled: bool = False,
        now_s: float | None = None,
        policy_timestamp_s: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SafetyDecision:
        """Return a safe command and explicit safety decision.

        ``operator_enabled`` should be controlled by the high-level bridge or a
        physical/manual enable mechanism. If ``SafetyConfig.require_operator_enable``
        is true, motion is refused until this flag is true.
        """

        now = float(self._clock() if now_s is None else now_s)
        reasons: list[str] = []
        warnings: list[str] = []
        meta: dict[str, Any] = dict(metadata or {})
        meta["now_s"] = now
        meta["sensor_ages_s"] = observation.sensor_ages_s()

        reasons.extend(self._hard_stop_reasons(observation, operator_enabled=operator_enabled, now_s=now))
        policy_age = _command_age_s(proposed_command, now_s=now, fallback_timestamp_s=policy_timestamp_s)
        meta["policy_age_s"] = policy_age
        if policy_age is not None and policy_age > self.watchdog.max_policy_age_s:
            reasons.append(f"stale_policy:{policy_age:.3f}s>{self.watchdog.max_policy_age_s:.3f}s")

        if reasons:
            safe_command = Command2D.stop(timestamp_s=now, source=self.config.stop_source)
            decision = self._make_decision(
                proposed_command=proposed_command,
                command=safe_command,
                reasons=tuple(reasons),
                warnings=tuple(warnings),
                metadata=meta,
                health=AdapterHealth.STALE if any("stale" in r or "missing" in r for r in reasons) else AdapterHealth.FAULT,
            )
            self._remember(decision)
            return decision

        safe_command = self.command_limits.clip(proposed_command, source=self.config.limited_source)
        if safe_command.tuple != proposed_command.tuple:
            warnings.append("velocity_clipped")

        if self.config.apply_accel_limits:
            accel_command, accel_warnings = self._apply_acceleration_limits(safe_command, now_s=now)
            if accel_warnings:
                warnings.extend(accel_warnings)
                safe_command = accel_command

        if self.config.apply_obstacle_slowdown:
            safe_command, obstacle_reasons, obstacle_warnings, obstacle_meta = self._apply_lidar_front_safety(
                safe_command,
                observation,
                now_s=now,
            )
            meta.update(obstacle_meta)
            reasons.extend(obstacle_reasons)
            warnings.extend(obstacle_warnings)

        health = AdapterHealth.OK
        if reasons:
            health = AdapterHealth.STALE if any("stale" in r or "missing" in r for r in reasons) else AdapterHealth.DEGRADED
        elif warnings:
            health = AdapterHealth.DEGRADED

        decision = self._make_decision(
            proposed_command=proposed_command,
            command=safe_command,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            metadata=meta,
            health=health,
        )
        self._remember(decision)
        return decision

    def _hard_stop_reasons(
        self,
        observation: RobotObservation,
        *,
        operator_enabled: bool,
        now_s: float,
    ) -> list[str]:
        reasons: list[str] = []

        if self.safety.require_operator_enable and not operator_enabled:
            reasons.append("operator_not_enabled")

        if not observation.ok:
            reasons.append("observation_fault")

        if self.safety.require_lidar_for_motion and observation.lidar is None:
            reasons.append("missing_lidar")
        elif observation.lidar is not None and self.watchdog.stop_on_stale_lidar:
            age = observation.lidar.age_s(now_s)
            if age > self.watchdog.max_lidar_age_s:
                reasons.append(f"stale_lidar:{age:.3f}s>{self.watchdog.max_lidar_age_s:.3f}s")

        if self.watchdog.stop_on_stale_rgbd:
            if observation.rgbd is None:
                reasons.append("missing_rgbd")
            else:
                age = observation.rgbd.age_s(now_s)
                if age > self.watchdog.max_rgbd_age_s:
                    reasons.append(f"stale_rgbd:{age:.3f}s>{self.watchdog.max_rgbd_age_s:.3f}s")

        if self.safety.require_robot_state_for_motion and observation.robot_state is None:
            reasons.append("missing_robot_state")
        elif observation.robot_state is not None:
            state = observation.robot_state
            if self.watchdog.stop_on_stale_robot_state:
                age = state.age_s(now_s)
                if age > self.watchdog.max_robot_state_age_s:
                    reasons.append(f"stale_robot_state:{age:.3f}s>{self.watchdog.max_robot_state_age_s:.3f}s")
            if self.safety.stop_on_estop and state.is_estopped:
                reasons.append("robot_estop")
            if self.safety.stop_on_robot_fault and state.is_faulted:
                reasons.append("robot_fault")
            if self.safety.min_battery_percent is not None and state.battery_percent is not None:
                if state.battery_percent < self.safety.min_battery_percent:
                    reasons.append(
                        f"low_battery:{state.battery_percent:.1f}%<{self.safety.min_battery_percent:.1f}%"
                    )

        return reasons

    def _apply_acceleration_limits(self, command: Command2D, *, now_s: float) -> tuple[Command2D, list[str]]:
        warnings: list[str] = []
        last = self._last_command
        if last is None:
            return replace(command, timestamp_s=command.timestamp_s if command.timestamp_s is not None else now_s), warnings

        last_t = last.timestamp_s
        if last_t is None:
            return replace(command, timestamp_s=command.timestamp_s if command.timestamp_s is not None else now_s), warnings

        dt = max(1e-6, now_s - float(last_t))
        vx = command.vx
        vy = command.vy
        wz = command.wz

        if self.command_limits.max_linear_accel_mps2 is not None:
            max_delta = self.command_limits.max_linear_accel_mps2 * dt
            vx_new = _limit_delta(vx, last.vx, max_delta)
            vy_new = _limit_delta(vy, last.vy, max_delta)
            if vx_new != vx or vy_new != vy:
                warnings.append("linear_accel_limited")
            vx, vy = vx_new, vy_new

        if self.command_limits.max_yaw_accel_radps2 is not None:
            max_delta_wz = self.command_limits.max_yaw_accel_radps2 * dt
            wz_new = _limit_delta(wz, last.wz, max_delta_wz)
            if wz_new != wz:
                warnings.append("yaw_accel_limited")
            wz = wz_new

        if not warnings:
            return replace(command, timestamp_s=command.timestamp_s if command.timestamp_s is not None else now_s), warnings
        return replace(
            command,
            vx=vx,
            vy=vy,
            wz=wz,
            timestamp_s=now_s,
            source=self.config.limited_source,
        ), warnings

    def _apply_lidar_front_safety(
        self,
        command: Command2D,
        observation: RobotObservation,
        *,
        now_s: float,
    ) -> tuple[Command2D, list[str], list[str], dict[str, Any]]:
        reasons: list[str] = []
        warnings: list[str] = []
        meta: dict[str, Any] = {"front_obstacle_distance_m": None}

        if observation.lidar is None or observation.lidar.point_count == 0:
            return command, reasons, warnings, meta
        if command.vx <= 0.0:
            return command, reasons, warnings, meta

        points = observation.lidar.points_body_xyz
        forward = points[:, 0]
        lateral = np.abs(points[:, 1])
        keep = (forward > 0.0) & (lateral <= self.config.front_lateral_half_width_m)
        if not np.any(keep):
            return command, reasons, warnings, meta

        front_min = float(np.min(forward[keep]))
        meta["front_obstacle_distance_m"] = front_min
        stop_d = float(self.safety.obstacle_stop_distance_m)
        slow_d = float(self.safety.obstacle_slow_distance_m)

        if front_min <= stop_d:
            reasons.append(f"front_obstacle_stop:{front_min:.3f}m<={stop_d:.3f}m")
            return Command2D.stop(timestamp_s=now_s, source=self.config.stop_source), reasons, warnings, meta

        if front_min < slow_d and slow_d > stop_d:
            scale = max(0.0, min(1.0, (front_min - stop_d) / (slow_d - stop_d)))
            vx = command.vx * scale
            warnings.append(f"front_obstacle_slow:{front_min:.3f}m<{slow_d:.3f}m scale={scale:.3f}")
            meta["front_obstacle_speed_scale"] = scale
            return replace(command, vx=vx, timestamp_s=now_s, source=self.config.slowed_source), reasons, warnings, meta

        return command, reasons, warnings, meta

    def _make_decision(
        self,
        *,
        proposed_command: Command2D,
        command: Command2D,
        reasons: tuple[str, ...],
        warnings: tuple[str, ...],
        metadata: Mapping[str, Any],
        health: AdapterHealth,
    ) -> SafetyDecision:
        message_parts: list[str] = []
        if reasons:
            message_parts.append("stop=" + ",".join(reasons))
        if warnings:
            message_parts.append("warn=" + ",".join(warnings))
        if not message_parts:
            message_parts.append("command accepted")
        status = AdapterStatus(
            name="safety_supervisor",
            health=health,
            timestamp_s=metadata.get("now_s") if isinstance(metadata, Mapping) else None,
            message="; ".join(message_parts),
            details={"reasons": reasons, "warnings": warnings, **dict(metadata)},
        )
        return SafetyDecision(
            proposed_command=proposed_command,
            command=command,
            status=status,
            reasons=reasons,
            warnings=warnings,
            metadata=dict(metadata),
        )

    def _remember(self, decision: SafetyDecision) -> None:
        self._last_decision = decision
        self._last_command = decision.command


def _command_age_s(
    command: Command2D,
    *,
    now_s: float,
    fallback_timestamp_s: float | None,
) -> float | None:
    timestamp = command.timestamp_s if command.timestamp_s is not None else fallback_timestamp_s
    if timestamp is None:
        return None
    return max(0.0, float(now_s) - float(timestamp))


def _limit_delta(value: float, previous: float, max_delta: float) -> float:
    lo = float(previous) - float(max_delta)
    hi = float(previous) + float(max_delta)
    return min(max(float(value), lo), hi)


__all__ = ["SafetyDecision", "SafetySupervisor", "SafetySupervisorConfig"]
