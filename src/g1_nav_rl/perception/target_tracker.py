"""Periodic 2D target tracking state machine.

This module is the next layer after one-shot target acquisition.  A caller can
refresh a measurement every N simulator steps, update this tracker, and pass the
tracker's ``active_target`` to navigation.  The tracker handles three practical
conditions for visual target following:

- ``TRACKING``: fresh valid target measurement accepted.
- ``OCCLUDED_RECENT``: measurement missing, but last target is still usable for
  a bounded grace interval.
- ``LOST``: no usable target remains; navigation should stop or switch policy.

The implementation is deterministic and does not depend on Isaac, YOLO, Torch,
or recurrent models.  It is intended as the stable baseline before adding
velocity prediction or learned temporal filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from g1_nav_rl.navigation.simple_nav import Target2D


class TargetTrackingState(str, Enum):
    TRACKING = "TRACKING"
    OCCLUDED_RECENT = "OCCLUDED_RECENT"
    LOST = "LOST"


@dataclass(frozen=True)
class TargetTrackerConfig:
    """Configuration for periodic target refresh and filtering."""

    refresh_period_steps: int = 30
    occlusion_grace_steps: int = 90
    smoothing_alpha: float = 0.35
    max_jump_m: float = 1.25
    min_update_distance_m: float = 0.01
    allow_first_measurement: bool = True

    def __post_init__(self) -> None:
        if int(self.refresh_period_steps) <= 0:
            raise ValueError("refresh_period_steps must be positive")
        if int(self.occlusion_grace_steps) < 0:
            raise ValueError("occlusion_grace_steps must be non-negative")
        if not (0.0 <= float(self.smoothing_alpha) <= 1.0):
            raise ValueError("smoothing_alpha must be in [0, 1]")
        if float(self.max_jump_m) < 0.0:
            raise ValueError("max_jump_m must be non-negative")
        if float(self.min_update_distance_m) < 0.0:
            raise ValueError("min_update_distance_m must be non-negative")


@dataclass(frozen=True)
class TargetMeasurement:
    """One target measurement from perception."""

    target: Target2D
    step: int
    confidence: float = 1.0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_xy(
        cls,
        xy: tuple[float, float],
        *,
        step: int,
        confidence: float = 1.0,
        source: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> "TargetMeasurement":
        return cls(
            target=Target2D(float(xy[0]), float(xy[1])),
            step=int(step),
            confidence=float(confidence),
            source=str(source),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class TargetTrackerUpdate:
    """Result of one tracker update."""

    state: TargetTrackingState
    active_target: Target2D | None
    accepted_measurement: bool
    rejected_measurement: bool
    reason: str
    last_seen_step: int | None
    steps_since_seen: int | None
    jump_distance_m: float | None = None

    @property
    def has_target(self) -> bool:
        return self.active_target is not None


class TargetTracker:
    """Stateful low-pass target tracker with occlusion grace and jump rejection."""

    def __init__(self, config: TargetTrackerConfig | None = None) -> None:
        self.config = config or TargetTrackerConfig()
        self._state = TargetTrackingState.LOST
        self._active_target: Target2D | None = None
        self._last_raw_target: Target2D | None = None
        self._last_seen_step: int | None = None
        self._last_update: TargetTrackerUpdate = TargetTrackerUpdate(
            state=TargetTrackingState.LOST,
            active_target=None,
            accepted_measurement=False,
            rejected_measurement=False,
            reason="init",
            last_seen_step=None,
            steps_since_seen=None,
        )

    @property
    def state(self) -> TargetTrackingState:
        return self._state

    @property
    def active_target(self) -> Target2D | None:
        return self._active_target

    @property
    def last_seen_step(self) -> int | None:
        return self._last_seen_step

    @property
    def last_update(self) -> TargetTrackerUpdate:
        return self._last_update

    def should_refresh(self, step: int) -> bool:
        """Return whether perception should be refreshed at this step."""
        step_i = int(step)
        if self._last_seen_step is None and self._active_target is None:
            return True
        return step_i % int(self.config.refresh_period_steps) == 0

    def reset(self) -> None:
        self.__init__(self.config)

    def update(
        self,
        *,
        step: int,
        measurement: TargetMeasurement | Target2D | tuple[float, float] | None,
    ) -> TargetTrackerUpdate:
        """Update state from a measurement or a missing-observation event."""
        step_i = int(step)
        measurement_obj = self._coerce_measurement(measurement, step=step_i)

        if measurement_obj is None:
            return self._mark_missing(step_i, reason="no_measurement")

        target = measurement_obj.target
        if self._active_target is None:
            if not bool(self.config.allow_first_measurement):
                update = TargetTrackerUpdate(
                    state=self._state,
                    active_target=self._active_target,
                    accepted_measurement=False,
                    rejected_measurement=True,
                    reason="first_measurement_disallowed",
                    last_seen_step=self._last_seen_step,
                    steps_since_seen=self._steps_since_seen(step_i),
                    jump_distance_m=None,
                )
                self._last_update = update
                return update
            return self._accept(step_i, target, raw_target=target, reason="first_measurement", jump_distance=None)

        jump = distance_xy(self._active_target, target)
        if jump > float(self.config.max_jump_m):
            # Reject the implausible measurement but keep last known target under
            # the normal occlusion grace semantics.
            return self._mark_missing(step_i, reason="jump_rejected", rejected=True, jump_distance=jump)

        if jump < float(self.config.min_update_distance_m):
            return self._accept(step_i, self._active_target, raw_target=target, reason="measurement_same", jump_distance=jump)

        alpha = float(self.config.smoothing_alpha)
        smoothed = Target2D(
            x=(1.0 - alpha) * float(self._active_target.x) + alpha * float(target.x),
            y=(1.0 - alpha) * float(self._active_target.y) + alpha * float(target.y),
        )
        return self._accept(step_i, smoothed, raw_target=target, reason="measurement_smoothed", jump_distance=jump)

    def _accept(
        self,
        step: int,
        target: Target2D,
        *,
        raw_target: Target2D,
        reason: str,
        jump_distance: float | None,
    ) -> TargetTrackerUpdate:
        self._active_target = target
        self._last_raw_target = raw_target
        self._last_seen_step = int(step)
        self._state = TargetTrackingState.TRACKING
        update = TargetTrackerUpdate(
            state=self._state,
            active_target=self._active_target,
            accepted_measurement=True,
            rejected_measurement=False,
            reason=reason,
            last_seen_step=self._last_seen_step,
            steps_since_seen=0,
            jump_distance_m=jump_distance,
        )
        self._last_update = update
        return update

    def _mark_missing(
        self,
        step: int,
        *,
        reason: str,
        rejected: bool = False,
        jump_distance: float | None = None,
    ) -> TargetTrackerUpdate:
        steps_since = self._steps_since_seen(step)
        if self._active_target is not None and steps_since is not None and steps_since <= int(self.config.occlusion_grace_steps):
            self._state = TargetTrackingState.OCCLUDED_RECENT
            active = self._active_target
        else:
            self._state = TargetTrackingState.LOST
            active = None
            self._active_target = None
        update = TargetTrackerUpdate(
            state=self._state,
            active_target=active,
            accepted_measurement=False,
            rejected_measurement=bool(rejected),
            reason=reason,
            last_seen_step=self._last_seen_step,
            steps_since_seen=steps_since,
            jump_distance_m=jump_distance,
        )
        self._last_update = update
        return update

    def _steps_since_seen(self, step: int) -> int | None:
        if self._last_seen_step is None:
            return None
        return max(0, int(step) - int(self._last_seen_step))

    @staticmethod
    def _coerce_measurement(
        measurement: TargetMeasurement | Target2D | tuple[float, float] | None,
        *,
        step: int,
    ) -> TargetMeasurement | None:
        if measurement is None:
            return None
        if isinstance(measurement, TargetMeasurement):
            return measurement
        if isinstance(measurement, Target2D):
            return TargetMeasurement(target=measurement, step=int(step))
        if isinstance(measurement, tuple) and len(measurement) == 2:
            return TargetMeasurement.from_xy((float(measurement[0]), float(measurement[1])), step=int(step))
        raise TypeError(f"unsupported measurement type: {type(measurement)!r}")


def distance_xy(a: Target2D, b: Target2D) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))
