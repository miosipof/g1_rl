"""Primary target-lock layer for detection-stream tracks.

This module keeps target identity management out of the navigation controller.
It consumes the shared DetectionStreamFrame / DetectionStreamReadResult schema
and returns either a Target2D to follow or an explicit missing/lost state.

Policy:
- initially lock the highest-confidence valid detection of the requested class;
- while locked, follow only the same track_id;
- ignore higher-confidence distractors with different IDs;
- during short occlusions, keep returning the last known target destination;
- after the lost budget expires, return no target so the controller stops.

The lock does not propagate targets with command odometry and does not smooth
positions. It only preserves target identity and last known destination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from g1_nav_rl.navigation.simple_nav import Target2D
from g1_nav_rl.perception.detection_stream import (
    DetectionStreamFrame,
    DetectionStreamHealth,
    DetectionStreamReadResult,
    TrackDetection,
)


class PrimaryTargetLockStateKind(str, Enum):
    """High-level target-lock state."""

    UNLOCKED = "unlocked"
    LOCKED_VISIBLE = "locked_visible"
    LOCKED_OCCLUDED = "locked_occluded"
    LOST = "lost"
    STALE = "stale"


@dataclass(frozen=True)
class PrimaryTargetLockConfig:
    """Configuration for the primary-track target lock."""

    target_class: str
    min_confidence: float = 0.10
    min_depth_valid_ratio: float = 0.0
    max_stream_age_s: float = 1.0
    max_lost_s: float = 5.0
    max_target_jump_m: float = 1.0
    require_track_id: bool = True
    allow_track_switch: bool = False
    initial_track_id: int | str | None = None

    def __post_init__(self) -> None:
        if not str(self.target_class).strip():
            raise ValueError("target_class must be non-empty")
        for name in (
            "min_confidence",
            "min_depth_valid_ratio",
            "max_stream_age_s",
            "max_lost_s",
            "max_target_jump_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"PrimaryTargetLockConfig.{name} must be finite and non-negative")
        if not (0.0 <= float(self.min_confidence) <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if not (0.0 <= float(self.min_depth_valid_ratio) <= 1.0):
            raise ValueError("min_depth_valid_ratio must be in [0, 1]")


@dataclass(frozen=True)
class PrimaryTargetState:
    """Result of one primary-target lock update."""

    state: PrimaryTargetLockStateKind | str
    locked_track_id: int | str | None
    target: Target2D | None
    using_last_known: bool
    lost_s: float
    reason: str
    selected_track: TrackDetection | None = None
    candidate_count: int = 0
    ignored_track_ids: tuple[int | str | None, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = self.state if isinstance(self.state, PrimaryTargetLockStateKind) else PrimaryTargetLockStateKind(str(self.state))
        object.__setattr__(self, "state", state)
        lost_s = float(self.lost_s)
        if not math.isfinite(lost_s) or lost_s < 0.0:
            raise ValueError("PrimaryTargetState.lost_s must be finite and non-negative")
        object.__setattr__(self, "lost_s", lost_s)

    @property
    def target_visible(self) -> bool:
        return self.state == PrimaryTargetLockStateKind.LOCKED_VISIBLE

    @property
    def has_target(self) -> bool:
        return self.target is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "locked_track_id": self.locked_track_id,
            "target_body_xy": None if self.target is None else [float(self.target.x), float(self.target.y)],
            "using_last_known": bool(self.using_last_known),
            "lost_s": float(self.lost_s),
            "reason": str(self.reason),
            "selected_track": None if self.selected_track is None else self.selected_track.to_dict(),
            "candidate_count": int(self.candidate_count),
            "ignored_track_ids": list(self.ignored_track_ids),
            "metadata": dict(self.metadata),
        }


def _track_id_key(track_id: int | str | None) -> str | None:
    if track_id is None:
        return None
    return str(track_id)


def _target_from_track(track: TrackDetection) -> Target2D:
    if track.target_body_xy is None:
        raise ValueError("track has no target_body_xy")
    return Target2D(float(track.target_body_xy[0]), float(track.target_body_xy[1]))


def _target_distance(a: Target2D | None, b: Target2D | None) -> float:
    if a is None or b is None:
        return math.inf
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


class PrimaryTargetLock:
    """Lock to the initial primary stream target and preserve identity."""

    def __init__(self, config: PrimaryTargetLockConfig) -> None:
        self.config = config
        self.locked_track_id: int | str | None = config.initial_track_id
        self._locked_track_key: str | None = _track_id_key(config.initial_track_id)
        self.last_known_target: Target2D | None = None
        self.last_seen_s: float | None = None
        self.last_frame_timestamp_s: float | None = None
        self.last_state: PrimaryTargetState = PrimaryTargetState(
            state=PrimaryTargetLockStateKind.UNLOCKED,
            locked_track_id=self.locked_track_id,
            target=None,
            using_last_known=False,
            lost_s=0.0,
            reason="initialized",
        )

    def reset(self) -> None:
        self.locked_track_id = self.config.initial_track_id
        self._locked_track_key = _track_id_key(self.config.initial_track_id)
        self.last_known_target = None
        self.last_seen_s = None
        self.last_frame_timestamp_s = None
        self.last_state = PrimaryTargetState(
            state=PrimaryTargetLockStateKind.UNLOCKED,
            locked_track_id=self.locked_track_id,
            target=None,
            using_last_known=False,
            lost_s=0.0,
            reason="reset",
        )

    @property
    def is_locked(self) -> bool:
        return self._locked_track_key is not None

    def _valid_candidates(self, frame: DetectionStreamFrame) -> list[TrackDetection]:
        candidates: list[TrackDetection] = []
        for track in frame.tracks:
            if track.class_name != self.config.target_class:
                continue
            if float(track.confidence) < float(self.config.min_confidence):
                continue
            if track.target_body_xy is None:
                continue
            x, y = float(track.target_body_xy[0]), float(track.target_body_xy[1])
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            if self.config.require_track_id and track.track_id is None:
                continue
            if track.depth_valid_ratio is not None and float(track.depth_valid_ratio) < float(self.config.min_depth_valid_ratio):
                continue
            if self.config.initial_track_id is not None and not self.is_locked:
                if _track_id_key(track.track_id) != _track_id_key(self.config.initial_track_id):
                    continue
            candidates.append(track)
        return candidates

    def _missing_state(
        self,
        *,
        now_s: float,
        reason: str,
        candidate_count: int = 0,
        ignored_track_ids: tuple[int | str | None, ...] = (),
        stream_metadata: Mapping[str, Any] | None = None,
        stale: bool = False,
    ) -> PrimaryTargetState:
        lost_s = 0.0 if self.last_seen_s is None else max(0.0, float(now_s) - float(self.last_seen_s))
        metadata = {
            "last_seen_s": self.last_seen_s,
            "last_frame_timestamp_s": self.last_frame_timestamp_s,
            "max_lost_s": float(self.config.max_lost_s),
        }
        if stream_metadata:
            metadata["stream"] = dict(stream_metadata)
        if self.last_known_target is not None and lost_s <= float(self.config.max_lost_s):
            state = PrimaryTargetState(
                state=PrimaryTargetLockStateKind.STALE if stale else PrimaryTargetLockStateKind.LOCKED_OCCLUDED,
                locked_track_id=self.locked_track_id,
                target=self.last_known_target,
                using_last_known=True,
                lost_s=lost_s,
                reason=reason,
                selected_track=None,
                candidate_count=candidate_count,
                ignored_track_ids=ignored_track_ids,
                metadata=metadata,
            )
        else:
            state = PrimaryTargetState(
                state=PrimaryTargetLockStateKind.LOST if self.is_locked else PrimaryTargetLockStateKind.UNLOCKED,
                locked_track_id=self.locked_track_id,
                target=None,
                using_last_known=False,
                lost_s=lost_s,
                reason=reason,
                selected_track=None,
                candidate_count=candidate_count,
                ignored_track_ids=ignored_track_ids,
                metadata=metadata,
            )
        self.last_state = state
        return state

    def update(
        self,
        frame: DetectionStreamFrame | None,
        *,
        now_s: float,
        stream_health: DetectionStreamHealth | str | None = None,
        allow_stale: bool = False,
        stream_metadata: Mapping[str, Any] | None = None,
    ) -> PrimaryTargetState:
        """Update lock state from one stream frame."""

        now_s = float(now_s)
        if not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if frame is None:
            return self._missing_state(now_s=now_s, reason="stream_unavailable", stream_metadata=stream_metadata)

        health_value = None
        if stream_health is not None:
            health_value = stream_health.value if isinstance(stream_health, DetectionStreamHealth) else str(stream_health)
        frame_age_s = max(0.0, now_s - float(frame.timestamp_s))
        is_stale = (health_value == DetectionStreamHealth.STALE.value) or frame_age_s > float(self.config.max_stream_age_s)
        if is_stale and not allow_stale:
            return self._missing_state(
                now_s=now_s,
                reason="stream_stale",
                stream_metadata={**dict(stream_metadata or {}), "frame_age_s": frame_age_s, "stream_health": health_value},
                stale=True,
            )

        candidates = self._valid_candidates(frame)
        candidate_count = len(candidates)
        ignored_track_ids: tuple[int | str | None, ...] = tuple(
            track.track_id for track in frame.tracks if track.class_name == self.config.target_class
        )

        selected: TrackDetection | None = None
        if self.is_locked:
            selected = next((track for track in candidates if _track_id_key(track.track_id) == self._locked_track_key), None)
            if selected is None and self.config.allow_track_switch:
                selected = max(candidates, key=lambda track: float(track.confidence), default=None)
        else:
            selected = max(candidates, key=lambda track: float(track.confidence), default=None)

        if selected is None:
            return self._missing_state(
                now_s=now_s,
                reason="locked_track_missing" if self.is_locked else "no_valid_candidate",
                candidate_count=candidate_count,
                ignored_track_ids=ignored_track_ids,
                stream_metadata={**dict(stream_metadata or {}), "frame_age_s": frame_age_s, "stream_health": health_value},
            )

        selected_target = _target_from_track(selected)
        jump_m = _target_distance(self.last_known_target, selected_target)
        if (
            self.last_known_target is not None
            and float(self.config.max_target_jump_m) > 0.0
            and jump_m > float(self.config.max_target_jump_m)
        ):
            return self._missing_state(
                now_s=now_s,
                reason="locked_track_jump_rejected",
                candidate_count=candidate_count,
                ignored_track_ids=ignored_track_ids,
                stream_metadata={
                    **dict(stream_metadata or {}),
                    "frame_age_s": frame_age_s,
                    "stream_health": health_value,
                    "jump_m": jump_m,
                    "max_target_jump_m": float(self.config.max_target_jump_m),
                    "rejected_track": selected.to_dict(),
                },
            )

        self.locked_track_id = selected.track_id
        self._locked_track_key = _track_id_key(selected.track_id)
        self.last_known_target = selected_target
        self.last_seen_s = now_s
        self.last_frame_timestamp_s = float(frame.timestamp_s)
        state = PrimaryTargetState(
            state=PrimaryTargetLockStateKind.LOCKED_VISIBLE,
            locked_track_id=self.locked_track_id,
            target=selected_target,
            using_last_known=False,
            lost_s=0.0,
            reason="initial_lock" if self.last_state.state == PrimaryTargetLockStateKind.UNLOCKED else "locked_track_visible",
            selected_track=selected,
            candidate_count=candidate_count,
            ignored_track_ids=tuple(track.track_id for track in frame.tracks if _track_id_key(track.track_id) != self._locked_track_key),
            metadata={
                **dict(stream_metadata or {}),
                "frame_age_s": frame_age_s,
                "stream_health": health_value,
                "selected_confidence": float(selected.confidence),
                "selected_depth_valid_ratio": selected.depth_valid_ratio,
            },
        )
        self.last_state = state
        return state

    def update_read_result(
        self,
        result: DetectionStreamReadResult,
        *,
        allow_stale: bool = False,
    ) -> PrimaryTargetState:
        return self.update(
            result.frame,
            now_s=float(result.status.timestamp_s),
            stream_health=result.status.health,
            allow_stale=allow_stale,
            stream_metadata={"status": result.status.message, "details": dict(result.status.details)},
        )


__all__ = [
    "PrimaryTargetLock",
    "PrimaryTargetLockConfig",
    "PrimaryTargetLockStateKind",
    "PrimaryTargetState",
]
