"""Detection-stream schema and file client for asynchronous target perception.

This module defines the hardware-neutral boundary between a future streaming
YOLO/tracker process and the G1 follow controller.  It intentionally has no
RealSense, YOLO, OpenCV, Torch, TensorRT, ZeroMQ, or Unitree imports.

The first transport is a debuggable atomic JSON file, e.g.::

    outputs/runtime/latest_tracks.json

A detector process can update that file at its own rate while the controller
polls the latest frame without running inference in the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence



class DetectionStreamHealth(str, Enum):
    """Health state for a detection stream read."""

    OK = "ok"
    STALE = "stale"
    UNAVAILABLE = "unavailable"
    FAULT = "fault"


@dataclass(frozen=True)
class DetectionStreamStatus:
    """Transport-neutral status for detection stream reads."""

    name: str
    health: DetectionStreamHealth
    timestamp_s: float
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


DETECTION_STREAM_SCHEMA_VERSION = 1
Clock = Callable[[], float]


@dataclass(frozen=True)
class TrackDetection:
    """One tracked 2D detection with optional body-frame target estimate.

    Coordinates:
    - ``bbox_xyxy`` is image-space ``[x1, y1, x2, y2]`` in pixels.
    - ``target_body_xy`` is robot body-frame ``[x, y]`` in meters, where
      ``+x`` is forward and ``+y`` is robot-left. It may be ``None`` when depth
      projection is unavailable or failed.
    """

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int | str | None = None
    target_body_xy: tuple[float, float] | None = None
    depth_m: float | None = None
    depth_valid_ratio: float | None = None
    status: str = "tracked"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.class_name:
            raise ValueError("TrackDetection.class_name must be non-empty")
        conf = float(self.confidence)
        if not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
            raise ValueError(f"TrackDetection.confidence must be finite in [0,1], got {self.confidence!r}")
        object.__setattr__(self, "confidence", conf)

        bbox = _tuple_floats(self.bbox_xyxy, 4, name="bbox_xyxy")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"TrackDetection.bbox_xyxy must have x2>x1 and y2>y1, got {bbox}")
        object.__setattr__(self, "bbox_xyxy", bbox)

        if self.target_body_xy is not None:
            target = _tuple_floats(self.target_body_xy, 2, name="target_body_xy")
            object.__setattr__(self, "target_body_xy", target)

        if self.depth_m is not None:
            depth = float(self.depth_m)
            if not math.isfinite(depth) or depth <= 0.0:
                raise ValueError(f"TrackDetection.depth_m must be positive finite, got {self.depth_m!r}")
            object.__setattr__(self, "depth_m", depth)

        if self.depth_valid_ratio is not None:
            ratio = float(self.depth_valid_ratio)
            if not math.isfinite(ratio) or not (0.0 <= ratio <= 1.0):
                raise ValueError(
                    "TrackDetection.depth_valid_ratio must be finite in [0,1], "
                    f"got {self.depth_valid_ratio!r}"
                )
            object.__setattr__(self, "depth_valid_ratio", ratio)

        if not self.status:
            raise ValueError("TrackDetection.status must be non-empty")

    @property
    def bbox_center_uv(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    @property
    def bbox_area_px(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, (x2 - x1) * (y2 - y1))

    @property
    def has_target_body_xy(self) -> bool:
        return self.target_body_xy is not None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "bbox_center_uv": [float(v) for v in self.bbox_center_uv],
            "bbox_area_px": float(self.bbox_area_px),
            "target_body_xy": None if self.target_body_xy is None else [float(v) for v in self.target_body_xy],
            "depth_m": None if self.depth_m is None else float(self.depth_m),
            "depth_valid_ratio": None if self.depth_valid_ratio is None else float(self.depth_valid_ratio),
            "status": self.status,
            "metadata": dict(self.metadata),
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackDetection":
        data = _mapping(data, "track")
        return cls(
            track_id=data.get("track_id"),
            class_name=str(data.get("class_name", "")),
            confidence=float(data.get("confidence", 0.0)),
            bbox_xyxy=_tuple_floats(data.get("bbox_xyxy"), 4, name="track.bbox_xyxy"),
            target_body_xy=(
                None
                if data.get("target_body_xy") is None
                else _tuple_floats(data.get("target_body_xy"), 2, name="track.target_body_xy")
            ),
            depth_m=None if data.get("depth_m") is None else float(data.get("depth_m")),
            depth_valid_ratio=(
                None if data.get("depth_valid_ratio") is None else float(data.get("depth_valid_ratio"))
            ),
            status=str(data.get("status", "tracked")),
            metadata=_mapping(data.get("metadata", {}), "track.metadata"),
        )


@dataclass(frozen=True)
class DetectionStreamFrame:
    """One published frame from a detector/tracker stream."""

    timestamp_s: float
    tracks: tuple[TrackDetection, ...] = ()
    frame_id: int | str | None = None
    image_shape: tuple[int, int, int] | tuple[int, int] | None = None
    source: str = "detection_stream"
    schema_version: int = DETECTION_STREAM_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ts = float(self.timestamp_s)
        if not math.isfinite(ts):
            raise ValueError("DetectionStreamFrame.timestamp_s must be finite")
        object.__setattr__(self, "timestamp_s", ts)
        if int(self.schema_version) != DETECTION_STREAM_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported detection stream schema_version={self.schema_version}; "
                f"expected {DETECTION_STREAM_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "tracks", tuple(self.tracks))
        if not all(isinstance(track, TrackDetection) for track in self.tracks):
            raise TypeError("DetectionStreamFrame.tracks must contain TrackDetection objects")
        if self.image_shape is not None:
            shape = tuple(int(v) for v in self.image_shape)
            if len(shape) not in {2, 3} or any(v <= 0 for v in shape):
                raise ValueError(f"DetectionStreamFrame.image_shape must be HxW or HxWxC positive ints, got {shape}")
            object.__setattr__(self, "image_shape", shape)
        if not self.source:
            raise ValueError("DetectionStreamFrame.source must be non-empty")

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - float(self.timestamp_s))

    def is_fresh(self, *, now_s: float, max_age_s: float) -> bool:
        return self.age_s(now_s) <= float(max_age_s)

    def tracks_for_class(self, class_name: str) -> tuple[TrackDetection, ...]:
        return tuple(track for track in self.tracks if track.class_name == class_name)

    def track_by_id(self, track_id: int | str) -> TrackDetection | None:
        return next((track for track in self.tracks if track.track_id == track_id), None)

    def best_track(
        self,
        *,
        class_name: str | None = None,
        min_confidence: float = 0.0,
        require_target_body_xy: bool = True,
        track_id: int | str | None = None,
    ) -> TrackDetection | None:
        """Return the highest-confidence matching track.

        This is intentionally a simple selector, not the target-lock manager.  A
        later layer should maintain target identity and decide when to switch or
        reacquire.  This helper is useful for basic stream-client smoke tests.
        """

        candidates: Iterable[TrackDetection] = self.tracks
        if track_id is not None:
            candidates = [track for track in candidates if track.track_id == track_id]
        if class_name is not None:
            candidates = [track for track in candidates if track.class_name == class_name]
        candidates = [track for track in candidates if track.confidence >= float(min_confidence)]
        if require_target_body_xy:
            candidates = [track for track in candidates if track.target_body_xy is not None]
        return max(candidates, key=lambda track: track.confidence, default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "timestamp_s": float(self.timestamp_s),
            "frame_id": self.frame_id,
            "source": self.source,
            "image_shape": None if self.image_shape is None else [int(v) for v in self.image_shape],
            "tracks": [track.to_dict() for track in self.tracks],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectionStreamFrame":
        data = _mapping(data, "detection_stream_frame")
        tracks_raw = data.get("tracks", [])
        if not isinstance(tracks_raw, Sequence) or isinstance(tracks_raw, (str, bytes, bytearray)):
            raise ValueError("detection_stream_frame.tracks must be a sequence")
        return cls(
            schema_version=int(data.get("schema_version", DETECTION_STREAM_SCHEMA_VERSION)),
            timestamp_s=float(data.get("timestamp_s")),
            frame_id=data.get("frame_id"),
            image_shape=None if data.get("image_shape") is None else tuple(int(v) for v in data.get("image_shape")),
            source=str(data.get("source", "detection_stream")),
            tracks=tuple(TrackDetection.from_dict(track) for track in tracks_raw),
            metadata=_mapping(data.get("metadata", {}), "detection_stream_frame.metadata"),
        )

    @classmethod
    def from_json_text(cls, text: str) -> "DetectionStreamFrame":
        data = json.loads(text)
        return cls.from_dict(data)

    def to_json_text(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class DetectionStreamReadResult:
    """Read result with status, allowing callers to distinguish stale/missing."""

    frame: DetectionStreamFrame | None
    status: DetectionStreamStatus

    @property
    def ok(self) -> bool:
        return self.frame is not None and self.status.health == DetectionStreamHealth.OK


@dataclass
class DetectionStreamClient:
    """Poll a latest-track JSON file written by a streaming detector.

    The producer should write atomically with :func:`write_detection_stream_frame_atomic`
    or equivalent ``tmp + rename`` semantics to avoid partial reads.
    """

    path: str | Path
    max_age_s: float = 1.0
    clock: Clock = time.monotonic
    source_name: str = "detection_stream_client"
    missing_ok: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if float(self.max_age_s) <= 0.0:
            raise ValueError("DetectionStreamClient.max_age_s must be positive")

    def read(self) -> DetectionStreamReadResult:
        now = float(self.clock())
        if not self.path.exists():
            status = DetectionStreamStatus(
                name=self.source_name,
                health=DetectionStreamHealth.UNAVAILABLE,
                timestamp_s=now,
                message=f"detection stream file not found: {self.path}",
                details={"path": str(self.path)},
            )
            if not self.missing_ok:
                raise FileNotFoundError(str(self.path))
            return DetectionStreamReadResult(frame=None, status=status)

        try:
            text = self.path.read_text(encoding="utf-8")
            frame = DetectionStreamFrame.from_json_text(text)
        except Exception as exc:  # noqa: BLE001 - surface parse errors as status
            return DetectionStreamReadResult(
                frame=None,
                status=DetectionStreamStatus(
                    name=self.source_name,
                    health=DetectionStreamHealth.FAULT,
                    timestamp_s=now,
                    message=f"failed to read detection stream: {exc}",
                    details={"path": str(self.path), "error": repr(exc)},
                ),
            )

        age = frame.age_s(now)
        health = DetectionStreamHealth.OK if age <= float(self.max_age_s) else DetectionStreamHealth.STALE
        message = "detection stream fresh" if health == DetectionStreamHealth.OK else f"detection stream stale: {age:.3f}s"
        return DetectionStreamReadResult(
            frame=frame,
            status=DetectionStreamStatus(
                name=self.source_name,
                health=health,
                timestamp_s=now,
                message=message,
                details={
                    "path": str(self.path),
                    "frame_timestamp_s": frame.timestamp_s,
                    "age_s": age,
                    "track_count": len(frame.tracks),
                    "schema_version": frame.schema_version,
                },
            ),
        )

    def read_latest(self, *, require_fresh: bool = True) -> DetectionStreamFrame | None:
        result = self.read()
        if result.frame is None:
            return None
        if require_fresh and result.status.health != DetectionStreamHealth.OK:
            return None
        return result.frame

    def best_track(
        self,
        *,
        class_name: str,
        min_confidence: float = 0.0,
        require_target_body_xy: bool = True,
        track_id: int | str | None = None,
        require_fresh: bool = True,
    ) -> TrackDetection | None:
        frame = self.read_latest(require_fresh=require_fresh)
        if frame is None:
            return None
        return frame.best_track(
            class_name=class_name,
            min_confidence=min_confidence,
            require_target_body_xy=require_target_body_xy,
            track_id=track_id,
        )


def write_detection_stream_frame_atomic(
    frame: DetectionStreamFrame,
    path: str | Path,
    *,
    indent: int | None = None,
) -> Path:
    """Atomically write one detection stream frame to ``path`` as JSON."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(frame.to_json_text(indent=indent) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    return out_path


def _mapping(data: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(data).__name__}")
    return data


def _tuple_floats(values: Any, n: int, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence of length {n}")
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}, got {len(values)}")
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} contains non-finite values: {out}")
    return out


__all__ = [
    "DETECTION_STREAM_SCHEMA_VERSION",
    "DetectionStreamClient",
    "DetectionStreamHealth",
    "DetectionStreamStatus",
    "DetectionStreamFrame",
    "DetectionStreamReadResult",
    "TrackDetection",
    "write_detection_stream_frame_atomic",
]
