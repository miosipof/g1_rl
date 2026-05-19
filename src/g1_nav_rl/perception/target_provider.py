"""Reusable RGB-D detection-to-navigation target provider.

This module separates the *policy* of acquiring a target from the mechanics of
Isaac, YOLO, or a specific depth tensor backend.  The provider accepts detector
outputs, optionally asks for a fallback detection, selects one detection, and
then delegates bbox/depth projection to a caller-supplied projection function.

The intent is to keep scripts such as ``smoke_test_yolo_person_nav.py`` focused
on simulation I/O while this module owns deterministic target acquisition
semantics:

- ``target_source='fallback'``: bypass detector detections and require/use the
  fallback detection factory.
- ``target_source='yolo'``: use detector detections only unless fallback is
  explicitly enabled.
- ``target_source='auto'``: use detector detections first, then fallback when
  enabled.

The module is deliberately dependency-light.  It does not import Isaac,
Ultralytics, Torch, or OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Protocol

from g1_nav_rl.navigation.simple_nav import Target2D
from g1_nav_rl.perception.target_from_bbox import Detection2D, select_detection_for_target


class TargetSource(str, Enum):
    """Supported target acquisition sources."""

    AUTO = "auto"
    YOLO = "yolo"
    FALLBACK = "fallback"


class TargetProviderStatus(str, Enum):
    """High-level acquisition outcome."""

    TARGET_READY = "target_ready"
    NO_DETECTION = "no_detection"
    PROJECTION_FAILED = "projection_failed"


class DetectionRunner(Protocol):
    """Callable detector protocol: RGB image in, detections out."""

    def __call__(self, rgb_image: Any) -> Iterable[Detection2D]:
        ...


class FallbackDetectionFactory(Protocol):
    """Callable fallback protocol used for synthetic/projected bboxes."""

    def __call__(self) -> Detection2D:
        ...


class TargetProjector(Protocol):
    """Callable projection protocol: selected bbox in, target result out."""

    def __call__(self, detection: Detection2D) -> Any | None:
        ...


@dataclass(frozen=True)
class TargetProviderConfig:
    """Deterministic target-acquisition policy."""

    target_source: TargetSource | str = TargetSource.AUTO
    target_class: str = "person"
    min_confidence: float = 0.25
    selection_policy: str = "nearest"
    fallback_enabled: bool = True
    assert_detection: bool = False
    assert_target: bool = False
    metadata_prefix: str = ""

    def normalized_source(self) -> TargetSource:
        if isinstance(self.target_source, TargetSource):
            return self.target_source
        try:
            return TargetSource(str(self.target_source))
        except ValueError as exc:
            raise ValueError(f"unknown target_source={self.target_source!r}; expected auto/yolo/fallback") from exc


@dataclass(frozen=True)
class SelectedDetection:
    """Selected detection plus all candidates and source label."""

    detection: Detection2D | None
    source: str
    detections: tuple[Detection2D, ...] = field(default_factory=tuple)
    reason: str = "ok"

    @property
    def found(self) -> bool:
        return self.detection is not None


@dataclass(frozen=True)
class TargetProviderResult:
    """Output of a target-acquisition attempt."""

    status: TargetProviderStatus
    selected: SelectedDetection
    target_result: Any | None = None
    reason: str = "ok"

    @property
    def target(self) -> Target2D | None:
        if self.target_result is None:
            return None
        return getattr(self.target_result, "target", None)

    @property
    def target_xy(self) -> tuple[float, float] | None:
        target = self.target
        if target is None:
            return None
        return float(target.x), float(target.y)

    @property
    def ready(self) -> bool:
        return self.status == TargetProviderStatus.TARGET_READY and self.target is not None

    def to_metadata(self, *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return JSON-friendly metadata for logs and smoke-test outputs."""
        detections = [detection_to_dict(d) for d in self.selected.detections]
        meta: dict[str, Any] = {
            "status": str(self.status.value),
            "reason": str(self.reason),
            "detection_source": str(self.selected.source),
            "detections": detections,
            "selected_detection": detection_to_dict(self.selected.detection) if self.selected.detection else None,
        }
        if self.target_result is not None:
            tr = self.target_result
            meta["target"] = {
                "depth_m": _maybe_float(getattr(tr, "depth_m", None)),
                "depth_source": getattr(tr, "depth_source", None),
                "object_xy_w": _xy_list(getattr(tr, "object_xy_w", None)),
                "target_xy_w": _target_xy_list(getattr(tr, "target", None)),
                "used_stand_off": _maybe_bool(getattr(tr, "used_stand_off", None)),
                "already_within_standoff": _maybe_bool(getattr(tr, "already_within_standoff", None)),
                "valid_depth_pixels": _maybe_int(getattr(tr, "valid_depth_pixels", None)),
                "depth_crop_xyxy": _seq_list(getattr(tr, "depth_crop_xyxy", None)),
                "reason": getattr(tr, "reason", None),
            }
        if extra:
            meta.update(dict(extra))
        return meta


def detection_to_dict(det: Detection2D | None) -> dict[str, Any] | None:
    if det is None:
        return None
    return {
        "class_name": str(det.class_name),
        "confidence": float(det.confidence),
        "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
        "area": float(det.area),
    }


def select_target_detection(
    detections: Iterable[Detection2D],
    *,
    config: TargetProviderConfig,
    depth_image: Any | None = None,
) -> Detection2D | None:
    """Select a detection according to ``TargetProviderConfig``."""
    return select_detection_for_target(
        detections,
        allowed_classes={str(config.target_class)},
        min_confidence=float(config.min_confidence),
        policy=str(config.selection_policy),
        depth_image=depth_image,
    )


class BBoxTargetProvider:
    """Acquire one navigation target from RGB detections and bbox projection.

    The provider itself does not know how to read camera tensors or compute
    depth.  Callers supply:

    - ``detector``: optional callable that returns ``Detection2D`` instances;
    - ``fallback_detection_factory``: optional callable for known/synthetic bbox;
    - ``target_projector``: callable that converts the selected detection into a
      ``TargetFromBBoxResult``-like object with a ``target`` attribute.
    """

    def __init__(self, config: TargetProviderConfig | None = None) -> None:
        self.config = config or TargetProviderConfig()

    def select(
        self,
        *,
        detections: Iterable[Detection2D] = (),
        fallback_detection_factory: FallbackDetectionFactory | None = None,
        depth_image: Any | None = None,
    ) -> SelectedDetection:
        cfg = self.config
        source = cfg.normalized_source()
        detections_tuple = tuple(detections)

        if source == TargetSource.FALLBACK:
            if fallback_detection_factory is None:
                if cfg.assert_detection:
                    raise RuntimeError("target_source=fallback but no fallback detection factory was provided")
                return SelectedDetection(None, "none", detections_tuple, "fallback_unavailable")
            return SelectedDetection(fallback_detection_factory(), "synthetic_projection", detections_tuple, "fallback_requested")

        selected = select_target_detection(detections_tuple, config=cfg, depth_image=depth_image)
        if selected is not None:
            return SelectedDetection(selected, "yolo", detections_tuple, "detector_selected")

        if cfg.fallback_enabled and source in (TargetSource.AUTO, TargetSource.YOLO):
            if fallback_detection_factory is not None:
                return SelectedDetection(fallback_detection_factory(), "synthetic_projection", detections_tuple, "fallback_after_no_detection")

        if cfg.assert_detection:
            raise RuntimeError("no detector/fallback target detection available")
        return SelectedDetection(None, "none", detections_tuple, "no_detection")

    def acquire(
        self,
        *,
        rgb_image: Any | None = None,
        detector: DetectionRunner | None = None,
        detections: Iterable[Detection2D] | None = None,
        fallback_detection_factory: FallbackDetectionFactory | None = None,
        target_projector: TargetProjector,
        depth_image_for_selection: Any | None = None,
    ) -> TargetProviderResult:
        """Run detection selection and project the selected bbox into a target."""
        cfg = self.config
        source = cfg.normalized_source()
        detection_list: tuple[Detection2D, ...]

        if detections is not None:
            detection_list = tuple(detections)
        elif detector is not None and source != TargetSource.FALLBACK:
            detection_list = tuple(detector(rgb_image))
        else:
            detection_list = tuple()

        selected = self.select(
            detections=detection_list,
            fallback_detection_factory=fallback_detection_factory,
            depth_image=depth_image_for_selection,
        )
        if selected.detection is None:
            return TargetProviderResult(TargetProviderStatus.NO_DETECTION, selected, None, selected.reason)

        target_result = target_projector(selected.detection)
        if target_result is None or getattr(target_result, "target", None) is None:
            if cfg.assert_target:
                raise RuntimeError("selected detection did not produce a valid Target2D")
            return TargetProviderResult(TargetProviderStatus.PROJECTION_FAILED, selected, None, "projection_failed")

        return TargetProviderResult(TargetProviderStatus.TARGET_READY, selected, target_result, "ok")


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _seq_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _xy_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        x, y = value[0], value[1]
        return [float(x), float(y)]
    except Exception:
        return None


def _target_xy_list(target: Any) -> list[float] | None:
    if target is None:
        return None
    try:
        return [float(target.x), float(target.y)]
    except Exception:
        return None
