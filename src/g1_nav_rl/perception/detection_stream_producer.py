"""Utilities for producing :mod:`detection_stream` frames from RGB-D detections.

The functions in this module are hardware-neutral.  They convert detector output
(``Detection2D`` + aligned RGB-D frame) into ``TrackDetection`` records with a
body-frame target estimate.  RealSense/Isaac/video producers can all reuse this
projection layer, while the transport remains the shared atomic JSON stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from g1_nav_rl.navigation.simple_nav import Pose2D
from g1_nav_rl.perception.detection_stream import DetectionStreamFrame, TrackDetection
from g1_nav_rl.perception.target_from_bbox import (
    Detection2D,
    TargetFromBBoxConfig,
    bbox_depth_to_target2d,
    center_crop_bbox_xyxy,
    make_forward_camera_T_body_camera,
)


@dataclass(frozen=True)
class DetectionStreamProjectionConfig:
    """RGB-D-to-stream projection and filtering settings."""

    target_class: str | None = None
    min_confidence: float = 0.10
    min_bbox_area_px: float = 0.0
    min_depth_valid_ratio: float = 0.0
    depth_min_m: float = 0.20
    depth_max_m: float = 8.0
    depth_crop_frac: float = 0.50
    depth_percentile: float = 0.50
    stand_off_m: float = 0.80
    min_target_distance_m: float = 0.05
    clamp_bbox_to_image: bool = True
    use_forward_camera_convention: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def target_from_bbox_config(self) -> TargetFromBBoxConfig:
        return TargetFromBBoxConfig(
            depth_min_m=float(self.depth_min_m),
            depth_max_m=float(self.depth_max_m),
            bbox_center_crop_frac=float(self.depth_crop_frac),
            depth_percentile=float(self.depth_percentile),
            stand_off_distance_m=float(self.stand_off_m),
            min_target_distance_m=float(self.min_target_distance_m),
            clamp_bbox_to_image=bool(self.clamp_bbox_to_image),
        )


def camera_to_body_matrix_for_frame(frame: Any, *, use_forward_camera_convention: bool = True) -> np.ndarray:
    """Return a camera-to-body transform for an RGBDFrame-like object.

    The real G1 config may still contain a placeholder identity camera transform.
    For live RealSense/Isaac tests we usually want the explicit forward-facing
    optical convention unless ``--use-config-camera-transform`` was requested.
    """

    T_cfg = np.asarray(frame.T_body_camera.matrix, dtype=np.float64)
    if not use_forward_camera_convention:
        return T_cfg
    translation = T_cfg[:3, 3]
    return make_forward_camera_T_body_camera(translation_body=translation)


def detection_passes_basic_filters(detection: Detection2D, config: DetectionStreamProjectionConfig) -> bool:
    """Check class/confidence/area before depth projection."""

    if config.target_class and detection.class_name.lower() != str(config.target_class).lower():
        return False
    if float(detection.confidence) < float(config.min_confidence):
        return False
    if float(detection.area) < float(config.min_bbox_area_px):
        return False
    return True


def depth_valid_ratio_in_bbox(
    depth_m: Any,
    bbox_xyxy: Sequence[float],
    *,
    depth_min_m: float,
    depth_max_m: float,
    crop_frac: float,
) -> tuple[float, int, int, tuple[int, int, int, int]]:
    """Return valid-depth ratio/counts in the central bbox crop."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be HxW or HxWx1, got {depth.shape}")
    h, w = depth.shape
    crop = center_crop_bbox_xyxy(bbox_xyxy, crop_frac=float(crop_frac), width=int(w), height=int(h))
    x1, y1, x2, y2 = crop
    values = depth[y1:y2, x1:x2]
    total = int(values.size)
    if total <= 0:
        return 0.0, 0, 0, crop
    finite = np.isfinite(values) & (values >= float(depth_min_m)) & (values <= float(depth_max_m))
    valid = int(finite.sum())
    return float(valid / max(total, 1)), valid, total, crop


def detection_to_track(
    detection: Detection2D,
    frame: Any,
    *,
    config: DetectionStreamProjectionConfig,
    track_id: int | str | None = None,
    status: str = "tracked",
    body_exclusion_box: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrackDetection | None:
    """Project a single RGB-D detection into a stream ``TrackDetection``.

    Returns ``None`` if class/confidence/area/depth/body-exclusion filters reject
    the detection.  ``target_body_xy`` contains the standoff target, not the raw
    object center; the raw object and 3D point are included in track metadata.
    """

    if not detection_passes_basic_filters(detection, config):
        return None

    ratio, valid_count, total_count, crop_xyxy = depth_valid_ratio_in_bbox(
        frame.depth_m,
        detection.bbox_xyxy,
        depth_min_m=config.depth_min_m,
        depth_max_m=config.depth_max_m,
        crop_frac=config.depth_crop_frac,
    )
    if ratio < float(config.min_depth_valid_ratio):
        return None

    T_body_camera = camera_to_body_matrix_for_frame(frame, use_forward_camera_convention=config.use_forward_camera_convention)
    result = bbox_depth_to_target2d(
        detection,
        frame.depth_m,
        frame.intrinsics,
        T_body_camera,
        robot_pose=Pose2D(0.0, 0.0, 0.0),
        config=config.target_from_bbox_config(),
    )
    if result is None:
        return None

    point_body = np.asarray(result.point_world, dtype=np.float64).reshape(3)
    if body_exclusion_box is not None and body_exclusion_box.contains_point(point_body):
        return None

    extra: dict[str, Any] = {
        "projection": "rgbd_bbox_depth_to_body",
        "object_body_xy": [float(result.object_xy_w[0]), float(result.object_xy_w[1])],
        "point_body_xyz": [float(v) for v in point_body],
        "depth_crop_xyxy": [int(v) for v in crop_xyxy],
        "depth_valid_pixel_count": int(valid_count),
        "depth_total_pixel_count": int(total_count),
        "used_stand_off": bool(result.used_stand_off),
        "already_within_standoff": bool(result.already_within_standoff),
    }
    extra.update(dict(config.metadata))
    if metadata:
        extra.update(dict(metadata))

    return TrackDetection(
        track_id=track_id,
        class_name=detection.class_name,
        confidence=float(detection.confidence),
        bbox_xyxy=detection.bbox_xyxy,
        target_body_xy=(float(result.target.x), float(result.target.y)),
        depth_m=float(result.depth_m),
        depth_valid_ratio=float(ratio),
        status=str(status),
        metadata=extra,
    )


def make_detection_stream_frame_from_detections(
    *,
    frame: Any,
    detections: Iterable[Detection2D],
    config: DetectionStreamProjectionConfig,
    frame_id: int | str | None = None,
    source: str = "rgbd_detection_stream_producer",
    track_ids: Sequence[int | str | None] | None = None,
    statuses: Sequence[str] | None = None,
    body_exclusion_box: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DetectionStreamFrame:
    """Convert detector outputs from one RGB-D frame into a stream frame."""

    detections_list = list(detections)
    ids = list(track_ids) if track_ids is not None else [None] * len(detections_list)
    status_list = list(statuses) if statuses is not None else ["tracked"] * len(detections_list)
    if len(ids) != len(detections_list):
        raise ValueError("track_ids length must match detections length")
    if len(status_list) != len(detections_list):
        raise ValueError("statuses length must match detections length")

    tracks: list[TrackDetection] = []
    rejected = 0
    for det, track_id, status in zip(detections_list, ids, status_list):
        track = detection_to_track(
            det,
            frame,
            config=config,
            track_id=track_id,
            status=status,
            body_exclusion_box=body_exclusion_box,
        )
        if track is None:
            rejected += 1
            continue
        tracks.append(track)

    image_shape = tuple(int(v) for v in np.asarray(frame.rgb).shape)
    meta = {
        "raw_detection_count": len(detections_list),
        "published_track_count": len(tracks),
        "rejected_detection_count": int(rejected),
        "projection_config": {
            "target_class": config.target_class,
            "min_confidence": float(config.min_confidence),
            "min_bbox_area_px": float(config.min_bbox_area_px),
            "min_depth_valid_ratio": float(config.min_depth_valid_ratio),
            "stand_off_m": float(config.stand_off_m),
            "depth_min_m": float(config.depth_min_m),
            "depth_max_m": float(config.depth_max_m),
            "depth_crop_frac": float(config.depth_crop_frac),
            "depth_percentile": float(config.depth_percentile),
            "use_forward_camera_convention": bool(config.use_forward_camera_convention),
        },
    }
    if metadata:
        meta.update(dict(metadata))

    return DetectionStreamFrame(
        timestamp_s=float(frame.timestamp_s),
        frame_id=frame_id if frame_id is not None else getattr(frame, "frame_id", None),
        image_shape=image_shape,
        source=source,
        tracks=tuple(tracks),
        metadata=meta,
    )


__all__ = [
    "DetectionStreamProjectionConfig",
    "camera_to_body_matrix_for_frame",
    "depth_valid_ratio_in_bbox",
    "detection_passes_basic_filters",
    "detection_to_track",
    "make_detection_stream_frame_from_detections",
]
