"""Convert RGB-D detections into 2D navigation targets.

This module is the first Phase-5 perception bridge:

    Detection2D bbox + aligned depth image + camera intrinsics + T_world_camera
        -> Target2D(x, y)

It deliberately does not depend on Isaac, RealSense, OpenCV, YOLO, or the
navigation runtime. Detector backends only need to output ``Detection2D``.
Navigation modules only receive ``Target2D``.

Coordinate convention
---------------------
The pinhole unprojection uses the common RGB-D camera convention:

- camera x: image right
- camera y: image down
- camera z: forward along optical axis

The caller must provide ``T_world_camera`` that maps homogeneous camera-frame
points into the same world/map frame used by ``Pose2D`` and ``Target2D``.
For a forward-facing camera mounted on a robot whose body frame is x-forward,
y-left, z-up, a typical camera-to-body rotation is:

    body_x = camera_z
    body_y = -camera_x
    body_z = -camera_y

Use ``make_forward_camera_T_body_camera`` as a convenient helper for that case.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from g1_nav_rl.navigation.simple_nav import Pose2D, Target2D


BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection2D:
    """Single 2D detector output in RGB image coordinates."""

    class_name: str
    bbox_xyxy: BBoxXYXY
    confidence: float = 1.0

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return 0.5 * (float(x1) + float(x2)), 0.5 * (float(y1) + float(y2))

    @property
    def width(self) -> float:
        x1, _, x2, _ = self.bbox_xyxy
        return max(0.0, float(x2) - float(x1))

    @property
    def height(self) -> float:
        _, y1, _, y2 = self.bbox_xyxy
        return max(0.0, float(y2) - float(y1))

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for an RGB/depth image pair."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass(frozen=True)
class TargetFromBBoxConfig:
    """Depth and target-construction parameters."""

    depth_min_m: float = 0.20
    depth_max_m: float = 8.0
    bbox_center_crop_frac: float = 0.50
    depth_percentile: float = 0.50
    stand_off_distance_m: float = 0.0
    min_target_distance_m: float = 0.05
    clamp_bbox_to_image: bool = True


@dataclass(frozen=True)
class DepthEstimate:
    """Robust depth estimate extracted from a bbox crop."""

    depth_m: float
    valid_pixel_count: int
    crop_xyxy: tuple[int, int, int, int]
    percentile: float


@dataclass(frozen=True)
class TargetFromBBoxResult:
    """Result of converting one bbox to a navigation target."""

    target: Target2D
    object_xy_w: tuple[float, float]
    detection: Detection2D
    depth_m: float
    point_camera: np.ndarray
    point_world: np.ndarray
    used_stand_off: bool
    already_within_standoff: bool
    reason: str = "ok"


def _as_numpy_depth(depth_image: Any) -> np.ndarray:
    """Convert numpy/torch-like depth image into a 2D float numpy array."""
    if hasattr(depth_image, "detach"):
        depth_image = depth_image.detach().cpu().numpy()
    arr = np.asarray(depth_image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"depth_image must be 2D or HxWx1, got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def _validate_T_world_camera(T_world_camera: Any) -> np.ndarray:
    T = np.asarray(T_world_camera, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_world_camera must have shape (4,4), got {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError("T_world_camera contains non-finite values")
    return T


def clamp_bbox_xyxy(bbox_xyxy: Sequence[float], width: int, height: int) -> BBoxXYXY:
    """Clamp a bbox to image bounds and preserve float coordinates."""
    if len(bbox_xyxy) != 4:
        raise ValueError(f"bbox must contain four values, got {bbox_xyxy!r}")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    x1 = min(max(x1, 0.0), float(width - 1))
    x2 = min(max(x2, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    y2 = min(max(y2, 0.0), float(height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def center_crop_bbox_xyxy(
    bbox_xyxy: Sequence[float],
    crop_frac: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Return integer image crop bounds for the central part of a bbox."""
    if not (0.0 < float(crop_frac) <= 1.0):
        raise ValueError(f"crop_frac must be in (0,1], got {crop_frac}")
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox_xyxy, width=width, height=height)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    half_w = 0.5 * (x2 - x1) * float(crop_frac)
    half_h = 0.5 * (y2 - y1) * float(crop_frac)
    ix1 = int(math.floor(max(0.0, cx - half_w)))
    ix2 = int(math.ceil(min(float(width), cx + half_w)))
    iy1 = int(math.floor(max(0.0, cy - half_h)))
    iy2 = int(math.ceil(min(float(height), cy + half_h)))
    # Ensure non-empty crop.
    ix2 = max(ix2, ix1 + 1)
    iy2 = max(iy2, iy1 + 1)
    ix2 = min(ix2, int(width))
    iy2 = min(iy2, int(height))
    return ix1, iy1, ix2, iy2


def estimate_depth_in_bbox(
    depth_image: Any,
    bbox_xyxy: Sequence[float],
    *,
    depth_min_m: float = 0.20,
    depth_max_m: float = 8.0,
    bbox_center_crop_frac: float = 0.50,
    depth_percentile: float = 0.50,
) -> DepthEstimate | None:
    """Estimate robust object depth from valid pixels inside the central bbox crop.

    ``depth_percentile=0.50`` gives the median. Use lower percentiles, e.g. 0.25,
    if the bbox often contains background behind the object.
    """
    depth = _as_numpy_depth(depth_image)
    h, w = depth.shape
    ix1, iy1, ix2, iy2 = center_crop_bbox_xyxy(
        bbox_xyxy,
        crop_frac=bbox_center_crop_frac,
        width=w,
        height=h,
    )
    crop = depth[iy1:iy2, ix1:ix2]
    valid = crop[np.isfinite(crop)]
    valid = valid[(valid >= float(depth_min_m)) & (valid <= float(depth_max_m))]
    if valid.size == 0:
        return None
    q = min(max(float(depth_percentile), 0.0), 1.0)
    z = float(np.quantile(valid.astype(np.float64), q))
    return DepthEstimate(
        depth_m=z,
        valid_pixel_count=int(valid.size),
        crop_xyxy=(ix1, iy1, ix2, iy2),
        percentile=q,
    )


def unproject_pixel_to_camera(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Unproject one RGB/depth pixel into camera-frame XYZ.

    Camera convention: x right, y down, z forward.
    """
    z = float(depth_m)
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError(f"depth_m must be positive and finite, got {depth_m}")
    x = (float(u) - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (float(v) - float(intrinsics.cy)) * z / float(intrinsics.fy)
    return np.array([x, y, z], dtype=np.float64)


def transform_point(T: Any, point_xyz: Sequence[float]) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to a 3D point."""
    T_np = _validate_T_world_camera(T)
    p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    out = T_np @ ph
    return out[:3]


def pose2d_to_T_world_body(pose: Pose2D, z: float = 0.0) -> np.ndarray:
    """Build a planar world-from-body transform from Pose2D."""
    c = math.cos(float(pose.yaw))
    s = math.sin(float(pose.yaw))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T[:3, 3] = np.array([float(pose.x), float(pose.y), float(z)], dtype=np.float64)
    return T


def make_forward_camera_T_body_camera(
    *,
    translation_body: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return body-from-camera transform for a forward-facing RGB-D camera.

    Assumes:
    - camera x is image-right;
    - camera y is image-down;
    - camera z is optical forward;
    - body x is forward;
    - body y is left;
    - body z is up.

    Therefore: body_x=camera_z, body_y=-camera_x, body_z=-camera_y.
    """
    t = np.asarray(translation_body, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    T[:3, 3] = t
    return T


def target_xy_with_standoff(
    object_xy_w: tuple[float, float],
    robot_pose: Pose2D | None,
    stand_off_distance_m: float,
    min_target_distance_m: float,
) -> tuple[Target2D, bool, bool]:
    """Construct target at object XY or at a standoff point from robot pose."""
    obj_x, obj_y = float(object_xy_w[0]), float(object_xy_w[1])
    stand_off = max(0.0, float(stand_off_distance_m))
    if robot_pose is None or stand_off <= 0.0:
        return Target2D(obj_x, obj_y), False, False

    dx = obj_x - float(robot_pose.x)
    dy = obj_y - float(robot_pose.y)
    dist = math.hypot(dx, dy)
    if dist <= max(float(min_target_distance_m), 1e-9):
        return Target2D(float(robot_pose.x), float(robot_pose.y)), True, True
    if dist <= stand_off:
        return Target2D(float(robot_pose.x), float(robot_pose.y)), True, True
    scale = (dist - stand_off) / dist
    return Target2D(float(robot_pose.x) + dx * scale, float(robot_pose.y) + dy * scale), True, False


def bbox_depth_to_target2d(
    detection: Detection2D,
    depth_image: Any,
    intrinsics: CameraIntrinsics,
    T_world_camera: Any,
    *,
    robot_pose: Pose2D | None = None,
    config: TargetFromBBoxConfig | None = None,
) -> TargetFromBBoxResult | None:
    """Convert one 2D bbox and aligned depth image into a ``Target2D``.

    Returns ``None`` if no valid depth is available inside the bbox crop.
    """
    cfg = config or TargetFromBBoxConfig()
    depth = _as_numpy_depth(depth_image)
    h, w = depth.shape
    if int(intrinsics.width) != int(w) or int(intrinsics.height) != int(h):
        raise ValueError(
            "intrinsics resolution does not match depth image: "
            f"intrinsics=({intrinsics.width},{intrinsics.height}) depth=({w},{h})"
        )

    bbox = detection.bbox_xyxy
    if cfg.clamp_bbox_to_image:
        bbox = clamp_bbox_xyxy(bbox, width=w, height=h)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None

    depth_est = estimate_depth_in_bbox(
        depth,
        bbox,
        depth_min_m=cfg.depth_min_m,
        depth_max_m=cfg.depth_max_m,
        bbox_center_crop_frac=cfg.bbox_center_crop_frac,
        depth_percentile=cfg.depth_percentile,
    )
    if depth_est is None:
        return None

    u = 0.5 * (bbox[0] + bbox[2])
    v = 0.5 * (bbox[1] + bbox[3])
    point_camera = unproject_pixel_to_camera(u, v, depth_est.depth_m, intrinsics)
    point_world = transform_point(T_world_camera, point_camera)
    object_xy_w = (float(point_world[0]), float(point_world[1]))
    target, used_stand_off, already_within_standoff = target_xy_with_standoff(
        object_xy_w,
        robot_pose=robot_pose,
        stand_off_distance_m=cfg.stand_off_distance_m,
        min_target_distance_m=cfg.min_target_distance_m,
    )
    return TargetFromBBoxResult(
        target=target,
        object_xy_w=object_xy_w,
        detection=Detection2D(
            class_name=detection.class_name,
            bbox_xyxy=bbox,
            confidence=float(detection.confidence),
        ),
        depth_m=float(depth_est.depth_m),
        point_camera=point_camera,
        point_world=point_world,
        used_stand_off=used_stand_off,
        already_within_standoff=already_within_standoff,
        reason="ok",
    )


# Alias used by higher-level target-provider code.
target_from_detection = bbox_depth_to_target2d


def select_detection_for_target(
    detections: Iterable[Detection2D],
    *,
    allowed_classes: set[str] | None = None,
    min_confidence: float = 0.0,
    policy: str = "nearest",
    depth_image: Any | None = None,
    depth_min_m: float = 0.20,
    depth_max_m: float = 8.0,
) -> Detection2D | None:
    """Select one detection using a minimal deterministic policy.

    Policies:
    - ``largest``: largest bbox area after class/conf filtering.
    - ``highest_confidence``: highest confidence.
    - ``nearest``: nearest robust bbox depth; requires ``depth_image``.
      Falls back to largest if depth is omitted or all depths are invalid.
    """
    candidates = []
    allowed_norm = {c.lower() for c in allowed_classes} if allowed_classes is not None else None
    for det in detections:
        if allowed_norm is not None and det.class_name.lower() not in allowed_norm:
            continue
        if float(det.confidence) < float(min_confidence):
            continue
        candidates.append(det)
    if not candidates:
        return None

    if policy == "highest_confidence":
        return max(candidates, key=lambda d: float(d.confidence))
    if policy == "largest":
        return max(candidates, key=lambda d: float(d.area))
    if policy != "nearest":
        raise ValueError(f"unknown target selection policy: {policy!r}")

    if depth_image is not None:
        depth = _as_numpy_depth(depth_image)
        scored: list[tuple[float, Detection2D]] = []
        for det in candidates:
            est = estimate_depth_in_bbox(
                depth,
                det.bbox_xyxy,
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
                bbox_center_crop_frac=0.50,
                depth_percentile=0.50,
            )
            if est is not None:
                scored.append((float(est.depth_m), det))
        if scored:
            return min(scored, key=lambda item: item[0])[1]
    return max(candidates, key=lambda d: float(d.area))
