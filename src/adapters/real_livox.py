"""Livox/MID-360 point-cloud adapter helpers for the shared interface.

The helpers mirror the conventions used in ``unitree_g1_vibes``:

- ``mount_mode='upside_down'`` applies a 180 degree correction about sensor X:
  ``(x, y, z) -> (x, -y, -z)``.
- ``T_body_lidar`` maps corrected LiDAR-frame points into the robot body frame.
- filtered body-frame points become :class:`adapters.types.LidarFrame`.

No Livox SDK, KISS-ICP, Open3D, or Unitree dependencies are imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Mapping

import numpy as np

from .config import BodyExclusionBox, LidarConfig
from .types import LidarFrame, Pose2D, Transform3D


_VALID_MOUNTS = {"normal", "upside_down"}


@dataclass(frozen=True)
class LivoxAdapterConfig:
    """Runtime options for converting Livox point clouds to LidarFrame."""

    source_frame_id: str = "livox_lidar"
    frame_id: str = "body"
    mount_mode: str = "upside_down"
    min_range_m: float = 0.15
    max_range_m: float = 8.0
    z_min_m: float = 0.05
    z_max_m: float = 1.30
    max_points: int | None = None
    T_body_lidar: Transform3D = field(
        default_factory=lambda: Transform3D(
            matrix=np.eye(4, dtype=np.float64),
            source_frame="livox_lidar",
            target_frame="body",
        )
    )
    body_exclusion_box: BodyExclusionBox | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_lidar_config(cls, config: LidarConfig) -> "LivoxAdapterConfig":
        return cls(
            source_frame_id=config.source_frame_id,
            frame_id=config.frame_id,
            mount_mode=config.mount_mode,
            min_range_m=config.min_range_m,
            max_range_m=config.max_range_m,
            z_min_m=config.z_min_m,
            z_max_m=config.z_max_m,
            max_points=config.max_points,
            T_body_lidar=config.T_body_lidar,
            body_exclusion_box=config.body_exclusion_box,
            metadata=dict(config.metadata),
        )

    def __post_init__(self) -> None:
        if self.mount_mode not in _VALID_MOUNTS:
            raise ValueError(f"mount_mode must be one of {_VALID_MOUNTS}, got {self.mount_mode!r}")
        if self.frame_id != "body":
            raise ValueError(f"LivoxAdapterConfig.frame_id must be 'body', got {self.frame_id!r}")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must be greater than min_range_m")
        if self.z_max_m <= self.z_min_m:
            raise ValueError("z_max_m must be greater than z_min_m")


def as_points_xyz(points_xyz: np.ndarray, *, name: str = "points_xyz") -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape Nx3, got {points.shape}")
    finite = np.isfinite(points).all(axis=1)
    return points[finite].astype(np.float32, copy=False)


def apply_mount_correction(points_lidar_xyz: np.ndarray, *, mount_mode: str) -> np.ndarray:
    """Apply the Livox mount correction used by unitree_g1_vibes."""

    if mount_mode not in _VALID_MOUNTS:
        raise ValueError(f"mount_mode must be one of {_VALID_MOUNTS}, got {mount_mode!r}")
    points = as_points_xyz(points_lidar_xyz, name="points_lidar_xyz")
    if mount_mode == "normal" or points.size == 0:
        return points.copy()
    return points * np.array([1.0, -1.0, -1.0], dtype=np.float32)


def transform_lidar_points_to_body(points_lidar_xyz: np.ndarray, *, T_body_lidar: Transform3D) -> np.ndarray:
    """Transform corrected LiDAR-frame points into robot body frame."""

    if T_body_lidar.target_frame != "body":
        raise ValueError(f"T_body_lidar target_frame must be 'body', got {T_body_lidar.target_frame!r}")
    points = as_points_xyz(points_lidar_xyz, name="points_lidar_xyz")
    return T_body_lidar.transform_points(points)


def filter_body_lidar_points(
    points_body_xyz: np.ndarray,
    *,
    min_range_m: float,
    max_range_m: float,
    z_min_m: float,
    z_max_m: float,
    max_points: int | None = None,
    body_exclusion_box: BodyExclusionBox | None = None,
) -> np.ndarray:
    """Filter body-frame LiDAR points for 2D navigation occupancy."""

    points = as_points_xyz(points_body_xyz, name="points_body_xyz")
    if points.size == 0:
        return points
    min_range = float(min_range_m)
    max_range = float(max_range_m)
    if max_range <= min_range:
        raise ValueError("max_range_m must be greater than min_range_m")
    if z_max_m <= z_min_m:
        raise ValueError("z_max_m must be greater than z_min_m")

    xy_range = np.linalg.norm(points[:, :2], axis=1)
    keep = (
        (xy_range >= min_range)
        & (xy_range <= max_range)
        & (points[:, 2] >= float(z_min_m))
        & (points[:, 2] <= float(z_max_m))
    )
    if body_exclusion_box is not None:
        keep = keep & ~body_exclusion_box.contains_points(points)
    out = points[keep].astype(np.float32, copy=False)
    if max_points is not None and int(max_points) > 0 and out.shape[0] > int(max_points):
        # Deterministic decimation keeps tests reproducible and avoids hidden RNG.
        idx = np.linspace(0, out.shape[0] - 1, num=int(max_points), dtype=np.int64)
        out = out[idx]
    return out


def make_lidar_frame_from_livox_points(
    *,
    timestamp_s: float,
    points_lidar_xyz: np.ndarray,
    T_body_lidar: Transform3D,
    source_frame_id: str = "livox_lidar",
    mount_mode: str = "upside_down",
    min_range_m: float = 0.15,
    max_range_m: float = 8.0,
    z_min_m: float = 0.05,
    z_max_m: float = 1.30,
    max_points: int | None = None,
    body_exclusion_box: BodyExclusionBox | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LidarFrame:
    """Normalize raw/corrected Livox points into a body-frame LidarFrame."""

    corrected = apply_mount_correction(points_lidar_xyz, mount_mode=mount_mode)
    points_body = transform_lidar_points_to_body(corrected, T_body_lidar=T_body_lidar)
    filtered = filter_body_lidar_points(
        points_body,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        max_points=max_points,
        body_exclusion_box=body_exclusion_box,
    )
    return LidarFrame(
        timestamp_s=float(timestamp_s),
        points_body_xyz=filtered,
        frame_id="body",
        source_frame_id=source_frame_id,
        T_body_lidar=T_body_lidar,
        metadata={
            **dict(metadata or {}),
            "mount_mode": mount_mode,
            "raw_points": int(as_points_xyz(points_lidar_xyz, name="points_lidar_xyz").shape[0]),
            "corrected_points": int(corrected.shape[0]),
            "filtered_points": int(filtered.shape[0]),
            "body_exclusion_box": None if body_exclusion_box is None else body_exclusion_box.to_dict(),
        },
    )


def make_lidar_frame_from_config(
    *,
    timestamp_s: float,
    points_lidar_xyz: np.ndarray,
    config: LivoxAdapterConfig,
    metadata: Mapping[str, Any] | None = None,
) -> LidarFrame:
    return make_lidar_frame_from_livox_points(
        timestamp_s=timestamp_s,
        points_lidar_xyz=points_lidar_xyz,
        T_body_lidar=config.T_body_lidar,
        source_frame_id=config.source_frame_id,
        mount_mode=config.mount_mode,
        min_range_m=config.min_range_m,
        max_range_m=config.max_range_m,
        z_min_m=config.z_min_m,
        z_max_m=config.z_max_m,
        max_points=config.max_points,
        body_exclusion_box=config.body_exclusion_box,
        metadata={**dict(config.metadata), **dict(metadata or {})},
    )


def pose2d_from_transform(T_map_body: np.ndarray) -> Pose2D:
    """Extract Pose2D(x, y, yaw) from a homogeneous map->body transform."""

    T = np.asarray(T_map_body, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_map_body must have shape (4,4), got {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError("T_map_body contains non-finite values")
    yaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    return Pose2D(x=float(T[0, 3]), y=float(T[1, 3]), yaw=yaw)


def body_pose_from_slam_lidar_pose(
    *,
    T_map_lidar: np.ndarray,
    T_body_lidar: Transform3D,
) -> tuple[Pose2D, Transform3D]:
    """Convert a SLAM LiDAR pose into a body pose.

    ``T_map_lidar`` maps LiDAR-frame points into the SLAM/map frame.  The
    configured ``T_body_lidar`` maps LiDAR-frame points into the body frame.
    Therefore ``T_map_body = T_map_lidar @ inv(T_body_lidar)``.
    """

    T_map_lidar_arr = np.asarray(T_map_lidar, dtype=np.float64)
    if T_map_lidar_arr.shape != (4, 4):
        raise ValueError(f"T_map_lidar must have shape (4,4), got {T_map_lidar_arr.shape}")
    T_lidar_body = np.linalg.inv(T_body_lidar.matrix)
    T_map_body = T_map_lidar_arr @ T_lidar_body
    transform = Transform3D(matrix=T_map_body, source_frame="body", target_frame="map")
    return pose2d_from_transform(T_map_body), transform


class LatestLivoxFrameBuffer:
    """Thread-compatible buffer for the latest Livox point cloud.

    This is intentionally SDK-agnostic.  A Livox callback can call ``update``;
    the bridge can call ``read_lidar_frame``.
    """

    def __init__(self, config: LivoxAdapterConfig) -> None:
        self.config = config
        self._points: np.ndarray | None = None
        self._timestamp_s: float | None = None

    def update(self, points_lidar_xyz: np.ndarray, *, timestamp_s: float | None = None) -> None:
        self._points = as_points_xyz(points_lidar_xyz, name="points_lidar_xyz").copy()
        self._timestamp_s = time.monotonic() if timestamp_s is None else float(timestamp_s)

    @property
    def has_frame(self) -> bool:
        return self._points is not None and self._timestamp_s is not None

    def read_lidar_frame(self) -> LidarFrame:
        if self._points is None or self._timestamp_s is None:
            raise RuntimeError("No Livox frame has been received yet")
        return make_lidar_frame_from_config(
            timestamp_s=self._timestamp_s,
            points_lidar_xyz=self._points,
            config=self.config,
            metadata={"source": "latest_livox_buffer"},
        )


__all__ = [
    "LatestLivoxFrameBuffer",
    "LivoxAdapterConfig",
    "apply_mount_correction",
    "as_points_xyz",
    "body_pose_from_slam_lidar_pose",
    "filter_body_lidar_points",
    "make_lidar_frame_from_config",
    "make_lidar_frame_from_livox_points",
    "pose2d_from_transform",
    "transform_lidar_points_to_body",
]


def lidar_frame_stats(frame: LidarFrame) -> dict[str, Any]:
    """Return compact numeric diagnostics for a normalized LiDAR frame."""

    points = np.asarray(frame.points_body_xyz, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1) if points.ndim == 2 and points.shape[1] == 3 else np.zeros((0,), dtype=bool)
    finite_points = points[finite] if points.size else points.reshape(0, 3)
    stats: dict[str, Any] = {
        "timestamp_s": float(frame.timestamp_s),
        "frame_id": frame.frame_id,
        "source_frame_id": frame.source_frame_id,
        "point_count": int(frame.point_count),
        "finite_point_count": int(finite_points.shape[0]),
        "metadata": dict(frame.metadata),
    }
    if finite_points.size:
        xy_range = np.linalg.norm(finite_points[:, :2], axis=1)
        stats.update(
            {
                "min_xyz_m": [float(v) for v in finite_points.min(axis=0)],
                "max_xyz_m": [float(v) for v in finite_points.max(axis=0)],
                "mean_xyz_m": [float(v) for v in finite_points.mean(axis=0)],
                "min_xy_range_m": float(xy_range.min()),
                "max_xy_range_m": float(xy_range.max()),
                "mean_xy_range_m": float(xy_range.mean()),
            }
        )
    return stats


class NpyLivoxReplayAdapter:
    """Small replay source for saved ``.npy`` Livox point clouds.

    The file must contain either ``Nx3`` points or ``FxNx3`` frames in Livox
    source-frame coordinates. Each call to ``read_lidar_frame`` returns one
    normalized :class:`LidarFrame` using the same conversion path as the live
    adapter. This is useful for local development without MID-360 hardware.
    """

    def __init__(self, path: str | Path, config: LivoxAdapterConfig, *, loop: bool = False, clock: Any = time.monotonic):
        from pathlib import Path as _Path

        self.path = _Path(path)
        self.config = config
        self.loop = bool(loop)
        self.clock = clock
        arr = np.load(self.path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            arr = arr[None, ...]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected .npy with shape Nx3 or FxNx3, got {arr.shape}")
        self.frames = arr
        self.index = 0

    def read_lidar_frame(self) -> LidarFrame:
        if self.index >= self.frames.shape[0]:
            if not self.loop:
                raise RuntimeError("NpyLivoxReplayAdapter reached end of replay")
            self.index = 0
        idx = self.index
        points = self.frames[idx]
        self.index += 1
        return make_lidar_frame_from_config(
            timestamp_s=float(self.clock()),
            points_lidar_xyz=points,
            config=self.config,
            metadata={"source": "npy_replay", "path": str(self.path), "frame_index": int(idx)},
        )


@dataclass
class UnitreeVibesLivox2AdapterConfig:
    """Configuration for lazily wrapping ``unitree_g1_vibes/livox2_python.py``."""

    livox_config_path: str
    host_ip: str
    adapter_config: LivoxAdapterConfig
    unitree_vibes_dir: str | None = None
    frame_time_s: float = 0.20
    frame_packets: int = 120
    read_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.livox_config_path:
            raise ValueError("livox_config_path is required")
        if not self.host_ip:
            raise ValueError("host_ip is required")
        if self.frame_time_s <= 0.0:
            raise ValueError("frame_time_s must be > 0")
        if self.frame_packets <= 0:
            raise ValueError("frame_packets must be > 0")
        if self.read_timeout_s <= 0.0:
            raise ValueError("read_timeout_s must be > 0")


class UnitreeVibesLivox2Adapter:
    """Lazy live Livox adapter backed by ``unitree_g1_vibes.Livox2``.

    No Livox SDK or ``unitree_g1_vibes`` import happens until ``start`` is
    called. This keeps local tests independent from the MID-360 driver stack.
    """

    def __init__(self, config: UnitreeVibesLivox2AdapterConfig) -> None:
        self.config = config
        self.buffer = LatestLivoxFrameBuffer(config.adapter_config)
        self._driver: Any | None = None
        self._driver_cls: Any | None = None

    @property
    def running(self) -> bool:
        return self._driver is not None

    def start(self) -> None:
        if self._driver is not None:
            return
        import importlib
        import sys

        if self.config.unitree_vibes_dir:
            if self.config.unitree_vibes_dir not in sys.path:
                sys.path.insert(0, self.config.unitree_vibes_dir)
        try:
            module = importlib.import_module("livox2_python")
        except Exception as exc:  # pragma: no cover - requires external SDK/module
            raise RuntimeError(
                "Could not import unitree_g1_vibes livox2_python. "
                "Pass --unitree-vibes-dir or set PYTHONPATH to the unitree_g1_vibes repo; "
                "Livox-SDK2 shared libraries must also be installed."
            ) from exc

        base_cls = getattr(module, "Livox2")
        buffer = self.buffer

        class _BufferingLivox2(base_cls):  # type: ignore[misc, valid-type]
            def handle_points(self, xyz: np.ndarray) -> None:  # noqa: D401
                buffer.update(xyz)

        self._driver_cls = _BufferingLivox2
        try:
            self._driver = _BufferingLivox2(
                self.config.livox_config_path,
                self.config.host_ip,
                frame_time=self.config.frame_time_s,
                frame_packets=self.config.frame_packets,
            )
        except Exception as exc:  # pragma: no cover - requires external hardware/SDK
            raise RuntimeError("Failed to start Livox2 driver; check config path, host IP, and SDK installation") from exc

    def stop(self) -> None:
        if self._driver is None:
            return
        try:
            self._driver.shutdown()
        finally:
            self._driver = None

    def read_lidar_frame(self, *, timeout_s: float | None = None) -> LidarFrame:
        if self._driver is None:
            raise RuntimeError("UnitreeVibesLivox2Adapter.start() must be called before reading")
        deadline = time.monotonic() + float(self.config.read_timeout_s if timeout_s is None else timeout_s)
        while time.monotonic() <= deadline:
            if self.buffer.has_frame:
                return self.buffer.read_lidar_frame()
            time.sleep(0.01)
        raise RuntimeError("Timed out waiting for Livox frame")
