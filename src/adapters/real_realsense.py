"""RealSense RGB-D adapter helpers for the shared sim/real interface.

This module normalizes data from the RealSense scripts in ``unitree_g1_vibes``
into :class:`adapters.types.RGBDFrame` without requiring pyrealsense2 at import
 time.  The pure conversion helpers are unit-testable on any machine:

- BGR uint8 -> RGB uint8
- raw uint16 depth units -> float32 meters
- RealSense intrinsics object/dict -> CameraIntrinsics
- normalized arrays -> RGBDFrame

A small ``RealSenseRGBDAdapter`` skeleton is included for live use.  It imports
``pyrealsense2`` only inside ``start()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

import numpy as np

from .config import CameraConfig
from .types import CameraIntrinsics, RGBDFrame, Transform3D


@dataclass(frozen=True)
class RealSenseRGBDAdapterConfig:
    """Runtime options for the RealSense RGB-D adapter.

    ``depth_scale_m_per_unit`` may be ``None``.  In that case the live adapter
    reads the authoritative scale from the RealSense depth sensor at startup.
    Offline helpers require an explicit scale when converting raw depth arrays.
    """

    width: int = 640
    height: int = 480
    fps: int = 30
    frame_id: str = "camera_color_optical"
    depth_frame_id: str = "camera_depth_optical"
    depth_scale_m_per_unit: float | None = None
    T_body_camera: Transform3D = field(
        default_factory=lambda: Transform3D(
            matrix=np.eye(4, dtype=np.float64),
            source_frame="camera_color_optical",
            target_frame="body",
        )
    )
    serial_number: str | None = None
    align_depth_to_color: bool = True
    use_spatial_filter: bool = True
    use_temporal_filter: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_camera_config(
        cls,
        config: CameraConfig,
        *,
        serial_number: str | None = None,
        use_spatial_filter: bool = True,
        use_temporal_filter: bool = True,
    ) -> "RealSenseRGBDAdapterConfig":
        return cls(
            width=int(config.width),
            height=int(config.height),
            fps=int(round(config.fps)),
            frame_id=config.frame_id,
            depth_frame_id=config.depth_frame_id,
            depth_scale_m_per_unit=config.depth_scale_m_per_unit,
            T_body_camera=config.T_body_camera,
            serial_number=serial_number,
            align_depth_to_color=bool(config.depth_aligned_to_color),
            use_spatial_filter=use_spatial_filter,
            use_temporal_filter=use_temporal_filter,
            metadata=dict(config.metadata),
        )


def bgr_to_rgb(color_bgr: np.ndarray) -> np.ndarray:
    """Convert a RealSense/OpenCV BGR image to RGB uint8.

    RealSense scripts in ``unitree_g1_vibes`` configure the color stream as
    ``rs.format.bgr8``.  YOLO and the sim-side code expect RGB order, so this
    conversion must be explicit at the adapter boundary.
    """

    arr = np.asarray(color_bgr)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"color_bgr must have shape HxWx3, got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"color_bgr must be uint8, got {arr.dtype}")
    return arr[..., ::-1].copy()


def depth_raw_to_meters(
    depth_raw: np.ndarray,
    *,
    depth_scale_m_per_unit: float,
    invalid_zero_to_nan: bool = False,
    max_depth_m: float | None = None,
) -> np.ndarray:
    """Convert RealSense raw depth units to float32 meters.

    RealSense ``z16`` frames are integer depth units.  The device-provided
    ``depth_scale`` maps those units to meters.
    """

    scale = float(depth_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"depth_scale_m_per_unit must be positive finite, got {scale}")

    raw = np.asarray(depth_raw)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"depth_raw must have shape HxW or HxWx1, got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer) and not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"depth_raw must be numeric, got dtype={raw.dtype}")

    depth_m = raw.astype(np.float32, copy=False) * np.float32(scale)
    if invalid_zero_to_nan:
        depth_m = depth_m.copy()
        depth_m[raw == 0] = np.nan
    if max_depth_m is not None:
        max_depth = float(max_depth_m)
        if not np.isfinite(max_depth) or max_depth <= 0.0:
            raise ValueError(f"max_depth_m must be positive finite, got {max_depth}")
        depth_m = depth_m.copy()
        depth_m[depth_m > max_depth] = np.nan if invalid_zero_to_nan else 0.0
    return depth_m.astype(np.float32, copy=False)


def camera_intrinsics_from_mapping(data: Mapping[str, Any]) -> CameraIntrinsics:
    """Create CameraIntrinsics from a dict-like RealSense metadata object.

    Accepted principal-point keys are either ``cx``/``cy`` or RealSense-style
    ``ppx``/``ppy``.
    """

    width = int(data["width"])
    height = int(data["height"])
    fx = float(data["fx"])
    fy = float(data["fy"])
    cx = float(data.get("cx", data.get("ppx")))
    cy = float(data.get("cy", data.get("ppy")))
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


def camera_intrinsics_from_realsense(intrinsics: Any) -> CameraIntrinsics:
    """Create CameraIntrinsics from a pyrealsense2 intrinsics object.

    This function uses duck typing, so importing this module does not require
    ``pyrealsense2``.
    """

    return CameraIntrinsics(
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.ppx),
        cy=float(intrinsics.ppy),
        width=int(intrinsics.width),
        height=int(intrinsics.height),
    )


def make_rgbd_frame_from_realsense_arrays(
    *,
    timestamp_s: float,
    color_bgr: np.ndarray,
    depth_raw: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_body_camera: Transform3D,
    depth_scale_m_per_unit: float,
    frame_id: str = "camera_color_optical",
    depth_frame_id: str = "camera_depth_optical",
    metadata: Mapping[str, Any] | None = None,
) -> RGBDFrame:
    """Normalize RealSense arrays into an RGBDFrame."""

    rgb = bgr_to_rgb(color_bgr)
    depth_m = depth_raw_to_meters(depth_raw, depth_scale_m_per_unit=depth_scale_m_per_unit)
    return RGBDFrame(
        timestamp_s=float(timestamp_s),
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=intrinsics,
        T_body_camera=T_body_camera,
        frame_id=frame_id,
        depth_frame_id=depth_frame_id,
        metadata=dict(metadata or {}),
    )


class RealSenseRGBDAdapter:
    """Minimal live RealSense adapter skeleton.

    The class is intentionally small.  It owns RealSense stream setup and frame
    normalization, but does not perform detection or navigation.  Hardware
    dependencies are imported lazily in ``start()`` so tests and sim runs remain
    dependency-light.
    """

    def __init__(self, config: RealSenseRGBDAdapterConfig) -> None:
        self.config = config
        self._rs: Any | None = None
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._spatial_filter: Any | None = None
        self._temporal_filter: Any | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._depth_scale_m_per_unit: float | None = config.depth_scale_m_per_unit

    @property
    def started(self) -> bool:
        return self._pipeline is not None

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    @property
    def depth_scale_m_per_unit(self) -> float | None:
        return self._depth_scale_m_per_unit

    def start(self) -> None:  # pragma: no cover - hardware-dependent
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for RealSenseRGBDAdapter.start()") from exc

        pipeline = rs.pipeline()
        config = rs.config()
        if self.config.serial_number:
            config.enable_device(self.config.serial_number)
        config.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps)
        config.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps)

        profile = pipeline.start(config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = camera_intrinsics_from_realsense(color_profile.get_intrinsics())

        if self._depth_scale_m_per_unit is None:
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale_m_per_unit = float(depth_sensor.get_depth_scale())

        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if self.config.align_depth_to_color else None
        self._spatial_filter = rs.spatial_filter() if self.config.use_spatial_filter else None
        self._temporal_filter = rs.temporal_filter() if self.config.use_temporal_filter else None

    def stop(self) -> None:  # pragma: no cover - hardware-dependent
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self._spatial_filter = None
        self._temporal_filter = None

    def read_frame(self, *, timeout_ms: int = 1000, timestamp_s: float | None = None) -> RGBDFrame:  # pragma: no cover
        if self._pipeline is None:
            raise RuntimeError("RealSenseRGBDAdapter is not started")
        if self._intrinsics is None or self._depth_scale_m_per_unit is None:
            raise RuntimeError("RealSenseRGBDAdapter missing intrinsics/depth scale")

        frames = self._pipeline.wait_for_frames(int(timeout_ms))
        if self._align is not None:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense frame set did not include both color and depth")

        if self._spatial_filter is not None:
            depth_frame = self._spatial_filter.process(depth_frame)
        if self._temporal_filter is not None:
            depth_frame = self._temporal_filter.process(depth_frame)

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        return make_rgbd_frame_from_realsense_arrays(
            timestamp_s=time.monotonic() if timestamp_s is None else float(timestamp_s),
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            intrinsics=self._intrinsics,
            T_body_camera=self.config.T_body_camera,
            depth_scale_m_per_unit=self._depth_scale_m_per_unit,
            frame_id=self.config.frame_id,
            depth_frame_id=self.config.depth_frame_id,
            metadata={**dict(self.config.metadata), "source": "realsense_live"},
        )

    def __enter__(self) -> "RealSenseRGBDAdapter":  # pragma: no cover - hardware-dependent
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:  # pragma: no cover - hardware-dependent
        self.stop()


__all__ = [
    "RealSenseRGBDAdapter",
    "RealSenseRGBDAdapterConfig",
    "bgr_to_rgb",
    "camera_intrinsics_from_mapping",
    "camera_intrinsics_from_realsense",
    "depth_raw_to_meters",
    "make_rgbd_frame_from_realsense_arrays",
]
