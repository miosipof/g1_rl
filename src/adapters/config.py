"""Configuration schema for sim/real adapter integration.

The schema in this module is deliberately dependency-light. It defines the
hardware-neutral configuration needed to normalize real Unitree G1 observations
into the adapter types and to bound commands before any real locomotion sink is
allowed to consume them.

Hardware-specific modules may extend this configuration, but the core config
must remain importable without Isaac, Unitree SDK, RealSense, Livox, ROS, or
OpenCV installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .types import CameraIntrinsics, Command2D, Transform3D


Number = int | float


@dataclass(frozen=True)
class BodyExclusionBox:
    """Axis-aligned body-frame exclusion box for self/near-field points.

    Coordinates are in the robot body frame: x forward, y left, z up. Points
    inside the inclusive box are treated as robot/self artifacts and ignored by
    the corresponding adapter/filter.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None, *, name: str) -> "BodyExclusionBox | None":
        if data is None:
            return None
        data = _mapping(data, name)
        if not bool(data.get("enabled", True)):
            return None
        box = cls(
            x_min=_number(data.get("x_min", -0.50), f"{name}.x_min"),
            x_max=_number(data.get("x_max", 0.20), f"{name}.x_max"),
            y_min=_number(data.get("y_min", -0.45), f"{name}.y_min"),
            y_max=_number(data.get("y_max", 0.45), f"{name}.y_max"),
            z_min=_number(data.get("z_min", -0.25), f"{name}.z_min"),
            z_max=_number(data.get("z_max", 1.30), f"{name}.z_max"),
        )
        box.validate(name=name)
        return box

    def validate(self, *, name: str = "body_exclusion_box") -> None:
        _check_bounds(self.x_min, self.x_max, f"{name}.x")
        _check_bounds(self.y_min, self.y_max, f"{name}.y")
        _check_bounds(self.z_min, self.z_max, f"{name}.z")

    def contains_points(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_xyz must have shape Nx3, got {points.shape}")
        return (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] <= self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] <= self.y_max)
            & (points[:, 2] >= self.z_min)
            & (points[:, 2] <= self.z_max)
        )

    def contains_point(self, point_xyz: Sequence[float]) -> bool:
        return bool(self.contains_points(np.asarray(point_xyz, dtype=np.float32).reshape(1, 3))[0])

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "z_min": float(self.z_min),
            "z_max": float(self.z_max),
        }


@dataclass(frozen=True)
class CameraConfig:
    """RGB-D camera normalization config.

    ``intrinsics`` may be ``None`` for real hardware because RealSense adapters
    should read authoritative intrinsics from the device at runtime. If provided,
    the resolution must match ``width``/``height``.
    """

    enabled: bool
    width: int
    height: int
    fps: float
    frame_id: str
    depth_frame_id: str
    depth_aligned_to_color: bool
    depth_scale_m_per_unit: float | None
    intrinsics: CameraIntrinsics | None
    T_body_camera: Transform3D
    body_exclusion_box: BodyExclusionBox | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CameraConfig":
        data = _mapping(data, "camera")
        resolution = _mapping(data.get("resolution", {}), "camera.resolution")
        width = _positive_int(resolution.get("width", data.get("width", 640)), "camera.resolution.width")
        height = _positive_int(resolution.get("height", data.get("height", 480)), "camera.resolution.height")
        fps = _positive_float(resolution.get("fps", data.get("fps", 30.0)), "camera.resolution.fps")

        intrinsics_data = data.get("intrinsics")
        intrinsics = None
        if intrinsics_data is not None:
            intrinsics_map = _mapping(intrinsics_data, "camera.intrinsics")
            intrinsics = CameraIntrinsics(
                fx=_positive_float(intrinsics_map.get("fx"), "camera.intrinsics.fx"),
                fy=_positive_float(intrinsics_map.get("fy"), "camera.intrinsics.fy"),
                cx=_number(intrinsics_map.get("cx"), "camera.intrinsics.cx"),
                cy=_number(intrinsics_map.get("cy"), "camera.intrinsics.cy"),
                width=width,
                height=height,
            )

        depth_scale = _optional_positive_float(
            data.get("depth_scale_m_per_unit"),
            "camera.depth_scale_m_per_unit",
        )

        transform = _transform_from_dict(
            data.get("T_body_camera"),
            field_name="camera.T_body_camera",
            default_source_frame=str(data.get("frame_id", "camera_color_optical")),
            default_target_frame="body",
        )
        if transform.target_frame != "body":
            raise ValueError("camera.T_body_camera.target_frame must be 'body'")

        return cls(
            enabled=bool(data.get("enabled", True)),
            width=width,
            height=height,
            fps=fps,
            frame_id=str(data.get("frame_id", "camera_color_optical")),
            depth_frame_id=str(data.get("depth_frame_id", "camera_depth_optical")),
            depth_aligned_to_color=bool(data.get("depth_aligned_to_color", True)),
            depth_scale_m_per_unit=depth_scale,
            intrinsics=intrinsics,
            T_body_camera=transform,
            body_exclusion_box=BodyExclusionBox.from_dict(data.get("body_exclusion_box"), name="camera.body_exclusion_box"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class LidarConfig:
    """LiDAR point-cloud normalization config.

    ``T_body_lidar`` maps LiDAR-frame points into the robot body frame.
    Height/range filters describe the 3D-to-2D projection band used by the
    navigation stack.
    """

    enabled: bool
    source_frame_id: str
    frame_id: str
    mount_mode: str
    min_range_m: float
    max_range_m: float
    z_min_m: float
    z_max_m: float
    max_points: int | None
    T_body_lidar: Transform3D
    body_exclusion_box: BodyExclusionBox | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LidarConfig":
        data = _mapping(data, "lidar")
        source_frame = str(data.get("source_frame_id", "livox_lidar"))
        transform = _transform_from_dict(
            data.get("T_body_lidar"),
            field_name="lidar.T_body_lidar",
            default_source_frame=source_frame,
            default_target_frame="body",
        )
        if transform.target_frame != "body":
            raise ValueError("lidar.T_body_lidar.target_frame must be 'body'")

        min_range = _nonnegative_float(data.get("min_range_m", 0.15), "lidar.min_range_m")
        max_range = _positive_float(data.get("max_range_m", 8.0), "lidar.max_range_m")
        if max_range <= min_range:
            raise ValueError("lidar.max_range_m must be greater than lidar.min_range_m")

        z_min = _number(data.get("z_min_m", 0.05), "lidar.z_min_m")
        z_max = _number(data.get("z_max_m", 1.30), "lidar.z_max_m")
        if z_max <= z_min:
            raise ValueError("lidar.z_max_m must be greater than lidar.z_min_m")

        max_points_raw = data.get("max_points")
        max_points = None if max_points_raw is None else _positive_int(max_points_raw, "lidar.max_points")

        return cls(
            enabled=bool(data.get("enabled", True)),
            source_frame_id=source_frame,
            frame_id=str(data.get("frame_id", "body")),
            mount_mode=str(data.get("mount_mode", "normal")),
            min_range_m=min_range,
            max_range_m=max_range,
            z_min_m=z_min,
            z_max_m=z_max,
            max_points=max_points,
            T_body_lidar=transform,
            body_exclusion_box=BodyExclusionBox.from_dict(data.get("body_exclusion_box"), name="lidar.body_exclusion_box"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class CommandLimitsConfig:
    """Conservative body-frame velocity and rate limits."""

    vx_min: float
    vx_max: float
    vy_min: float
    vy_max: float
    wz_min: float
    wz_max: float
    max_linear_accel_mps2: float | None
    max_yaw_accel_radps2: float | None
    command_rate_hz: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommandLimitsConfig":
        data = _mapping(data, "command_limits")
        cfg = cls(
            vx_min=_number(data.get("vx_min", -0.05), "command_limits.vx_min"),
            vx_max=_number(data.get("vx_max", 0.10), "command_limits.vx_max"),
            vy_min=_number(data.get("vy_min", -0.05), "command_limits.vy_min"),
            vy_max=_number(data.get("vy_max", 0.05), "command_limits.vy_max"),
            wz_min=_number(data.get("wz_min", -0.20), "command_limits.wz_min"),
            wz_max=_number(data.get("wz_max", 0.20), "command_limits.wz_max"),
            max_linear_accel_mps2=_optional_positive_float(
                data.get("max_linear_accel_mps2"), "command_limits.max_linear_accel_mps2"
            ),
            max_yaw_accel_radps2=_optional_positive_float(
                data.get("max_yaw_accel_radps2"), "command_limits.max_yaw_accel_radps2"
            ),
            command_rate_hz=_positive_float(data.get("command_rate_hz", 10.0), "command_limits.command_rate_hz"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        _check_bounds(self.vx_min, self.vx_max, "command_limits.vx")
        _check_bounds(self.vy_min, self.vy_max, "command_limits.vy")
        _check_bounds(self.wz_min, self.wz_max, "command_limits.wz")

    def clip(self, command: Command2D, *, source: str = "safety_limited") -> Command2D:
        """Return a command clipped to this limit profile."""

        return command.clipped(
            vx_min=self.vx_min,
            vx_max=self.vx_max,
            vy_min=self.vy_min,
            vy_max=self.vy_max,
            wz_min=self.wz_min,
            wz_max=self.wz_max,
            source=source,
        )


@dataclass(frozen=True)
class WatchdogConfig:
    """Freshness limits used before sending motion to the real robot."""

    max_rgbd_age_s: float
    max_lidar_age_s: float
    max_robot_state_age_s: float
    max_policy_age_s: float
    stop_on_stale_rgbd: bool
    stop_on_stale_lidar: bool
    stop_on_stale_robot_state: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WatchdogConfig":
        data = _mapping(data, "watchdog")
        return cls(
            max_rgbd_age_s=_positive_float(data.get("max_rgbd_age_s", 0.50), "watchdog.max_rgbd_age_s"),
            max_lidar_age_s=_positive_float(data.get("max_lidar_age_s", 0.30), "watchdog.max_lidar_age_s"),
            max_robot_state_age_s=_positive_float(
                data.get("max_robot_state_age_s", 0.30), "watchdog.max_robot_state_age_s"
            ),
            max_policy_age_s=_positive_float(data.get("max_policy_age_s", 0.20), "watchdog.max_policy_age_s"),
            stop_on_stale_rgbd=bool(data.get("stop_on_stale_rgbd", False)),
            stop_on_stale_lidar=bool(data.get("stop_on_stale_lidar", True)),
            stop_on_stale_robot_state=bool(data.get("stop_on_stale_robot_state", True)),
        )


@dataclass(frozen=True)
class PerceptionConfig:
    """Target-detection/tracking config shared by sim and real runs."""

    target_source: str
    target_depth_source: str
    yolo_model: str
    yolo_device: str
    target_class: str
    yolo_confidence_threshold: float
    redetect_every_steps: int
    occlusion_grace_steps: int
    smoothing_alpha: float
    max_jump_m: float
    stand_off_m: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PerceptionConfig":
        data = _mapping(data, "perception")
        alpha = _number(data.get("smoothing_alpha", 0.35), "perception.smoothing_alpha")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("perception.smoothing_alpha must be in [0, 1]")
        conf = _number(data.get("yolo_confidence_threshold", 0.25), "perception.yolo_confidence_threshold")
        if not (0.0 <= conf <= 1.0):
            raise ValueError("perception.yolo_confidence_threshold must be in [0, 1]")
        return cls(
            target_source=str(data.get("target_source", "yolo")),
            target_depth_source=str(data.get("target_depth_source", "bbox")),
            yolo_model=str(data.get("yolo_model", "yolov8n.pt")),
            yolo_device=str(data.get("yolo_device", "auto")),
            target_class=str(data.get("target_class", "person")),
            yolo_confidence_threshold=conf,
            redetect_every_steps=_nonnegative_int(data.get("redetect_every_steps", 50), "perception.redetect_every_steps"),
            occlusion_grace_steps=_nonnegative_int(data.get("occlusion_grace_steps", 600), "perception.occlusion_grace_steps"),
            smoothing_alpha=alpha,
            max_jump_m=_positive_float(data.get("max_jump_m", 1.25), "perception.max_jump_m"),
            stand_off_m=_nonnegative_float(data.get("stand_off_m", 0.80), "perception.stand_off_m"),
        )


@dataclass(frozen=True)
class SafetyConfig:
    """High-level safety gates used by the real bridge."""

    require_operator_enable: bool
    require_lidar_for_motion: bool
    require_robot_state_for_motion: bool
    stop_on_estop: bool
    stop_on_robot_fault: bool
    min_battery_percent: float | None
    obstacle_stop_distance_m: float
    obstacle_slow_distance_m: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SafetyConfig":
        data = _mapping(data, "safety")
        min_battery = data.get("min_battery_percent")
        min_battery_percent = None if min_battery is None else _number(min_battery, "safety.min_battery_percent")
        if min_battery_percent is not None and not (0.0 <= min_battery_percent <= 100.0):
            raise ValueError("safety.min_battery_percent must be in [0, 100]")
        stop_dist = _positive_float(data.get("obstacle_stop_distance_m", 0.55), "safety.obstacle_stop_distance_m")
        slow_dist = _positive_float(data.get("obstacle_slow_distance_m", 0.90), "safety.obstacle_slow_distance_m")
        if slow_dist < stop_dist:
            raise ValueError("safety.obstacle_slow_distance_m must be >= safety.obstacle_stop_distance_m")
        return cls(
            require_operator_enable=bool(data.get("require_operator_enable", True)),
            require_lidar_for_motion=bool(data.get("require_lidar_for_motion", True)),
            require_robot_state_for_motion=bool(data.get("require_robot_state_for_motion", True)),
            stop_on_estop=bool(data.get("stop_on_estop", True)),
            stop_on_robot_fault=bool(data.get("stop_on_robot_fault", True)),
            min_battery_percent=min_battery_percent,
            obstacle_stop_distance_m=stop_dist,
            obstacle_slow_distance_m=slow_dist,
        )


@dataclass(frozen=True)
class AdapterRuntimeConfig:
    """Full adapter runtime configuration profile."""

    profile_name: str
    clock_source: str
    camera: CameraConfig
    lidar: LidarConfig
    command_limits: CommandLimitsConfig
    watchdog: WatchdogConfig
    perception: PerceptionConfig
    safety: SafetyConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdapterRuntimeConfig":
        data = _mapping(data, "config")
        return cls(
            profile_name=str(data.get("profile_name", "unnamed_adapter_profile")),
            clock_source=str(data.get("clock_source", "monotonic")),
            camera=CameraConfig.from_dict(data.get("camera", {})),
            lidar=LidarConfig.from_dict(data.get("lidar", {})),
            command_limits=CommandLimitsConfig.from_dict(data.get("command_limits", {})),
            watchdog=WatchdogConfig.from_dict(data.get("watchdog", {})),
            perception=PerceptionConfig.from_dict(data.get("perception", {})),
            safety=SafetyConfig.from_dict(data.get("safety", {})),
            metadata=dict(data.get("metadata", {})),
        )


def load_adapter_config(path: str | Path) -> AdapterRuntimeConfig:
    """Load and validate an adapter config from YAML."""

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("PyYAML is required to load adapter YAML configs") from exc

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return AdapterRuntimeConfig.from_dict(data)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _positive_float(value: Any, name: str) -> float:
    out = _number(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out}")
    return out


def _nonnegative_float(value: Any, name: str) -> float:
    out = _number(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0, got {out}")
    return out


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name)


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _check_bounds(lo: float, hi: float, name: str) -> None:
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound, got {lo} > {hi}")


def _transform_from_dict(
    value: Any,
    *,
    field_name: str,
    default_source_frame: str,
    default_target_frame: str,
) -> Transform3D:
    data = _mapping(value, field_name)
    matrix_raw = data.get("matrix", np.eye(4, dtype=np.float64).tolist())
    matrix = np.asarray(matrix_raw, dtype=np.float64)
    return Transform3D(
        matrix=matrix,
        source_frame=str(data.get("source_frame", default_source_frame)),
        target_frame=str(data.get("target_frame", default_target_frame)),
    )


__all__ = [
    "AdapterRuntimeConfig",
    "CameraConfig",
    "CommandLimitsConfig",
    "LidarConfig",
    "PerceptionConfig",
    "SafetyConfig",
    "WatchdogConfig",
    "load_adapter_config",
]
