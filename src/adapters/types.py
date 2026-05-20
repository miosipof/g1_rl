"""Shared adapter interface types for sim/real G1 autonomy.

This module defines the narrow contracts that let the same autonomy code consume
observations from Isaac Sim, a real Unitree G1, or a replay log, and emit planar
velocity commands to either a simulator proxy, a real locomotion client, or a dry
run logger.

The classes deliberately avoid Isaac, Unitree SDK, RealSense, Livox, ROS, and
OpenCV imports. Hardware-specific code must adapt into these types at the
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Any, Mapping, Sequence

import numpy as np

from g1_nav_rl.navigation.simple_nav import MotionCommand, Pose2D, Target2D
from g1_nav_rl.perception.target_from_bbox import CameraIntrinsics


class AdapterHealth(str, Enum):
    """Coarse health state reported by observation/command adapters."""

    OK = "ok"
    DEGRADED = "degraded"
    STALE = "stale"
    FAULT = "fault"
    UNAVAILABLE = "unavailable"


class BridgeMode(str, Enum):
    """High-level bridge operating mode shared by sim and real runners."""

    IDLE = "idle"
    ARMED = "armed"
    FOLLOWING = "following"
    PAUSED = "paused"
    STOPPING = "stopping"
    FAULT = "fault"
    E_STOP = "e_stop"


@dataclass(frozen=True)
class AdapterStatus:
    """Status metadata returned by hardware, sim, or replay adapters.

    ``timestamp_s`` must use a monotonic clock when available. Wall-clock values
    are acceptable for offline logs, but all producers in one run must use the
    same clock basis.
    """

    name: str
    health: AdapterHealth = AdapterHealth.OK
    timestamp_s: float | None = None
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.health == AdapterHealth.OK

    def age_s(self, now_s: float) -> float | None:
        if self.timestamp_s is None:
            return None
        return max(0.0, float(now_s) - float(self.timestamp_s))

    def is_fresh(self, *, now_s: float, max_age_s: float) -> bool:
        age = self.age_s(now_s)
        return age is not None and age <= float(max_age_s)


@dataclass(frozen=True)
class Transform3D:
    """Homogeneous 4x4 transform with named source/target frames.

    Convention: ``matrix`` maps homogeneous points from ``source_frame`` into
    ``target_frame``.
    """

    matrix: np.ndarray
    source_frame: str
    target_frame: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform3D.matrix must be shape (4,4), got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Transform3D.matrix contains non-finite values")
        object.__setattr__(self, "matrix", matrix)
        if not self.source_frame:
            raise ValueError("Transform3D.source_frame must be non-empty")
        if not self.target_frame:
            raise ValueError("Transform3D.target_frame must be non-empty")

    @classmethod
    def identity(cls, *, frame: str) -> "Transform3D":
        return cls(matrix=np.eye(4, dtype=np.float64), source_frame=frame, target_frame=frame)

    def inverse(self) -> "Transform3D":
        return Transform3D(
            matrix=np.linalg.inv(self.matrix),
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )

    def transform_points(self, points_xyz: np.ndarray) -> np.ndarray:
        points = _as_points_xyz(points_xyz, name="points_xyz")
        if points.size == 0:
            return points.copy()
        hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
        out = (self.matrix @ hom.T).T[:, :3]
        return out.astype(np.float32, copy=False)


@dataclass(frozen=True)
class RGBDFrame:
    """Timestamped RGB-D observation normalized for the autonomy stack.

    Required conventions:
    - ``rgb`` is ``HxWx3`` uint8 in **RGB** order, not BGR.
    - ``depth_m`` is ``HxW`` float32/float64 in meters, aligned to the RGB image.
    - ``T_body_camera`` maps camera-frame points into the robot body frame.
    """

    timestamp_s: float
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    T_body_camera: Transform3D
    frame_id: str = "camera_color_optical"
    depth_frame_id: str = "camera_depth_optical"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGBDFrame.rgb must have shape HxWx3, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"RGBDFrame.rgb must be uint8 RGB, got dtype={rgb.dtype}")

        depth = np.asarray(self.depth_m)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"RGBDFrame.depth_m must have shape HxW or HxWx1, got {depth.shape}")
        if depth.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"RGB/depth resolution mismatch: rgb={rgb.shape[:2]}, depth={depth.shape[:2]}")
        if int(self.intrinsics.width) != int(rgb.shape[1]) or int(self.intrinsics.height) != int(rgb.shape[0]):
            raise ValueError(
                "intrinsics resolution does not match rgb: "
                f"intrinsics=({self.intrinsics.width},{self.intrinsics.height}), "
                f"rgb=({rgb.shape[1]},{rgb.shape[0]})"
            )
        if not isinstance(self.T_body_camera, Transform3D):
            raise TypeError("RGBDFrame.T_body_camera must be a Transform3D")
        if self.T_body_camera.target_frame != "body":
            raise ValueError(
                "RGBDFrame.T_body_camera must target frame 'body', "
                f"got {self.T_body_camera.target_frame!r}"
            )
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("RGBDFrame.timestamp_s must be finite")

        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth.astype(np.float32, copy=False))

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - float(self.timestamp_s))

    def is_fresh(self, *, now_s: float, max_age_s: float) -> bool:
        return self.age_s(now_s) <= float(max_age_s)


@dataclass(frozen=True)
class LidarFrame:
    """Timestamped LiDAR observation normalized to robot body frame.

    ``points_body_xyz`` must be ``Nx3`` points in meters in a body frame whose
    x-axis is forward, y-axis is left, and z-axis is up.
    """

    timestamp_s: float
    points_body_xyz: np.ndarray
    frame_id: str = "body"
    source_frame_id: str = "lidar"
    T_body_lidar: Transform3D | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = _as_points_xyz(self.points_body_xyz, name="points_body_xyz")
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("LidarFrame.timestamp_s must be finite")
        if self.frame_id != "body":
            raise ValueError(f"LidarFrame.frame_id must be 'body', got {self.frame_id!r}")
        if self.T_body_lidar is not None and self.T_body_lidar.target_frame != "body":
            raise ValueError(
                "LidarFrame.T_body_lidar must target frame 'body', "
                f"got {self.T_body_lidar.target_frame!r}"
            )
        object.__setattr__(self, "points_body_xyz", points)

    @property
    def points_body_xy(self) -> np.ndarray:
        return self.points_body_xyz[:, :2]

    @property
    def point_count(self) -> int:
        return int(self.points_body_xyz.shape[0])

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - float(self.timestamp_s))

    def is_fresh(self, *, now_s: float, max_age_s: float) -> bool:
        return self.age_s(now_s) <= float(max_age_s)

    def filtered_by_height(self, *, z_min_m: float, z_max_m: float) -> "LidarFrame":
        z = self.points_body_xyz[:, 2]
        keep = (z >= float(z_min_m)) & (z <= float(z_max_m))
        return replace(self, points_body_xyz=self.points_body_xyz[keep])


@dataclass(frozen=True)
class RobotState:
    """Robot status information required by a safety supervisor."""

    timestamp_s: float
    pose: Pose2D | None = None
    mode: str = "unknown"
    fsm_state: str | int | None = None
    battery_percent: float | None = None
    is_estopped: bool = False
    is_faulted: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("RobotState.timestamp_s must be finite")
        if self.battery_percent is not None:
            pct = float(self.battery_percent)
            if not (0.0 <= pct <= 100.0):
                raise ValueError(f"battery_percent must be in [0,100], got {pct}")
            object.__setattr__(self, "battery_percent", pct)

    def age_s(self, now_s: float) -> float:
        return max(0.0, float(now_s) - float(self.timestamp_s))

    def is_fresh(self, *, now_s: float, max_age_s: float) -> bool:
        return self.age_s(now_s) <= float(max_age_s)


@dataclass(frozen=True)
class RobotObservation:
    """Single policy input assembled from sim, real robot, or replay data."""

    timestamp_s: float
    pose: Pose2D | None = None
    rgbd: RGBDFrame | None = None
    lidar: LidarFrame | None = None
    robot_state: RobotState | None = None
    target_hint: Target2D | None = None
    statuses: tuple[AdapterStatus, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.timestamp_s)):
            raise ValueError("RobotObservation.timestamp_s must be finite")
        if self.rgbd is not None and self.rgbd.timestamp_s > self.timestamp_s + 1e-6:
            raise ValueError("RGBDFrame timestamp cannot be newer than RobotObservation timestamp")
        if self.lidar is not None and self.lidar.timestamp_s > self.timestamp_s + 1e-6:
            raise ValueError("LidarFrame timestamp cannot be newer than RobotObservation timestamp")
        if self.robot_state is not None and self.robot_state.timestamp_s > self.timestamp_s + 1e-6:
            raise ValueError("RobotState timestamp cannot be newer than RobotObservation timestamp")
        object.__setattr__(self, "statuses", tuple(self.statuses))

    @property
    def has_rgbd(self) -> bool:
        return self.rgbd is not None

    @property
    def has_lidar(self) -> bool:
        return self.lidar is not None

    @property
    def has_pose(self) -> bool:
        return self.pose is not None

    @property
    def ok(self) -> bool:
        if self.robot_state is not None and (self.robot_state.is_estopped or self.robot_state.is_faulted):
            return False
        return all(status.health not in {AdapterHealth.FAULT, AdapterHealth.UNAVAILABLE} for status in self.statuses)

    def sensor_ages_s(self) -> dict[str, float | None]:
        return {
            "rgbd": None if self.rgbd is None else self.timestamp_s - self.rgbd.timestamp_s,
            "lidar": None if self.lidar is None else self.timestamp_s - self.lidar.timestamp_s,
            "robot_state": None if self.robot_state is None else self.timestamp_s - self.robot_state.timestamp_s,
        }


@dataclass(frozen=True)
class Command2D:
    """Planar body-frame velocity command used by sim and real command sinks.

    Convention matches ``MotionCommand`` and Unitree locomotion calls:
    ``vx`` forward m/s, ``vy`` left m/s, ``wz`` positive yaw rate rad/s.
    """

    vx: float
    vy: float
    wz: float
    timestamp_s: float | None = None
    duration_s: float | None = None
    source: str = "policy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = (float(self.vx), float(self.vy), float(self.wz))
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Command2D values must be finite, got {values}")
        object.__setattr__(self, "vx", values[0])
        object.__setattr__(self, "vy", values[1])
        object.__setattr__(self, "wz", values[2])
        if self.timestamp_s is not None and not math.isfinite(float(self.timestamp_s)):
            raise ValueError("Command2D.timestamp_s must be finite when provided")
        if self.duration_s is not None:
            duration = float(self.duration_s)
            if duration <= 0.0 or not math.isfinite(duration):
                raise ValueError(f"Command2D.duration_s must be positive finite, got {duration}")
            object.__setattr__(self, "duration_s", duration)

    @classmethod
    def stop(cls, *, timestamp_s: float | None = None, source: str = "stop") -> "Command2D":
        return cls(0.0, 0.0, 0.0, timestamp_s=timestamp_s, source=source)

    @classmethod
    def from_motion_command(
        cls,
        command: MotionCommand,
        *,
        timestamp_s: float | None = None,
        duration_s: float | None = None,
        source: str = "policy",
    ) -> "Command2D":
        return cls(
            vx=float(command.vx),
            vy=float(command.vy),
            wz=float(command.wz),
            timestamp_s=timestamp_s,
            duration_s=duration_s,
            source=source,
        )

    @property
    def tuple(self) -> tuple[float, float, float]:
        return float(self.vx), float(self.vy), float(self.wz)

    @property
    def is_stop(self) -> bool:
        return abs(self.vx) < 1e-9 and abs(self.vy) < 1e-9 and abs(self.wz) < 1e-9

    def to_motion_command(self) -> MotionCommand:
        return MotionCommand(vx=float(self.vx), vy=float(self.vy), wz=float(self.wz))

    def clipped(
        self,
        *,
        vx_min: float,
        vx_max: float,
        vy_min: float,
        vy_max: float,
        wz_min: float,
        wz_max: float,
        source: str | None = None,
    ) -> "Command2D":
        return replace(
            self,
            vx=_clip(self.vx, vx_min, vx_max),
            vy=_clip(self.vy, vy_min, vy_max),
            wz=_clip(self.wz, wz_min, wz_max),
            source=self.source if source is None else source,
        )


@dataclass(frozen=True)
class AdapterCommandResult:
    """Result from a command sink after sending or logging a command."""

    accepted: bool
    command: Command2D
    status: AdapterStatus

    @property
    def ok(self) -> bool:
        return bool(self.accepted) and self.status.ok


def _clip(value: float, lo: float, hi: float) -> float:
    if float(lo) > float(hi):
        raise ValueError(f"invalid clip bounds: lo={lo}, hi={hi}")
    return min(max(float(value), float(lo)), float(hi))


def _as_points_xyz(points_xyz: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape Nx3, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


__all__ = [
    "AdapterCommandResult",
    "AdapterHealth",
    "AdapterStatus",
    "BridgeMode",
    "CameraIntrinsics",
    "Command2D",
    "LidarFrame",
    "MotionCommand",
    "Pose2D",
    "RGBDFrame",
    "RobotObservation",
    "RobotState",
    "Target2D",
    "Transform3D",
]
