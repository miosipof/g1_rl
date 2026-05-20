"""Real-sensor observation providers for bridge dry-run and staged deployment.

This module composes already-normalized sensor adapters into the shared
``RobotObservation`` contract.  It is intentionally hardware-light: no
RealSense, Livox, Unitree, Isaac, ROS, or OpenCV modules are imported here.
Live hardware objects are injected and only need to expose the small methods
used by the provider, such as ``read_frame()`` for RGB-D.

Step 9 wires real RGB-D into the dry-run bridge while keeping LiDAR and robot
state synthetic by default.  That is deliberate: it lets us validate camera
streaming, timestamping, and bridge integration without allowing robot motion or
requiring a live MID-360/SLAM source yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from .config import AdapterRuntimeConfig
from .types import AdapterHealth, AdapterStatus, LidarFrame, Pose2D, RGBDFrame, RobotObservation, RobotState


Clock = Callable[[], float]


@runtime_checkable
class RGBDAdapterLike(Protocol):
    """Small protocol implemented by RealSenseRGBDAdapter and test fakes."""

    def read_frame(self) -> RGBDFrame:
        ...


@dataclass(frozen=True)
class SyntheticSupportConfig:
    """Synthetic support data used while only part of the real stack is live."""

    pose: Pose2D = field(default_factory=lambda: Pose2D(0.0, 0.0, 0.0))
    battery_percent: float = 100.0
    mode: str = "rgbd_dryrun"
    clear_lidar_points_body_xyz: tuple[tuple[float, float, float], ...] = (
        (2.0, 1.1, 0.35),
        (3.0, -1.2, 0.45),
    )
    robot_state_details: Mapping[str, Any] = field(default_factory=lambda: {"synthetic": True})
    lidar_metadata: Mapping[str, Any] = field(default_factory=lambda: {"synthetic": True, "role": "safety_placeholder"})

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.battery_percent) <= 100.0):
            raise ValueError("battery_percent must be in [0, 100]")


@dataclass
class RealSenseDryRunObservationProvider:
    """Observation provider using live RealSense RGB-D plus synthetic support data.

    This is the first real-sensor bridge stage:

    ``RealSenseRGBDAdapter.read_frame() -> RobotObservation.rgbd``

    LiDAR and robot state are still synthetic by default so that the existing
    safety supervisor can run without requiring Livox/Unitree connections.  The
    observation metadata makes this explicit.
    """

    rgbd_adapter: RGBDAdapterLike
    config: AdapterRuntimeConfig
    support: SyntheticSupportConfig = field(default_factory=SyntheticSupportConfig)
    clock: Clock = time.monotonic
    include_synthetic_lidar: bool = True
    include_synthetic_robot_state: bool = True
    source_name: str = "RealSenseDryRunObservationProvider"

    def read_observation(self) -> RobotObservation:
        rgbd = self.rgbd_adapter.read_frame()
        now_s = max(float(self.clock()), float(rgbd.timestamp_s))
        lidar = self._make_synthetic_lidar(now_s) if self.include_synthetic_lidar else None
        robot_state = self._make_synthetic_robot_state(now_s) if self.include_synthetic_robot_state else None
        statuses = [
            AdapterStatus(
                name="rgbd",
                health=AdapterHealth.OK,
                timestamp_s=rgbd.timestamp_s,
                message="live RGB-D frame available",
                details={
                    "frame_id": rgbd.frame_id,
                    "width": rgbd.width,
                    "height": rgbd.height,
                    "synthetic": False,
                },
            )
        ]
        if lidar is not None:
            statuses.append(
                AdapterStatus(
                    name="lidar",
                    health=AdapterHealth.OK,
                    timestamp_s=lidar.timestamp_s,
                    message="synthetic LiDAR placeholder used for dry-run safety",
                    details=dict(lidar.metadata),
                )
            )
        if robot_state is not None:
            statuses.append(
                AdapterStatus(
                    name="robot_state",
                    health=AdapterHealth.OK,
                    timestamp_s=robot_state.timestamp_s,
                    message="synthetic robot state placeholder used for dry-run safety",
                    details=dict(robot_state.details),
                )
            )
        return RobotObservation(
            timestamp_s=now_s,
            pose=self.support.pose,
            rgbd=rgbd,
            lidar=lidar,
            robot_state=robot_state,
            statuses=tuple(statuses),
            metadata={
                "source": self.source_name,
                "rgbd_live": True,
                "lidar_synthetic": lidar is not None,
                "robot_state_synthetic": robot_state is not None,
            },
        )

    def _make_synthetic_lidar(self, timestamp_s: float) -> LidarFrame:
        points = np.asarray(self.support.clear_lidar_points_body_xyz, dtype=np.float32)
        return LidarFrame(
            timestamp_s=float(timestamp_s),
            points_body_xyz=points,
            source_frame_id=self.config.lidar.source_frame_id,
            T_body_lidar=self.config.lidar.T_body_lidar,
            metadata=dict(self.support.lidar_metadata),
        )

    def _make_synthetic_robot_state(self, timestamp_s: float) -> RobotState:
        return RobotState(
            timestamp_s=float(timestamp_s),
            pose=self.support.pose,
            mode=self.support.mode,
            battery_percent=float(self.support.battery_percent),
            is_estopped=False,
            is_faulted=False,
            details=dict(self.support.robot_state_details),
        )


def rgbd_stats(frame: RGBDFrame) -> dict[str, Any]:
    """Return compact numeric diagnostics for an RGB-D frame."""

    rgb = np.asarray(frame.rgb)
    depth = np.asarray(frame.depth_m)
    finite_depth = depth[np.isfinite(depth) & (depth > 0.0)]
    stats: dict[str, Any] = {
        "timestamp_s": float(frame.timestamp_s),
        "width": int(frame.width),
        "height": int(frame.height),
        "rgb_dtype": str(rgb.dtype),
        "depth_dtype": str(depth.dtype),
        "rgb_min": int(rgb.min()) if rgb.size else None,
        "rgb_max": int(rgb.max()) if rgb.size else None,
        "rgb_mean": float(rgb.mean()) if rgb.size else None,
        "rgb_channel_mean": [float(v) for v in rgb.reshape(-1, 3).mean(axis=0)] if rgb.size else [],
        "depth_valid_count": int(finite_depth.size),
        "depth_min_m": None if finite_depth.size == 0 else float(finite_depth.min()),
        "depth_max_m": None if finite_depth.size == 0 else float(finite_depth.max()),
        "depth_mean_m": None if finite_depth.size == 0 else float(finite_depth.mean()),
        "intrinsics": {
            "fx": float(frame.intrinsics.fx),
            "fy": float(frame.intrinsics.fy),
            "cx": float(frame.intrinsics.cx),
            "cy": float(frame.intrinsics.cy),
            "width": int(frame.intrinsics.width),
            "height": int(frame.intrinsics.height),
        },
        "metadata": dict(frame.metadata),
    }
    return stats


__all__ = [
    "RGBDAdapterLike",
    "RealSenseDryRunObservationProvider",
    "SyntheticSupportConfig",
    "rgbd_stats",
]


@runtime_checkable
class LidarAdapterLike(Protocol):
    """Small protocol implemented by live/replay Livox adapters and test fakes."""

    def read_lidar_frame(self) -> LidarFrame:
        ...


@dataclass
class LidarDryRunObservationProvider:
    """Observation provider using live/replay LiDAR plus synthetic support data.

    This is the LiDAR equivalent of ``RealSenseDryRunObservationProvider``:
    ``lidar_adapter.read_lidar_frame() -> RobotObservation.lidar`` while robot
    state remains synthetic unless explicitly disabled. RGB-D can remain absent
    because the default real-G1 watchdog does not require fresh RGB-D for motion.
    """

    lidar_adapter: LidarAdapterLike
    config: AdapterRuntimeConfig
    support: SyntheticSupportConfig = field(default_factory=lambda: SyntheticSupportConfig(mode="lidar_dryrun"))
    clock: Clock = time.monotonic
    include_synthetic_robot_state: bool = True
    source_name: str = "LidarDryRunObservationProvider"

    def read_observation(self) -> RobotObservation:
        lidar = self.lidar_adapter.read_lidar_frame()
        now_s = max(float(self.clock()), float(lidar.timestamp_s))
        robot_state = self._make_synthetic_robot_state(now_s) if self.include_synthetic_robot_state else None
        statuses = [
            AdapterStatus(
                name="lidar",
                health=AdapterHealth.OK,
                timestamp_s=lidar.timestamp_s,
                message="live/replay LiDAR frame available",
                details={"point_count": lidar.point_count, **dict(lidar.metadata)},
            )
        ]
        if robot_state is not None:
            statuses.append(
                AdapterStatus(
                    name="robot_state",
                    health=AdapterHealth.OK,
                    timestamp_s=robot_state.timestamp_s,
                    message="synthetic robot state placeholder used for dry-run safety",
                    details=dict(robot_state.details),
                )
            )
        return RobotObservation(
            timestamp_s=now_s,
            pose=self.support.pose,
            rgbd=None,
            lidar=lidar,
            robot_state=robot_state,
            statuses=tuple(statuses),
            metadata={
                "source": self.source_name,
                "rgbd_live": False,
                "lidar_synthetic": False,
                "robot_state_synthetic": robot_state is not None,
            },
        )

    def _make_synthetic_robot_state(self, timestamp_s: float) -> RobotState:
        return RobotState(
            timestamp_s=float(timestamp_s),
            pose=self.support.pose,
            mode=self.support.mode,
            battery_percent=float(self.support.battery_percent),
            is_estopped=False,
            is_faulted=False,
            details=dict(self.support.robot_state_details),
        )


@dataclass
class CompositeDryRunObservationProvider:
    """Observation provider combining optional live RGB-D and optional live LiDAR.

    Missing streams can be replaced by synthetic placeholders for dry-run bridge
    validation. This keeps staged deployment explicit: metadata reports which
    streams are live and which are synthetic.
    """

    config: AdapterRuntimeConfig
    rgbd_adapter: RGBDAdapterLike | None = None
    lidar_adapter: LidarAdapterLike | None = None
    support: SyntheticSupportConfig = field(default_factory=SyntheticSupportConfig)
    clock: Clock = time.monotonic
    include_synthetic_lidar_if_missing: bool = True
    include_synthetic_robot_state: bool = True
    source_name: str = "CompositeDryRunObservationProvider"

    def read_observation(self) -> RobotObservation:
        rgbd = self.rgbd_adapter.read_frame() if self.rgbd_adapter is not None else None
        lidar = self.lidar_adapter.read_lidar_frame() if self.lidar_adapter is not None else None
        timestamps = [float(self.clock())]
        if rgbd is not None:
            timestamps.append(float(rgbd.timestamp_s))
        if lidar is not None:
            timestamps.append(float(lidar.timestamp_s))
        now_s = max(timestamps)
        if lidar is None and self.include_synthetic_lidar_if_missing:
            lidar = self._make_synthetic_lidar(now_s)
        robot_state = self._make_synthetic_robot_state(now_s) if self.include_synthetic_robot_state else None

        statuses: list[AdapterStatus] = []
        if rgbd is not None:
            statuses.append(
                AdapterStatus(
                    name="rgbd",
                    health=AdapterHealth.OK,
                    timestamp_s=rgbd.timestamp_s,
                    message="live RGB-D frame available",
                    details={"width": rgbd.width, "height": rgbd.height, "synthetic": False},
                )
            )
        if lidar is not None:
            statuses.append(
                AdapterStatus(
                    name="lidar",
                    health=AdapterHealth.OK,
                    timestamp_s=lidar.timestamp_s,
                    message="LiDAR frame available",
                    details=dict(lidar.metadata),
                )
            )
        if robot_state is not None:
            statuses.append(
                AdapterStatus(
                    name="robot_state",
                    health=AdapterHealth.OK,
                    timestamp_s=robot_state.timestamp_s,
                    message="synthetic robot state placeholder used for dry-run safety",
                    details=dict(robot_state.details),
                )
            )
        return RobotObservation(
            timestamp_s=now_s,
            pose=self.support.pose,
            rgbd=rgbd,
            lidar=lidar,
            robot_state=robot_state,
            statuses=tuple(statuses),
            metadata={
                "source": self.source_name,
                "rgbd_live": rgbd is not None,
                "lidar_synthetic": lidar is not None and self.lidar_adapter is None,
                "robot_state_synthetic": robot_state is not None,
            },
        )

    def _make_synthetic_lidar(self, timestamp_s: float) -> LidarFrame:
        points = np.asarray(self.support.clear_lidar_points_body_xyz, dtype=np.float32)
        return LidarFrame(
            timestamp_s=float(timestamp_s),
            points_body_xyz=points,
            source_frame_id=self.config.lidar.source_frame_id,
            T_body_lidar=self.config.lidar.T_body_lidar,
            metadata=dict(self.support.lidar_metadata),
        )

    def _make_synthetic_robot_state(self, timestamp_s: float) -> RobotState:
        return RobotState(
            timestamp_s=float(timestamp_s),
            pose=self.support.pose,
            mode=self.support.mode,
            battery_percent=float(self.support.battery_percent),
            is_estopped=False,
            is_faulted=False,
            details=dict(self.support.robot_state_details),
        )


# Keep __all__ in sync with the staged providers added after the Step 9 API.
__all__ = [
    "RGBDAdapterLike",
    "LidarAdapterLike",
    "RealSenseDryRunObservationProvider",
    "LidarDryRunObservationProvider",
    "CompositeDryRunObservationProvider",
    "SyntheticSupportConfig",
    "rgbd_stats",
]
