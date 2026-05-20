"""Isaac-side adapter skeleton for the shared sim/real interface.

This module intentionally does **not** import Isaac Lab, Isaac Sim, USD, Torch,
or Omniverse packages at module import time.  It provides a small set of helper
conversions that turn ordinary Python/NumPy values extracted from an Isaac scene
into the hardware-neutral adapter contracts defined in :mod:`adapters.types`.

The goal of this step is not to replace ``scripts/smoke_test_yolo_person_nav.py``
yet.  The goal is to establish the seam that the large smoke script can later
call:

    Isaac scene/proxy/camera/RayCaster -> IsaacAdapterSkeleton -> RobotObservation

A future concrete Isaac adapter can own the scene-specific reads.  This skeleton
only owns validation, frame convention, and object construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .types import (
    AdapterHealth,
    AdapterStatus,
    CameraIntrinsics,
    Command2D,
    LidarFrame,
    Pose2D,
    RGBDFrame,
    RobotObservation,
    RobotState,
    Transform3D,
)


@dataclass(frozen=True)
class IsaacAdapterSkeletonConfig:
    """Configuration for the dependency-light Isaac adapter skeleton.

    ``body_frame_z_w`` is the world-frame z coordinate that should be subtracted
    from Isaac world hit points when constructing body-frame LiDAR points.  For
    the current 2D navigation layer this value is usually not critical because
    only x/y are rasterized, but keeping z meaningful makes the observation
    contract compatible with real 3D LiDAR data.
    """

    body_frame: str = "body"
    camera_frame: str = "camera_color_optical"
    depth_frame: str = "camera_depth_optical"
    lidar_source_frame: str = "isaac_lidar"
    body_frame_z_w: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IsaacAdapterSkeleton:
    """Small builder for Isaac-derived :class:`RobotObservation` objects.

    The class accepts already-extracted arrays and scalar pose values.  It does
    not read from Isaac itself.  This keeps unit tests headless and lets the
    large smoke scripts migrate one call-site at a time.
    """

    config: IsaacAdapterSkeletonConfig = field(default_factory=IsaacAdapterSkeletonConfig)

    def make_pose(self, position_xy_yaw: Sequence[float] | Pose2D) -> Pose2D:
        return pose2d_from_xy_yaw(position_xy_yaw)

    def make_rgbd(
        self,
        *,
        timestamp_s: float,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        T_body_camera: Transform3D,
        metadata: Mapping[str, Any] | None = None,
    ) -> RGBDFrame:
        return make_rgbd_frame(
            timestamp_s=timestamp_s,
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
            T_body_camera=T_body_camera,
            frame_id=self.config.camera_frame,
            depth_frame_id=self.config.depth_frame,
            metadata=metadata,
        )

    def make_lidar_from_world_points(
        self,
        *,
        timestamp_s: float,
        points_world_xyz: np.ndarray,
        pose: Pose2D,
        T_body_lidar: Transform3D | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LidarFrame:
        return make_lidar_frame_from_world_points(
            timestamp_s=timestamp_s,
            points_world_xyz=points_world_xyz,
            pose=pose,
            body_frame_z_w=self.config.body_frame_z_w,
            source_frame_id=self.config.lidar_source_frame,
            T_body_lidar=T_body_lidar,
            metadata=metadata,
        )

    def make_observation(
        self,
        *,
        timestamp_s: float,
        pose: Pose2D | None = None,
        rgbd: RGBDFrame | None = None,
        lidar: LidarFrame | None = None,
        robot_state: RobotState | None = None,
        statuses: Sequence[AdapterStatus] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RobotObservation:
        return make_robot_observation(
            timestamp_s=timestamp_s,
            pose=pose,
            rgbd=rgbd,
            lidar=lidar,
            robot_state=robot_state,
            statuses=statuses,
            metadata={**dict(self.config.metadata), **dict(metadata or {})},
        )

    def command_to_motion_tuple(self, command: Command2D) -> tuple[float, float, float]:
        """Return the body-frame command tuple expected by the sim proxy.

        The current Isaac proxy and the real Unitree locomotion API both consume
        the same conceptual command: forward velocity, lateral velocity, yaw rate.
        """

        return command.tuple


def pose2d_from_xy_yaw(value: Sequence[float] | Pose2D) -> Pose2D:
    """Normalize a pose-like value into :class:`Pose2D`.

    Accepted values are an existing ``Pose2D`` or a 3-element sequence
    ``(x_w, y_w, yaw_w)``.
    """

    if isinstance(value, Pose2D):
        return value
    if len(value) != 3:
        raise ValueError(f"Pose requires three values (x, y, yaw), got {value!r}")
    x, y, yaw = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(v) for v in (x, y, yaw)):
        raise ValueError(f"Pose contains non-finite values: {value!r}")
    return Pose2D(x=x, y=y, yaw=yaw)


def make_rgbd_frame(
    *,
    timestamp_s: float,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    T_body_camera: Transform3D,
    frame_id: str = "camera_color_optical",
    depth_frame_id: str = "camera_depth_optical",
    metadata: Mapping[str, Any] | None = None,
) -> RGBDFrame:
    """Construct a normalized RGB-D frame from Isaac camera arrays.

    Required convention: ``rgb`` must already be ``HxWx3`` RGB uint8 and
    ``depth_m`` must already be meters.  The current Isaac camera path satisfies
    this after moving tensors to CPU NumPy arrays.
    """

    return RGBDFrame(
        timestamp_s=float(timestamp_s),
        rgb=np.asarray(rgb),
        depth_m=np.asarray(depth_m),
        intrinsics=intrinsics,
        T_body_camera=T_body_camera,
        frame_id=frame_id,
        depth_frame_id=depth_frame_id,
        metadata=dict(metadata or {}),
    )


def world_points_to_body_xyz(
    points_world_xyz: np.ndarray,
    pose: Pose2D,
    *,
    body_frame_z_w: float = 0.0,
) -> np.ndarray:
    """Convert Isaac world-frame points into the robot body frame.

    Convention:
    - world/body x is forward at yaw=0;
    - body y is left;
    - positive yaw rotates body x toward world y;
    - z is translated by ``body_frame_z_w`` but otherwise not rotated.

    The z simplification is intentional for the current 2D navigation layer and
    mirrors how the smoke scripts use LiDAR for local occupancy.  A future full
    3D adapter can replace this with a calibrated 6-DoF transform.
    """

    points = _as_points_xyz(points_world_xyz, name="points_world_xyz")
    if points.size == 0:
        return points.copy()

    c = math.cos(float(pose.yaw))
    s = math.sin(float(pose.yaw))
    dx = points[:, 0].astype(np.float64) - float(pose.x)
    dy = points[:, 1].astype(np.float64) - float(pose.y)

    body_x = c * dx + s * dy
    body_y = -s * dx + c * dy
    body_z = points[:, 2].astype(np.float64) - float(body_frame_z_w)
    return np.stack([body_x, body_y, body_z], axis=1).astype(np.float32)


def make_lidar_frame_from_world_points(
    *,
    timestamp_s: float,
    points_world_xyz: np.ndarray,
    pose: Pose2D,
    body_frame_z_w: float = 0.0,
    source_frame_id: str = "isaac_lidar",
    T_body_lidar: Transform3D | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> LidarFrame:
    """Construct a normalized body-frame LiDAR frame from Isaac hit points."""

    points_body = world_points_to_body_xyz(points_world_xyz, pose, body_frame_z_w=body_frame_z_w)
    return LidarFrame(
        timestamp_s=float(timestamp_s),
        points_body_xyz=points_body,
        frame_id="body",
        source_frame_id=source_frame_id,
        T_body_lidar=T_body_lidar,
        metadata=dict(metadata or {}),
    )


def make_robot_state(
    *,
    timestamp_s: float,
    pose: Pose2D | None,
    mode: str = "isaac_sim",
    details: Mapping[str, Any] | None = None,
) -> RobotState:
    """Create a nominal sim robot state."""

    return RobotState(
        timestamp_s=float(timestamp_s),
        pose=pose,
        mode=mode,
        is_estopped=False,
        is_faulted=False,
        details=dict(details or {}),
    )


def make_robot_observation(
    *,
    timestamp_s: float,
    pose: Pose2D | None = None,
    rgbd: RGBDFrame | None = None,
    lidar: LidarFrame | None = None,
    robot_state: RobotState | None = None,
    statuses: Sequence[AdapterStatus] = (),
    metadata: Mapping[str, Any] | None = None,
) -> RobotObservation:
    """Create a complete observation with nominal sim status if none is given."""

    status_tuple = tuple(statuses) or (
        AdapterStatus(name="isaac_adapter", health=AdapterHealth.OK, timestamp_s=float(timestamp_s)),
    )
    if robot_state is None and pose is not None:
        robot_state = make_robot_state(timestamp_s=timestamp_s, pose=pose)
    return RobotObservation(
        timestamp_s=float(timestamp_s),
        pose=pose,
        rgbd=rgbd,
        lidar=lidar,
        robot_state=robot_state,
        statuses=status_tuple,
        metadata=dict(metadata or {}),
    )


def _as_points_xyz(points_xyz: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape Nx3, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite values")
    return points


__all__ = [
    "IsaacAdapterSkeleton",
    "IsaacAdapterSkeletonConfig",
    "make_lidar_frame_from_world_points",
    "make_rgbd_frame",
    "make_robot_observation",
    "make_robot_state",
    "pose2d_from_xy_yaw",
    "world_points_to_body_xyz",
]
