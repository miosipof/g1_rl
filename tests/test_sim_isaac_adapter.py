from __future__ import annotations

import numpy as np
import pytest

from adapters import CameraIntrinsics, Command2D, Transform3D
from adapters.sim_isaac import (
    IsaacAdapterSkeleton,
    IsaacAdapterSkeletonConfig,
    make_lidar_frame_from_world_points,
    make_rgbd_frame,
    pose2d_from_xy_yaw,
    world_points_to_body_xyz,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=1.5, cy=1.0, width=4, height=3)


def test_world_points_to_body_xyz_uses_pose_yaw() -> None:
    pose = pose2d_from_xy_yaw((1.0, 2.0, np.pi / 2.0))
    points_w = np.array(
        [
            [1.0, 3.0, 0.5],  # one meter along world +y -> body +x at yaw=90deg
            [2.0, 2.0, 0.5],  # one meter along world +x -> body -y at yaw=90deg
        ],
        dtype=np.float32,
    )

    points_b = world_points_to_body_xyz(points_w, pose, body_frame_z_w=0.1)

    assert points_b.shape == (2, 3)
    assert points_b[0] == pytest.approx((1.0, 0.0, 0.4), abs=1e-6)
    assert points_b[1] == pytest.approx((0.0, -1.0, 0.4), abs=1e-6)


def test_make_lidar_frame_from_world_points_contract() -> None:
    pose = pose2d_from_xy_yaw((0.5, -0.5, 0.0))
    points_w = np.array([[1.5, -0.5, 0.3], [2.5, 0.0, 1.4]], dtype=np.float32)

    lidar = make_lidar_frame_from_world_points(
        timestamp_s=12.0,
        points_world_xyz=points_w,
        pose=pose,
        body_frame_z_w=0.1,
        source_frame_id="raycaster",
    )

    assert lidar.frame_id == "body"
    assert lidar.source_frame_id == "raycaster"
    assert lidar.point_count == 2
    assert lidar.points_body_xy[0] == pytest.approx((1.0, 0.0))
    assert lidar.filtered_by_height(z_min_m=0.0, z_max_m=0.5).point_count == 1


def test_make_rgbd_and_observation_without_isaac_imports() -> None:
    adapter = IsaacAdapterSkeleton(IsaacAdapterSkeletonConfig(body_frame_z_w=0.2))
    pose = adapter.make_pose((0.0, 0.0, 0.0))
    T = Transform3D.identity(frame="body")
    T_body_camera = Transform3D(matrix=T.matrix, source_frame="camera_color_optical", target_frame="body")

    rgbd = adapter.make_rgbd(
        timestamp_s=2.0,
        rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        depth_m=np.ones((3, 4), dtype=np.float32),
        intrinsics=_intrinsics(),
        T_body_camera=T_body_camera,
    )
    lidar = adapter.make_lidar_from_world_points(
        timestamp_s=2.1,
        points_world_xyz=np.array([[1.0, 0.0, 0.4]], dtype=np.float32),
        pose=pose,
    )
    obs = adapter.make_observation(timestamp_s=2.2, pose=pose, rgbd=rgbd, lidar=lidar)

    assert obs.ok
    assert obs.has_rgbd
    assert obs.has_lidar
    assert obs.has_pose
    assert obs.robot_state is not None
    assert obs.sensor_ages_s()["rgbd"] == pytest.approx(0.2)
    assert adapter.command_to_motion_tuple(Command2D(0.1, -0.2, 0.3)) == pytest.approx((0.1, -0.2, 0.3))


def test_make_rgbd_rejects_non_rgb_shape() -> None:
    T = Transform3D(np.eye(4), source_frame="camera_color_optical", target_frame="body")
    with pytest.raises(ValueError):
        make_rgbd_frame(
            timestamp_s=0.0,
            rgb=np.zeros((3, 4), dtype=np.uint8),
            depth_m=np.zeros((3, 4), dtype=np.float32),
            intrinsics=_intrinsics(),
            T_body_camera=T,
        )
