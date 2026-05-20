from __future__ import annotations

import numpy as np
import pytest

from adapters import (
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
from g1_nav_rl.navigation.simple_nav import MotionCommand


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=1.5, cy=1.0, width=4, height=3)


def test_rgbd_lidar_observation_contracts() -> None:
    T_body_camera = Transform3D.identity(frame="body")
    T_body_camera = Transform3D(
        matrix=T_body_camera.matrix,
        source_frame="camera_color_optical",
        target_frame="body",
    )

    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth_m = np.ones((3, 4), dtype=np.float32)
    rgbd = RGBDFrame(
        timestamp_s=10.0,
        rgb=rgb,
        depth_m=depth_m,
        intrinsics=_intrinsics(),
        T_body_camera=T_body_camera,
    )

    points = np.array([[1.0, 0.0, 0.2], [2.0, 0.5, 1.2]], dtype=np.float32)
    lidar = LidarFrame(timestamp_s=10.1, points_body_xyz=points)
    robot_state = RobotState(timestamp_s=10.2, pose=Pose2D(0.0, 0.0, 0.0), mode="test")

    obs = RobotObservation(timestamp_s=10.2, pose=robot_state.pose, rgbd=rgbd, lidar=lidar, robot_state=robot_state)

    assert obs.ok
    assert obs.has_rgbd
    assert obs.has_lidar
    assert obs.has_pose
    assert obs.sensor_ages_s()["rgbd"] == pytest.approx(0.2)
    assert lidar.points_body_xy.shape == (2, 2)
    assert lidar.filtered_by_height(z_min_m=0.0, z_max_m=0.5).point_count == 1


def test_command2d_motion_command_conversion_and_clipping() -> None:
    motion = MotionCommand(vx=0.3, vy=-0.2, wz=0.6)
    command = Command2D.from_motion_command(motion, timestamp_s=1.0, duration_s=0.1)
    clipped = command.clipped(vx_min=-0.1, vx_max=0.1, vy_min=-0.05, vy_max=0.05, wz_min=-0.2, wz_max=0.2)

    assert clipped.tuple == pytest.approx((0.1, -0.05, 0.2))
    assert clipped.to_motion_command().tuple == pytest.approx((0.1, -0.05, 0.2))
    assert Command2D.stop().is_stop


def test_validation_rejects_bad_shapes_and_faulted_status() -> None:
    with pytest.raises(ValueError):
        LidarFrame(timestamp_s=0.0, points_body_xyz=np.zeros((3, 2), dtype=np.float32))

    T_body_camera = Transform3D(np.eye(4), source_frame="camera", target_frame="body")
    with pytest.raises(ValueError):
        RGBDFrame(
            timestamp_s=0.0,
            rgb=np.zeros((3, 4, 3), dtype=np.uint8),
            depth_m=np.zeros((2, 4), dtype=np.float32),
            intrinsics=_intrinsics(),
            T_body_camera=T_body_camera,
        )

    obs = RobotObservation(
        timestamp_s=1.0,
        statuses=(AdapterStatus(name="camera", health=AdapterHealth.FAULT),),
    )
    assert not obs.ok
