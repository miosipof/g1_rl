from __future__ import annotations

import numpy as np
import pytest

from adapters import CameraIntrinsics, Transform3D
from adapters.real_livox import (
    LatestLivoxFrameBuffer,
    LivoxAdapterConfig,
    apply_mount_correction,
    body_pose_from_slam_lidar_pose,
    filter_body_lidar_points,
    make_lidar_frame_from_livox_points,
)
from adapters.real_realsense import (
    RealSenseRGBDAdapterConfig,
    bgr_to_rgb,
    camera_intrinsics_from_mapping,
    depth_raw_to_meters,
    make_rgbd_frame_from_realsense_arrays,
)


def _T_body_camera() -> Transform3D:
    return Transform3D(
        matrix=np.eye(4, dtype=np.float64),
        source_frame="camera_color_optical",
        target_frame="body",
    )


def _T_body_lidar() -> Transform3D:
    return Transform3D(
        matrix=np.eye(4, dtype=np.float64),
        source_frame="livox_lidar",
        target_frame="body",
    )


def test_realsense_bgr_depth_conversion_and_rgbd_frame() -> None:
    color_bgr = np.array(
        [
            [[10, 20, 30], [1, 2, 3]],
            [[4, 5, 6], [7, 8, 9]],
        ],
        dtype=np.uint8,
    )
    rgb = bgr_to_rgb(color_bgr)
    assert rgb[0, 0].tolist() == [30, 20, 10]
    assert rgb.flags.c_contiguous or rgb.base is None

    depth_raw = np.array([[1000, 0], [2500, 500]], dtype=np.uint16)
    depth_m = depth_raw_to_meters(depth_raw, depth_scale_m_per_unit=0.001)
    assert depth_m.dtype == np.float32
    assert depth_m[0, 0] == pytest.approx(1.0)
    assert depth_m[1, 0] == pytest.approx(2.5)

    intr = CameraIntrinsics(fx=100.0, fy=110.0, cx=0.5, cy=0.5, width=2, height=2)
    frame = make_rgbd_frame_from_realsense_arrays(
        timestamp_s=3.0,
        color_bgr=color_bgr,
        depth_raw=depth_raw,
        intrinsics=intr,
        T_body_camera=_T_body_camera(),
        depth_scale_m_per_unit=0.001,
    )

    assert frame.width == 2
    assert frame.height == 2
    assert frame.rgb[0, 0].tolist() == [30, 20, 10]
    assert frame.depth_m[1, 0] == pytest.approx(2.5)


def test_realsense_intrinsics_from_mapping_and_config() -> None:
    intr = camera_intrinsics_from_mapping(
        {"width": 640, "height": 480, "fx": 600.0, "fy": 601.0, "ppx": 319.5, "ppy": 239.5}
    )
    cfg = RealSenseRGBDAdapterConfig(width=640, height=480, fps=30, T_body_camera=_T_body_camera())

    assert intr.width == 640
    assert intr.height == 480
    assert intr.cx == pytest.approx(319.5)
    assert cfg.T_body_camera.target_frame == "body"


def test_livox_mount_correction_filtering_and_frame() -> None:
    points = np.array(
        [
            [1.0, 2.0, 3.0],       # becomes z=-3 after upside-down -> filtered out
            [1.0, -0.2, -0.5],     # becomes z=0.5 -> kept
            [0.01, 0.0, -0.5],     # too close after correction
            [9.0, 0.0, -0.5],      # too far after correction
        ],
        dtype=np.float32,
    )
    corrected = apply_mount_correction(points, mount_mode="upside_down")
    assert corrected[0].tolist() == pytest.approx([1.0, -2.0, -3.0])

    frame = make_lidar_frame_from_livox_points(
        timestamp_s=4.0,
        points_lidar_xyz=points,
        T_body_lidar=_T_body_lidar(),
        mount_mode="upside_down",
        min_range_m=0.15,
        max_range_m=5.0,
        z_min_m=0.05,
        z_max_m=1.30,
    )

    assert frame.frame_id == "body"
    assert frame.source_frame_id == "livox_lidar"
    assert frame.point_count == 1
    assert frame.points_body_xyz[0] == pytest.approx([1.0, 0.2, 0.5])
    assert frame.metadata["raw_points"] == 4
    assert frame.metadata["filtered_points"] == 1


def test_livox_filter_decimates_deterministically() -> None:
    points = np.array([[float(i), 0.0, 0.5] for i in range(1, 11)], dtype=np.float32)
    out = filter_body_lidar_points(
        points,
        min_range_m=0.0,
        max_range_m=20.0,
        z_min_m=0.0,
        z_max_m=1.0,
        max_points=3,
    )
    assert out.shape == (3, 3)
    assert out[:, 0].tolist() == pytest.approx([1.0, 5.0, 10.0])


def test_livox_slam_pose_and_latest_buffer() -> None:
    T_body_lidar = Transform3D(
        matrix=np.array(
            [
                [1.0, 0.0, 0.0, 0.2],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        source_frame="livox_lidar",
        target_frame="body",
    )
    T_map_lidar = np.eye(4, dtype=np.float64)
    T_map_lidar[0, 3] = 2.0
    pose, T_map_body = body_pose_from_slam_lidar_pose(T_map_lidar=T_map_lidar, T_body_lidar=T_body_lidar)
    assert pose.x == pytest.approx(1.8)
    assert T_map_body.source_frame == "body"
    assert T_map_body.target_frame == "map"

    cfg = LivoxAdapterConfig(
        T_body_lidar=_T_body_lidar(),
        mount_mode="normal",
        min_range_m=0.0,
        max_range_m=5.0,
        z_min_m=-1.0,
        z_max_m=1.0,
    )
    buffer = LatestLivoxFrameBuffer(cfg)
    buffer.update(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), timestamp_s=10.0)
    frame = buffer.read_lidar_frame()
    assert buffer.has_frame
    assert frame.timestamp_s == pytest.approx(10.0)
    assert frame.point_count == 1


def test_livox_npy_replay_adapter_and_stats(tmp_path) -> None:
    from adapters.real_livox import LivoxAdapterConfig, NpyLivoxReplayAdapter, lidar_frame_stats

    path = tmp_path / "livox.npy"
    np.save(path, np.array([[[1.0, 0.0, 0.4], [2.0, 0.5, 0.5]]], dtype=np.float32))
    cfg = LivoxAdapterConfig(
        T_body_lidar=_T_body_lidar(),
        mount_mode="normal",
        min_range_m=0.0,
        max_range_m=5.0,
        z_min_m=0.0,
        z_max_m=1.0,
    )
    replay = NpyLivoxReplayAdapter(path, cfg, clock=lambda: 12.0)
    frame = replay.read_lidar_frame()
    stats = lidar_frame_stats(frame)

    assert frame.timestamp_s == pytest.approx(12.0)
    assert frame.point_count == 2
    assert stats["point_count"] == 2
    assert stats["max_xy_range_m"] > 2.0


def test_unitree_vibes_livox_adapter_requires_start_before_read() -> None:
    from adapters.real_livox import LivoxAdapterConfig, UnitreeVibesLivox2Adapter, UnitreeVibesLivox2AdapterConfig

    cfg = UnitreeVibesLivox2AdapterConfig(
        livox_config_path="mid360_config.json",
        host_ip="192.168.123.222",
        adapter_config=LivoxAdapterConfig(T_body_lidar=_T_body_lidar()),
    )
    adapter = UnitreeVibesLivox2Adapter(cfg)
    with pytest.raises(RuntimeError, match="start"):
        adapter.read_lidar_frame(timeout_s=0.01)
