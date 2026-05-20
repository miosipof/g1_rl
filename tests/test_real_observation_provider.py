from __future__ import annotations

import numpy as np

from adapters import CameraIntrinsics, RealSenseDryRunObservationProvider, Transform3D, rgbd_stats, load_adapter_config
from adapters.types import RGBDFrame


class FakeRGBDAdapter:
    def __init__(self, frame: RGBDFrame) -> None:
        self.frame = frame
        self.calls = 0

    def read_frame(self) -> RGBDFrame:
        self.calls += 1
        return self.frame


def _frame(timestamp_s: float = 10.0) -> RGBDFrame:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 20
    rgb[..., 2] = 30
    depth = np.array(
        [
            [0.0, 1.0, 1.5, 2.0, 0.0],
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [2.0, 2.1, 2.2, 2.3, 2.4],
            [0.0, 3.0, 3.1, 3.2, 3.3],
        ],
        dtype=np.float32,
    )
    return RGBDFrame(
        timestamp_s=timestamp_s,
        rgb=rgb,
        depth_m=depth,
        intrinsics=CameraIntrinsics(fx=100.0, fy=100.0, cx=2.0, cy=1.5, width=5, height=4),
        T_body_camera=Transform3D.identity(frame="body"),
        metadata={"source": "fake"},
    )


def test_realsense_dryrun_provider_builds_observation_with_live_rgbd() -> None:
    cfg = load_adapter_config("configs/real_g1.yaml")
    fake = FakeRGBDAdapter(_frame(timestamp_s=10.0))
    provider = RealSenseDryRunObservationProvider(rgbd_adapter=fake, config=cfg, clock=lambda: 10.1)

    obs = provider.read_observation()

    assert fake.calls == 1
    assert obs.has_rgbd is True
    assert obs.has_lidar is True
    assert obs.robot_state is not None
    assert obs.metadata["rgbd_live"] is True
    assert obs.metadata["lidar_synthetic"] is True
    assert obs.metadata["robot_state_synthetic"] is True
    assert obs.rgbd is fake.frame
    assert obs.lidar.point_count == 2
    assert obs.robot_state.mode == "rgbd_dryrun"


def test_realsense_dryrun_provider_can_omit_synthetic_support() -> None:
    cfg = load_adapter_config("configs/real_g1.yaml")
    provider = RealSenseDryRunObservationProvider(
        rgbd_adapter=FakeRGBDAdapter(_frame()),
        config=cfg,
        include_synthetic_lidar=False,
        include_synthetic_robot_state=False,
    )

    obs = provider.read_observation()

    assert obs.has_rgbd is True
    assert obs.has_lidar is False
    assert obs.robot_state is None


def test_rgbd_stats_reports_depth_and_intrinsics() -> None:
    stats = rgbd_stats(_frame())

    assert stats["width"] == 5
    assert stats["height"] == 4
    assert stats["rgb_channel_mean"] == [10.0, 20.0, 30.0]
    assert stats["depth_valid_count"] == 17
    assert stats["depth_min_m"] == 1.0
    assert stats["intrinsics"]["fx"] == 100.0


class FakeLidarAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def read_lidar_frame(self):
        from adapters import LidarFrame, Transform3D

        self.calls += 1
        return LidarFrame(
            timestamp_s=20.0,
            points_body_xyz=np.array([[1.0, 0.0, 0.4]], dtype=np.float32),
            source_frame_id="livox_lidar",
            T_body_lidar=Transform3D.identity(frame="body"),
            metadata={"source": "fake_livox"},
        )


def test_lidar_dryrun_provider_builds_observation_with_live_lidar() -> None:
    from adapters import LidarDryRunObservationProvider, load_adapter_config

    cfg = load_adapter_config("configs/real_g1.yaml")
    fake = FakeLidarAdapter()
    provider = LidarDryRunObservationProvider(lidar_adapter=fake, config=cfg, clock=lambda: 20.1)

    obs = provider.read_observation()

    assert fake.calls == 1
    assert obs.has_rgbd is False
    assert obs.has_lidar is True
    assert obs.robot_state is not None
    assert obs.metadata["lidar_synthetic"] is False
    assert obs.metadata["robot_state_synthetic"] is True
    assert obs.lidar.point_count == 1


def test_composite_dryrun_provider_can_combine_rgbd_and_lidar() -> None:
    from adapters import CompositeDryRunObservationProvider, load_adapter_config

    cfg = load_adapter_config("configs/real_g1.yaml")
    provider = CompositeDryRunObservationProvider(
        config=cfg,
        rgbd_adapter=FakeRGBDAdapter(_frame(timestamp_s=19.0)),
        lidar_adapter=FakeLidarAdapter(),
        clock=lambda: 20.1,
    )

    obs = provider.read_observation()

    assert obs.has_rgbd is True
    assert obs.has_lidar is True
    assert obs.metadata["rgbd_live"] is True
    assert obs.metadata["lidar_synthetic"] is False
