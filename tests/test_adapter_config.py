from __future__ import annotations

from pathlib import Path

import pytest

from adapters import AdapterRuntimeConfig, Command2D, load_adapter_config
from adapters.config import CommandLimitsConfig, LidarConfig


def test_load_real_g1_config_profile() -> None:
    cfg = load_adapter_config(Path("configs/real_g1.yaml"))

    assert cfg.profile_name == "real_g1_conservative"
    assert cfg.camera.width == 640
    assert cfg.camera.height == 480
    assert cfg.camera.intrinsics is None
    assert cfg.camera.T_body_camera.target_frame == "body"
    assert cfg.lidar.T_body_lidar.target_frame == "body"
    assert cfg.command_limits.vx_max <= 0.10
    assert cfg.watchdog.stop_on_stale_lidar
    assert cfg.perception.target_class == "person"
    assert cfg.safety.require_operator_enable


def test_command_limits_clip_command() -> None:
    limits = CommandLimitsConfig.from_dict(
        {
            "vx_min": -0.1,
            "vx_max": 0.2,
            "vy_min": -0.05,
            "vy_max": 0.05,
            "wz_min": -0.3,
            "wz_max": 0.3,
            "command_rate_hz": 20,
        }
    )
    cmd = Command2D(vx=1.0, vy=-0.2, wz=0.9, timestamp_s=1.0)
    clipped = limits.clip(cmd)

    assert clipped.tuple == pytest.approx((0.2, -0.05, 0.3))
    assert clipped.source == "safety_limited"


def test_config_rejects_invalid_bounds_and_transforms() -> None:
    with pytest.raises(ValueError):
        CommandLimitsConfig.from_dict({"vx_min": 0.2, "vx_max": -0.2})

    with pytest.raises(ValueError):
        LidarConfig.from_dict(
            {
                "T_body_lidar": {
                    "source_frame": "livox_lidar",
                    "target_frame": "map",
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            }
        )


def test_config_from_dict_allows_runtime_intrinsics() -> None:
    cfg = AdapterRuntimeConfig.from_dict(
        {
            "profile_name": "test_runtime_intrinsics",
            "camera": {
                "resolution": {"width": 320, "height": 240, "fps": 30},
                "intrinsics": None,
                "T_body_camera": {
                    "source_frame": "camera_color_optical",
                    "target_frame": "body",
                    "matrix": [
                        [1.0, 0.0, 0.0, 0.3],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.4],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            },
        }
    )

    assert cfg.camera.intrinsics is None
    assert cfg.camera.width == 320
    assert cfg.camera.T_body_camera.source_frame == "camera_color_optical"
