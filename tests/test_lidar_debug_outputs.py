from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from adapters import LidarDebugWriter, LidarFrame, Pose2D, RobotObservation, RobotState, Target2D, Transform3D, load_adapter_config
from adapters.g1_policy_runtime import LidarTargetNavPolicyRuntime, LidarTargetNavPolicyRuntimeConfig
from adapters.real_bridge import BridgeStepResult, RealBridge, RealBridgeConfig
from adapters.safety_supervisor import SafetySupervisor
from adapters.real_unitree_loco import DryRunCommandSink


class _StaticObservationProvider:
    def __init__(self) -> None:
        self.obs = _observation()

    def read_observation(self) -> RobotObservation:
        return self.obs


def _observation() -> RobotObservation:
    timestamp_s = 123.0
    pose = Pose2D(0.0, 0.0, 0.0)
    lidar = LidarFrame(
        timestamp_s=timestamp_s,
        points_body_xyz=np.array([[2.0, 1.0, 0.35], [2.5, -1.0, 0.40]], dtype=np.float32),
        source_frame_id="livox_lidar",
        T_body_lidar=Transform3D.identity(frame="body"),
        metadata={"test": True},
    )
    state = RobotState(timestamp_s=timestamp_s, pose=pose, mode="test", battery_percent=100.0)
    return RobotObservation(
        timestamp_s=timestamp_s,
        pose=pose,
        lidar=lidar,
        robot_state=state,
        target_hint=Target2D(2.0, 0.0),
        metadata={"source": "test"},
    )


def test_lidar_debug_writer_saves_expected_artifacts(tmp_path: Path) -> None:
    cfg = load_adapter_config("configs/real_g1.yaml")
    policy_cfg = LidarTargetNavPolicyRuntimeConfig.from_runtime_config(
        cfg,
        target=Target2D(2.0, 0.0),
        inflation_radius_m=0.20,
    )
    policy = LidarTargetNavPolicyRuntime(policy_cfg)
    bridge = RealBridge(
        observation_provider=_StaticObservationProvider(),
        policy_runtime=policy,
        safety_supervisor=SafetySupervisor.from_runtime_config(cfg),
        command_sink=DryRunCommandSink(),
        config=RealBridgeConfig(send_commands=True),
    )
    assert bridge.follow().ok
    result = bridge.step()

    writer = LidarDebugWriter(tmp_path, grid_cfg=policy_cfg.grid_cfg)
    paths = writer.write_step(0, result, policy=policy)

    assert Path(paths["lidar_points_body_step"]).exists()
    assert Path(paths["occupancy_grid_step"]).exists()
    assert Path(paths["occupancy_grid_meta_step"]).exists()
    assert (tmp_path / "lidar_points_body.npy").exists()
    assert (tmp_path / "occupancy_grid.png").exists()
    assert (tmp_path / "occupancy_grid_meta.json").exists()
    assert (tmp_path / "target_nav_command_log.csv").exists()

    points = np.load(tmp_path / "lidar_points_body.npy")
    assert points.shape == (2, 3)
    meta = json.loads((tmp_path / "occupancy_grid_meta.json").read_text())
    assert meta["grid"]["occupied_cells"] == 2
    assert meta["grid"]["coordinate_convention"]["y_axis"].startswith("+Y robot-left")
    assert "mirrored relative" in meta["grid"]["coordinate_convention"]["note"]
    assert meta["policy_debug"]["plan_found"] is True


def test_follow_dryrun_save_lidar_debug_cli(monkeypatch, capsys, tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import real_g1_follow_dryrun

    debug_dir = tmp_path / "debug"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "real_g1_follow_dryrun.py",
            "--policy",
            "target-nav",
            "--target-x",
            "2.0",
            "--target-y",
            "0.0",
            "--steps",
            "1",
            "--dt",
            "0",
            "--save-lidar-debug",
            str(debug_dir),
            "--json-lines",
        ],
    )
    assert real_g1_follow_dryrun.main() == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert "lidar_debug" in row
    assert (debug_dir / "occupancy_grid.png").exists()
    assert (debug_dir / "target_nav_command_log.csv").exists()
