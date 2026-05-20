from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from adapters import (
    Command2D,
    ConstantVelocityPolicy,
    LidarFrame,
    LidarTargetNavPolicyRuntime,
    LidarTargetNavPolicyRuntimeConfig,
    Pose2D,
    RobotObservation,
    RobotState,
    Target2D,
    Transform3D,
    load_adapter_config,
    occupancy_grid_from_lidar_frame,
    pose_from_observation,
)


def _observation(*, timestamp_s: float = 10.0, with_lidar: bool = True) -> RobotObservation:
    pose = Pose2D(0.0, 0.0, 0.0)
    state = RobotState(timestamp_s=timestamp_s, pose=pose, mode="test", battery_percent=100.0)
    lidar = None
    if with_lidar:
        lidar = LidarFrame(
            timestamp_s=timestamp_s,
            points_body_xyz=np.array([[2.0, 1.0, 0.35], [3.0, -1.0, 0.40]], dtype=np.float32),
            source_frame_id="livox_lidar",
            T_body_lidar=Transform3D.identity(frame="body"),
        )
    return RobotObservation(timestamp_s=timestamp_s, pose=pose, lidar=lidar, robot_state=state)


def test_constant_velocity_policy_returns_command() -> None:
    policy = ConstantVelocityPolicy(0.03, 0.01, -0.02)
    command = policy.step(_observation())

    assert isinstance(command, Command2D)
    assert command.tuple == (0.03, 0.01, -0.02)
    assert command.source == "constant_velocity_policy"


def test_pose_from_observation_prefers_observation_pose() -> None:
    obs = _observation()
    assert pose_from_observation(obs).xy == (0.0, 0.0)


def test_occupancy_grid_from_lidar_frame_rasterizes_points() -> None:
    cfg = load_adapter_config("configs/real_g1.yaml")
    obs = _observation()
    grid = occupancy_grid_from_lidar_frame(obs.lidar, grid_cfg=cfg_to_grid())

    assert grid.ndim == 2
    assert int((grid > 0).sum().item()) == 2


def cfg_to_grid():
    from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg

    return OccupancyGridCfg(x_min=-1.0, x_max=4.0, y_min=-2.0, y_max=2.0, resolution=0.1)


def test_lidar_target_nav_policy_emits_forward_command() -> None:
    cfg = load_adapter_config("configs/real_g1.yaml")
    policy_cfg = LidarTargetNavPolicyRuntimeConfig.from_runtime_config(
        cfg,
        target=Target2D(2.0, 0.0),
        inflation_radius_m=0.20,
    )
    policy = LidarTargetNavPolicyRuntime(policy_cfg)

    command = policy.step(_observation())

    assert command.vx > 0.0
    assert abs(command.vy) <= cfg.command_limits.vy_max
    assert command.source == "lidar_target_nav_policy"
    assert command.metadata["policy"] == "target-nav"
    assert command.metadata["plan_found"] is True
    assert policy.last_debug is not None
    assert policy.last_debug.lidar_points == 2


def test_lidar_target_nav_policy_requires_lidar_by_default() -> None:
    cfg = load_adapter_config("configs/real_g1.yaml")
    policy = LidarTargetNavPolicyRuntime(
        LidarTargetNavPolicyRuntimeConfig.from_runtime_config(cfg, target=Target2D(2.0, 0.0))
    )

    try:
        policy.step(_observation(with_lidar=False))
    except ValueError as exc:
        assert "lidar" in str(exc).lower()
    else:  # pragma: no cover - defensive
        raise AssertionError("expected missing-lidar ValueError")


def test_lidar_target_nav_policy_allows_empty_grid_when_configured() -> None:
    cfg = load_adapter_config("configs/real_g1.yaml")
    policy_cfg = LidarTargetNavPolicyRuntimeConfig.from_runtime_config(cfg, target=Target2D(1.0, 0.0))
    policy_cfg = LidarTargetNavPolicyRuntimeConfig(
        target=policy_cfg.target,
        grid_cfg=policy_cfg.grid_cfg,
        astar_cfg=policy_cfg.astar_cfg,
        nav_cfg=policy_cfg.nav_cfg,
        runtime_cfg=policy_cfg.runtime_cfg,
        require_lidar=False,
    )
    policy = LidarTargetNavPolicyRuntime(policy_cfg)

    command = policy.step(_observation(with_lidar=False))

    assert command.vx > 0.0
    assert command.metadata["lidar_points"] == 0


def test_follow_dryrun_target_nav_script_runs(monkeypatch, capsys) -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import real_g1_follow_dryrun

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
            "--json-lines",
        ],
    )
    assert real_g1_follow_dryrun.main() == 0
    row = json.loads(capsys.readouterr().out.strip())
    assert row["status_health"] in {"ok", "degraded"}
    assert row["proposed_command"][0] > 0.0
    assert row["has_lidar"] is True
