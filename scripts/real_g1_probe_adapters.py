#!/usr/bin/env python3
"""Probe the sim-to-real adapter contracts without moving the robot.

This script is intentionally hardware-neutral by default.  It loads the real-G1
adapter config, builds a synthetic but fully-formed RobotObservation, runs one
SafetySupervisor check, and prints a compact status report.  It is the first
entry point to verify that the shared adapter layer is importable on the target
machine before wiring RealSense, Livox, or Unitree SDK objects.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters import Command2D, LidarFrame, Pose2D, RobotObservation, RobotState, load_adapter_config
from adapters.safety_supervisor import SafetySupervisor


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


def make_synthetic_observation(*, now_s: float, config_path: Path) -> tuple[RobotObservation, Any]:
    """Create a minimal observation that satisfies real-robot safety gates."""

    cfg = load_adapter_config(config_path)
    pose = Pose2D(0.0, 0.0, 0.0)
    robot_state = RobotState(
        timestamp_s=now_s,
        pose=pose,
        mode="synthetic_probe",
        fsm_state=cfg.metadata.get("expected_fsm_state"),
        battery_percent=100.0,
        is_estopped=False,
        is_faulted=False,
        details={"source": "real_g1_probe_adapters.py", "synthetic": True},
    )

    # Points are deliberately outside the front safety corridor, so the command
    # check below exercises normal acceptance rather than obstacle stop.
    points_body_xyz = np.array(
        [
            [2.0, 1.2, 0.35],
            [3.0, -1.3, 0.45],
            [1.5, 0.9, 0.60],
        ],
        dtype=np.float32,
    )
    lidar = LidarFrame(
        timestamp_s=now_s,
        points_body_xyz=points_body_xyz,
        source_frame_id=cfg.lidar.source_frame_id,
        T_body_lidar=cfg.lidar.T_body_lidar,
        metadata={"source": "synthetic_probe", "point_count_before_filter": int(points_body_xyz.shape[0])},
    )
    obs = RobotObservation(
        timestamp_s=now_s,
        pose=pose,
        lidar=lidar,
        robot_state=robot_state,
        metadata={"source": "synthetic_probe", "config": str(config_path)},
    )
    return obs, cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    parser.add_argument("--probe-command-vx", type=float, default=0.05, help="Synthetic command vx used for safety probing.")
    parser.add_argument("--probe-command-vy", type=float, default=0.0, help="Synthetic command vy used for safety probing.")
    parser.add_argument("--probe-command-wz", type=float, default=0.0, help="Synthetic command wz used for safety probing.")
    args = parser.parse_args()

    now_s = time.monotonic()
    obs, cfg = make_synthetic_observation(now_s=now_s, config_path=args.config)
    supervisor = SafetySupervisor.from_runtime_config(cfg)
    command = Command2D(args.probe_command_vx, args.probe_command_vy, args.probe_command_wz, timestamp_s=now_s, source="probe")
    decision = supervisor.filter_command(command, obs, operator_enabled=True, now_s=now_s, policy_timestamp_s=now_s)

    report: dict[str, Any] = {
        "ok": decision.ok,
        "profile_name": cfg.profile_name,
        "config_path": str(args.config),
        "camera_enabled": cfg.camera.enabled,
        "camera_resolution": [cfg.camera.width, cfg.camera.height],
        "lidar_enabled": cfg.lidar.enabled,
        "lidar_mount_mode": cfg.lidar.mount_mode,
        "command_limits": {
            "vx": [cfg.command_limits.vx_min, cfg.command_limits.vx_max],
            "vy": [cfg.command_limits.vy_min, cfg.command_limits.vy_max],
            "wz": [cfg.command_limits.wz_min, cfg.command_limits.wz_max],
            "rate_hz": cfg.command_limits.command_rate_hz,
        },
        "watchdog": {
            "max_lidar_age_s": cfg.watchdog.max_lidar_age_s,
            "max_robot_state_age_s": cfg.watchdog.max_robot_state_age_s,
        },
        "observation": {
            "timestamp_s": obs.timestamp_s,
            "has_lidar": obs.has_lidar,
            "has_robot_state": obs.robot_state is not None,
            "sensor_ages_s": obs.sensor_ages_s(),
        },
        "safety_decision": {
            "health": decision.status.health.value,
            "message": decision.status.message,
            "reasons": list(decision.reasons),
            "warnings": list(decision.warnings),
            "proposed_command": decision.proposed_command.tuple,
            "safe_command": decision.command.tuple,
            "allow_motion": decision.allow_motion,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("[INFO]: adapter probe complete")
        print(f"[INFO]: profile: {report['profile_name']}")
        print(f"[INFO]: config: {report['config_path']}")
        print(f"[INFO]: camera: enabled={cfg.camera.enabled} resolution={cfg.camera.width}x{cfg.camera.height}")
        print(f"[INFO]: lidar: enabled={cfg.lidar.enabled} mount={cfg.lidar.mount_mode}")
        print(f"[INFO]: proposed command: {decision.proposed_command.tuple}")
        print(f"[INFO]: safe command: {decision.command.tuple}")
        print(f"[INFO]: safety health: {decision.status.health.value} message={decision.status.message}")
        if decision.reasons:
            print(f"[WARN]: safety reasons: {list(decision.reasons)}")
        if decision.warnings:
            print(f"[WARN]: safety warnings: {list(decision.warnings)}")
    return 0 if decision.status.health.value in {"ok", "degraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
