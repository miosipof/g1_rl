#!/usr/bin/env python3
"""Run the high-level bridge in dry-run mode.

Default mode uses synthetic observations and is fully hardware-free. RealSense
and Livox/MID-360 sources can be enabled independently for dry-run bridge
validation. Commands are still logged only through ``DryRunCommandSink``; no
robot motion is possible from this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adapters import (
    Command2D,
    ConstantVelocityPolicy,
    DryRunCommandSink,
    LidarFrame,
    LidarDebugWriter,
    LazyYoloDebugDetector,
    Pose2D,
    RGBDDebugWriter,
    Target2D,
    LidarTargetNavPolicyRuntime,
    LidarTargetNavPolicyRuntimeConfig,
    RLResidualTargetNavPolicyRuntime,
    RLResidualTargetNavPolicyRuntimeConfig,
    RealBridge,
    RealBridgeConfig,
    CompositeDryRunObservationProvider,
    RealSenseDryRunObservationProvider,
    RealSenseRGBDAdapter,
    RealSenseRGBDAdapterConfig,
    RobotObservation,
    RobotState,
    load_adapter_config,
    make_run_id,
    timestamped_output_dir,
    write_run_info,
)
from adapters.safety_supervisor import SafetySupervisor, SafetySupervisorConfig
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicyCfg
from adapters.real_livox import LivoxAdapterConfig, NpyLivoxReplayAdapter, UnitreeVibesLivox2Adapter, UnitreeVibesLivox2AdapterConfig


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


@dataclass
class SyntheticObservationProvider:
    """Minimal observation provider used until live real adapters are wired."""

    config: Any
    clock: Any = time.monotonic
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    obstacle_front_m: float | None = None

    def read_observation(self) -> RobotObservation:
        now_s = float(self.clock())
        pose = Pose2D(self.x, self.y, self.yaw)
        robot_state = RobotState(
            timestamp_s=now_s,
            pose=pose,
            mode="synthetic_dryrun",
            battery_percent=100.0,
            is_estopped=False,
            is_faulted=False,
            details={"synthetic": True},
        )

        if self.obstacle_front_m is None:
            points = np.array([[2.0, 1.1, 0.35], [3.0, -1.2, 0.45]], dtype=np.float32)
        else:
            points = np.array([[float(self.obstacle_front_m), 0.0, 0.35]], dtype=np.float32)

        lidar = LidarFrame(
            timestamp_s=now_s,
            points_body_xyz=points,
            source_frame_id=self.config.lidar.source_frame_id,
            T_body_lidar=self.config.lidar.T_body_lidar,
            metadata={"synthetic": True},
        )
        return RobotObservation(
            timestamp_s=now_s,
            pose=pose,
            lidar=lidar,
            robot_state=robot_state,
            metadata={"source": "SyntheticObservationProvider"},
        )


class _TimeoutRealSenseAdapter:
    """Small wrapper so RealSenseDryRunObservationProvider can use a timeout."""

    def __init__(self, adapter: RealSenseRGBDAdapter, *, timeout_ms: int) -> None:
        self.adapter = adapter
        self.timeout_ms = int(timeout_ms)

    def read_frame(self):
        return self.adapter.read_frame(timeout_ms=self.timeout_ms)


def cfg_to_default_grid_cfg():
    # Import lazily so the dry-run script remains lightweight for non-target-nav use.
    from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg

    return OccupancyGridCfg()


def _result_to_dict(result: Any) -> dict[str, Any]:
    decision = result.safety_decision
    command_result = result.command_result
    obs = result.observation
    return {
        "mode": result.mode.value,
        "action": result.action,
        "status_health": result.status.health.value,
        "status_message": result.status.message,
        "proposed_command": None if result.proposed_command is None else list(result.proposed_command.tuple),
        "safe_command": None if decision is None else list(decision.command.tuple),
        "safety_reasons": [] if decision is None else list(decision.reasons),
        "safety_warnings": [] if decision is None else list(decision.warnings),
        "command_accepted": None if command_result is None else command_result.accepted,
        "observation_source": None if obs is None else obs.metadata.get("source"),
        "has_rgbd": None if obs is None else obs.has_rgbd,
        "has_lidar": None if obs is None else obs.has_lidar,
        "has_robot_state": None if obs is None else obs.robot_state is not None,
        "sensor_ages_s": None if obs is None else obs.sensor_ages_s(),
        "rgbd_shape": None if obs is None or obs.rgbd is None else [obs.rgbd.height, obs.rgbd.width, 3],
        "rgbd_live": None if obs is None else obs.metadata.get("rgbd_live", False),
        "lidar_synthetic": None if obs is None else obs.metadata.get("lidar_synthetic", obs.lidar is not None),
        "robot_state_synthetic": None if obs is None else obs.metadata.get("robot_state_synthetic", obs.robot_state is not None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--steps", type=int, default=5, help="Number of bridge iterations.")
    parser.add_argument("--dt", type=float, default=0.10, help="Sleep between iterations.")
    parser.add_argument("--policy", choices=("constant", "target-nav", "rl-residual"), default="constant", help="Policy/runtime used to propose commands.")
    parser.add_argument("--vx", type=float, default=0.05, help="Requested policy vx in m/s for --policy constant.")
    parser.add_argument("--vy", type=float, default=0.0, help="Requested policy vy in m/s for --policy constant.")
    parser.add_argument("--wz", type=float, default=0.0, help="Requested policy wz in rad/s for --policy constant.")
    parser.add_argument("--target-x", type=float, default=2.0, help="Manual world-frame target x for --policy target-nav.")
    parser.add_argument("--target-y", type=float, default=0.0, help="Manual world-frame target y for --policy target-nav.")
    parser.add_argument("--target-nav-replan-every", type=int, default=0, help="Replan cadence for --policy target-nav; 0 disables periodic replanning.")
    parser.add_argument("--target-nav-enable-replanning", action="store_true", help="Enable path conflict/periodic replanning in --policy target-nav.")
    parser.add_argument("--target-nav-inflation", type=float, default=0.35, help="A* inflation radius in meters for --policy target-nav.")
    parser.add_argument("--target-nav-max-wz", type=float, default=None, help="Override target-nav policy max yaw rate before final Unitree command scaling.")
    parser.add_argument("--target-nav-command-mode", choices=("holonomic", "heading_aligned"), default="holonomic", help="Waypoint-to-command controller for --policy target-nav. heading_aligned prefers vx+wz and damps vy.")
    parser.add_argument("--rl-residual-checkpoint", type=Path, default=None, help="Optional Torch checkpoint for --policy rl-residual; omitted means zero residual and deterministic nominal behavior.")
    parser.add_argument("--rl-residual-limit-vx", type=float, default=0.10, help="Max |Δvx| residual for --policy rl-residual.")
    parser.add_argument("--rl-residual-limit-vy", type=float, default=0.10, help="Max |Δvy| residual for --policy rl-residual.")
    parser.add_argument("--rl-residual-limit-wz", type=float, default=0.25, help="Max |Δwz| residual for --policy rl-residual.")
    parser.add_argument("--rl-path-points", type=int, default=8, help="Number of A* path samples used by --policy rl-residual features.")
    parser.add_argument("--heading-lookahead-m", type=float, default=0.60, help="Path lookahead distance used by heading_aligned mode to smooth local grid-cell turns.")
    parser.add_argument("--heading-slow-angle-rad", type=float, default=0.35, help="Bearing error above which heading_aligned starts reducing vx/vy.")
    parser.add_argument("--heading-rotate-in-place-angle-rad", type=float, default=0.80, help="Bearing error above which heading_aligned commands turn-first behavior.")
    parser.add_argument("--heading-rotate-vx-scale", type=float, default=0.0, help="Forward scale during turn-first heading alignment.")
    parser.add_argument("--heading-rotate-lateral-scale", type=float, default=0.0, help="Lateral scale during turn-first heading alignment.")
    parser.add_argument("--heading-lateral-scale", type=float, default=0.20, help="Policy-level lateral damping used by --target-nav-command-mode heading_aligned.")
    parser.add_argument("--heading-min-vx-scale", type=float, default=0.25, help="Minimum forward scale used by heading_aligned mode when bearing error is large.")
    parser.add_argument("--obstacle-front-m", type=float, default=None, help="Optional synthetic front obstacle distance.")
    parser.add_argument(
        "--observation-source",
        choices=("synthetic", "realsense", "livox-npy", "livox-live-vibes", "realsense-livox-npy", "realsense-livox-live-vibes"),
        default="synthetic",
        help="Observation provider. Real-sensor modes remain dry-run and never send robot motion.",
    )
    parser.add_argument("--realsense-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--realsense-timeout-ms", type=int, default=1000, help="RealSense wait_for_frames timeout.")
    parser.add_argument("--no-realsense-spatial-filter", action="store_true", help="Disable RealSense spatial filter.")
    parser.add_argument("--no-realsense-temporal-filter", action="store_true", help="Disable RealSense temporal filter.")
    parser.add_argument("--livox-npy-path", type=Path, default=None, help=".npy with Nx3 or FxNx3 Livox-frame points for replay modes.")
    parser.add_argument(
        "--loop-livox-npy",
        dest="loop_livox_npy",
        action="store_true",
        default=True,
        help="Loop Livox .npy replay if steps exceed frames. Enabled by default for multi-step dry-runs.",
    )
    parser.add_argument(
        "--no-loop-livox-npy",
        dest="loop_livox_npy",
        action="store_false",
        help="Disable Livox .npy looping and fail when replay frames are exhausted.",
    )
    parser.add_argument("--unitree-vibes-dir", default=None, help="Path to unitree_g1_vibes repo for live-vibes LiDAR mode.")
    parser.add_argument("--livox-config", default=None, help="Livox-SDK2 JSON config path for live-vibes LiDAR mode.")
    parser.add_argument("--host-ip", default=None, help="Host IP passed to Livox2 in live-vibes LiDAR mode.")
    parser.add_argument("--livox-frame-time", type=float, default=0.20, help="Livox packet aggregation window.")
    parser.add_argument("--livox-frame-packets", type=int, default=120, help="Livox packet aggregation count.")
    parser.add_argument("--livox-timeout", type=float, default=2.0, help="Timeout waiting for live Livox frames.")
    parser.add_argument("--save-lidar-debug", type=Path, default=None, help="Base directory for LiDAR occupancy/debug artifacts; a timestamped run subdir is added by default.")
    parser.add_argument("--save-rgbd-debug", type=Path, default=None, help="Base directory for RGB-D snapshot/depth/YOLO debug artifacts; a timestamped run subdir is added by default.")
    parser.add_argument("--run-id", default=None, help="Optional run id used for timestamped debug subdirectories.")
    parser.add_argument("--no-timestamp-debug-dirs", action="store_true", help="Write debug artifacts directly into requested directories instead of a run_<timestamp> subdir.")
    parser.add_argument("--rgbd-debug-yolo", action="store_true", help="Run YOLO on RGB-D debug snapshots and save detection overlays/metadata.")
    parser.add_argument("--rgbd-debug-yolo-model", default="yolov8n.pt", help="YOLO model used for --rgbd-debug-yolo.")
    parser.add_argument("--rgbd-debug-yolo-confidence", type=float, default=0.25, help="YOLO confidence threshold for RGB-D debug detection.")
    parser.add_argument("--rgbd-debug-yolo-iou", type=float, default=0.45, help="YOLO IoU threshold for RGB-D debug detection.")
    parser.add_argument("--rgbd-debug-yolo-device", default=None, help="Optional YOLO device for RGB-D debug detection, e.g. cuda:0 or cpu.")
    parser.add_argument("--rgbd-debug-yolo-classes", default="person", help="Comma-separated classes to keep for RGB-D debug YOLO; use 'all' to keep every class.")
    parser.add_argument("--allow-yaw-on-front-stop", action="store_true", help="Allow yaw-only escape commands when front obstacle stop would otherwise zero all motion.")
    parser.add_argument("--front-stop-yaw-escape-min-wz", type=float, default=0.05, help="Minimum proposed |wz| required to preserve yaw during front-obstacle stop.")
    parser.add_argument("--front-stop-yaw-escape-max-wz", type=float, default=0.60, help="Maximum |wz| allowed during front-obstacle yaw escape.")
    parser.add_argument("--front-stop-yaw-escape-max-linear", type=float, default=0.05, help="Maximum proposed linear speed norm allowed for yaw escape during front-obstacle stop.")
    parser.add_argument("--json-lines", action="store_true", help="Print one JSON object per step.")
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.dt < 0.0:
        raise SystemExit("--dt must be non-negative")
    if args.realsense_timeout_ms <= 0:
        raise SystemExit("--realsense-timeout-ms must be > 0")
    if args.livox_timeout <= 0.0:
        raise SystemExit("--livox-timeout must be > 0")
    if args.target_nav_replan_every < 0:
        raise SystemExit("--target-nav-replan-every must be >= 0")
    if args.target_nav_inflation < 0.0:
        raise SystemExit("--target-nav-inflation must be >= 0")
    if args.target_nav_max_wz is not None and args.target_nav_max_wz <= 0.0:
        raise SystemExit("--target-nav-max-wz must be positive when provided")
    if args.rl_residual_limit_vx < 0.0 or args.rl_residual_limit_vy < 0.0 or args.rl_residual_limit_wz < 0.0:
        raise SystemExit("--rl-residual-limit-vx/vy/wz must be non-negative")
    if args.rl_path_points < 1:
        raise SystemExit("--rl-path-points must be >= 1")
    if args.heading_lookahead_m < 0.0:
        raise SystemExit("--heading-lookahead-m must be >= 0")
    if args.heading_slow_angle_rad < 0.0:
        raise SystemExit("--heading-slow-angle-rad must be >= 0")
    if args.heading_rotate_in_place_angle_rad < args.heading_slow_angle_rad:
        raise SystemExit("--heading-rotate-in-place-angle-rad must be >= --heading-slow-angle-rad")
    if args.heading_rotate_vx_scale < 0.0 or args.heading_rotate_lateral_scale < 0.0:
        raise SystemExit("--heading-rotate-vx-scale and --heading-rotate-lateral-scale must be >= 0")
    if not (0.0 <= args.rgbd_debug_yolo_confidence <= 1.0):
        raise SystemExit("--rgbd-debug-yolo-confidence must be in [0, 1]")
    if not (0.0 <= args.rgbd_debug_yolo_iou <= 1.0):
        raise SystemExit("--rgbd-debug-yolo-iou must be in [0, 1]")
    if args.front_stop_yaw_escape_min_wz < 0.0:
        raise SystemExit("--front-stop-yaw-escape-min-wz must be >= 0")
    if args.front_stop_yaw_escape_max_wz <= 0.0:
        raise SystemExit("--front-stop-yaw-escape-max-wz must be > 0")
    if args.front_stop_yaw_escape_max_linear < 0.0:
        raise SystemExit("--front-stop-yaw-escape-max-linear must be >= 0")

    cfg = load_adapter_config(args.config)
    realsense_adapter: RealSenseRGBDAdapter | None = None
    livox_adapter: Any | None = None

    uses_realsense = args.observation_source in {"realsense", "realsense-livox-npy", "realsense-livox-live-vibes"}
    uses_livox_npy = args.observation_source in {"livox-npy", "realsense-livox-npy"}
    uses_livox_live = args.observation_source in {"livox-live-vibes", "realsense-livox-live-vibes"}

    rgbd_source = None
    if uses_realsense:
        rs_cfg = RealSenseRGBDAdapterConfig.from_camera_config(
            cfg.camera,
            serial_number=args.realsense_serial,
            use_spatial_filter=not args.no_realsense_spatial_filter,
            use_temporal_filter=not args.no_realsense_temporal_filter,
        )
        realsense_adapter = RealSenseRGBDAdapter(rs_cfg)
        realsense_adapter.start()
        rgbd_source = _TimeoutRealSenseAdapter(realsense_adapter, timeout_ms=args.realsense_timeout_ms)

    lidar_source = None
    if uses_livox_npy:
        if args.livox_npy_path is None:
            raise SystemExit("--livox-npy-path is required for livox-npy modes")
        lidar_source = NpyLivoxReplayAdapter(
            args.livox_npy_path,
            LivoxAdapterConfig.from_lidar_config(cfg.lidar),
            loop=args.loop_livox_npy,
        )
    elif uses_livox_live:
        if not args.livox_config:
            raise SystemExit("--livox-config is required for live-vibes LiDAR modes")
        if not args.host_ip:
            raise SystemExit("--host-ip is required for live-vibes LiDAR modes")
        livox_adapter = UnitreeVibesLivox2Adapter(
            UnitreeVibesLivox2AdapterConfig(
                livox_config_path=args.livox_config,
                host_ip=args.host_ip,
                adapter_config=LivoxAdapterConfig.from_lidar_config(cfg.lidar),
                unitree_vibes_dir=args.unitree_vibes_dir,
                frame_time_s=args.livox_frame_time,
                frame_packets=args.livox_frame_packets,
                read_timeout_s=args.livox_timeout,
            )
        )
        livox_adapter.start()
        lidar_source = livox_adapter

    if args.observation_source == "synthetic":
        provider = SyntheticObservationProvider(config=cfg, obstacle_front_m=args.obstacle_front_m)
    elif args.observation_source == "realsense":
        provider = RealSenseDryRunObservationProvider(rgbd_adapter=rgbd_source, config=cfg)
    else:
        provider = CompositeDryRunObservationProvider(
            config=cfg,
            rgbd_adapter=rgbd_source,
            lidar_adapter=lidar_source,
            include_synthetic_lidar_if_missing=False,
        )

    if args.policy == "constant":
        policy = ConstantVelocityPolicy(args.vx, args.vy, args.wz)
    elif args.policy == "target-nav":
        policy_cfg = LidarTargetNavPolicyRuntimeConfig.from_runtime_config(
            cfg,
            target=Target2D(float(args.target_x), float(args.target_y)),
            enable_replanning=bool(args.target_nav_enable_replanning or args.target_nav_replan_every > 0),
            replan_every=int(args.target_nav_replan_every),
            inflation_radius_m=float(args.target_nav_inflation),
            max_wz=args.target_nav_max_wz,
            command_mode=args.target_nav_command_mode,
            heading_lateral_scale=float(args.heading_lateral_scale),
            heading_min_vx_scale=float(args.heading_min_vx_scale),
            heading_slow_angle_rad=float(args.heading_slow_angle_rad),
            heading_rotate_in_place_angle_rad=float(args.heading_rotate_in_place_angle_rad),
            heading_rotate_vx_scale=float(args.heading_rotate_vx_scale),
            heading_rotate_lateral_scale=float(args.heading_rotate_lateral_scale),
            path_lookahead_m=float(args.heading_lookahead_m),
        )
        policy = LidarTargetNavPolicyRuntime(policy_cfg)
    else:
        residual_cfg = RLResidualPolicyCfg(
            path_points=int(args.rl_path_points),
            residual_limit_vx=float(args.rl_residual_limit_vx),
            residual_limit_vy=float(args.rl_residual_limit_vy),
            residual_limit_wz=float(args.rl_residual_limit_wz),
        )
        policy_cfg = RLResidualTargetNavPolicyRuntimeConfig.from_runtime_config(
            cfg,
            target=Target2D(float(args.target_x), float(args.target_y)),
            checkpoint_path=None if args.rl_residual_checkpoint is None else str(args.rl_residual_checkpoint),
            residual_policy_cfg=residual_cfg,
            enable_replanning=bool(args.target_nav_enable_replanning or args.target_nav_replan_every > 0),
            replan_every=int(args.target_nav_replan_every),
            inflation_radius_m=float(args.target_nav_inflation),
            max_wz=args.target_nav_max_wz,
            command_mode=args.target_nav_command_mode,
        )
        policy = RLResidualTargetNavPolicyRuntime(policy_cfg)

    supervisor = SafetySupervisor.from_runtime_config(
        cfg,
        config=SafetySupervisorConfig(
            allow_yaw_on_front_stop=bool(args.allow_yaw_on_front_stop),
            front_stop_yaw_escape_min_wz=float(args.front_stop_yaw_escape_min_wz),
            front_stop_yaw_escape_max_wz=float(args.front_stop_yaw_escape_max_wz),
            front_stop_yaw_escape_max_linear_mps=float(args.front_stop_yaw_escape_max_linear),
        ),
    )
    sink = DryRunCommandSink(name="dryrun_bridge_sink")
    bridge = RealBridge(
        observation_provider=provider,
        policy_runtime=policy,
        safety_supervisor=supervisor,
        command_sink=sink,
        config=RealBridgeConfig(name="real_g1_follow_dryrun", send_commands=True),
    )

    run_id = args.run_id or make_run_id()
    timestamp_debug_dirs = not bool(args.no_timestamp_debug_dirs)
    lidar_debug_dir = timestamped_output_dir(args.save_lidar_debug, run_id=run_id, enabled=timestamp_debug_dirs)
    rgbd_debug_dir = timestamped_output_dir(args.save_rgbd_debug, run_id=run_id, enabled=timestamp_debug_dirs)

    lidar_debug_writer = None
    if lidar_debug_dir is not None:
        debug_grid_cfg = policy.config.grid_cfg if hasattr(policy, "config") and hasattr(policy.config, "grid_cfg") else cfg_to_default_grid_cfg()
        lidar_debug_writer = LidarDebugWriter(lidar_debug_dir, grid_cfg=debug_grid_cfg)

    rgbd_debug_writer = RGBDDebugWriter(rgbd_debug_dir) if rgbd_debug_dir is not None else None
    run_info = {
        "run_id": run_id,
        "timestamped_debug_dirs": timestamp_debug_dirs,
        "script": "real_g1_follow_dryrun.py",
        "config": str(args.config),
        "observation_source": str(args.observation_source),
        "policy": str(args.policy),
        "body_exclusion": {
            "camera": None if cfg.camera.body_exclusion_box is None else cfg.camera.body_exclusion_box.to_dict(),
            "lidar": None if cfg.lidar.body_exclusion_box is None else cfg.lidar.body_exclusion_box.to_dict(),
        },
    }
    write_run_info(lidar_debug_dir, run_info)
    write_run_info(rgbd_debug_dir, run_info)
    rgbd_debug_detector = None
    if args.rgbd_debug_yolo:
        allowed_classes = None if str(args.rgbd_debug_yolo_classes).strip().lower() == "all" else [
            item.strip() for item in str(args.rgbd_debug_yolo_classes).split(",") if item.strip()
        ]
        rgbd_debug_detector = LazyYoloDebugDetector(
            model=args.rgbd_debug_yolo_model,
            confidence=float(args.rgbd_debug_yolo_confidence),
            iou=float(args.rgbd_debug_yolo_iou),
            device=args.rgbd_debug_yolo_device,
            allowed_classes=allowed_classes,
        )

    try:
        status = bridge.follow()
        if not args.json_lines:
            print(f"[INFO]: bridge command: follow -> {status.health.value}: {status.message}")
            print(f"[INFO]: observation_source={args.observation_source}")
            print(f"[INFO]: policy={args.policy}")
            if args.policy in {"target-nav", "rl-residual"}:
                print(
                    f"[INFO]: target-nav target=({args.target_x:.3f}, {args.target_y:.3f}) "
                    f"replan_every={args.target_nav_replan_every} inflation={args.target_nav_inflation:.3f} "
                    f"command_mode={args.target_nav_command_mode} heading_lateral_scale={args.heading_lateral_scale:.3f} "
                    f"heading_min_vx_scale={args.heading_min_vx_scale:.3f}"
                )
            if uses_realsense and not (uses_livox_npy or uses_livox_live):
                print("[INFO]: using live RealSense RGB-D; LiDAR and robot state remain synthetic placeholders")
            elif (uses_livox_npy or uses_livox_live) and not uses_realsense:
                print("[INFO]: using Livox/MID-360 LiDAR source; robot state remains synthetic placeholder")
            elif uses_realsense and (uses_livox_npy or uses_livox_live):
                print("[INFO]: using live RealSense RGB-D and Livox/MID-360 LiDAR source; robot state remains synthetic placeholder")
            if uses_livox_npy:
                print(f"[INFO]: Livox .npy replay loop={args.loop_livox_npy}")
            if lidar_debug_writer is not None:
                print(f"[INFO]: run_id={run_id}")
                print(f"[INFO]: LiDAR debug artifacts: {lidar_debug_writer.output_dir}")
            if rgbd_debug_writer is not None:
                if lidar_debug_writer is None:
                    print(f"[INFO]: run_id={run_id}")
                print(f"[INFO]: RGB-D debug artifacts: {rgbd_debug_writer.output_dir}")
            if rgbd_debug_detector is not None:
                print(f"[INFO]: RGB-D debug YOLO enabled model={args.rgbd_debug_yolo_model!r} classes={args.rgbd_debug_yolo_classes!r}")

        exit_code = 0
        for i in range(args.steps):
            result = bridge.step()
            debug_paths = None
            if lidar_debug_writer is not None:
                debug_paths = lidar_debug_writer.write_step(i, result, policy=policy if hasattr(policy, "last_debug") else None)
            rgbd_debug_paths = None
            if rgbd_debug_writer is not None:
                rgbd_debug_paths = rgbd_debug_writer.write_step(i, result, detector=rgbd_debug_detector)
            row = {"step": i, **_result_to_dict(result)}
            if debug_paths:
                row["lidar_debug"] = debug_paths
            if rgbd_debug_paths:
                row["rgbd_debug"] = rgbd_debug_paths
            if args.json_lines:
                print(json.dumps(row, sort_keys=True))
            else:
                print(
                    f"[INFO]: step={i} mode={row['mode']} action={row['action']} "
                    f"health={row['status_health']} proposed={row['proposed_command']} safe={row['safe_command']} "
                    f"accepted={row['command_accepted']} has_rgbd={row['has_rgbd']} "
                    f"reasons={row['safety_reasons']} warnings={row['safety_warnings']}"
                )
            if result.status.health.value == "fault":
                exit_code = 1
            if args.dt > 0.0 and i + 1 < args.steps:
                time.sleep(args.dt)

        stop_status = bridge.stop()
        if not args.json_lines:
            print(f"[INFO]: bridge command: stop -> {stop_status.health.value}: {stop_status.message}")
            print(f"[INFO]: dry-run commands logged: {len(sink.sent_commands)} stop_count={sink.stop_count}")
        return exit_code
    finally:
        if realsense_adapter is not None:
            realsense_adapter.stop()
        if livox_adapter is not None and hasattr(livox_adapter, "stop"):
            livox_adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
