#!/usr/bin/env python3
"""Synchronized stepwise detect-and-follow runner for real Unitree G1 tests.

This script is the real-robot counterpart of the Isaac stepwise smoke tests.  It
keeps motion disabled by default.  In dry-run mode it still reads live sensors and
prints/logs the commands that would be sent.

Pipeline, inline YOLO mode::

    RealSense RGB-D + YOLO + depth projection
      + Livox/MID-360 LiDAR snapshot
      -> RobotObservation(target_hint, lidar)
      -> StepwiseTargetNavPolicyRuntime
      -> bounded Unitree Move command for action.duration_s
      -> hard stop
      -> repeat

Pipeline, streaming YOLO mode::

    external real_g1_yolo_track_stream.py latest_tracks.json
      + Livox/MID-360 LiDAR snapshot
      -> PrimaryTargetLock
      -> RobotObservation(target_hint, lidar)
      -> StepwiseTargetNavPolicyRuntime
      -> bounded Unitree Move command for action.duration_s
      -> hard stop
      -> repeat

The stepwise controller has no command odometry and does not use the continuous
SafetySupervisor.  It relies on A* obstacle inflation plus immediate forward
sweep/front guards.  The script stops before each observation cycle and after each
executed action so observations and commands remain synchronized.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from adapters import (
    Command2D,
    DetectionStreamClient,
    LidarFrame,
    PrimaryTargetLock,
    PrimaryTargetLockConfig,
    RealSenseRGBDAdapter,
    RealSenseRGBDAdapterConfig,
    RobotObservation,
    StepwiseTargetNavPolicyRuntime,
    StepwiseTargetNavPolicyRuntimeConfig,
    SyntheticSupportConfig,
    Target2D,
    UnitreeLocoAdapter,
    UnitreeLocoAdapterConfig,
    make_run_id,
    timestamped_output_dir,
    write_run_info,
    load_adapter_config,
)
from adapters.real_livox import LivoxAdapterConfig, UnitreeVibesLivox2Adapter, UnitreeVibesLivox2AdapterConfig
from adapters.real_observation import CompositeDryRunObservationProvider
from adapters.lidar_debug import save_occupancy_debug_image
from adapters.rgbd_debug import RGBDDebugWriter
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.stepwise import StepwiseNavConfig
from g1_nav_rl.planning.grid_astar import GridAStarCfg

# Reuse the mature inline RGB-D/YOLO target provider from the existing real script.
from real_g1_follow_detect_live import DetectTargetObservationProvider, _json_default  # noqa: E402
from real_g1_follow_dryrun import _TimeoutRealSenseAdapter  # noqa: E402


class StreamTargetObservationProvider:
    """Inject target hints from a detection-stream JSON file into observations."""

    def __init__(
        self,
        *,
        base_provider: Any,
        stream_client: DetectionStreamClient,
        target_lock: PrimaryTargetLock,
        source_name: str = "StreamTargetObservationProvider",
    ) -> None:
        self.base_provider = base_provider
        self.stream_client = stream_client
        self.target_lock = target_lock
        self.source_name = source_name
        self.last_read_result: Any | None = None
        self.last_target_state: Any | None = None

    def read_observation(self) -> RobotObservation:
        obs: RobotObservation = self.base_provider.read_observation()
        read_result = self.stream_client.read()
        target_state = self.target_lock.update_read_result(read_result, allow_stale=False)
        self.last_read_result = read_result
        self.last_target_state = target_state
        target = target_state.target
        meta = dict(obs.metadata)
        meta["source"] = self.source_name
        meta["target_lock"] = target_state.to_dict()
        meta["detection_stream_status"] = {
            "health": read_result.status.health.value,
            "message": read_result.status.message,
            "timestamp_s": float(read_result.status.timestamp_s),
            "details": dict(read_result.status.details),
        }
        if read_result.frame is not None:
            meta["detection_stream_frame"] = {
                "timestamp_s": float(read_result.frame.timestamp_s),
                "frame_id": read_result.frame.frame_id,
                "source": read_result.frame.source,
                "track_count": len(read_result.frame.tracks),
            }
        return RobotObservation(
            timestamp_s=obs.timestamp_s,
            pose=obs.pose,
            rgbd=obs.rgbd,
            lidar=obs.lidar,
            robot_state=obs.robot_state,
            target_hint=target,
            statuses=obs.statuses,
            metadata=meta,
        )


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        return asdict(value)
    except Exception:
        return str(value)


def _xy_from_target(target: Target2D | None) -> list[float] | None:
    if target is None:
        return None
    return [float(target.x), float(target.y)]


def _command_dict(cmd: Command2D | None) -> dict[str, Any] | None:
    if cmd is None:
        return None
    return {
        "vx": float(cmd.vx),
        "vy": float(cmd.vy),
        "wz": float(cmd.wz),
        "duration_s": None if cmd.duration_s is None else float(cmd.duration_s),
        "source": cmd.source,
        "metadata": _json_safe(dict(cmd.metadata)),
    }


def _make_grid_cfg(args: argparse.Namespace) -> OccupancyGridCfg:
    return OccupancyGridCfg(
        x_min=float(args.grid_x_min),
        x_max=float(args.grid_x_max),
        y_min=float(args.grid_y_min),
        y_max=float(args.grid_y_max),
        resolution=float(args.grid_resolution),
    )


def _make_stepwise_cfg(args: argparse.Namespace, grid_cfg: OccupancyGridCfg) -> StepwiseNavConfig:
    if args.astar_inflation_m is not None:
        inflation = float(args.astar_inflation_m)
    else:
        inflation = float(args.body_radius_m) + float(args.safety_margin_m)
    return StepwiseNavConfig(
        grid_cfg=grid_cfg,
        astar_cfg=GridAStarCfg(
            inflation_radius=inflation,
            occupancy_threshold=float(args.occupancy_threshold),
            allow_diagonal=not bool(args.no_diagonal_astar),
            heuristic_weight=float(args.astar_heuristic_weight),
            max_expansions=int(args.astar_max_expansions),
        ),
        forward_segment_m=float(args.forward_segment_m),
        forward_speed_mps=float(args.forward_speed_mps),
        turn_speed_radps=float(args.turn_speed_radps),
        turn_angle_max_rad=math.radians(float(args.turn_angle_max_deg)),
        min_turn_angle_rad=math.radians(float(args.min_turn_angle_deg)),
        min_forward_m=float(args.min_forward_m),
        replan_every_actions=int(args.replan_every_actions),
        replan_every_forward_m=float(args.replan_every_forward_m),
        target_change_replan_m=float(args.target_change_replan_m),
        path_block_check=not bool(args.disable_path_block_check),
        sweep_start_ignore_m=float(args.sweep_start_ignore_m),
        emergency_front_stop_m=float(args.emergency_front_stop_m),
        success_radius_m=float(args.target_reached_radius_m),
        allow_goal_relocation=not bool(args.disable_goal_relocation),
        allow_start_relocation=not bool(args.disable_start_relocation),
        source="real_g1_follow_stepwise",
    )


def _build_base_provider(
    *,
    args: argparse.Namespace,
    cfg: Any,
    rs_adapter: RealSenseRGBDAdapter | None,
    livox_adapter: UnitreeVibesLivox2Adapter,
) -> CompositeDryRunObservationProvider:
    return CompositeDryRunObservationProvider(
        config=cfg,
        rgbd_adapter=None if rs_adapter is None else _TimeoutRealSenseAdapter(rs_adapter, timeout_ms=args.realsense_timeout_ms),
        lidar_adapter=livox_adapter,
        support=SyntheticSupportConfig(mode="real_g1_stepwise"),
        include_synthetic_lidar_if_missing=False,
        include_synthetic_robot_state=True,
        source_name="RealG1StepwiseBaseObservationProvider",
    )


def _execute_for_duration(
    *,
    sink: UnitreeLocoAdapter,
    command: Command2D,
    real_motion: bool,
    command_period_s: float,
    require_state_gate: bool,
    json_lines: bool,
) -> list[dict[str, Any]]:
    """Send/log a command for its requested duration, periodically refreshing Move()."""

    duration = float(command.duration_s or 0.0)
    if command.is_stop or duration <= 0.0:
        result = sink.stop(timestamp_s=time.monotonic())
        return [{"phase": "stop", "accepted": result.accepted, "status": _json_safe(result.status)}]

    deadline = time.monotonic() + duration
    sends: list[dict[str, Any]] = []
    # In dry-run mode, one send is enough to preserve fast offline iteration while
    # still logging the intended duration.  Real motion refreshes the command.
    first = True
    while first or (real_motion and time.monotonic() < deadline):
        first = False
        result = sink.send_command(command, require_state_gate=require_state_gate)
        sends.append(
            {
                "phase": "send",
                "accepted": bool(result.accepted),
                "status": _json_safe(result.status),
                "sent_command": _command_dict(result.command),
            }
        )
        if not result.accepted:
            break
        if not real_motion:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(float(command_period_s), remaining))
    return sends


def _write_step_debug(
    *,
    step: int,
    obs: RobotObservation,
    command: Command2D,
    stepwise_runtime: StepwiseTargetNavPolicyRuntime,
    lidar_debug_dir: Path | None,
    rgbd_writer: RGBDDebugWriter | None,
    detections: list[Any] | None,
) -> dict[str, Any]:
    paths: dict[str, str] = {}
    decision = stepwise_runtime.last_decision
    if lidar_debug_dir is not None and obs.lidar is not None:
        from adapters.g1_policy_runtime import occupancy_grid_from_observation

        grid = occupancy_grid_from_observation(
            obs,
            grid_cfg=stepwise_runtime.config.stepwise_cfg.grid_cfg,
            require_lidar=True,
        )[0]
        plan_xy_b = []
        if decision is not None and decision.plan is not None:
            plan_xy_b = decision.plan.path_xy_b
        target_xy_b = None if obs.target_hint is None else obs.target_hint.xy
        step_tag = f"{int(step):06d}"
        lidar_debug_dir.mkdir(parents=True, exist_ok=True)
        npy_path = lidar_debug_dir / f"lidar_points_body_step_{step_tag}.npy"
        np.save(npy_path, obs.lidar.points_body_xyz.astype(np.float32, copy=False))
        img_path = lidar_debug_dir / f"occupancy_grid_step_{step_tag}.png"
        save_occupancy_debug_image(
            img_path,
            grid,
            grid_cfg=stepwise_runtime.config.stepwise_cfg.grid_cfg,
            plan_xy_b=plan_xy_b,
            target_xy_b=target_xy_b,
        )
        meta = {
            "step": int(step),
            "timestamp_s": float(obs.timestamp_s),
            "lidar_point_count": int(obs.lidar.point_count),
            "target_xy_b": None if target_xy_b is None else [float(v) for v in target_xy_b],
            "occupied_cells": int((grid > 0.0).sum().item()),
            "stepwise_decision": None if decision is None else decision.to_dict(),
        }
        meta_path = lidar_debug_dir / f"occupancy_grid_meta_step_{step_tag}.json"
        meta_path.write_text(json.dumps(_json_safe(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.update({"lidar_points": str(npy_path), "occupancy_grid": str(img_path), "occupancy_meta": str(meta_path)})

    if rgbd_writer is not None and obs.rgbd is not None:
        artifacts = rgbd_writer.write_frame(
            step=step,
            frame=obs.rgbd,
            detections=[] if detections is None else detections,
            source=obs.metadata.get("source"),
            extra_metadata={
                "target_tracking": obs.metadata.get("target_tracking"),
                "target_lock": obs.metadata.get("target_lock"),
                "stepwise_decision": None if decision is None else decision.to_dict(),
            },
        )
        paths["rgbd"] = artifacts.to_dict()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--iface", default="eth0", help="Network interface for Unitree SDK ChannelFactoryInitialize.")
    parser.add_argument("--unitree-vibes-dir", required=True, help="Path to unitree_g1_vibes repo for live Livox mode.")
    parser.add_argument("--livox-config", required=True, help="Livox-SDK2 JSON config path.")
    parser.add_argument("--host-ip", default="192.168.123.222", help="Host IP passed to Livox2.")
    parser.add_argument("--livox-frame-time", type=float, default=0.20, help="Livox aggregation window.")
    parser.add_argument("--livox-frame-packets", type=int, default=120, help="Livox aggregation packet count.")
    parser.add_argument("--livox-timeout", type=float, default=2.0, help="Timeout waiting for live Livox frames.")
    parser.add_argument("--target-source", choices=("yolo", "stream"), default="yolo", help="Inline RGB-D YOLO or external latest_tracks.json stream.")
    parser.add_argument("--target-class", default="person", help="Target class to follow.")
    parser.add_argument("--stand-off-m", type=float, default=0.80, help="Standoff from detected object; inline YOLO applies this before planning.")
    parser.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=30, help="Maximum stop-observe-plan-execute cycles.")
    parser.add_argument("--settle-s", type=float, default=0.50, help="Sleep after stop before observation.")
    parser.add_argument("--stop-after-action-s", type=float, default=0.05, help="Short pause after post-action stop.")

    # RealSense / inline YOLO.
    parser.add_argument("--realsense-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--realsense-timeout-ms", type=int, default=1000, help="RealSense wait_for_frames timeout.")
    parser.add_argument("--no-realsense-spatial-filter", action="store_true", help="Disable RealSense spatial filter.")
    parser.add_argument("--no-realsense-temporal-filter", action="store_true", help="Disable RealSense temporal filter.")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name for inline mode.")
    parser.add_argument("--yolo-confidence", type=float, default=0.10, help="YOLO confidence threshold for inline mode.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO IoU threshold for inline mode.")
    parser.add_argument("--yolo-device", default="cpu", help="YOLO device for inline mode.")
    parser.add_argument("--redetect-every", type=int, default=1, help="Run inline YOLO every N cycles while tracking.")
    parser.add_argument("--occluded-redetect-every", type=int, default=1, help="Run inline YOLO every N cycles while occluded_recent.")
    parser.add_argument("--lost-redetect-every", type=int, default=1, help="Run inline YOLO every N cycles while lost.")
    parser.add_argument("--occlusion-grace-steps", type=int, default=5, help="Keep last inline target for this many missed cycles.")
    parser.add_argument("--target-smoothing-alpha", type=float, default=1.0, help="Inline target smoothing alpha.")
    parser.add_argument("--target-max-jump-m", type=float, default=1.25, help="Reject inline target jumps larger than this distance.")
    parser.add_argument("--target-depth-min-m", type=float, default=0.20, help="Minimum valid inline target depth.")
    parser.add_argument("--target-depth-max-m", type=float, default=8.0, help="Maximum valid inline target depth.")
    parser.add_argument("--depth-crop-frac", type=float, default=0.50, help="Central bbox crop fraction for inline depth extraction.")
    parser.add_argument("--depth-percentile", type=float, default=0.50, help="Depth percentile inside inline bbox crop.")
    parser.add_argument("--use-config-camera-transform", action="store_true", help="Use config T_body_camera exactly instead of forward-camera convention.")

    # Stream target lock.
    parser.add_argument("--detection-stream-path", type=Path, default=None, help="latest_tracks.json path for --target-source stream.")
    parser.add_argument("--detection-stream-max-age-s", type=float, default=1.0, help="Maximum detection stream age.")
    parser.add_argument("--target-lock-mode", choices=("primary_track", "none"), default="primary_track", help="Target identity policy for stream mode.")
    parser.add_argument("--target-lock-min-confidence", type=float, default=0.10)
    parser.add_argument("--target-lock-min-depth-valid-ratio", type=float, default=0.0)
    parser.add_argument("--target-lock-max-lost-s", type=float, default=5.0)
    parser.add_argument("--target-lock-max-jump-m", type=float, default=1.0)
    parser.add_argument("--target-lock-allow-track-switch", action="store_true")
    parser.add_argument("--target-lock-no-require-track-id", action="store_true")

    # Stepwise nav/planning.
    parser.add_argument("--grid-x-min", type=float, default=-2.0)
    parser.add_argument("--grid-x-max", type=float, default=6.0)
    parser.add_argument("--grid-y-min", type=float, default=-4.0)
    parser.add_argument("--grid-y-max", type=float, default=4.0)
    parser.add_argument("--grid-resolution", type=float, default=0.05)
    parser.add_argument("--body-radius-m", type=float, default=0.34, help="Robot footprint radius used when deriving A* inflation.")
    parser.add_argument("--safety-margin-m", type=float, default=0.35, help="Extra clearance used when deriving A* inflation.")
    parser.add_argument("--astar-inflation-m", type=float, default=None, help="Override A* inflation; otherwise body_radius+safety_margin.")
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--no-diagonal-astar", action="store_true")
    parser.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    parser.add_argument("--astar-max-expansions", type=int, default=0)
    parser.add_argument("--forward-segment-m", type=float, default=0.35)
    parser.add_argument("--forward-speed-mps", type=float, default=0.25)
    parser.add_argument("--turn-speed-radps", type=float, default=0.35)
    parser.add_argument("--turn-angle-max-deg", type=float, default=35.0)
    parser.add_argument("--min-turn-angle-deg", type=float, default=3.0)
    parser.add_argument("--min-forward-m", type=float, default=0.02)
    parser.add_argument("--replan-every-actions", type=int, default=0)
    parser.add_argument("--replan-every-forward-m", type=float, default=0.60)
    parser.add_argument("--target-change-replan-m", type=float, default=0.50)
    parser.add_argument("--disable-path-block-check", action="store_true")
    parser.add_argument("--sweep-start-ignore-m", type=float, default=0.05)
    parser.add_argument("--emergency-front-stop-m", type=float, default=0.40)
    parser.add_argument("--target-reached-radius-m", type=float, default=0.75)
    parser.add_argument("--disable-goal-relocation", action="store_true")
    parser.add_argument("--disable-start-relocation", action="store_true")
    parser.add_argument("--front-x-min-m", type=float, default=0.0)
    parser.add_argument("--front-x-max-m", type=float, default=1.0)
    parser.add_argument("--front-y-abs-max-m", type=float, default=0.35)
    parser.add_argument("--front-z-min-m", type=float, default=None)
    parser.add_argument("--front-z-max-m", type=float, default=None)

    # Command / safety gate.
    parser.add_argument("--expected-fsm-id", type=int, default=200)
    parser.add_argument("--allow-non-200", action="store_true")
    parser.add_argument("--state-samples", type=int, default=3)
    parser.add_argument("--state-interval-s", type=float, default=0.05)
    parser.add_argument("--command-vx-scale", type=float, default=1.0)
    parser.add_argument("--command-vy-scale", type=float, default=0.0, help="Default disables lateral motion on the real G1.")
    parser.add_argument("--command-wz-scale", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.10, help="Refresh period while executing a real Move command.")
    parser.add_argument("--yes-i-understand-this-moves-the-robot", action="store_true")
    parser.add_argument("--allow-synthetic-robot-state-motion", action="store_true", help="Required for real motion until real RobotState provider is implemented.")

    # Output.
    parser.add_argument("--save-lidar-debug", type=Path, default=None)
    parser.add_argument("--save-rgbd-debug", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-timestamp-debug-dirs", action="store_true")
    parser.add_argument("--json-lines", action="store_true")
    args = parser.parse_args()

    if args.max_steps < 1:
        raise SystemExit("--max-steps/--steps must be >= 1")
    if args.settle_s < 0.0 or args.stop_after_action_s < 0.0:
        raise SystemExit("--settle-s and --stop-after-action-s must be non-negative")
    if args.command_period_s <= 0.0:
        raise SystemExit("--command-period-s must be positive")
    if args.target_source == "stream" and args.detection_stream_path is None:
        raise SystemExit("--target-source stream requires --detection-stream-path")
    real_motion = bool(args.yes_i_understand_this_moves_the_robot)
    if real_motion and not args.allow_synthetic_robot_state_motion:
        raise SystemExit(
            "Real motion currently uses synthetic RobotState in the observation provider. "
            "Pass --allow-synthetic-robot-state-motion after external verification."
        )

    cfg = load_adapter_config(args.config)
    run_id = args.run_id or make_run_id()
    timestamp_debug_dirs = not bool(args.no_timestamp_debug_dirs)
    lidar_debug_dir = timestamped_output_dir(args.save_lidar_debug, run_id=run_id, enabled=timestamp_debug_dirs)
    rgbd_debug_dir = timestamped_output_dir(args.save_rgbd_debug, run_id=run_id, enabled=timestamp_debug_dirs)
    rgbd_writer = RGBDDebugWriter(rgbd_debug_dir) if rgbd_debug_dir else None

    grid_cfg = _make_grid_cfg(args)
    stepwise_cfg = _make_stepwise_cfg(args, grid_cfg)
    stepwise_runtime = StepwiseTargetNavPolicyRuntime(
        StepwiseTargetNavPolicyRuntimeConfig(
            stepwise_cfg=stepwise_cfg,
            require_lidar=True,
            front_x_min_m=float(args.front_x_min_m),
            front_x_max_m=float(args.front_x_max_m),
            front_y_abs_max_m=float(args.front_y_abs_max_m),
            front_z_min_m=args.front_z_min_m,
            front_z_max_m=args.front_z_max_m,
            source="real_g1_follow_stepwise",
        )
    )

    rs_adapter: RealSenseRGBDAdapter | None = None
    if args.target_source == "yolo" or rgbd_writer is not None:
        rs_adapter = RealSenseRGBDAdapter(
            RealSenseRGBDAdapterConfig.from_camera_config(
                cfg.camera,
                serial_number=args.realsense_serial,
                use_spatial_filter=not bool(args.no_realsense_spatial_filter),
                use_temporal_filter=not bool(args.no_realsense_temporal_filter),
            )
        )
        rs_adapter.start()

    livox_adapter = UnitreeVibesLivox2Adapter(
        UnitreeVibesLivox2AdapterConfig(
            unitree_vibes_dir=str(args.unitree_vibes_dir),
            livox_config_path=str(args.livox_config),
            host_ip=str(args.host_ip),
            frame_time_s=float(args.livox_frame_time),
            frame_packets=int(args.livox_frame_packets),
            read_timeout_s=float(args.livox_timeout),
            adapter_config=LivoxAdapterConfig.from_lidar_config(cfg.lidar),
        )
    )
    livox_adapter.start()

    base_provider = _build_base_provider(args=args, cfg=cfg, rs_adapter=rs_adapter, livox_adapter=livox_adapter)
    if args.target_source == "yolo":
        observation_provider: Any = DetectTargetObservationProvider(
            base_provider=base_provider,
            yolo_model=args.yolo_model,
            yolo_confidence=float(args.yolo_confidence),
            yolo_iou=float(args.yolo_iou),
            yolo_device=args.yolo_device,
            target_class=args.target_class,
            redetect_every_steps=int(args.redetect_every),
            occluded_redetect_every_steps=int(args.occluded_redetect_every),
            lost_redetect_every_steps=int(args.lost_redetect_every),
            occlusion_grace_steps=int(args.occlusion_grace_steps),
            smoothing_alpha=float(args.target_smoothing_alpha),
            max_jump_m=float(args.target_max_jump_m),
            stand_off_m=float(args.stand_off_m),
            depth_min_m=float(args.target_depth_min_m),
            depth_max_m=float(args.target_depth_max_m),
            depth_crop_frac=float(args.depth_crop_frac),
            depth_percentile=float(args.depth_percentile),
            use_forward_camera_convention=not bool(args.use_config_camera_transform),
            rgbd_body_exclusion_box=cfg.camera.body_exclusion_box,
        )
    else:
        target_lock = PrimaryTargetLock(
            PrimaryTargetLockConfig(
                target_class=str(args.target_class),
                min_confidence=float(args.target_lock_min_confidence),
                min_depth_valid_ratio=float(args.target_lock_min_depth_valid_ratio),
                max_stream_age_s=float(args.detection_stream_max_age_s),
                max_lost_s=float(args.target_lock_max_lost_s),
                max_target_jump_m=float(args.target_lock_max_jump_m),
                require_track_id=not bool(args.target_lock_no_require_track_id),
                allow_track_switch=bool(args.target_lock_allow_track_switch),
            )
        )
        observation_provider = StreamTargetObservationProvider(
            base_provider=base_provider,
            stream_client=DetectionStreamClient(
                path=args.detection_stream_path,
                max_age_s=float(args.detection_stream_max_age_s),
                missing_ok=True,
            ),
            target_lock=target_lock,
        )

    require_fsm_id = None if args.allow_non_200 else int(args.expected_fsm_id)
    sink = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(
            iface=args.iface,
            dry_run=not real_motion,
            require_arm=True,
            require_fsm_id=require_fsm_id,
            command_period_s=float(args.command_period_s),
            command_vx_scale=float(args.command_vx_scale),
            command_vy_scale=float(args.command_vy_scale),
            command_wz_scale=float(args.command_wz_scale),
            state_samples=int(args.state_samples),
            state_interval_s=float(args.state_interval_s),
        ),
        command_limits=cfg.command_limits,
    )
    sink.connect()
    sink.arm()

    run_info = {
        "run_id": run_id,
        "script": "real_g1_follow_stepwise.py",
        "real_motion": real_motion,
        "target_source": args.target_source,
        "target_class": args.target_class,
        "stepwise_cfg": stepwise_cfg,
        "unitree": {
            "iface": args.iface,
            "require_fsm_id": require_fsm_id,
            "command_vx_scale": args.command_vx_scale,
            "command_vy_scale": args.command_vy_scale,
            "command_wz_scale": args.command_wz_scale,
        },
    }
    write_run_info(lidar_debug_dir, run_info)
    write_run_info(rgbd_debug_dir, run_info)

    print("[INFO]: real stepwise detect-and-follow starting")
    print(f"[INFO]: target_source={args.target_source} target_class={args.target_class}")
    print(f"[INFO]: real_motion={real_motion} run_id={run_id}")
    print(f"[INFO]: stepwise inflation={stepwise_cfg.astar_cfg.inflation_radius:.3f} grid_resolution={grid_cfg.resolution:.3f}")
    print(f"[INFO]: motion: forward_speed={args.forward_speed_mps} turn_speed={args.turn_speed_radps} replan_forward={args.replan_every_forward_m}")
    print(f"[INFO]: command mapping: vx_scale={args.command_vx_scale} vy_scale={args.command_vy_scale} wz_scale={args.command_wz_scale}")
    if not real_motion:
        print("[INFO]: dry-run mode: no Unitree Move command will be sent")
    else:
        print("[WARN]: REAL MOTION ENABLED; operator must be ready to stop the robot")

    exit_code = 0
    command_rows_path = None
    if lidar_debug_dir is not None:
        command_rows_path = lidar_debug_dir / "stepwise_real_trace.jsonl"
        command_rows_path.parent.mkdir(parents=True, exist_ok=True)
        if command_rows_path.exists():
            command_rows_path.unlink()

    try:
        for step in range(int(args.max_steps)):
            stop_before = sink.stop(timestamp_s=time.monotonic())
            if float(args.settle_s) > 0.0:
                time.sleep(float(args.settle_s))

            obs = observation_provider.read_observation()
            command = stepwise_runtime.step(obs)
            decision = stepwise_runtime.last_decision
            detections = None
            if args.target_source == "yolo":
                detections = getattr(observation_provider, "last_detections", None)

            debug_paths = _write_step_debug(
                step=step,
                obs=obs,
                command=command,
                stepwise_runtime=stepwise_runtime,
                lidar_debug_dir=lidar_debug_dir,
                rgbd_writer=rgbd_writer,
                detections=detections,
            )

            send_events: list[dict[str, Any]] = []
            if command.is_stop or command.duration_s is None:
                send_events.append({"phase": "no_motion", "reason": None if decision is None else decision.reason})
            else:
                send_events.extend(
                    _execute_for_duration(
                        sink=sink,
                        command=command,
                        real_motion=real_motion,
                        command_period_s=float(args.command_period_s),
                        require_state_gate=real_motion,
                        json_lines=bool(args.json_lines),
                    )
                )
            stop_after = sink.stop(timestamp_s=time.monotonic())
            if float(args.stop_after_action_s) > 0.0:
                time.sleep(float(args.stop_after_action_s))

            tracking = obs.metadata.get("target_tracking")
            target_lock = obs.metadata.get("target_lock")
            row = {
                "step": int(step),
                "timestamp_s": float(obs.timestamp_s),
                "target_xy_b": _xy_from_target(obs.target_hint),
                "has_lidar": bool(obs.has_lidar),
                "lidar_point_count": None if obs.lidar is None else int(obs.lidar.point_count),
                "sensor_ages_s": obs.sensor_ages_s(),
                "target_tracking": tracking,
                "target_lock": target_lock,
                "stepwise_decision": None if decision is None else decision.to_dict(),
                "command": _command_dict(command),
                "stop_before": {"accepted": stop_before.accepted, "status": _json_safe(stop_before.status)},
                "send_events": send_events,
                "stop_after": {"accepted": stop_after.accepted, "status": _json_safe(stop_after.status)},
                "debug_paths": debug_paths,
            }
            if command_rows_path is not None:
                with command_rows_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            if args.json_lines:
                print(json.dumps(_json_safe(row), sort_keys=True))
            else:
                reason = None if decision is None else decision.reason
                action = None if decision is None else decision.action.kind.value
                target_state = None
                if isinstance(tracking, Mapping):
                    target_state = tracking.get("state")
                if isinstance(target_lock, Mapping):
                    target_state = target_lock.get("state")
                print(
                    f"[INFO]: step={step} reason={reason} action={action} "
                    f"target={_xy_from_target(obs.target_hint)} target_state={target_state} "
                    f"cmd={[round(v, 4) for v in command.tuple]} duration={command.duration_s} "
                    f"lidar_points={None if obs.lidar is None else obs.lidar.point_count}"
                )

            if decision is not None and decision.target_reached:
                print(f"[INFO]: target reached at step={step}")
                break
            if real_motion and any(event.get("accepted") is False for event in send_events):
                print("[ERROR]: command rejected during real motion; stopping")
                exit_code = 1
                break
    except KeyboardInterrupt:
        print("[WARN]: interrupted by user; stopping robot")
        exit_code = 130
    except Exception as exc:
        print(f"[ERROR]: real_g1_follow_stepwise failed: {exc!r}")
        exit_code = 1
    finally:
        try:
            sink.stop(timestamp_s=time.monotonic())
        finally:
            sink.close()
            livox_adapter.stop()
            if rs_adapter is not None:
                rs_adapter.stop()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
