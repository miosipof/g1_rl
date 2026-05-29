#!/usr/bin/env python3
"""Minimal real G1 target-follow script.

This script is intentionally NOT a wrapper around the stepwise runtime.  It is a
clean hardware test for the simplest contract:

  first/latest valid target from a YOLO stream
  + current Livox occupancy grid
  -> A* path in the robot body frame
  -> one yaw-first / forward command burst
  -> stop, observe again after 2--3 seconds

Deliberately omitted in this script:
  - PrimaryTargetLock / track-ID persistence
  - target-change replanning during a burst
  - path_block inflated-grid execution guard
  - command odometry
  - continuous SafetySupervisor
  - lateral velocity commands
  - start/goal relocation by default
  - stop after every primitive

The only runtime guard kept is an optional raw front-distance emergency stop.
Real motion still requires --yes-i-understand-this-moves-the-robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch

from adapters import (
    Command2D,
    DetectionStreamClient,
    Target2D,
    UnitreeLocoAdapter,
    UnitreeLocoAdapterConfig,
    load_adapter_config,
    make_run_id,
    timestamped_output_dir,
    write_run_info,
)
from adapters.config import CommandLimitsConfig
from adapters.lidar_debug import save_occupancy_debug_image
from adapters.real_livox import LivoxAdapterConfig, UnitreeVibesLivox2Adapter, UnitreeVibesLivox2AdapterConfig
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg, rasterize_points_to_occupancy_grid
from g1_nav_rl.planning.grid_astar import GridAStarCfg, GridAStarResult, plan_grid_path_xy


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _default_config_path() -> Path:
    return ROOT / "configs" / "real_g1.yaml"


def _make_grid_cfg(args: argparse.Namespace) -> OccupancyGridCfg:
    return OccupancyGridCfg(
        x_min=float(args.grid_x_min),
        x_max=float(args.grid_x_max),
        y_min=float(args.grid_y_min),
        y_max=float(args.grid_y_max),
        resolution=float(args.grid_resolution),
    )


def _read_best_stream_target(
    *,
    stream_client: DetectionStreamClient,
    target_class: str,
    min_confidence: float,
    min_depth_valid_ratio: float,
    require_fresh: bool,
) -> tuple[Target2D | None, dict[str, Any]]:
    """Return the highest-confidence current target without target-lock state."""

    result = stream_client.read()
    meta: dict[str, Any] = {
        "stream_status": result.status.health.value,
        "stream_message": result.status.message,
        "stream_details": dict(result.status.details),
        "candidate_count": 0,
        "selected_track": None,
    }
    if result.frame is None:
        return None, meta
    if require_fresh and result.status.health.value != "ok":
        return None, meta

    candidates = []
    for track in result.frame.tracks:
        if track.class_name != str(target_class):
            continue
        if float(track.confidence) < float(min_confidence):
            continue
        if track.target_body_xy is None:
            continue
        if track.depth_valid_ratio is not None and float(track.depth_valid_ratio) < float(min_depth_valid_ratio):
            continue
        x, y = track.target_body_xy
        if not (math.isfinite(float(x)) and math.isfinite(float(y))):
            continue
        candidates.append(track)

    meta["candidate_count"] = len(candidates)
    if not candidates:
        return None, meta

    best = max(candidates, key=lambda t: float(t.confidence))
    x, y = best.target_body_xy  # type: ignore[misc]
    target = Target2D(x=float(x), y=float(y), confidence=float(best.confidence), source="stream_best_detection")
    meta["selected_track"] = best.to_dict()
    return target, meta


def _wait_for_initial_target(
    *,
    stream_client: DetectionStreamClient,
    target_class: str,
    min_confidence: float,
    min_depth_valid_ratio: float,
    timeout_s: float,
    poll_s: float,
    json_lines: bool,
) -> tuple[Target2D, dict[str, Any]]:
    deadline = time.monotonic() + float(timeout_s)
    last_meta: dict[str, Any] = {}
    while time.monotonic() <= deadline:
        target, meta = _read_best_stream_target(
            stream_client=stream_client,
            target_class=target_class,
            min_confidence=min_confidence,
            min_depth_valid_ratio=min_depth_valid_ratio,
            require_fresh=True,
        )
        last_meta = meta
        if target is not None:
            if json_lines:
                print(json.dumps({"event": "initial_target_acquired", "target_xy_b": [target.x, target.y], "stream": meta}, sort_keys=True))
            else:
                print(f"[INFO]: initial target acquired: xy_b=({target.x:.3f}, {target.y:.3f})")
            return target, meta
        if json_lines:
            print(json.dumps({"event": "waiting_for_initial_target", "stream": meta}, sort_keys=True))
        else:
            print(f"[INFO]: waiting for first target: {meta.get('stream_message')} candidates={meta.get('candidate_count')}")
        time.sleep(float(poll_s))
    raise RuntimeError(f"timed out waiting for initial {target_class!r} target; last_stream={last_meta}")


def _occupancy_from_lidar(points_body_xyz: np.ndarray, grid_cfg: OccupancyGridCfg) -> torch.Tensor:
    points = np.asarray(points_body_xyz, dtype=np.float32)
    if points.size == 0:
        pts = torch.zeros((1, 0, 2), dtype=torch.float32)
    else:
        pts = torch.as_tensor(points[:, :2], dtype=torch.float32).unsqueeze(0)
    return rasterize_points_to_occupancy_grid(pts, valid=None, cfg=grid_cfg)[0].detach().cpu()


def _path_length(path_xy_b: Sequence[tuple[float, float]]) -> float:
    if len(path_xy_b) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path_xy_b[:-1], path_xy_b[1:]):
        total += math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
    return total


def _sample_path(path_xy_b: Sequence[tuple[float, float]], distance_m: float) -> tuple[float, float] | None:
    if not path_xy_b:
        return None
    if len(path_xy_b) == 1 or distance_m <= 0.0:
        return path_xy_b[0]
    remaining = float(distance_m)
    prev = path_xy_b[0]
    for curr in path_xy_b[1:]:
        seg = math.hypot(float(curr[0]) - float(prev[0]), float(curr[1]) - float(prev[1]))
        if seg <= 1e-9:
            prev = curr
            continue
        if remaining <= seg:
            t = remaining / seg
            return (float(prev[0]) + t * (float(curr[0]) - float(prev[0])), float(prev[1]) + t * (float(curr[1]) - float(prev[1])))
        remaining -= seg
        prev = curr
    return path_xy_b[-1]


def _raw_front_min(points_body_xyz: np.ndarray, *, corridor_half_width_m: float, x_min_m: float, x_max_m: float) -> float | None:
    points = np.asarray(points_body_xyz, dtype=np.float32)
    if points.size == 0:
        return None
    x = points[:, 0]
    y = points[:, 1]
    keep = (x >= float(x_min_m)) & (x <= float(x_max_m)) & (np.abs(y) <= float(corridor_half_width_m))
    if not np.any(keep):
        return None
    return float(np.min(x[keep]))


def _make_burst_commands(
    *,
    plan: GridAStarResult,
    target: Target2D,
    forward_speed_mps: float,
    checkpoint_period_s: float,
    lookahead_m: float,
    min_turn_angle_rad: float,
    turn_speed_radps: float,
    turn_angle_max_rad: float,
    target_reached_radius_m: float,
) -> tuple[list[Command2D], dict[str, Any]]:
    dist_to_target = math.hypot(float(target.x), float(target.y))
    if dist_to_target <= float(target_reached_radius_m):
        return [], {"reason": "target_reached", "target_distance_m": dist_to_target}
    if not plan.found or len(plan.path_xy_b) < 2:
        return [], {"reason": "no_plan", "plan_reason": plan.reason}

    sample = _sample_path(plan.path_xy_b, float(lookahead_m))
    if sample is None:
        return [], {"reason": "empty_path"}
    sx, sy = float(sample[0]), float(sample[1])
    heading = math.atan2(sy, max(1e-6, sx))
    commands: list[Command2D] = []
    remaining = float(checkpoint_period_s)

    turn_angle = 0.0
    if abs(heading) >= float(min_turn_angle_rad):
        turn_angle = max(-float(turn_angle_max_rad), min(float(turn_angle_max_rad), heading))
        turn_duration = abs(turn_angle) / max(1e-6, float(turn_speed_radps))
        turn_duration = min(turn_duration, remaining)
        wz = math.copysign(float(turn_speed_radps), turn_angle)
        commands.append(Command2D(0.0, 0.0, wz, duration_s=max(0.05, turn_duration), source="simplified_target_follow_turn", metadata={"turn_angle_rad": turn_angle, "lookahead_xy_b": [sx, sy]}))
        remaining -= turn_duration

    forward_duration = max(0.0, remaining)
    # Commit at least one useful forward interval if the path is roughly ahead.
    if forward_duration >= 0.20:
        commands.append(Command2D(float(forward_speed_mps), 0.0, 0.0, duration_s=forward_duration, source="simplified_target_follow_forward", metadata={"lookahead_xy_b": [sx, sy]}))

    return commands, {
        "reason": "commands_ready" if commands else "no_time_for_forward",
        "target_distance_m": dist_to_target,
        "lookahead_xy_b": [sx, sy],
        "lookahead_heading_rad": heading,
        "turn_angle_rad": turn_angle,
        "checkpoint_period_s": checkpoint_period_s,
        "path_length_m": _path_length(plan.path_xy_b),
        "command_count": len(commands),
    }


def _send_command_for_duration(
    *,
    sink: UnitreeLocoAdapter,
    command: Command2D,
    command_period_s: float,
    real_motion: bool,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if command.is_stop:
        result = sink.stop(timestamp_s=time.monotonic())
        return [{"phase": "stop", "accepted": result.accepted, "status": _json_safe(result.status)}]

    duration = float(command.duration_s or 0.0)
    if duration <= 0.0:
        return []

    if not real_motion:
        result = sink.send_command(command, require_state_gate=False)
        return [{"phase": "dry_send", "accepted": result.accepted, "sent_command": _json_safe(result.command), "status": _json_safe(result.status)}]

    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        now = time.monotonic()
        cmd = Command2D(command.vx, command.vy, command.wz, timestamp_s=now, duration_s=command.duration_s, source=command.source, metadata=dict(command.metadata))
        result = sink.send_command(cmd, require_state_gate=True)
        events.append({"phase": "send", "accepted": result.accepted, "sent_command": _json_safe(result.command), "status": _json_safe(result.status)})
        if not result.accepted:
            break
        sleep_s = min(float(command_period_s), max(0.0, deadline - time.monotonic()))
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    return events


def _execute_burst(*, sink: UnitreeLocoAdapter, commands: Sequence[Command2D], command_period_s: float, real_motion: bool) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, command in enumerate(commands):
        command_events = _send_command_for_duration(sink=sink, command=command, command_period_s=command_period_s, real_motion=real_motion)
        events.append({"burst_index": idx, "command": _json_safe(command), "events": command_events})
        if any(not bool(e.get("accepted", True)) for e in command_events):
            break
    stop = sink.stop(timestamp_s=time.monotonic())
    events.append({"phase": "burst_stop", "accepted": stop.accepted, "status": _json_safe(stop.status)})
    return events


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal real G1 target follower: stream target + LiDAR A* + rare command bursts.")
    p.add_argument("--config", default=str(_default_config_path()))
    p.add_argument("--unitree-vibes-dir", required=True)
    p.add_argument("--livox-config", required=True)
    p.add_argument("--host-ip", required=True)
    p.add_argument("--iface", default="eth0")
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-root", default="outputs/real_runs")
    p.add_argument("--no-timestamp-debug-dirs", action="store_true")
    p.add_argument("--json-lines", action="store_true")

    p.add_argument("--detection-stream-path", required=True)
    p.add_argument("--detection-stream-max-age-s", type=float, default=5.0)
    p.add_argument("--target-class", default="bottle")
    p.add_argument("--target-min-confidence", type=float, default=0.10)
    p.add_argument("--target-min-depth-valid-ratio", type=float, default=0.0)
    p.add_argument("--initial-target-timeout-s", type=float, default=20.0)
    p.add_argument("--target-missing-policy", choices=["reuse_last", "stop"], default="reuse_last")

    p.add_argument("--max-cycles", type=int, default=10)
    p.add_argument("--checkpoint-period-s", type=float, default=2.5)
    p.add_argument("--settle-s", type=float, default=0.50)
    p.add_argument("--command-period-s", type=float, default=0.10)
    p.add_argument("--stop-after-burst-s", type=float, default=0.20)

    p.add_argument("--forward-speed-mps", type=float, default=0.20)
    p.add_argument("--turn-speed-radps", type=float, default=0.25)
    p.add_argument("--turn-angle-max-deg", type=float, default=30.0)
    p.add_argument("--min-turn-angle-deg", type=float, default=10.0)
    p.add_argument("--lookahead-m", type=float, default=0.80)
    p.add_argument("--target-reached-radius-m", type=float, default=0.75)

    p.add_argument("--grid-x-min", type=float, default=-2.0)
    p.add_argument("--grid-x-max", type=float, default=6.0)
    p.add_argument("--grid-y-min", type=float, default=-4.0)
    p.add_argument("--grid-y-max", type=float, default=4.0)
    p.add_argument("--grid-resolution", type=float, default=0.05)
    p.add_argument("--astar-inflation-m", type=float, default=0.35)
    p.add_argument("--occupancy-threshold", type=float, default=0.5)
    p.add_argument("--astar-heuristic-weight", type=float, default=1.0)
    p.add_argument("--astar-max-expansions", type=int, default=0)
    p.add_argument("--no-diagonal-astar", action="store_true")
    p.add_argument("--allow-start-relocation", action="store_true")
    p.add_argument("--allow-goal-relocation", action="store_true")

    p.add_argument("--emergency-front-stop-m", type=float, default=0.45)
    p.add_argument("--front-corridor-half-width-m", type=float, default=0.25)
    p.add_argument("--disable-emergency-front-stop", action="store_true")

    p.add_argument("--livox-frame-time", type=float, default=0.20)
    p.add_argument("--livox-frame-packets", type=int, default=120)
    p.add_argument("--livox-timeout", type=float, default=2.0)

    p.add_argument("--command-vx-scale", type=float, default=1.0)
    p.add_argument("--command-vy-scale", type=float, default=0.0)
    p.add_argument("--command-wz-scale", type=float, default=1.0)
    p.add_argument("--allow-non-200", action="store_true")
    p.add_argument("--expected-fsm-id", type=int, default=200)
    p.add_argument("--state-samples", type=int, default=1)
    p.add_argument("--state-interval-s", type=float, default=0.05)
    p.add_argument("--yes-i-understand-this-moves-the-robot", action="store_true")

    p.add_argument("--save-lidar-debug", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_adapter_config(args.config)
    run_id = args.run_id or make_run_id(prefix="simplified_target_follow")
    debug_base = None
    if args.save_lidar_debug is not None:
        debug_base = timestamped_output_dir(args.save_lidar_debug, run_id=run_id, enabled=not bool(args.no_timestamp_debug_dirs))
        assert debug_base is not None
        debug_base.mkdir(parents=True, exist_ok=True)
    else:
        debug_base = Path(args.output_root) / run_id / "simplified_target_follow"
        debug_base.mkdir(parents=True, exist_ok=True)

    real_motion = bool(args.yes_i_understand_this_moves_the_robot)
    grid_cfg = _make_grid_cfg(args)
    astar_cfg = GridAStarCfg(
        occupancy_threshold=float(args.occupancy_threshold),
        inflation_radius=float(args.astar_inflation_m),
        allow_diagonal=not bool(args.no_diagonal_astar),
        heuristic_weight=float(args.astar_heuristic_weight),
        max_expansions=int(args.astar_max_expansions),
    )

    stream_client = DetectionStreamClient(
        path=args.detection_stream_path,
        max_age_s=float(args.detection_stream_max_age_s),
        missing_ok=True,
    )

    livox_adapter = UnitreeVibesLivox2Adapter(
        UnitreeVibesLivox2AdapterConfig(
            livox_config_path=str(args.livox_config),
            unitree_vibes_dir=str(args.unitree_vibes_dir),
            host_ip=str(args.host_ip),
            frame_time_s=float(args.livox_frame_time),
            frame_packets=int(args.livox_frame_packets),
            read_timeout_s=float(args.livox_timeout),
            adapter_config=LivoxAdapterConfig.from_lidar_config(cfg.lidar),
        )
    )

    require_fsm_id = None if args.allow_non_200 else int(args.expected_fsm_id)
    sink = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(
            iface=str(args.iface),
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

    run_info = {
        "script": "scripts/simplified_target_follow.py",
        "run_id": run_id,
        "real_motion": real_motion,
        "simple_contract": {
            "target_selection": "best current stream detection, no target lock, no track-id persistence",
            "planning": "A* on current Livox occupancy grid from body origin",
            "execution": "yaw-first/forward-only burst, stop only at checkpoint",
            "disabled": [
                "StepwiseTargetNavPolicyRuntime",
                "PrimaryTargetLock",
                "SafetySupervisor",
                "command_odometry",
                "target_change_replanning",
                "path_block_guard",
                "lateral_commands",
            ],
        },
        "args": vars(args),
        "grid_cfg": grid_cfg,
        "astar_cfg": astar_cfg,
    }
    write_run_info(debug_base, run_info)
    trace_path = debug_base / "simplified_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    print("[INFO]: simplified_target_follow starting")
    print(f"[INFO]: run_id={run_id} real_motion={real_motion}")
    print(f"[INFO]: stream={args.detection_stream_path}")
    print(f"[INFO]: no target lock, no stepwise runtime, no path_block guard")
    if real_motion:
        print("[WARN]: REAL MOTION ENABLED; operator must be ready to stop the robot")
    else:
        print("[INFO]: dry-run mode: no real Move commands will be sent")

    exit_code = 0
    last_target: Target2D | None = None
    livox_adapter.start()
    sink.connect()
    sink.arm()
    try:
        # Initial target acquisition is explicit.  This is the only required
        # target precondition for the simple test.
        last_target, target_meta = _wait_for_initial_target(
            stream_client=stream_client,
            target_class=str(args.target_class),
            min_confidence=float(args.target_min_confidence),
            min_depth_valid_ratio=float(args.target_min_depth_valid_ratio),
            timeout_s=float(args.initial_target_timeout_s),
            poll_s=0.25,
            json_lines=bool(args.json_lines),
        )

        for cycle in range(int(args.max_cycles)):
            stop_before = sink.stop(timestamp_s=time.monotonic())
            if float(args.settle_s) > 0.0:
                time.sleep(float(args.settle_s))

            now = time.monotonic()
            current_target, target_meta = _read_best_stream_target(
                stream_client=stream_client,
                target_class=str(args.target_class),
                min_confidence=float(args.target_min_confidence),
                min_depth_valid_ratio=float(args.target_min_depth_valid_ratio),
                require_fresh=True,
            )
            target_source = "current_stream"
            if current_target is not None:
                last_target = current_target
            elif args.target_missing_policy == "reuse_last" and last_target is not None:
                current_target = last_target
                target_source = "last_known_reused"
            else:
                current_target = None
                target_source = "missing_stop"

            lidar = livox_adapter.read_lidar_frame(timeout_s=float(args.livox_timeout))
            occupancy = _occupancy_from_lidar(lidar.points_body_xyz, grid_cfg)
            front_min = _raw_front_min(
                lidar.points_body_xyz,
                corridor_half_width_m=float(args.front_corridor_half_width_m),
                x_min_m=0.0,
                x_max_m=float(args.emergency_front_stop_m),
            )

            plan = None
            commands: list[Command2D] = []
            command_meta: dict[str, Any] = {"reason": "no_target"}
            emergency_stop = False
            if current_target is None:
                command_meta = {"reason": "target_missing"}
            elif (not args.disable_emergency_front_stop) and front_min is not None and front_min <= float(args.emergency_front_stop_m):
                emergency_stop = True
                command_meta = {"reason": "emergency_front_stop", "front_min_m": front_min}
            else:
                plan = plan_grid_path_xy(
                    occupancy,
                    start_xy_b=(0.0, 0.0),
                    goal_xy_b=(float(current_target.x), float(current_target.y)),
                    grid_cfg=grid_cfg,
                    planner_cfg=astar_cfg,
                    allow_start_relocation=bool(args.allow_start_relocation),
                    allow_goal_relocation=bool(args.allow_goal_relocation),
                )
                commands, command_meta = _make_burst_commands(
                    plan=plan,
                    target=current_target,
                    forward_speed_mps=float(args.forward_speed_mps),
                    checkpoint_period_s=float(args.checkpoint_period_s),
                    lookahead_m=float(args.lookahead_m),
                    min_turn_angle_rad=math.radians(float(args.min_turn_angle_deg)),
                    turn_speed_radps=float(args.turn_speed_radps),
                    turn_angle_max_rad=math.radians(float(args.turn_angle_max_deg)),
                    target_reached_radius_m=float(args.target_reached_radius_m),
                )

            # Debug image before motion.
            image_path = debug_base / f"occupancy_grid_cycle_{cycle:06d}.png"
            try:
                save_occupancy_debug_image(
                    image_path,
                    occupancy,
                    grid_cfg=grid_cfg,
                    plan_xy_b=None if plan is None else plan.path_xy_b,
                    target_xy_b=None if current_target is None else (float(current_target.x), float(current_target.y)),
                )
            except Exception as exc:
                image_path = None  # type: ignore[assignment]
                command_meta = dict(command_meta) | {"debug_image_error": repr(exc)}

            send_events: list[dict[str, Any]] = []
            if commands:
                send_events = _execute_burst(sink=sink, commands=commands, command_period_s=float(args.command_period_s), real_motion=real_motion)
            else:
                stop = sink.stop(timestamp_s=time.monotonic())
                send_events = [{"phase": "no_motion", "accepted": stop.accepted, "status": _json_safe(stop.status), "reason": command_meta.get("reason")}]

            if float(args.stop_after_burst_s) > 0.0:
                time.sleep(float(args.stop_after_burst_s))

            row = {
                "cycle": int(cycle),
                "timestamp_s": now,
                "target_xy_b": None if current_target is None else [float(current_target.x), float(current_target.y)],
                "target_source": target_source,
                "target_stream": target_meta,
                "lidar_point_count": int(lidar.point_count),
                "front_min_m": front_min,
                "emergency_stop": emergency_stop,
                "occupancy_grid_path": None if image_path is None else str(image_path),
                "plan": None if plan is None else {
                    "found": bool(plan.found),
                    "reason": plan.reason,
                    "start_cell": list(plan.start_cell),
                    "goal_cell": list(plan.goal_cell),
                    "path_len": len(plan.path_xy_b),
                    "path_length_m": _path_length(plan.path_xy_b),
                },
                "command_meta": command_meta,
                "commands": [_json_safe(c) for c in commands],
                "stop_before": {"accepted": stop_before.accepted, "status": _json_safe(stop_before.status)},
                "send_events": send_events,
            }
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
            if args.json_lines:
                print(json.dumps(_json_safe(row), sort_keys=True))
            else:
                print(
                    f"[INFO]: cycle={cycle} target={row['target_xy_b']} source={target_source} "
                    f"front={front_min} plan={None if plan is None else plan.reason} "
                    f"cmd_reason={command_meta.get('reason')} commands={len(commands)}"
                )

            if command_meta.get("reason") == "target_reached":
                print(f"[INFO]: target reached at cycle={cycle}")
                break
            if real_motion:
                rejected = False
                for burst in send_events:
                    for event in burst.get("events", []) if isinstance(burst, Mapping) else []:
                        if event.get("accepted") is False:
                            rejected = True
                if rejected:
                    print("[ERROR]: command rejected during real motion; stopping")
                    exit_code = 1
                    break
    except KeyboardInterrupt:
        print("[WARN]: interrupted by user; stopping robot")
        exit_code = 130
    except Exception as exc:
        print(f"[ERROR]: simplified_target_follow failed: {exc!r}")
        exit_code = 1
    finally:
        try:
            sink.stop(timestamp_s=time.monotonic())
        finally:
            sink.close()
            livox_adapter.stop()
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
