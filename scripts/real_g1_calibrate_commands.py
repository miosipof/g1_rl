#!/usr/bin/env python3
"""Manual real-G1 command calibration helper.

This script sends a constant body command for a bounded duration, then sends a
stop. It is intended for supervised calibration of the final Unitree
``Move(vx, vy, wz)`` mapping, deadbands, signs, and rough response magnitudes.

It does not read odometry yet. Measure motion externally/manual-observed and
record the result from the printed notes. Use one axis at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from adapters import Command2D, UnitreeLocoAdapter, UnitreeLocoAdapterConfig, load_adapter_config


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _command_from_args(args: argparse.Namespace) -> tuple[float, float, float]:
    vx = float(args.vx)
    vy = float(args.vy)
    wz = float(args.wz)
    if args.yaw_degrees is not None:
        if args.duration <= 0.0:
            raise SystemExit("--duration must be positive when --yaw-degrees is used")
        wz = math.radians(float(args.yaw_degrees)) / float(args.duration)
    return vx, vy, wz


def _default_for_axis(axis: str) -> tuple[float, float, float]:
    if axis == "vx":
        return 0.5, 0.0, 0.0
    if axis == "vy":
        return 0.0, 0.1, 0.0
    if axis == "wz":
        return 0.0, 0.0, 0.2
    raise ValueError(axis)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--iface", default="eth0", help="Network interface for Unitree SDK ChannelFactoryInitialize.")
    parser.add_argument("--axis", choices=["vx", "vy", "wz", "custom"], default="vx", help="Calibration axis; sets default command if vx/vy/wz are not provided.")
    parser.add_argument("--vx", type=float, default=None, help="Commanded vx in m/s. If omitted, selected by --axis.")
    parser.add_argument("--vy", type=float, default=None, help="Commanded vy in m/s. If omitted, selected by --axis.")
    parser.add_argument("--wz", type=float, default=None, help="Commanded wz in rad/s. If omitted, selected by --axis or --yaw-degrees.")
    parser.add_argument("--yaw-degrees", type=float, default=None, help="For yaw tests, compute wz=yaw_degrees/duration, e.g. 360 over 3 s.")
    parser.add_argument("--duration", type=float, default=1.0, help="Command duration in seconds.")
    parser.add_argument("--rate-hz", type=float, default=None, help="Command send rate. Default comes from config command_limits.command_rate_hz.")
    parser.add_argument("--command-vx-scale", type=float, default=1.0, help="Final vx scale/sign before Unitree Move().")
    parser.add_argument("--command-vy-scale", type=float, default=1.0, help="Final vy scale/sign before Unitree Move().")
    parser.add_argument("--command-wz-scale", type=float, default=1.0, help="Final wz scale/sign before Unitree Move().")
    parser.add_argument("--expected-fsm-id", type=int, default=200, help="Expected Unitree loco fsm_id when not using --allow-non-200.")
    parser.add_argument("--allow-non-200", action="store_true", help="Disable fsm_id equality gate; fsm_mode gate remains active.")
    parser.add_argument("--state-samples", type=int, default=3, help="Number of Unitree loco state samples before each command.")
    parser.add_argument("--state-interval-s", type=float, default=0.05, help="Delay between Unitree loco state samples.")
    parser.add_argument("--allow-outside-config-limits", action="store_true", help="DANGEROUS: bypass real_g1.yaml command clipping. Needed for high-yaw tests such as 360 deg / 3 s.")
    parser.add_argument("--yes-i-understand-this-moves-the-robot", action="store_true", help="Actually send Unitree Move commands. Without this, dry-run only.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary at the end.")
    args = parser.parse_args()

    if args.duration < 0.0 or not math.isfinite(args.duration):
        raise SystemExit("--duration must be non-negative and finite")
    if args.rate_hz is not None and (args.rate_hz <= 0.0 or not math.isfinite(args.rate_hz)):
        raise SystemExit("--rate-hz must be positive and finite")
    if args.state_samples < 1:
        raise SystemExit("--state-samples must be >= 1")
    if args.state_interval_s < 0.0:
        raise SystemExit("--state-interval-s must be non-negative")

    default_vx, default_vy, default_wz = _default_for_axis("vx" if args.axis == "custom" else args.axis)
    if args.vx is None:
        args.vx = 0.0 if args.axis == "custom" else default_vx
    if args.vy is None:
        args.vy = 0.0 if args.axis == "custom" else default_vy
    if args.wz is None:
        args.wz = 0.0 if args.axis == "custom" else default_wz
    vx, vy, wz = _command_from_args(args)

    cfg = load_adapter_config(args.config)
    rate_hz = float(args.rate_hz if args.rate_hz is not None else cfg.command_limits.command_rate_hz)
    period_s = 1.0 / rate_hz
    real_motion = bool(args.yes_i_understand_this_moves_the_robot)
    require_fsm_id = None if args.allow_non_200 else int(args.expected_fsm_id)
    command_limits = None if args.allow_outside_config_limits else cfg.command_limits

    if real_motion and args.allow_outside_config_limits:
        print("[WARN]: --allow-outside-config-limits is enabled; config velocity clipping is bypassed")
    if real_motion:
        print("[WARN]: REAL MOTION ENABLED; keep operator/stop command ready")
    else:
        print("[INFO]: dry-run mode: no Unitree Move command will be sent")

    print(f"[INFO]: config={args.config}")
    print(f"[INFO]: iface={args.iface}")
    print(f"[INFO]: command request vx={vx:.6f} vy={vy:.6f} wz={wz:.6f} duration={args.duration:.3f}s rate={rate_hz:.2f}Hz")
    print(f"[INFO]: command mapping vx_scale={args.command_vx_scale} vy_scale={args.command_vy_scale} wz_scale={args.command_wz_scale}")
    print(f"[INFO]: fsm gate require_fsm_id={require_fsm_id} state_samples={args.state_samples}")
    if args.yaw_degrees is not None:
        print(f"[INFO]: yaw target {args.yaw_degrees:.1f} deg over {args.duration:.3f}s -> wz={wz:.6f} rad/s")
    print("[INFO]: Manual measurement suggestion: record actual distance/angle, drift, smoothness, and stop latency.")

    sink = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(
            iface=args.iface,
            dry_run=not real_motion,
            require_arm=True,
            require_fsm_id=require_fsm_id,
            command_vx_scale=args.command_vx_scale,
            command_vy_scale=args.command_vy_scale,
            command_wz_scale=args.command_wz_scale,
            state_samples=args.state_samples,
            state_interval_s=args.state_interval_s,
        ),
        command_limits=command_limits,
    )
    sink.arm()

    iterations = 0 if args.duration == 0.0 else max(1, int(math.ceil(args.duration * rate_hz)))
    command = Command2D(vx=vx, vy=vy, wz=wz, source="real_g1_calibrate_commands")
    sent: list[dict[str, Any]] = []
    exit_code = 0
    try:
        start = time.monotonic()
        for i in range(iterations):
            cmd = Command2D(vx=command.vx, vy=command.vy, wz=command.wz, timestamp_s=time.monotonic(), source=command.source)
            result = sink.send_command(cmd)
            status = result.status
            details = dict(status.details)
            sent.append(
                {
                    "i": i,
                    "accepted": bool(result.accepted),
                    "health": status.health.value,
                    "message": status.message,
                    "details": details,
                    "command": result.command.tuple,
                }
            )
            print(f"[INFO]: i={i} accepted={result.accepted} health={status.health.value} message={status.message!r} command={result.command.tuple}")
            if details:
                print(f"[INFO]: details={json.dumps(details, sort_keys=True, default=_json_default)}")
            if not result.accepted:
                exit_code = 1
                break
            next_t = start + (i + 1) * period_s
            delay = next_t - time.monotonic()
            if delay > 0.0 and i + 1 < iterations:
                time.sleep(delay)
    finally:
        stop = sink.stop(timestamp_s=time.monotonic())
        print(f"[INFO]: stop accepted={stop.accepted} health={stop.status.health.value} message={stop.status.message!r}")
        sink.close()

    summary = {
        "real_motion": real_motion,
        "command_request": {"vx": vx, "vy": vy, "wz": wz, "duration_s": args.duration, "rate_hz": rate_hz},
        "command_mapping": {"vx_scale": args.command_vx_scale, "vy_scale": args.command_vy_scale, "wz_scale": args.command_wz_scale},
        "bypassed_config_limits": bool(args.allow_outside_config_limits),
        "iterations": iterations,
        "accepted_count": sum(1 for row in sent if row["accepted"]),
        "sent": sent,
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
