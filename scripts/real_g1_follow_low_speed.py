#!/usr/bin/env python3
"""Low-speed real-robot bridge entry point.

This script is intentionally conservative.  By default it runs in Unitree
locomotion dry-run mode and sends no robot commands.  Real motion requires the
explicit flag ``--yes-i-understand-this-moves-the-robot`` and should only be used
after the probe and dry-run scripts are clean.

The current implementation still uses a synthetic observation provider.  This is
suitable for command-sink validation and low-speed nudge-style tests, not for
closed-loop autonomous following.  Live RealSense/Livox providers should be wired
before using this for real target following.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from adapters import RealBridge, RealBridgeConfig, UnitreeLocoAdapter, UnitreeLocoAdapterConfig, load_adapter_config
from adapters.safety_supervisor import SafetySupervisor

# Reuse the safe synthetic provider/policy from the dry-run script.  Importing a
# script module is acceptable here because both files are explicit entry points.
from real_g1_follow_dryrun import ConstantVelocityPolicy, SyntheticObservationProvider, _result_to_dict


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--iface", default="eth0", help="Network interface for Unitree SDK ChannelFactoryInitialize.")
    parser.add_argument("--steps", type=int, default=3, help="Number of bridge iterations.")
    parser.add_argument("--dt", type=float, default=0.10, help="Sleep between iterations.")
    parser.add_argument("--vx", type=float, default=0.03, help="Requested low-speed vx in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Requested low-speed vy in m/s.")
    parser.add_argument("--wz", type=float, default=0.0, help="Requested low-speed wz in rad/s.")
    parser.add_argument("--obstacle-front-m", type=float, default=None, help="Optional synthetic front obstacle distance.")
    parser.add_argument(
        "--yes-i-understand-this-moves-the-robot",
        action="store_true",
        help="Disable dry-run and allow Unitree Move commands. Use only in a clear test area.",
    )
    parser.add_argument(
        "--allow-synthetic-observation-motion",
        action="store_true",
        help="Required together with the motion flag while this script still uses synthetic observations.",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.dt < 0.0:
        raise SystemExit("--dt must be non-negative")

    real_motion = bool(args.yes_i_understand_this_moves_the_robot)
    if real_motion and not args.allow_synthetic_observation_motion:
        raise SystemExit(
            "Refusing real motion with synthetic observations. Add "
            "--allow-synthetic-observation-motion only for a supervised low-speed command-sink test."
        )

    cfg = load_adapter_config(args.config)
    provider = SyntheticObservationProvider(config=cfg, obstacle_front_m=args.obstacle_front_m)
    policy = ConstantVelocityPolicy(args.vx, args.vy, args.wz, source="low_speed_constant_policy")
    supervisor = SafetySupervisor.from_runtime_config(cfg)
    sink = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(iface=args.iface, dry_run=not real_motion, require_arm=True),
        command_limits=cfg.command_limits,
    )

    print("[INFO]: low-speed bridge starting")
    print(f"[INFO]: config={args.config}")
    print(f"[INFO]: iface={args.iface}")
    print(f"[INFO]: real_motion={real_motion}")
    if not real_motion:
        print("[INFO]: dry-run mode: no Unitree Move command will be sent")
    else:
        print("[WARN]: REAL MOTION ENABLED; keep an operator ready to stop the robot")

    bridge = RealBridge(
        observation_provider=provider,
        policy_runtime=policy,
        safety_supervisor=supervisor,
        command_sink=sink,
        config=RealBridgeConfig(name="real_g1_follow_low_speed", send_commands=True, auto_disarm_sink_on_stop=True),
    )

    status = bridge.follow()
    print(f"[INFO]: bridge command: follow -> {status.health.value}: {status.message}")

    exit_code = 0
    try:
        for i in range(args.steps):
            result = bridge.step()
            row: dict[str, Any] = _result_to_dict(result)
            print(
                f"[INFO]: step={i} mode={row['mode']} action={row['action']} "
                f"health={row['status_health']} proposed={row['proposed_command']} safe={row['safe_command']} "
                f"accepted={row['command_accepted']} reasons={row['safety_reasons']} warnings={row['safety_warnings']}"
            )
            if result.status.health.value == "fault":
                exit_code = 1
                break
            if args.dt > 0.0 and i + 1 < args.steps:
                time.sleep(args.dt)
    finally:
        stop_status = bridge.stop()
        print(f"[INFO]: bridge command: stop -> {stop_status.health.value}: {stop_status.message}")
        sink.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
