#!/usr/bin/env python3
"""Execute one conservative G1 forward primitive using measured odometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters import (  # noqa: E402
    Command2D,
    OdometryActionExecutor,
    OdometryActionExecutorConfig,
    TerminalLocomotionFault,
    UnitreeLocoAdapter,
    UnitreeLocoAdapterConfig,
    UnitreeOdometryAdapter,
    UnitreeOdometryConfig,
)
from g1_nav_rl.navigation.stepwise import MotionCommand, StepwiseAction, StepwiseActionKind  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--distance-m", type=float, default=0.10)
    parser.add_argument("--speed-mps", type=float, default=0.15)
    parser.add_argument("--command-period-s", type=float, default=0.10)
    parser.add_argument("--expected-fsm-id", type=int, default=200)
    parser.add_argument("--yes-i-understand-this-moves-the-robot", action="store_true")
    args = parser.parse_args()
    if not args.yes_i_understand_this_moves_the_robot:
        raise SystemExit("Refusing motion without --yes-i-understand-this-moves-the-robot")
    if args.distance_m <= 0.0 or args.speed_mps <= 0.0:
        raise SystemExit("distance and speed must be positive")

    odom = UnitreeOdometryAdapter(UnitreeOdometryConfig(iface=args.iface))
    sink = UnitreeLocoAdapter(
        config=UnitreeLocoAdapterConfig(
            iface=args.iface,
            command_period_s=args.command_period_s,
            require_fsm_id=args.expected_fsm_id,
        )
    )
    try:
        status = odom.connect()
        if not status.ok:
            raise RuntimeError(status.message)
        odom.capture_origin()
        status = sink.connect()
        if not status.ok:
            raise RuntimeError(status.message)
        state_ok, reasons, _ = sink.check_state_gate()
        if not state_ok:
            raise RuntimeError(f"locomotion state gate rejected motion: {reasons}")
        sink.arm()
        executor = OdometryActionExecutor(
            sink=sink,
            odometry=odom,
            config=OdometryActionExecutorConfig(command_refresh_period_s=args.command_period_s),
        )
        start = odom.snapshot()
        executor.checked_stop()
        stationary = executor.wait_until_stationary(newer_than_sequence=start.sequence)
        if not stationary.stationary:
            raise RuntimeError(f"robot is not stationary: {stationary.reason}")
        duration = args.distance_m / args.speed_mps
        action = StepwiseAction(
            StepwiseActionKind.FORWARD,
            MotionCommand(args.speed_mps, 0.0, 0.0),
            duration_s=duration,
            distance_m=args.distance_m,
            source="real_g1_odom_nudge",
        )
        command = Command2D(
            args.speed_mps,
            0.0,
            0.0,
            duration_s=duration,
            source="real_g1_odom_nudge",
        )
        result = executor.execute(action=action, command=command, require_state_gate=True)
        print(json.dumps(result.to_dict(), sort_keys=True))
        return 0 if result.completed else 1
    except TerminalLocomotionFault as exc:
        print(json.dumps({"terminal_fault": str(exc)}, sort_keys=True))
        return 2
    finally:
        try:
            sink.stop()
        finally:
            sink.close()
            odom.close()


if __name__ == "__main__":
    raise SystemExit(main())
