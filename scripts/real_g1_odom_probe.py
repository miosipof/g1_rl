#!/usr/bin/env python3
"""Read and optionally log Unitree G1 odometry without sending motion commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapters import UnitreeOdometryAdapter, UnitreeOdometryConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--topic", default="rt/odommodestate")
    parser.add_argument("--startup-timeout-s", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=0.0, help="Zero runs until interrupted.")
    parser.add_argument("--jsonl", type=Path, default=None)
    args = parser.parse_args()
    if args.rate_hz <= 0.0 or args.duration_s < 0.0:
        raise SystemExit("--rate-hz must be positive and --duration-s non-negative")

    odom = UnitreeOdometryAdapter(
        UnitreeOdometryConfig(
            iface=args.iface,
            topic=args.topic,
            startup_timeout_s=args.startup_timeout_s,
        )
    )
    status = odom.connect()
    if not status.ok:
        print(json.dumps({"error": status.message, "status": dict(status.details)}, sort_keys=True))
        odom.close()
        return 1
    odom.capture_origin()
    started = time.monotonic()
    period = 1.0 / args.rate_hz
    try:
        while args.duration_s <= 0.0 or time.monotonic() - started < args.duration_s:
            snapshot = odom.snapshot()
            row = snapshot.to_dict()
            row["age_s"] = snapshot.raw.age_s(time.monotonic())
            line = json.dumps(row, sort_keys=True)
            print(line)
            if args.jsonl is not None:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.jsonl.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            time.sleep(period)
    except KeyboardInterrupt:
        return 130
    finally:
        odom.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
