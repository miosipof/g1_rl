#!/usr/bin/env python3
"""Clean real G1 target-follow entrypoint.

This is the public script for the simplified real-robot test.  It deliberately
exposes only the parameters that should be tuned during robot sessions.

Fixed policy:
  - Use the best current YOLO-stream detection as the target.
  - Build a Livox occupancy grid.
  - Plan with A* in the robot body frame.
  - Execute yaw-first / forward-only command bursts.
  - Replan because current LiDAR says the active path is blocked or a fresh
    target is available at a checkpoint.
  - Do not replan merely because the target is missing.
  - Never replan to stale body-frame target coordinates.
  - Clear queued commands on emergency front stop.

The old flag-heavy implementation is kept as
`scripts/simplified_target_follow_advanced.py` for debugging and exact replay.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simplified_target_follow_advanced import main as _advanced_main  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Clean target-follow test: YOLO stream target + Livox occupancy + "
            "A* + yaw-first/forward-only G1 command bursts."
        )
    )

    # Required hardware / stream inputs.
    p.add_argument("--unitree-vibes-dir", required=True, help="Path to unitree_g1_vibes on the robot.")
    p.add_argument("--livox-config", required=True, help="Livox SDK config JSON.")
    p.add_argument("--host-ip", required=True, help="Host IP used by the Livox/Unitree adapters.")
    p.add_argument("--iface", default="eth0", help="Network interface for Unitree locomotion commands.")
    p.add_argument("--detection-stream-path", required=True, help="Path to latest_tracks.json produced by real_g1_yolo_track_stream.py.")

    # Target selection.
    p.add_argument("--target-class", default="bottle", help="YOLO class to follow, e.g. bottle or person.")
    p.add_argument("--target-min-confidence", type=float, default=0.10, help="Minimum confidence for accepting a stream detection.")
    p.add_argument("--stream-max-age-s", type=float, default=5.0, help="Maximum accepted age of latest_tracks.json.")
    p.add_argument("--initial-target-timeout-s", type=float, default=20.0, help="How long to wait for the first valid target.")

    # Motion policy.  These are actual SDK-level command magnitudes; no separate
    # vx scale is exposed in this clean entrypoint.
    p.add_argument("--max-cycles", type=int, default=20, help="Number of stop/observe/act cycles.")
    p.add_argument("--burst-s", type=float, default=2.0, help="Maximum open-loop motion time before the next LiDAR check.")
    p.add_argument("--settle-s", type=float, default=0.50, help="Stop-settle time before reading sensors.")
    p.add_argument("--sdk-forward-vx-mps", type=float, default=0.20, help="Forward velocity sent to the Unitree SDK.")
    p.add_argument("--sdk-turn-wz-radps", type=float, default=0.25, help="Yaw rate sent to the Unitree SDK.")
    p.add_argument("--turn-angle-max-deg", type=float, default=30.0, help="Maximum single yaw command angle.")
    p.add_argument("--min-turn-angle-deg", type=float, default=10.0, help="Ignore smaller yaw corrections.")

    # Planning / safety knobs.  These are the only geometry values expected to be
    # tuned during field tests.
    p.add_argument("--planner-inflation-m", type=float, default=0.25, help="A* obstacle inflation radius.")
    p.add_argument("--front-replan-m", type=float, default=0.75, help="Obstacle distance that triggers a LiDAR-driven replan.")
    p.add_argument("--front-stop-m", type=float, default=0.45, help="Emergency raw-front stop distance; no motion is sent below this.")
    p.add_argument("--goal-relocation-radius-m", type=float, default=0.80, help="Allowed goal-only relocation radius when the target stand-off cell is occupied.")
    p.add_argument("--target-reached-radius-m", type=float, default=0.75, help="Stop when the current target is within this distance.")

    # Logging / execution.
    p.add_argument("--run-id", default=None)
    p.add_argument("--output-root", default="outputs/real_runs")
    p.add_argument("--save-lidar-debug", default=None, help="Directory for occupancy debug images and trace logs.")
    p.add_argument("--no-timestamp-debug-dirs", action="store_true")
    p.add_argument("--json-lines", action="store_true")
    p.add_argument("--allow-non-200", action="store_true", help="Allow locomotion even if FSM id is not 200.")
    p.add_argument("--yes-i-understand-this-moves-the-robot", action="store_true", help="Actually send G1 motion commands. Omit for dry-run.")

    return p


def _to_advanced_args(args: argparse.Namespace) -> list[str]:
    """Translate the small public CLI into the advanced implementation args."""

    burst_s = max(0.0, float(args.burst_s))
    front_replan_m = float(args.front_replan_m)
    front_stop_m = float(args.front_stop_m)
    if front_replan_m <= front_stop_m:
        raise SystemExit(
            f"--front-replan-m must be larger than --front-stop-m; got {front_replan_m} <= {front_stop_m}"
        )

    out: list[str] = [
        "--unitree-vibes-dir", str(args.unitree_vibes_dir),
        "--livox-config", str(args.livox_config),
        "--host-ip", str(args.host_ip),
        "--iface", str(args.iface),
        "--detection-stream-path", str(args.detection_stream_path),
        "--detection-stream-max-age-s", str(float(args.stream_max_age_s)),
        "--target-class", str(args.target_class),
        "--target-min-confidence", str(float(args.target_min_confidence)),
        "--initial-target-timeout-s", str(float(args.initial_target_timeout_s)),

        # Fixed simplified behavior.
        "--target-missing-policy", "continue_plan",
        "--max-cycles", str(int(args.max_cycles)),
        # The advanced script still has both checkpoint period and burst cap.
        # Set both to the same public value so there is one timing concept.
        "--checkpoint-period-s", str(burst_s),
        "--max-burst-duration-s", str(burst_s),
        "--settle-s", str(float(args.settle_s)),
        "--command-period-s", "0.10",
        "--stop-after-burst-s", "0.20",

        # Actual SDK commands: no extra scaling in the clean entrypoint.
        "--forward-speed-mps", str(float(args.sdk_forward_vx_mps)),
        "--turn-speed-radps", str(float(args.sdk_turn_wz_radps)),
        "--command-vx-scale", "1.0",
        "--command-vy-scale", "0.0",
        "--command-wz-scale", "1.0",
        "--turn-angle-max-deg", str(float(args.turn_angle_max_deg)),
        "--min-turn-angle-deg", str(float(args.min_turn_angle_deg)),
        "--lookahead-m", "0.80",
        "--target-reached-radius-m", str(float(args.target_reached_radius_m)),

        # Fixed grid defaults known to work for the current real test volume.
        "--grid-x-min", "-2.0",
        "--grid-x-max", "6.0",
        "--grid-y-min", "-4.0",
        "--grid-y-max", "4.0",
        "--grid-resolution", "0.05",
        "--astar-inflation-m", str(float(args.planner_inflation_m)),
        "--occupancy-threshold", "0.5",
        "--astar-heuristic-weight", "1.0",
        "--astar-max-expansions", "0",
        "--goal-relocation-radius-m", str(float(args.goal_relocation_radius_m)),

        # Obstacle-driven replanning and burst validation.  These are fixed on.
        "--front-corridor-half-width-m", "0.25",
        "--obstacle-replan-front-m", str(front_replan_m),
        "--emergency-front-stop-m", str(front_stop_m),
        "--clear-queue-on-emergency-stop",
        "--obstacle-replan-with-estimated-goal",
        "--validate-active-queue-with-lidar",
        "--validate-executed-burst-with-lidar",
        "--queued-segment-guard-inflation-m", "0.25",
        "--queued-segment-start-ignore-m", "0.15",
        "--queued-segment-sample-step-m", "0.05",

        # Adapter sampling defaults.
        "--livox-frame-time", "0.20",
        "--livox-frame-packets", "120",
        "--livox-timeout", "2.0",
        "--expected-fsm-id", "200",
        "--state-samples", "1",
        "--state-interval-s", "0.05",
    ]

    if args.run_id:
        out += ["--run-id", str(args.run_id)]
    if args.output_root:
        out += ["--output-root", str(args.output_root)]
    if args.save_lidar_debug:
        out += ["--save-lidar-debug", str(args.save_lidar_debug)]
    if args.no_timestamp_debug_dirs:
        out.append("--no-timestamp-debug-dirs")
    if args.json_lines:
        out.append("--json-lines")
    if args.allow_non_200:
        out.append("--allow-non-200")
    if args.yes_i_understand_this_moves_the_robot:
        out.append("--yes-i-understand-this-moves-the-robot")

    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(_advanced_main(_to_advanced_args(args)))


if __name__ == "__main__":
    raise SystemExit(main())
