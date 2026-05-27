#!/usr/bin/env python3
"""Run the Isaac streaming-target smoke test with the stepwise controller.

This is the Step-A entry point for the simplified architecture:

    detection stream target_body_xy
    + Isaac RayCaster/LiDAR
    -> StepwiseTargetNavPolicyRuntime
    -> G1 material-point command proxy

By default the script writes a deterministic static DetectionStreamFrame before
launching Isaac.  For a real asynchronous YOLO stream, pass
``--no-static-stream --detection-stream-path <latest_tracks.json>`` and run a
producer that continuously writes the same stream schema.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from g1_nav_rl.perception.detection_stream import (  # noqa: E402
    DetectionStreamFrame,
    TrackDetection,
    write_detection_stream_frame_atomic,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Isaac target navigation with --controller_mode=stepwise and --target_source=stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke-script", type=str, default=str(PROJECT_ROOT / "scripts" / "smoke_test_yolo_person_nav.py"))
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "outputs" / "isaac_stepwise_stream"))
    parser.add_argument("--run-id", type=str, default="", help="Run id folder. Default: stepwise_YYYYMMDD_HHMMSS.")
    parser.add_argument("--mode-timeout-s", type=float, default=0.0, help="Optional child-process timeout. 0 disables timeout.")
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--tail-lines", type=int, default=80)

    parser.add_argument("--target-class", type=str, default="person")
    parser.add_argument("--detection-stream-path", type=str, default="", help="Path to latest_tracks.json. Required with --no-static-stream.")
    parser.add_argument("--stream-max-age-s", type=float, default=1.0e9, help="Large default supports deterministic static stream files.")
    parser.add_argument("--no-stream-allow-stale", action="store_true", help="Do not pass --detection_stream_allow_stale to the child.")

    parser.add_argument("--static-stream", dest="static_stream", action="store_true", default=True)
    parser.add_argument("--no-static-stream", dest="static_stream", action="store_false")
    parser.add_argument("--stream-track-id", type=str, default="sim_static_target")
    parser.add_argument("--stream-confidence", type=float, default=1.0)
    parser.add_argument("--stream-bbox", type=float, nargs=4, default=(300.0, 120.0, 380.0, 360.0))
    parser.add_argument("--stream-target-body-x", type=float, default=3.2)
    parser.add_argument("--stream-target-body-y", type=float, default=0.0)
    parser.add_argument("--stream-depth-m", type=float, default=3.2)
    parser.add_argument("--stream-depth-valid-ratio", type=float, default=1.0)

    parser.add_argument("--forward-segment-m", type=float, default=0.35)
    parser.add_argument("--forward-speed-mps", type=float, default=0.35)
    parser.add_argument("--turn-speed-radps", type=float, default=0.50)
    parser.add_argument("--turn-angle-max-deg", type=float, default=35.0)
    parser.add_argument("--blind-plan-actions", type=int, default=1)
    parser.add_argument("--replan-every-actions", type=int, default=2)
    parser.add_argument("--emergency-front-stop-m", type=float, default=0.35)
    parser.add_argument("--action-duration-max-s", type=float, default=1.0)

    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments after -- are passed through to smoke_test_yolo_person_nav.py, e.g. -- --scenario front_box --max_steps 40.",
    )
    return parser


def _clean_remainder(values: Sequence[str]) -> list[str]:
    args = list(values)
    if args and args[0] == "--":
        args = args[1:]
    return args


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _tail_text(path: Path, *, max_lines: int) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, int(max_lines)) :]
    except OSError:
        return []


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()


def _write_static_stream(args: argparse.Namespace, run_dir: Path) -> Path:
    stream_path = Path(args.detection_stream_path).expanduser() if str(args.detection_stream_path).strip() else run_dir / "stream" / "latest_tracks.json"
    track = TrackDetection(
        track_id=str(args.stream_track_id),
        class_name=str(args.target_class),
        confidence=float(args.stream_confidence),
        bbox_xyxy=tuple(float(v) for v in args.stream_bbox),
        target_body_xy=(float(args.stream_target_body_x), float(args.stream_target_body_y)),
        depth_m=float(args.stream_depth_m),
        depth_valid_ratio=float(args.stream_depth_valid_ratio),
        metadata={"producer": "smoke_test_yolo_stepwise_nav", "mode": "static_isaac_stepwise_stream"},
    )
    frame = DetectionStreamFrame(
        timestamp_s=time.time(),
        frame_id="static-stepwise-0",
        image_shape=(480, 640, 3),
        source="static_isaac_stepwise_stream",
        tracks=(track,),
        metadata={"note": "Deterministic stream target for Isaac stepwise smoke tests."},
    )
    write_detection_stream_frame_atomic(frame, stream_path, indent=2)
    return stream_path


def _run_command(command: list[str], *, run_dir: Path, timeout_s: float, poll_interval_s: float, tail_lines: int) -> dict[str, Any]:
    stdout_path = run_dir / "child_stdout.log"
    stderr_path = run_dir / "child_stderr.log"
    command_path = run_dir / "child_command.json"
    result_path = run_dir / "child_result.json"
    _write_json_atomic(command_path, {"status": "starting", "command": command})

    start = time.monotonic()
    timed_out = False
    proc: subprocess.Popen[Any] | None = None
    launch_error: str | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout_f,
                stderr=stderr_f,
                start_new_session=True,
            )
            _write_json_atomic(command_path, {"status": "running", "pid": proc.pid, "command": command})
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if timeout_s > 0 and elapsed >= timeout_s:
                    timed_out = True
                    _terminate_process_group(proc)
                    break
                time.sleep(max(0.1, float(poll_interval_s)))
    except KeyboardInterrupt:
        if proc is not None:
            _terminate_process_group(proc)
        raise
    except Exception as exc:
        launch_error = repr(exc)
        if proc is not None:
            _terminate_process_group(proc)

    elapsed = time.monotonic() - start
    returncode = int(proc.returncode) if proc is not None and proc.returncode is not None else None
    if launch_error is not None:
        status = "launch_error"
        effective_returncode = -1
    elif timed_out:
        status = "timeout"
        effective_returncode = returncode if returncode is not None else -9
    elif returncode == 0:
        status = "ok"
        effective_returncode = 0
    else:
        status = "failed"
        effective_returncode = returncode if returncode is not None else -1

    result = {
        "status": status,
        "returncode": int(effective_returncode),
        "elapsed_s": float(elapsed),
        "timed_out": bool(timed_out),
        "launch_error": launch_error,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": _tail_text(stdout_path, max_lines=tail_lines),
        "stderr_tail": _tail_text(stderr_path, max_lines=tail_lines),
    }
    _write_json_atomic(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    passthrough = _clean_remainder(args.script_args)

    run_id = str(args.run_id).strip() or time.strftime("stepwise_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.static_stream):
        stream_path = _write_static_stream(args, run_dir)
    else:
        if not str(args.detection_stream_path).strip():
            raise SystemExit("--detection-stream-path is required with --no-static-stream")
        stream_path = Path(args.detection_stream_path).expanduser()

    child_output_dir = run_dir / "yolo_stepwise_nav"
    child_perception_dir = run_dir / "perception"
    cmd = [
        sys.executable,
        str(Path(args.smoke_script)),
        "--controller_mode",
        "stepwise",
        "--target_source",
        "stream",
        "--target_class",
        str(args.target_class),
        "--detection_stream_path",
        str(stream_path),
        "--detection_stream_max_age_s",
        str(float(args.stream_max_age_s)),
        "--output_dir",
        str(child_output_dir),
        "--perception_image_dir",
        str(child_perception_dir),
        "--stepwise_forward_segment_m",
        str(float(args.forward_segment_m)),
        "--stepwise_forward_speed_mps",
        str(float(args.forward_speed_mps)),
        "--stepwise_turn_speed_radps",
        str(float(args.turn_speed_radps)),
        "--stepwise_turn_angle_max_deg",
        str(float(args.turn_angle_max_deg)),
        "--stepwise_blind_plan_actions",
        str(int(args.blind_plan_actions)),
        "--stepwise_replan_every_actions",
        str(int(args.replan_every_actions)),
        "--stepwise_emergency_front_stop_m",
        str(float(args.emergency_front_stop_m)),
        "--stepwise_action_duration_max_s",
        str(float(args.action_duration_max_s)),
    ]
    if not bool(args.no_stream_allow_stale):
        cmd.append("--detection_stream_allow_stale")
    if bool(args.static_stream):
        cmd.append("--stepwise_static_stream_world_target")
    cmd.extend(passthrough)

    summary_path = run_dir / "stepwise_summary.json"
    _write_json_atomic(
        summary_path,
        {
            "status": "running",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "stream_path": str(stream_path),
            "static_stream": bool(args.static_stream),
            "command": cmd,
        },
    )
    print(f"[INFO]: stepwise Isaac stream run dir: {run_dir}", flush=True)
    print(f"[INFO]: detection stream path: {stream_path}", flush=True)
    print(f"[INFO]: summary: {summary_path}", flush=True)

    result = _run_command(
        cmd,
        run_dir=run_dir,
        timeout_s=float(args.mode_timeout_s),
        poll_interval_s=float(args.poll_interval_s),
        tail_lines=int(args.tail_lines),
    )
    _write_json_atomic(
        summary_path,
        {
            "status": result["status"],
            "run_id": run_id,
            "run_dir": str(run_dir),
            "stream_path": str(stream_path),
            "static_stream": bool(args.static_stream),
            "child_result": result,
        },
    )
    print(f"[INFO]: child status={result['status']} returncode={result['returncode']}", flush=True)
    if int(result["returncode"]) != 0:
        print("[ERROR]: child stdout tail:", flush=True)
        print("\n".join(result.get("stdout_tail", [])[-20:]), flush=True)
        print("[ERROR]: child stderr tail:", flush=True)
        print("\n".join(result.get("stderr_tail", [])[-20:]), flush=True)
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
