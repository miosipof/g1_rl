#!/usr/bin/env python3
"""Run Isaac person-target navigation with the synchronized stepwise controller.

This is the stepwise-person entry point.  Unlike
``smoke_test_yolo_stepwise_nav.py`` it does not write a synthetic detection
stream by default.  It delegates target acquisition to
``smoke_test_yolo_person_nav.py`` and enables the existing person-target
perception/tracking path:

    RGB-D/YOLO or projected proxy-person fallback
    -> TargetTracker / last-known target handling
    -> StepwiseTargetNavPolicyRuntime
    -> G1 material-point command proxy

Use this when the proxy person is not YOLO-detectable but the old person smoke
script can still provide its projected fallback target.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch smoke_test_yolo_person_nav.py in synchronized stepwise person-tracking mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--smoke-script", type=str, default=str(PROJECT_ROOT / "scripts" / "smoke_test_yolo_person_nav.py"))
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "outputs" / "isaac_stepwise_person"))
    parser.add_argument("--run-id", type=str, default="", help="Run id folder. Default: stepwise_person_YYYYMMDD_HHMMSS.")
    parser.add_argument("--mode-timeout-s", type=float, default=0.0, help="Optional child-process timeout. 0 disables timeout.")
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--tail-lines", type=int, default=80)

    parser.add_argument("--target-class", type=str, default="person")
    parser.add_argument(
        "--person-target-source",
        choices=("auto", "yolo", "fallback"),
        default="auto",
        help="Target source passed to the child. auto runs YOLO then projected proxy-person fallback; fallback skips YOLO.",
    )
    parser.add_argument("--no-fallback-synthetic-bbox", action="store_true", help="Disable projected proxy-person fallback in the child.")
    parser.add_argument("--actual-detections-only", action="store_true", help="Strict YOLO-only target acquisition: force --target_source=yolo, disable projected synthetic fallback, and assert initial target.")
    parser.add_argument("--keep-existing-run-dir", action="store_true", help="Do not delete stale logs/artifacts from an existing run directory before launch.")
    parser.add_argument("--assert-initial-target", action="store_true", default=True, help="Assert that initial target projection succeeds.")
    parser.add_argument("--no-assert-initial-target", dest="assert_initial_target", action="store_false")
    parser.add_argument("--redetect-every", type=int, default=1, help="Refresh person target every N stepwise cycles through old RGB-D/YOLO/fallback logic.")
    parser.add_argument("--target-occlusion-grace-steps", type=int, default=20, help="How many missed refreshes can continue toward last known person target.")
    parser.add_argument("--target-smoothing-alpha", type=float, default=1.0, help="Tracker update alpha. 1.0 accepts current measurement directly.")
    parser.add_argument("--target-max-jump-m", type=float, default=1.25)

    parser.add_argument("--forward-segment-m", type=float, default=0.35)
    parser.add_argument("--forward-speed-mps", type=float, default=0.35)
    parser.add_argument("--turn-speed-radps", type=float, default=0.50)
    parser.add_argument("--turn-angle-max-deg", type=float, default=35.0)
    parser.add_argument("--blind-plan-actions", type=int, default=1)
    parser.add_argument("--replan-every-actions", "--stepwise_replan_every_actions", "--stepwise-replan-every-actions", dest="replan_every_actions", type=int, default=0)
    parser.add_argument("--replan-every-forward-m", "--stepwise_replan_every_forward_m", "--stepwise-replan-every-forward-m", dest="replan_every_forward_m", type=float, default=0.7)
    parser.add_argument("--emergency-front-stop-m", type=float, default=0.35)
    parser.add_argument("--sweep-start-ignore-m", "--stepwise_sweep_start_ignore_m", "--stepwise-sweep-start-ignore-m", dest="sweep_start_ignore_m", type=float, default=0.05)
    parser.add_argument("--action-duration-max-s", type=float, default=0.0, help="Compatibility; synchronized stepwise controller does not clip action durations.")
    parser.add_argument("--save-rgb-screenshots", "--save-perception-images", dest="save_rgb_screenshots", action="store_true")

    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed through to smoke_test_yolo_person_nav.py.")
    return parser


def _clean_remainder(values: Sequence[str]) -> list[str]:
    args = list(values)
    if args and args[0] == "--":
        args = args[1:]
    # ``argparse.REMAINDER`` preserves accidental blank/whitespace-only tokens.
    # Passing those through to the child script produces opaque failures such as
    # ``unrecognized arguments:  `` and can leave stale result metadata in a
    # reused run directory.  Strip only empty shell artifacts, not normal args.
    return [str(a) for a in args if str(a).strip()]


def _prepare_run_dir(run_dir: Path, *, keep_existing: bool = False) -> None:
    if keep_existing:
        run_dir.mkdir(parents=True, exist_ok=True)
        return
    if run_dir.exists():
        for name in (
            "child_stdout.log",
            "child_stderr.log",
            "child_command.json",
            "child_result.json",
            "stepwise_person_summary.json",
        ):
            path = run_dir / name
            if path.exists():
                path.unlink()
        for name in ("yolo_stepwise_person_nav", "perception"):
            path = run_dir / name
            if path.exists():
                shutil.rmtree(path)
    run_dir.mkdir(parents=True, exist_ok=True)


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
            proc = subprocess.Popen(command, cwd=str(PROJECT_ROOT), stdout=stdout_f, stderr=stderr_f, start_new_session=True)
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
    _write_json_atomic(command_path, {"status": "completed" if int(effective_returncode) == 0 else status, "returncode": int(effective_returncode), "elapsed_s": float(elapsed), "pid": None if proc is None else proc.pid, "command": command})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    if bool(args.actual_detections_only):
        args.person_target_source = "yolo"
        args.no_fallback_synthetic_bbox = True
        args.assert_initial_target = True
    passthrough = _clean_remainder(args.script_args)

    run_id = str(args.run_id).strip() or time.strftime("stepwise_person_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root).expanduser() / run_id
    _prepare_run_dir(run_dir, keep_existing=bool(args.keep_existing_run_dir))
    child_output_dir = run_dir / "yolo_stepwise_person_nav"
    child_perception_dir = run_dir / "perception"

    cmd = [
        sys.executable,
        str(Path(args.smoke_script)),
        "--controller_mode", "stepwise",
        "--target_source", str(args.person_target_source),
        "--target_class", str(args.target_class),
        "--output_dir", str(child_output_dir),
        "--perception_image_dir", str(child_perception_dir),
        "--stepwise_refresh_person_target",
        "--redetect_every", str(int(args.redetect_every)),
        "--target_occlusion_grace_steps", str(int(args.target_occlusion_grace_steps)),
        "--target_smoothing_alpha", str(float(args.target_smoothing_alpha)),
        "--target_max_jump_m", str(float(args.target_max_jump_m)),
        "--stepwise_forward_segment_m", str(float(args.forward_segment_m)),
        "--stepwise_forward_speed_mps", str(float(args.forward_speed_mps)),
        "--stepwise_turn_speed_radps", str(float(args.turn_speed_radps)),
        "--stepwise_turn_angle_max_deg", str(float(args.turn_angle_max_deg)),
        "--stepwise_blind_plan_actions", str(int(args.blind_plan_actions)),
        "--stepwise_replan_every_actions", str(int(args.replan_every_actions)),
        "--stepwise_replan_every_forward_m", str(float(args.replan_every_forward_m)),
        "--stepwise_emergency_front_stop_m", str(float(args.emergency_front_stop_m)),
        "--stepwise_sweep_start_ignore_m", str(float(args.sweep_start_ignore_m)),
        "--stepwise_action_duration_max_s", str(float(args.action_duration_max_s)),
    ]
    if not bool(args.no_fallback_synthetic_bbox):
        cmd.append("--fallback_synthetic_bbox")
    else:
        cmd.append("--no_fallback_synthetic_bbox")
    if bool(args.assert_initial_target):
        cmd.append("--assert_target")
    if bool(args.save_rgb_screenshots):
        cmd.append("--save_perception_images")
    cmd.extend(passthrough)

    summary_path = run_dir / "stepwise_person_summary.json"
    _write_json_atomic(summary_path, {
        "status": "running",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "target_mode": "person_perception_tracking",
        "child_target_source": str(args.person_target_source),
        "actual_detections_only": bool(args.actual_detections_only),
        "fallback_synthetic_bbox_enabled": not bool(args.no_fallback_synthetic_bbox),
        "command": cmd,
    })
    print(f"[INFO]: stepwise Isaac person run dir: {run_dir}", flush=True)
    print(f"[INFO]: target acquisition delegated to child --target_source={args.person_target_source} with --stepwise_refresh_person_target", flush=True)
    print(f"[INFO]: summary: {summary_path}", flush=True)

    result = _run_command(cmd, run_dir=run_dir, timeout_s=float(args.mode_timeout_s), poll_interval_s=float(args.poll_interval_s), tail_lines=int(args.tail_lines))
    _write_json_atomic(summary_path, {
        "status": "completed" if int(result["returncode"]) == 0 else result["status"],
        "returncode": int(result["returncode"]),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "target_mode": "person_perception_tracking",
        "child_target_source": str(args.person_target_source),
        "actual_detections_only": bool(args.actual_detections_only),
        "fallback_synthetic_bbox_enabled": not bool(args.no_fallback_synthetic_bbox),
        "trace_path": str(child_output_dir / "nav_lidar_trace.csv"),
        "stepwise_trace_path": str(child_output_dir / "stepwise_trace.jsonl"),
        "perception_image_dir": str(child_perception_dir),
        "child_result": result,
    })
    print(f"[INFO]: child status={result['status']} returncode={result['returncode']}", flush=True)
    if int(result["returncode"]) != 0:
        print("[ERROR]: child stdout tail:", flush=True)
        print("\n".join(result.get("stdout_tail", [])[-20:]), flush=True)
        print("[ERROR]: child stderr tail:", flush=True)
        print("\n".join(result.get("stderr_tail", [])[-20:]), flush=True)
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
