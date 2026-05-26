#!/usr/bin/env python3
"""Compare Isaac detect-and-follow target paths: direct YOLO/RGB-D vs stream client.

This script is intentionally only an orchestrator. It does not import Isaac Lab.
It launches ``scripts/smoke_test_yolo_person_nav.py`` once with the previous
in-loop target extraction architecture and once with ``--target_source=stream``.

The wrapper is deliberately crash/hang-tolerant: it creates the comparison run
folder before launching Isaac, writes command/status JSON before each child run,
redirects stdout/stderr into per-mode logs, and periodically rewrites
``compare_summary.json`` while the run is in progress. If Isaac hangs on shutdown,
``--mode-timeout-s`` lets the wrapper terminate the child and still preserve the
logs and result JSON.
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
        description="Run Isaac target-navigation smoke tests in direct and detection-stream modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("both", "direct", "stream"), default="both")
    parser.add_argument("--smoke-script", type=str, default=str(PROJECT_ROOT / "scripts" / "smoke_test_yolo_person_nav.py"))
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "outputs" / "isaac_stream_compare"))
    parser.add_argument("--run-id", type=str, default="", help="Run id folder. Default: timestamped compare_YYYYMMDD_HHMMSS.")
    parser.add_argument("--continue-on-fail", action="store_true", help="Run both modes even if one mode fails.")
    parser.add_argument(
        "--mode-timeout-s",
        type=float,
        default=1800.0,
        help="Per-mode child-process timeout. Use 0 to disable. Useful when Isaac prints success but hangs on shutdown.",
    )
    parser.add_argument("--poll-interval-s", type=float, default=1.0, help="Child-process polling interval.")
    parser.add_argument("--tail-lines", type=int, default=80, help="Lines of stdout/stderr to embed in result JSON.")

    parser.add_argument("--direct-target-source", choices=("auto", "yolo", "fallback"), default="fallback")
    parser.add_argument("--target-class", type=str, default="person")
    parser.add_argument("--stream-track-id", type=str, default="sim_static_target")
    parser.add_argument("--stream-confidence", type=float, default=1.0)
    parser.add_argument("--stream-bbox", type=float, nargs=4, default=(300.0, 120.0, 380.0, 360.0))
    parser.add_argument("--stream-target-body-x", type=float, default=3.2, help="Body-frame target x used by stream mode.")
    parser.add_argument("--stream-target-body-y", type=float, default=0.0, help="Body-frame target y used by stream mode.")
    parser.add_argument("--stream-depth-m", type=float, default=3.2)
    parser.add_argument("--stream-depth-valid-ratio", type=float, default=1.0)
    parser.add_argument("--stream-max-age-s", type=float, default=1.0e9, help="Large by default for deterministic static smoke files.")

    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments passed through to smoke_test_yolo_person_nav.py. "
            "Use '--' before them, e.g. -- --scenario clear --max_steps 200."
        ),
    )
    return parser


def _clean_remainder(args: Sequence[str]) -> list[str]:
    out = list(args)
    if out and out[0] == "--":
        out = out[1:]
    return out


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _now_s() -> float:
    return float(time.monotonic())


def _write_static_stream(args: argparse.Namespace, run_dir: Path) -> Path:
    stream_path = run_dir / "stream" / "latest_tracks.json"
    track = TrackDetection(
        track_id=str(args.stream_track_id),
        class_name=str(args.target_class),
        confidence=float(args.stream_confidence),
        bbox_xyxy=tuple(float(v) for v in args.stream_bbox),
        target_body_xy=(float(args.stream_target_body_x), float(args.stream_target_body_y)),
        depth_m=float(args.stream_depth_m),
        depth_valid_ratio=float(args.stream_depth_valid_ratio),
        status="tracked",
        metadata={"producer": "smoke_test_yolo_person_nav_stream_compare", "mode": "static_isaac_smoke"},
    )
    frame = DetectionStreamFrame(
        timestamp_s=time.monotonic(),
        frame_id="static_isaac_smoke_0",
        source="static_isaac_smoke_stream",
        image_shape=(480, 640, 3),
        tracks=(track,),
        metadata={"note": "Deterministic static stream target for Isaac smoke comparison."},
    )
    write_detection_stream_frame_atomic(frame, stream_path, indent=2)
    return stream_path


def _tail_text(path: Path, *, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-max_lines:]


def _terminate_process_group(proc: subprocess.Popen[Any], *, grace_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _summarize_results(
    *,
    run_id: str,
    run_dir: Path,
    mode: str,
    script_args: list[str],
    results: list[dict[str, Any]],
    status: str,
    skipped: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": mode,
        "status": status,
        "script_args": script_args,
        "results": results,
        "skipped": skipped or [],
        "updated_at_s": _now_s(),
    }


def _write_summary(
    *,
    run_id: str,
    run_dir: Path,
    mode: str,
    script_args: list[str],
    results: list[dict[str, Any]],
    status: str,
    skipped: list[str] | None = None,
) -> None:
    summary = _summarize_results(
        run_id=run_id,
        run_dir=run_dir,
        mode=mode,
        script_args=script_args,
        results=results,
        status=status,
        skipped=skipped,
    )
    _write_json_atomic(run_dir / "compare_summary.json", summary)


def _run_command(
    name: str,
    command: list[str],
    *,
    run_id: str,
    run_dir: Path,
    mode: str,
    script_args: list[str],
    results: list[dict[str, Any]],
    timeout_s: float,
    poll_interval_s: float,
    tail_lines: int,
) -> dict[str, Any]:
    mode_dir = run_dir / name
    mode_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = mode_dir / "stdout.log"
    stderr_path = mode_dir / "stderr.log"
    command_path = mode_dir / "command.json"
    result_path = run_dir / f"{name}_result.json"

    command_record = {
        "name": name,
        "command": command,
        "cwd": str(PROJECT_ROOT),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at_s": _now_s(),
        "status": "starting",
    }
    _write_json_atomic(command_path, command_record)

    running_result: dict[str, Any] = {
        "name": name,
        "status": "running",
        "returncode": None,
        "elapsed_s": 0.0,
        "timed_out": False,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command_path": str(command_path),
    }
    _write_json_atomic(result_path, running_result)
    _write_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode=mode,
        script_args=script_args,
        results=[*results, running_result],
        status=f"running_{name}",
    )

    print(f"[INFO]: running {name} mode", flush=True)
    print("[INFO]: " + " ".join(command), flush=True)
    print(f"[INFO]: {name} stdout log: {stdout_path}", flush=True)
    print(f"[INFO]: {name} stderr log: {stderr_path}", flush=True)
    print(f"[INFO]: {name} command record: {command_path}", flush=True)

    start = time.monotonic()
    timed_out = False
    launch_error: str | None = None
    proc: subprocess.Popen[Any] | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f:
            proc = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout_f,
                stderr=stderr_f,
                start_new_session=True,
            )
            command_record["pid"] = int(proc.pid)
            command_record["status"] = "running"
            _write_json_atomic(command_path, command_record)

            while proc.poll() is None:
                elapsed = time.monotonic() - start
                if timeout_s > 0 and elapsed >= timeout_s:
                    timed_out = True
                    print(f"[ERROR]: {name} mode timed out after {elapsed:.1f}s; terminating child process", flush=True)
                    _terminate_process_group(proc)
                    break
                time.sleep(max(0.1, float(poll_interval_s)))
    except KeyboardInterrupt:
        if proc is not None:
            _terminate_process_group(proc)
        raise
    except Exception as exc:  # preserve diagnostics even on launch/file errors
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
        "name": name,
        "status": status,
        "returncode": int(effective_returncode),
        "elapsed_s": float(elapsed),
        "timed_out": bool(timed_out),
        "launch_error": launch_error,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command_path": str(command_path),
        "stdout_tail": _tail_text(stdout_path, max_lines=tail_lines),
        "stderr_tail": _tail_text(stderr_path, max_lines=tail_lines),
    }
    _write_json_atomic(result_path, result)

    command_record["status"] = status
    command_record["returncode"] = int(effective_returncode)
    command_record["elapsed_s"] = float(elapsed)
    _write_json_atomic(command_path, command_record)

    if effective_returncode != 0:
        print(f"[ERROR]: {name} mode ended with status={status} returncode={effective_returncode}", flush=True)
        tail = result.get("stdout_tail") or []
        if tail:
            print(f"[ERROR]: {name} stdout tail:", flush=True)
            print("\n".join(str(line) for line in tail[-20:]), flush=True)
        err_tail = result.get("stderr_tail") or []
        if err_tail:
            print(f"[ERROR]: {name} stderr tail:", flush=True)
            print("\n".join(str(line) for line in err_tail[-20:]), flush=True)
    else:
        print(f"[INFO]: {name} mode completed successfully in {elapsed:.1f}s", flush=True)
    return result


def _append_output_args(base_cmd: list[str], *, output_dir: Path, perception_dir: Path) -> list[str]:
    # smoke_test_yolo_person_nav.py writes grid images and nav_lidar_trace.csv
    # under --output_dir. Perception images are controlled separately. Keep both
    # inside the comparison run directory so direct/stream artifacts are grouped.
    return [
        *base_cmd,
        "--output_dir",
        str(output_dir),
        "--perception_image_dir",
        str(perception_dir),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    script_args = _clean_remainder(args.script_args)

    run_id = str(args.run_id).strip() or time.strftime("compare_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    smoke_script = str(Path(args.smoke_script))
    common = [sys.executable, smoke_script, *script_args]
    results: list[dict[str, Any]] = []

    _write_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode=str(args.mode),
        script_args=script_args,
        results=results,
        status="starting",
        skipped=[],
    )
    print(f"[INFO]: comparison run dir: {run_dir}", flush=True)
    print(f"[INFO]: comparison summary: {run_dir / 'compare_summary.json'}", flush=True)

    try:
        if args.mode in {"both", "direct"}:
            direct_cmd = _append_output_args(
                [
                    *common,
                    "--target_source",
                    str(args.direct_target_source),
                    "--target_class",
                    str(args.target_class),
                ],
                output_dir=run_dir / "direct" / "yolo_person_nav",
                perception_dir=run_dir / "direct" / "perception",
            )
            direct_result = _run_command(
                "direct",
                direct_cmd,
                run_id=run_id,
                run_dir=run_dir,
                mode=str(args.mode),
                script_args=script_args,
                results=results,
                timeout_s=float(args.mode_timeout_s),
                poll_interval_s=float(args.poll_interval_s),
                tail_lines=int(args.tail_lines),
            )
            results.append(direct_result)
            _write_summary(
                run_id=run_id,
                run_dir=run_dir,
                mode=str(args.mode),
                script_args=script_args,
                results=results,
                status="direct_done",
                skipped=["stream"] if args.mode == "both" and int(direct_result["returncode"]) != 0 and not bool(args.continue_on_fail) else [],
            )
            if int(direct_result["returncode"]) != 0 and not bool(args.continue_on_fail):
                print(f"[INFO]: comparison summary: {run_dir / 'compare_summary.json'}", flush=True)
                return int(direct_result["returncode"])

        if args.mode in {"both", "stream"}:
            stream_path = _write_static_stream(args, run_dir)
            print(f"[INFO]: wrote static stream file: {stream_path}", flush=True)
            _write_summary(
                run_id=run_id,
                run_dir=run_dir,
                mode=str(args.mode),
                script_args=script_args,
                results=results,
                status="stream_file_written",
            )
            stream_cmd = _append_output_args(
                [
                    *common,
                    "--target_source",
                    "stream",
                    "--target_class",
                    str(args.target_class),
                    "--detection_stream_path",
                    str(stream_path),
                    "--detection_stream_max_age_s",
                    str(float(args.stream_max_age_s)),
                    "--detection_stream_allow_stale",
                ],
                output_dir=run_dir / "stream" / "yolo_person_nav",
                perception_dir=run_dir / "stream" / "perception",
            )
            stream_result = _run_command(
                "stream",
                stream_cmd,
                run_id=run_id,
                run_dir=run_dir,
                mode=str(args.mode),
                script_args=script_args,
                results=results,
                timeout_s=float(args.mode_timeout_s),
                poll_interval_s=float(args.poll_interval_s),
                tail_lines=int(args.tail_lines),
            )
            results.append(stream_result)
    except KeyboardInterrupt:
        _write_summary(
            run_id=run_id,
            run_dir=run_dir,
            mode=str(args.mode),
            script_args=script_args,
            results=results,
            status="interrupted",
        )
        print(f"[ERROR]: interrupted; partial comparison summary: {run_dir / 'compare_summary.json'}", flush=True)
        raise
    except Exception as exc:
        _write_summary(
            run_id=run_id,
            run_dir=run_dir,
            mode=str(args.mode),
            script_args=script_args,
            results=results,
            status=f"wrapper_error:{type(exc).__name__}",
        )
        print(f"[ERROR]: wrapper failed: {exc!r}", flush=True)
        print(f"[ERROR]: partial comparison summary: {run_dir / 'compare_summary.json'}", flush=True)
        return 1

    final_status = "ok" if results and all(int(r["returncode"]) == 0 for r in results) else "failed"
    _write_summary(
        run_id=run_id,
        run_dir=run_dir,
        mode=str(args.mode),
        script_args=script_args,
        results=results,
        status=final_status,
    )
    print(f"[INFO]: comparison summary: {run_dir / 'compare_summary.json'}", flush=True)
    return 0 if final_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
