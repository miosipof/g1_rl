#!/usr/bin/env python3
"""Probe the live RealSense RGB-D adapter without moving the robot.

This is Step 9's first real-sensor executable.  It starts a RealSense stream,
captures one or more aligned RGB-D frames, converts them through
``adapters.real_realsense`` into ``RGBDFrame``, and prints compact diagnostics.
No Unitree locomotion, Livox, or bridge motion code is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from adapters import RealSenseRGBDAdapter, RealSenseRGBDAdapterConfig, load_adapter_config, rgbd_stats


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


def _save_frame_outputs(frame: Any, *, save_dir: Path, index: int) -> list[str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    rgb_path = save_dir / f"realsense_rgb_{index:04d}.png"
    depth_path = save_dir / f"realsense_depth_m_{index:04d}.npy"
    try:
        from PIL import Image

        Image.fromarray(frame.rgb).save(rgb_path)
        written.append(str(rgb_path))
    except Exception as exc:  # pragma: no cover - depends on optional Pillow runtime
        print(f"[WARN]: failed to save RGB PNG: {exc}", file=sys.stderr)
    np.save(depth_path, frame.depth_m)
    written.append(str(depth_path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--frames", type=int, default=1, help="Number of RGB-D frames to capture.")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="RealSense wait_for_frames timeout.")
    parser.add_argument("--no-spatial-filter", action="store_true", help="Disable RealSense spatial depth filter.")
    parser.add_argument("--no-temporal-filter", action="store_true", help="Disable RealSense temporal depth filter.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory for RGB PNG and depth .npy dumps.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    if args.timeout_ms <= 0:
        raise SystemExit("--timeout-ms must be > 0")

    cfg = load_adapter_config(args.config)
    rs_cfg = RealSenseRGBDAdapterConfig.from_camera_config(
        cfg.camera,
        serial_number=args.serial,
        use_spatial_filter=not args.no_spatial_filter,
        use_temporal_filter=not args.no_temporal_filter,
    )
    adapter = RealSenseRGBDAdapter(rs_cfg)

    rows: list[dict[str, Any]] = []
    start_s = time.monotonic()
    try:
        adapter.start()
        for i in range(args.frames):
            frame = adapter.read_frame(timeout_ms=args.timeout_ms)
            row = rgbd_stats(frame)
            row["index"] = i
            row["age_s_at_report"] = max(0.0, time.monotonic() - frame.timestamp_s)
            if args.save_dir is not None:
                row["saved_files"] = _save_frame_outputs(frame, save_dir=args.save_dir, index=i)
            rows.append(row)
    finally:
        adapter.stop()

    payload = {
        "ok": True,
        "config_path": str(args.config),
        "profile_name": cfg.profile_name,
        "frames": rows,
        "elapsed_s": time.monotonic() - start_s,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"[INFO]: config={args.config}")
        print(f"[INFO]: profile={cfg.profile_name}")
        print(f"[INFO]: captured_frames={len(rows)}")
        for row in rows:
            print(
                "[INFO]: frame={index} size={width}x{height} "
                "rgb_mean={rgb_mean:.1f} depth_valid={depth_valid_count} "
                "depth_min={depth_min_m} depth_mean={depth_mean_m} depth_max={depth_max_m}".format(**row)
            )
            if row.get("saved_files"):
                for path in row["saved_files"]:
                    print(f"[INFO]: saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
