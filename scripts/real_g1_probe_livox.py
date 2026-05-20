#!/usr/bin/env python3
"""Probe the LiDAR adapter path without robot motion.

Default mode is synthetic so this script can run on a development laptop. Use
``--source npy`` for replayed point clouds, or ``--source live-vibes`` on the
robot/Jetson with unitree_g1_vibes + Livox-SDK2 installed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters import load_adapter_config
from adapters.real_livox import (
    LivoxAdapterConfig,
    NpyLivoxReplayAdapter,
    UnitreeVibesLivox2Adapter,
    UnitreeVibesLivox2AdapterConfig,
    lidar_frame_stats,
    make_lidar_frame_from_config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    return _repo_root() / "configs" / "real_g1.yaml"


def _synthetic_livox_points() -> np.ndarray:
    # Points are in Livox source frame. The default real_g1.yaml T_body_lidar
    # translates z by +0.10, so these become body-frame obstacle-height points.
    return np.array(
        [
            [1.0, 0.0, 0.20],
            [2.0, 0.5, 0.30],
            [3.0, -0.6, 0.40],
            [0.05, 0.0, 0.30],   # filtered: too close
            [9.0, 0.0, 0.40],    # filtered: too far
            [2.0, 0.0, -0.20],   # filtered by z after transform if normal mount
        ],
        dtype=np.float32,
    )


def _frame_from_source(args: argparse.Namespace, adapter_config: LivoxAdapterConfig):
    if args.source == "synthetic":
        return make_lidar_frame_from_config(
            timestamp_s=time.monotonic(),
            points_lidar_xyz=_synthetic_livox_points(),
            config=adapter_config,
            metadata={"source": "synthetic_probe"},
        ), None

    if args.source == "npy":
        if args.npy_path is None:
            raise SystemExit("--npy-path is required when --source npy")
        adapter = NpyLivoxReplayAdapter(args.npy_path, adapter_config, loop=args.loop_npy)
        return adapter.read_lidar_frame(), adapter

    if args.source == "live-vibes":
        if not args.livox_config:
            raise SystemExit("--livox-config is required when --source live-vibes")
        if not args.host_ip:
            raise SystemExit("--host-ip is required when --source live-vibes")
        live_cfg = UnitreeVibesLivox2AdapterConfig(
            livox_config_path=args.livox_config,
            host_ip=args.host_ip,
            adapter_config=adapter_config,
            unitree_vibes_dir=args.unitree_vibes_dir,
            frame_time_s=args.frame_time,
            frame_packets=args.frame_packets,
            read_timeout_s=args.timeout,
        )
        adapter = UnitreeVibesLivox2Adapter(live_cfg)
        adapter.start()
        try:
            return adapter.read_lidar_frame(timeout_s=args.timeout), adapter
        except Exception:
            adapter.stop()
            raise

    raise AssertionError(f"unhandled source {args.source!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--source", choices=("synthetic", "npy", "live-vibes"), default="synthetic")
    parser.add_argument("--json", action="store_true", help="Print JSON diagnostics only.")
    parser.add_argument("--npy-path", type=Path, default=None, help="Path to .npy with Nx3 or FxNx3 Livox-frame points.")
    parser.add_argument("--loop-npy", action="store_true", help="Loop replay source when reading multiple frames.")
    parser.add_argument("--unitree-vibes-dir", default=None, help="Path to unitree_g1_vibes repo for live-vibes mode.")
    parser.add_argument("--livox-config", default=None, help="Livox-SDK2 JSON config path for live-vibes mode.")
    parser.add_argument("--host-ip", default=None, help="Host IP passed to Livox2 in live-vibes mode.")
    parser.add_argument("--frame-time", type=float, default=0.20, help="Livox packet aggregation window.")
    parser.add_argument("--frame-packets", type=int, default=120, help="Livox packet aggregation count.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Timeout waiting for a live LiDAR frame.")
    args = parser.parse_args()

    cfg = load_adapter_config(args.config)
    adapter_config = LivoxAdapterConfig.from_lidar_config(cfg.lidar)
    adapter = None
    try:
        frame, adapter = _frame_from_source(args, adapter_config)
        payload: dict[str, Any] = {
            "ok": True,
            "config_path": str(Path(args.config).resolve()),
            "profile_name": cfg.profile_name,
            "source": args.source,
            "lidar_config": {
                "source_frame_id": adapter_config.source_frame_id,
                "frame_id": adapter_config.frame_id,
                "mount_mode": adapter_config.mount_mode,
                "min_range_m": adapter_config.min_range_m,
                "max_range_m": adapter_config.max_range_m,
                "z_min_m": adapter_config.z_min_m,
                "z_max_m": adapter_config.z_max_m,
                "max_points": adapter_config.max_points,
            },
            "frame": lidar_frame_stats(frame),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"[INFO]: source={args.source}")
            print(f"[INFO]: profile={cfg.profile_name}")
            print(f"[INFO]: mount={adapter_config.mount_mode} frame={frame.frame_id} source_frame={frame.source_frame_id}")
            print(f"[INFO]: points={frame.point_count} metadata={dict(frame.metadata)}")
            stats = payload["frame"]
            if stats["point_count"]:
                print(
                    "[INFO]: xyz min/max/mean: "
                    f"{stats.get('min_xyz_m')} / {stats.get('max_xyz_m')} / {stats.get('mean_xyz_m')}"
                )
        return 0
    finally:
        if adapter is not None and hasattr(adapter, "stop"):
            adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
