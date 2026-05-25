#!/usr/bin/env python3
"""Diagnose camera/LiDAR/body-frame left-right conventions.

This script is read-only: it captures one RGB-D frame and/or one LiDAR frame,
runs optional YOLO target detection, projects the selected bbox into body-frame
2D, and saves explicit debug artifacts. It is intended to distinguish a pure
visualization mirror issue from a real transform/sign bug.

Expected project convention:
    body +X = forward
    body +Y = robot-left
    body +Z = up

Occupancy debug PNG convention:
    image top   = +X forward
    image right = +Y robot-left

Therefore an object on RGB-left can legitimately appear on image-right in the
occupancy PNG. The saved JSON states this explicitly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters import (
    RGBDDebugWriter,
    RealSenseRGBDAdapter,
    RealSenseRGBDAdapterConfig,
    load_adapter_config,
)
from adapters.g1_policy_runtime import LidarTargetNavPolicyRuntimeConfig, occupancy_grid_from_lidar_frame
from adapters.lidar_debug import save_occupancy_debug_image
from adapters.real_livox import (
    LivoxAdapterConfig,
    NpyLivoxReplayAdapter,
    UnitreeVibesLivox2Adapter,
    UnitreeVibesLivox2AdapterConfig,
    lidar_frame_stats,
)
from g1_nav_rl.perception.target_from_bbox import (
    TargetFromBBoxConfig,
    bbox_depth_to_target2d,
    make_forward_camera_T_body_camera,
    select_detection_for_target,
)
from g1_nav_rl.perception.yolo_detector import YoloDetector, YoloDetectorConfig, filter_detections


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "real_g1.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _side_from_pixel(u: float, width: int) -> str:
    center = 0.5 * float(width)
    if abs(float(u) - center) < 0.02 * float(width):
        return "center"
    return "rgb_left" if float(u) < center else "rgb_right"


def _side_from_body_y(y: float, *, eps: float = 0.03) -> str:
    if abs(float(y)) <= eps:
        return "body_center"
    return "robot_left(+Y)" if float(y) > 0.0 else "robot_right(-Y)"


def _camera_transform(frame: Any, *, use_config_camera_transform: bool) -> np.ndarray:
    matrix = np.asarray(frame.T_body_camera.matrix, dtype=np.float64)
    if use_config_camera_transform:
        return matrix
    # The real_g1.yaml camera matrix is currently a placeholder. By default use
    # the known forward-facing optical convention and only keep translation.
    return make_forward_camera_T_body_camera(translation_body=matrix[:3, 3])


def _read_lidar(args: argparse.Namespace, cfg: Any) -> Any | None:
    if args.lidar_source == "none":
        return None
    livox_cfg = LivoxAdapterConfig.from_lidar_config(cfg.lidar)
    if args.lidar_source == "npy":
        if args.livox_npy_path is None:
            raise SystemExit("--livox-npy-path is required with --lidar-source npy")
        adapter = NpyLivoxReplayAdapter(args.livox_npy_path, config=livox_cfg, loop=True)
        return adapter.read_lidar_frame()
    if args.lidar_source == "live-vibes":
        if not args.unitree_vibes_dir or not args.livox_config:
            raise SystemExit("--unitree-vibes-dir and --livox-config are required with --lidar-source live-vibes")
        adapter = UnitreeVibesLivox2Adapter(
            UnitreeVibesLivox2AdapterConfig(
                livox_config_path=str(args.livox_config),
                host_ip=str(args.host_ip),
                unitree_vibes_dir=str(args.unitree_vibes_dir),
                frame_time_s=float(args.livox_frame_time),
                frame_packets=int(args.livox_frame_packets),
                read_timeout_s=float(args.livox_timeout),
                adapter_config=livox_cfg,
            )
        )
        adapter.start()
        try:
            return adapter.read_lidar_frame(timeout_s=float(args.livox_timeout))
        finally:
            adapter.stop()
    raise SystemExit(f"unknown lidar source: {args.lidar_source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Adapter YAML config path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real_g1/frame_diagnosis"), help="Output directory.")
    parser.add_argument("--skip-rgbd", action="store_true", help="Do not capture RealSense RGB-D / run YOLO.")
    parser.add_argument("--realsense-serial", default=None, help="Optional RealSense serial number.")
    parser.add_argument("--realsense-timeout-ms", type=int, default=1000, help="RealSense wait timeout.")
    parser.add_argument("--no-realsense-spatial-filter", action="store_true", help="Disable RealSense spatial filter.")
    parser.add_argument("--no-realsense-temporal-filter", action="store_true", help="Disable RealSense temporal filter.")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name.")
    parser.add_argument("--yolo-device", default="cpu", help="YOLO device; CPU is preferred on the Jetson unless CUDA is validated.")
    parser.add_argument("--yolo-confidence", type=float, default=0.10, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO IoU threshold.")
    parser.add_argument("--target-class", default="person", help="YOLO class to diagnose, e.g. person or bottle.")
    parser.add_argument("--stand-off-m", type=float, default=0.80, help="Standoff used for projected target.")
    parser.add_argument("--target-depth-min-m", type=float, default=0.20, help="Minimum valid target depth.")
    parser.add_argument("--target-depth-max-m", type=float, default=8.0, help="Maximum valid target depth.")
    parser.add_argument("--depth-crop-frac", type=float, default=0.50, help="Central bbox depth crop fraction.")
    parser.add_argument("--depth-percentile", type=float, default=0.50, help="Depth percentile inside crop.")
    parser.add_argument("--use-config-camera-transform", action="store_true", help="Use real_g1.yaml T_body_camera exactly; default uses forward optical convention.")
    parser.add_argument("--lidar-source", choices=["none", "npy", "live-vibes"], default="live-vibes", help="LiDAR source for occupancy diagnosis.")
    parser.add_argument("--livox-npy-path", type=Path, default=None, help="Nx3 or FxNx3 Livox .npy replay path.")
    parser.add_argument("--unitree-vibes-dir", default=None, help="Path to unitree_g1_vibes repo for live-vibes LiDAR.")
    parser.add_argument("--livox-config", type=Path, default=None, help="Livox-SDK2 JSON config path for live-vibes.")
    parser.add_argument("--host-ip", default="192.168.123.222", help="Host IP passed to Livox2 live-vibes wrapper.")
    parser.add_argument("--livox-frame-time", type=float, default=0.20, help="Livox aggregation window.")
    parser.add_argument("--livox-frame-packets", type=int, default=120, help="Livox aggregation packet count.")
    parser.add_argument("--livox-timeout", type=float, default=2.0, help="Timeout waiting for live Livox frame.")
    parser.add_argument("--json", action="store_true", help="Print JSON diagnosis to stdout.")
    args = parser.parse_args()

    if args.realsense_timeout_ms <= 0:
        raise SystemExit("--realsense-timeout-ms must be positive")
    if args.livox_timeout <= 0.0:
        raise SystemExit("--livox-timeout must be positive")

    cfg = load_adapter_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis: dict[str, Any] = {
        "timestamp_s": time.monotonic(),
        "config": str(args.config),
        "coordinate_convention": {
            "body": "+X forward, +Y robot-left, +Z up",
            "occupancy_png": "top=+X forward, right=+Y robot-left, left=-Y robot-right",
            "camera_note": "RGB visual-left normally maps to body +Y, which appears on the right side of the top-down occupancy PNG.",
        },
    }

    frame = None
    detections = []
    selected = None
    target_result = None
    if not args.skip_rgbd:
        rs = RealSenseRGBDAdapter(
            RealSenseRGBDAdapterConfig.from_camera_config(
                cfg.camera,
                serial_number=args.realsense_serial,
                use_spatial_filter=not args.no_realsense_spatial_filter,
                use_temporal_filter=not args.no_realsense_temporal_filter,
            )
        )
        rs.start()
        try:
            frame = rs.read_frame(timeout_ms=int(args.realsense_timeout_ms))
        finally:
            rs.stop()

        detector = YoloDetector(YoloDetectorConfig(model=args.yolo_model, confidence=args.yolo_confidence, iou=args.yolo_iou, device=args.yolo_device, verbose=False))
        detections = filter_detections(
            detector.detect(frame.rgb),
            allowed_classes={str(args.target_class)},
            min_confidence=float(args.yolo_confidence),
        )
        selected = select_detection_for_target(detections, target_class=str(args.target_class))
        T_body_camera = _camera_transform(frame, use_config_camera_transform=args.use_config_camera_transform)
        target_cfg = TargetFromBBoxConfig(
            stand_off_m=float(args.stand_off_m),
            depth_min_m=float(args.target_depth_min_m),
            depth_max_m=float(args.target_depth_max_m),
            depth_crop_frac=float(args.depth_crop_frac),
            depth_percentile=float(args.depth_percentile),
        )
        if selected is not None:
            target_result = bbox_depth_to_target2d(selected, frame.depth_m, frame.intrinsics, T_body_camera, config=target_cfg)

        writer = RGBDDebugWriter(args.output_dir / "rgbd")
        writer.write_frame(step=0, frame=frame, detections=detections, source="diagnose_frames")
        detection_payload = []
        for det in detections:
            x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
            u = 0.5 * (x1 + x2)
            detection_payload.append(
                {
                    "class_name": det.class_name,
                    "confidence": float(det.confidence),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_center_u": float(u),
                    "rgb_side": _side_from_pixel(u, frame.width),
                }
            )
        target_payload = None
        if target_result is not None:
            object_xy_b = tuple(float(v) for v in target_result.object_xy_w)
            target_xy_b = tuple(float(v) for v in target_result.target.xy)
            target_payload = {
                "object_xy_b": list(object_xy_b),
                "object_side": _side_from_body_y(object_xy_b[1]),
                "target_xy_b": list(target_xy_b),
                "target_side": _side_from_body_y(target_xy_b[1]),
                "depth_m": float(target_result.depth_m),
            }
        diagnosis["rgbd"] = {
            "captured": True,
            "width": frame.width,
            "height": frame.height,
            "target_class": args.target_class,
            "detection_count": len(detections),
            "detections": detection_payload,
            "selected_detection": None
            if selected is None
            else {
                "class_name": selected.class_name,
                "confidence": float(selected.confidence),
                "bbox_xyxy": [float(v) for v in selected.bbox_xyxy],
            },
            "target_projection": target_payload,
            "T_body_camera_used": T_body_camera.tolist(),
            "used_config_camera_transform": bool(args.use_config_camera_transform),
        }
    else:
        diagnosis["rgbd"] = {"captured": False}

    lidar = _read_lidar(args, cfg)
    if lidar is not None:
        nav_cfg = LidarTargetNavPolicyRuntimeConfig.from_runtime_config(cfg).grid_cfg
        grid = occupancy_grid_from_lidar_frame(lidar, grid_cfg=nav_cfg)
        target_xy_b = None
        if target_result is not None:
            target_xy_b = tuple(float(v) for v in target_result.target.xy)
        occupancy_path = args.output_dir / "occupancy_grid_axes.png"
        save_occupancy_debug_image(occupancy_path, grid, grid_cfg=nav_cfg, target_xy_b=target_xy_b, show_axes=True)
        np.save(args.output_dir / "lidar_points_body.npy", lidar.points_body_xyz.astype(np.float32, copy=False))
        pts = lidar.points_body_xyz
        diagnosis["lidar"] = {
            "captured": True,
            "stats": lidar_frame_stats(lidar),
            "occupancy_grid_path": str(occupancy_path),
            "body_y_summary": None
            if pts.size == 0
            else {
                "mean_y_m": float(np.mean(pts[:, 1])),
                "min_y_m": float(np.min(pts[:, 1])),
                "max_y_m": float(np.max(pts[:, 1])),
                "positive_y_count": int(np.sum(pts[:, 1] > 0.0)),
                "negative_y_count": int(np.sum(pts[:, 1] < 0.0)),
            },
        }
    else:
        diagnosis["lidar"] = {"captured": False}

    out_json = args.output_dir / "frame_diagnosis.json"
    out_json.write_text(json.dumps(diagnosis, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(diagnosis, indent=2, sort_keys=True, default=_json_default))
    else:
        print(f"[INFO]: diagnosis saved to {out_json}")
        if diagnosis.get("rgbd", {}).get("target_projection"):
            print(f"[INFO]: target_projection={diagnosis['rgbd']['target_projection']}")
        if lidar is not None:
            print(f"[INFO]: occupancy={args.output_dir / 'occupancy_grid_axes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
