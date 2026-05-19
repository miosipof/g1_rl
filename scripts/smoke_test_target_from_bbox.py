#!/usr/bin/env python3
"""Offline smoke test for Phase-5 RGB-D bbox -> Target2D conversion.

No Isaac, RealSense, OpenCV, or YOLO dependency. This validates the geometry
adapter that detector backends will use later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

try:
    from g1_nav_rl.navigation.simple_nav import Pose2D
    from g1_nav_rl.perception.target_from_bbox import (
        CameraIntrinsics,
        Detection2D,
        TargetFromBBoxConfig,
        bbox_depth_to_target2d,
        make_forward_camera_T_body_camera,
        pose2d_to_T_world_body,
        select_detection_for_target,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - convenience for direct script use
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from g1_nav_rl.navigation.simple_nav import Pose2D
    from g1_nav_rl.perception.target_from_bbox import (
        CameraIntrinsics,
        Detection2D,
        TargetFromBBoxConfig,
        bbox_depth_to_target2d,
        make_forward_camera_T_body_camera,
        pose2d_to_T_world_body,
        select_detection_for_target,
    )


def assert_close(name: str, value: float, expected: float, tol: float = 1e-6) -> None:
    if abs(float(value) - float(expected)) > float(tol):
        raise AssertionError(f"{name}: got {value}, expected {expected} ± {tol}")


def make_synthetic_depth(width: int, height: int, default_depth: float = 6.0) -> np.ndarray:
    depth = np.full((height, width), default_depth, dtype=np.float32)
    # Add invalid border pixels to verify robust filtering.
    depth[:5, :] = np.nan
    depth[:, :5] = 0.0
    return depth


def run_synthetic_test(args: argparse.Namespace) -> None:
    intr = CameraIntrinsics(
        fx=float(args.fx),
        fy=float(args.fy),
        cx=float(args.cx),
        cy=float(args.cy),
        width=int(args.width),
        height=int(args.height),
    )
    robot_pose = Pose2D(float(args.robot_x), float(args.robot_y), float(args.robot_yaw))
    T_world_body = pose2d_to_T_world_body(robot_pose)
    T_body_camera = make_forward_camera_T_body_camera(
        translation_body=(float(args.camera_x_b), float(args.camera_y_b), float(args.camera_z_b))
    )
    T_world_camera = T_world_body @ T_body_camera

    depth = make_synthetic_depth(intr.width, intr.height, default_depth=float(args.background_depth))

    # Synthetic target appears as a central bbox at fixed object depth. The bbox
    # center is offset horizontally, so the test validates lateral projection too.
    bbox = (
        float(args.bbox_cx) - float(args.bbox_w) / 2.0,
        float(args.bbox_cy) - float(args.bbox_h) / 2.0,
        float(args.bbox_cx) + float(args.bbox_w) / 2.0,
        float(args.bbox_cy) + float(args.bbox_h) / 2.0,
    )
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    depth[max(0, y1):min(intr.height, y2), max(0, x1):min(intr.width, x2)] = float(args.object_depth)

    det = Detection2D(class_name=args.class_name, bbox_xyxy=bbox, confidence=0.99)
    cfg = TargetFromBBoxConfig(
        depth_min_m=float(args.depth_min),
        depth_max_m=float(args.depth_max),
        bbox_center_crop_frac=float(args.crop_frac),
        depth_percentile=float(args.depth_percentile),
        stand_off_distance_m=float(args.stand_off),
    )
    result = bbox_depth_to_target2d(
        det,
        depth,
        intr,
        T_world_camera,
        robot_pose=robot_pose,
        config=cfg,
    )
    if result is None:
        raise RuntimeError("bbox_depth_to_target2d returned None on synthetic valid depth")

    print("[INFO]: synthetic RGB-D bbox -> Target2D")
    print(f"  intrinsics: {intr.to_dict()}")
    print(f"  bbox_xyxy: {tuple(round(v, 3) for v in det.bbox_xyxy)}")
    print(f"  depth_m: {result.depth_m:.3f}")
    print(f"  point_camera: {np.round(result.point_camera, 4).tolist()}")
    print(f"  point_world: {np.round(result.point_world, 4).tolist()}")
    print(f"  object_xy_w: ({result.object_xy_w[0]:.3f}, {result.object_xy_w[1]:.3f})")
    print(f"  target: ({result.target.x:.3f}, {result.target.y:.3f})")
    print(f"  used_stand_off: {result.used_stand_off}")

    # For a forward-facing body transform and yaw=0:
    # camera z -> world/body x, camera x right -> -world/body y.
    expected_cam_x = (float(args.bbox_cx) - intr.cx) * float(args.object_depth) / intr.fx
    expected_obj_x = robot_pose.x + float(args.camera_x_b) + float(args.object_depth)
    expected_obj_y = robot_pose.y + float(args.camera_y_b) - expected_cam_x
    assert_close("depth_m", result.depth_m, float(args.object_depth), tol=1e-5)
    assert_close("object_x", result.object_xy_w[0], expected_obj_x, tol=1e-5)
    assert_close("object_y", result.object_xy_w[1], expected_obj_y, tol=1e-5)

    if args.stand_off <= 0.0:
        assert_close("target_x", result.target.x, expected_obj_x, tol=1e-5)
        assert_close("target_y", result.target.y, expected_obj_y, tol=1e-5)
    else:
        target_dist = float(np.hypot(result.target.x - robot_pose.x, result.target.y - robot_pose.y))
        obj_dist = float(np.hypot(expected_obj_x - robot_pose.x, expected_obj_y - robot_pose.y))
        assert_close("target stand-off distance", target_dist, max(0.0, obj_dist - args.stand_off), tol=1e-5)

    # Check selection helper.
    detections = [
        Detection2D("chair", (10, 10, 100, 100), 0.95),
        det,
        Detection2D("person", (300, 200, 340, 260), 0.40),
    ]
    selected = select_detection_for_target(
        detections,
        allowed_classes={args.class_name},
        min_confidence=0.50,
        policy="largest",
    )
    if selected != det:
        raise AssertionError("select_detection_for_target did not choose the expected detection")

    if args.save_snapshot:
        out_dir = Path(args.save_snapshot)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "synthetic_depth_m.npy", depth)
        with open(out_dir / "synthetic_intrinsics.json", "w", encoding="utf-8") as f:
            json.dump(intr.to_dict(), f, indent=2)
        with open(out_dir / "synthetic_detection.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "class_name": det.class_name,
                    "bbox_xyxy": list(det.bbox_xyxy),
                    "confidence": det.confidence,
                    "target_xy": [result.target.x, result.target.y],
                    "object_xy": list(result.object_xy_w),
                    "depth_m": result.depth_m,
                },
                f,
                indent=2,
            )
        print(f"[INFO]: saved synthetic snapshot to {out_dir}")

    print("[INFO]: target-from-bbox smoke test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    # Defaults from the G1 RealSense D435I test log.
    parser.add_argument("--fx", type=float, default=604.2)
    parser.add_argument("--fy", type=float, default=603.8)
    parser.add_argument("--cx", type=float, default=320.7)
    parser.add_argument("--cy", type=float, default=256.5)
    parser.add_argument("--robot_x", type=float, default=0.0)
    parser.add_argument("--robot_y", type=float, default=0.0)
    parser.add_argument("--robot_yaw", type=float, default=0.0)
    parser.add_argument("--camera_x_b", type=float, default=0.10)
    parser.add_argument("--camera_y_b", type=float, default=0.00)
    parser.add_argument("--camera_z_b", type=float, default=0.55)
    parser.add_argument("--bbox_cx", type=float, default=350.0)
    parser.add_argument("--bbox_cy", type=float, default=240.0)
    parser.add_argument("--bbox_w", type=float, default=80.0)
    parser.add_argument("--bbox_h", type=float, default=160.0)
    parser.add_argument("--class_name", type=str, default="person")
    parser.add_argument("--object_depth", type=float, default=3.0)
    parser.add_argument("--background_depth", type=float, default=6.0)
    parser.add_argument("--depth_min", type=float, default=0.20)
    parser.add_argument("--depth_max", type=float, default=8.0)
    parser.add_argument("--crop_frac", type=float, default=0.50)
    parser.add_argument("--depth_percentile", type=float, default=0.50)
    parser.add_argument("--stand_off", type=float, default=0.0)
    parser.add_argument("--save_snapshot", type=str, default="")
    args = parser.parse_args()
    run_synthetic_test(args)


if __name__ == "__main__":
    main()
