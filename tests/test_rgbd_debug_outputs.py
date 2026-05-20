from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from adapters import (
    CameraIntrinsics,
    RGBDFrame,
    RGBDDebugDetection,
    RGBDDebugWriter,
    Transform3D,
    depth_to_uint8_visualization,
    rgbd_debug_frame_stats,
)


def _frame() -> RGBDFrame:
    h, w = 48, 64
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 100
    rgb[..., 1] = np.arange(w, dtype=np.uint8)[None, :]
    rgb[..., 2] = 200
    depth = np.linspace(0.2, 2.0, h * w, dtype=np.float32).reshape(h, w)
    return RGBDFrame(
        timestamp_s=1.0,
        rgb=rgb,
        depth_m=depth,
        intrinsics=CameraIntrinsics(fx=50.0, fy=50.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h),
        T_body_camera=Transform3D.identity(frame="body"),
        frame_id="camera_color_optical",
        depth_frame_id="camera_depth_optical",
        metadata={"unit_test": True},
    )


def test_depth_visualization_has_expected_shape_and_dtype():
    frame = _frame()
    vis = depth_to_uint8_visualization(frame.depth_m)
    assert vis.shape == frame.depth_m.shape
    assert vis.dtype == np.uint8
    assert vis.max() > vis.min()


def test_rgbd_debug_writer_saves_snapshot_overlay_and_metadata(tmp_path: Path):
    writer = RGBDDebugWriter(tmp_path)
    det = RGBDDebugDetection(class_name="person", confidence=0.9, bbox_xyxy=(5.0, 6.0, 30.0, 40.0))

    artifacts = writer.write_frame(step=3, frame=_frame(), detections=[det], source="unit_test")

    paths = artifacts.paths
    assert Path(paths["rgb"]).exists()
    assert Path(paths["rgb_latest"]).exists()
    assert Path(paths["depth_m"]).exists()
    assert Path(paths["depth_viz"]).exists()
    assert Path(paths["yolo_overlay"]).exists()
    assert Path(paths["detections"]).exists()
    assert Path(paths["metadata"]).exists()

    depth = np.load(paths["depth_m"])
    assert depth.shape == _frame().depth_m.shape

    detections = json.loads(Path(paths["detections"]).read_text())
    assert detections["detection_count"] == 1
    assert detections["detections"][0]["class_name"] == "person"

    meta = json.loads(Path(paths["metadata"]).read_text())
    assert meta["step"] == 3
    assert meta["source"] == "unit_test"
    assert meta["detection_count"] == 1
    assert meta["depth_valid_count"] > 0


def test_rgbd_frame_stats_contains_intrinsics():
    stats = rgbd_debug_frame_stats(_frame())
    assert stats["rgb_shape"] == [48, 64, 3]
    assert stats["intrinsics"]["width"] == 64
    assert stats["depth_valid_count"] == 48 * 64


def test_dryrun_script_accepts_rgbd_debug_options_help():
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "real_g1_follow_dryrun.py"), "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--save-rgbd-debug" in proc.stdout
    assert "--rgbd-debug-yolo" in proc.stdout
