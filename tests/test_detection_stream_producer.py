from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from adapters import RGBDFrame, Transform3D
from g1_nav_rl.perception import CameraIntrinsics, Detection2D
from g1_nav_rl.perception.detection_stream_producer import (
    DetectionStreamProjectionConfig,
    depth_valid_ratio_in_bbox,
    detection_to_track,
    make_detection_stream_frame_from_detections,
)


def _rgbd_frame(depth_value: float = 2.0) -> RGBDFrame:
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.full((100, 100), depth_value, dtype=np.float32)
    intr = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)
    return RGBDFrame(
        timestamp_s=123.0,
        rgb=rgb,
        depth_m=depth,
        intrinsics=intr,
        T_body_camera=Transform3D(matrix=np.eye(4, dtype=np.float64), source_frame="camera_color_optical", target_frame="body"),
    )


def test_detection_to_track_projects_center_with_standoff() -> None:
    frame = _rgbd_frame(2.0)
    detection = Detection2D(class_name="bottle", confidence=0.9, bbox_xyxy=(45.0, 40.0, 55.0, 60.0))
    cfg = DetectionStreamProjectionConfig(target_class="bottle", stand_off_m=0.8, min_confidence=0.1)

    track = detection_to_track(detection, frame, config=cfg, track_id=7)

    assert track is not None
    assert track.track_id == 7
    assert track.class_name == "bottle"
    assert track.depth_m == 2.0
    assert track.target_body_xy is not None
    assert abs(track.target_body_xy[0] - 1.2) < 1e-6
    assert abs(track.target_body_xy[1]) < 1e-6
    assert track.metadata["used_stand_off"] is True
    assert track.metadata["object_body_xy"] == [2.0, 0.0]


def test_detection_to_track_rejects_small_or_wrong_class() -> None:
    frame = _rgbd_frame(2.0)
    cfg = DetectionStreamProjectionConfig(target_class="bottle", min_bbox_area_px=100.0)

    small = Detection2D(class_name="bottle", confidence=0.9, bbox_xyxy=(10.0, 10.0, 15.0, 15.0))
    wrong = Detection2D(class_name="person", confidence=0.9, bbox_xyxy=(30.0, 30.0, 60.0, 80.0))

    assert detection_to_track(small, frame, config=cfg) is None
    assert detection_to_track(wrong, frame, config=cfg) is None


def test_depth_valid_ratio_in_bbox_handles_invalid_depth() -> None:
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth[8:12, 8:12] = np.nan
    ratio, valid, total, crop = depth_valid_ratio_in_bbox(
        depth,
        (5.0, 5.0, 15.0, 15.0),
        depth_min_m=0.2,
        depth_max_m=8.0,
        crop_frac=1.0,
    )
    assert crop == (5, 5, 15, 15)
    assert total == 100
    assert valid == 84
    assert abs(ratio - 0.84) < 1e-6


def test_make_detection_stream_frame_from_detections() -> None:
    frame = _rgbd_frame(3.0)
    dets = [Detection2D(class_name="bottle", confidence=0.7, bbox_xyxy=(45.0, 45.0, 55.0, 55.0))]
    cfg = DetectionStreamProjectionConfig(target_class="bottle", stand_off_m=1.0)

    stream = make_detection_stream_frame_from_detections(
        frame=frame,
        detections=dets,
        config=cfg,
        frame_id="f0",
        source="test_producer",
        track_ids=["track-a"],
    )

    assert stream.frame_id == "f0"
    assert stream.source == "test_producer"
    assert len(stream.tracks) == 1
    assert stream.tracks[0].track_id == "track-a"
    assert stream.tracks[0].target_body_xy == (2.0, 0.0)
    assert stream.metadata["raw_detection_count"] == 1
    assert stream.metadata["published_track_count"] == 1


def test_yolo_track_stream_script_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "real_g1_yolo_track_stream.py"
    result = subprocess.run([sys.executable, str(script), "--help"], text=True, capture_output=True, check=True, env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
    assert "--output" in result.stdout
    assert "--mode" in result.stdout
    assert "--tracker-config" in result.stdout
