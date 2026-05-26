from __future__ import annotations

import json

import pytest

from adapters import (
    DetectionStreamClient,
    DetectionStreamHealth,
    DetectionStreamFrame,
    TrackDetection,
    write_detection_stream_frame_atomic,
)


def test_track_detection_round_trip_and_properties() -> None:
    track = TrackDetection(
        track_id=7,
        class_name="bottle",
        confidence=0.72,
        bbox_xyxy=(10.0, 20.0, 50.0, 80.0),
        target_body_xy=(1.5, -0.2),
        depth_m=1.7,
        depth_valid_ratio=0.65,
        metadata={"backend": "test"},
    )

    assert track.bbox_center_uv == (30.0, 50.0)
    assert track.bbox_area_px == 2400.0
    restored = TrackDetection.from_dict(track.to_dict())
    assert restored == track


def test_detection_stream_frame_best_track() -> None:
    frame = DetectionStreamFrame(
        timestamp_s=10.0,
        frame_id=3,
        image_shape=(480, 640, 3),
        tracks=(
            TrackDetection(class_name="bottle", confidence=0.3, bbox_xyxy=(1, 2, 10, 20), track_id=1, target_body_xy=(2.0, 0.3)),
            TrackDetection(class_name="bottle", confidence=0.8, bbox_xyxy=(2, 3, 20, 30), track_id=2, target_body_xy=(1.0, -0.1)),
            TrackDetection(class_name="person", confidence=0.9, bbox_xyxy=(4, 5, 40, 50), track_id=3, target_body_xy=(3.0, 0.0)),
        ),
    )

    best = frame.best_track(class_name="bottle", min_confidence=0.5)
    assert best is not None
    assert best.track_id == 2
    assert frame.track_by_id(3).class_name == "person"
    assert len(frame.tracks_for_class("bottle")) == 2


def test_detection_stream_client_reads_atomic_json(tmp_path) -> None:
    path = tmp_path / "latest_tracks.json"
    frame = DetectionStreamFrame(
        timestamp_s=100.0,
        tracks=(
            TrackDetection(
                track_id="abc",
                class_name="bottle",
                confidence=0.5,
                bbox_xyxy=(10, 10, 40, 60),
                target_body_xy=(2.0, 0.1),
            ),
        ),
    )
    write_detection_stream_frame_atomic(frame, path, indent=2)

    client = DetectionStreamClient(path=path, max_age_s=1.0, clock=lambda: 100.2)
    result = client.read()

    assert result.ok
    assert result.status.health == DetectionStreamHealth.OK
    assert result.frame is not None
    assert result.frame.tracks[0].track_id == "abc"
    assert client.best_track(class_name="bottle").target_body_xy == (2.0, 0.1)


def test_detection_stream_client_reports_stale_and_missing(tmp_path) -> None:
    missing = DetectionStreamClient(path=tmp_path / "missing.json", clock=lambda: 5.0)
    assert missing.read().status.health == DetectionStreamHealth.UNAVAILABLE
    assert missing.read_latest() is None

    path = tmp_path / "latest_tracks.json"
    write_detection_stream_frame_atomic(DetectionStreamFrame(timestamp_s=1.0), path)
    stale = DetectionStreamClient(path=path, max_age_s=0.5, clock=lambda: 2.0)
    result = stale.read()
    assert result.frame is not None
    assert result.status.health == DetectionStreamHealth.STALE
    assert stale.read_latest(require_fresh=True) is None
    assert stale.read_latest(require_fresh=False) is not None


def test_detection_stream_validation_errors() -> None:
    with pytest.raises(ValueError):
        TrackDetection(class_name="", confidence=0.5, bbox_xyxy=(0, 0, 1, 1))
    with pytest.raises(ValueError):
        TrackDetection(class_name="bottle", confidence=1.5, bbox_xyxy=(0, 0, 1, 1))
    with pytest.raises(ValueError):
        TrackDetection(class_name="bottle", confidence=0.5, bbox_xyxy=(0, 0, 0, 1))
    with pytest.raises(ValueError):
        DetectionStreamFrame.from_dict({"schema_version": 999, "timestamp_s": 1.0, "tracks": []})


def test_detection_stream_json_shape_is_plain() -> None:
    frame = DetectionStreamFrame(
        timestamp_s=1.0,
        image_shape=(480, 640, 3),
        tracks=(TrackDetection(class_name="bottle", confidence=0.2, bbox_xyxy=(1, 2, 3, 4)),),
    )
    data = json.loads(frame.to_json_text())
    assert data["schema_version"] == 1
    assert data["tracks"][0]["bbox_center_uv"] == [2.0, 3.0]
    assert data["tracks"][0]["target_body_xy"] is None


def test_detection_stream_contract_is_available_from_general_perception_package() -> None:
    from g1_nav_rl.perception.detection_stream import DetectionStreamFrame as GeneralFrame
    from g1_nav_rl.perception.detection_stream import DetectionStreamHealth as GeneralHealth
    from adapters import DetectionStreamFrame as AdapterFrame

    assert GeneralFrame is AdapterFrame
    assert GeneralHealth.OK.value == "ok"
