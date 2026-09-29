from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from adapters import PrimaryTargetLock, PrimaryTargetLockConfig, RobotObservation, Target2D
from adapters.detection_stream import (
    DetectionStreamFrame,
    DetectionStreamHealth,
    DetectionStreamReadResult,
    DetectionStreamStatus,
    TrackDetection,
)
from real_g1_follow_detect_live import DetectTargetObservationProvider, TargetTrackingStatus
from real_g1_follow_stepwise import StreamTargetObservationProvider


class _Base:
    def read_observation(self) -> RobotObservation:
        return RobotObservation(timestamp_s=3.0)


class _Stream:
    def __init__(self, result: DetectionStreamReadResult) -> None:
        self.result = result

    def read(self) -> DetectionStreamReadResult:
        return self.result


def _result(timestamp_s: float) -> DetectionStreamReadResult:
    track = TrackDetection(
        track_id=7,
        class_name="person",
        confidence=0.9,
        bbox_xyxy=(1.0, 1.0, 2.0, 3.0),
        target_body_xy=(1.5, 0.1),
        depth_m=1.5,
        depth_valid_ratio=1.0,
    )
    frame = DetectionStreamFrame(timestamp_s=timestamp_s, frame_id=f"f{timestamp_s}", tracks=(track,))
    return DetectionStreamReadResult(
        frame=frame,
        status=DetectionStreamStatus("stream", DetectionStreamHealth.OK, timestamp_s),
    )


def test_stream_target_requires_a_frame_newer_than_motion_boundary() -> None:
    stream = _Stream(_result(1.0))
    provider = StreamTargetObservationProvider(
        base_provider=_Base(),
        stream_client=stream,
        target_lock=PrimaryTargetLock(PrimaryTargetLockConfig(target_class="person")),
    )
    assert provider.read_observation().target_hint is not None
    provider.invalidate_body_target(after_timestamp_s=1.5)
    stale = provider.read_observation()
    assert stale.target_hint is None
    assert stale.metadata["target_measurement"]["fresh_after_motion"] is False
    stream.result = _result(2.0)
    fresh = provider.read_observation()
    assert fresh.target_hint is not None
    assert fresh.metadata["target_measurement"]["fresh_after_motion"] is True


def test_inline_provider_invalidation_drops_cached_body_coordinates() -> None:
    provider = DetectTargetObservationProvider.__new__(DetectTargetObservationProvider)
    provider.active_target = Target2D(2.0, 0.2)
    provider.last_seen_step = 4
    provider.last_detections = [object()]
    provider.last_status = TargetTrackingStatus("tracking", "measurement", (2.0, 0.2))
    provider.invalidate_body_target(reason="physical_motion")
    assert provider.active_target is None
    assert provider.last_seen_step is None
    assert provider.last_detections == []
    assert provider.last_status.state == "lost"
