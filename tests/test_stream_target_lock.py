from __future__ import annotations

from g1_nav_rl.perception.detection_stream import (
    DetectionStreamFrame,
    DetectionStreamHealth,
    DetectionStreamReadResult,
    DetectionStreamStatus,
    TrackDetection,
)
from adapters.stream_target_lock import PrimaryTargetLock, PrimaryTargetLockConfig


def _track(track_id, conf, xy, *, cls="bottle", depth_valid_ratio=1.0):
    return TrackDetection(
        track_id=track_id,
        class_name=cls,
        confidence=conf,
        bbox_xyxy=(10.0, 10.0, 20.0, 30.0),
        target_body_xy=xy,
        depth_m=xy[0] if xy is not None else None,
        depth_valid_ratio=depth_valid_ratio,
    )


def _frame(ts: float, *tracks: TrackDetection) -> DetectionStreamFrame:
    return DetectionStreamFrame(timestamp_s=ts, frame_id=f"f{ts}", tracks=tracks)


def test_primary_target_lock_initially_selects_highest_confidence_valid_track() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", min_confidence=0.1))
    state = lock.update(_frame(1.0, _track(1, 0.4, (2.0, 0.0)), _track(2, 0.8, (3.0, 0.2))), now_s=1.1)
    assert state.state == "locked_visible"
    assert state.locked_track_id == 2
    assert state.target is not None
    assert state.target.xy == (3.0, 0.2)
    assert state.reason == "initial_lock"


def test_primary_target_lock_ignores_higher_confidence_distractor_after_lock() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", min_confidence=0.1))
    first = lock.update(_frame(1.0, _track(1, 0.6, (2.0, 0.0))), now_s=1.0)
    assert first.locked_track_id == 1
    second = lock.update(_frame(2.0, _track(1, 0.5, (2.1, 0.0)), _track(2, 0.99, (4.0, 0.0))), now_s=2.0)
    assert second.state == "locked_visible"
    assert second.locked_track_id == 1
    assert second.target is not None
    assert second.target.xy == (2.1, 0.0)


def test_primary_target_lock_uses_last_known_target_during_short_occlusion() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", max_lost_s=3.0))
    lock.update(_frame(1.0, _track("a", 0.9, (2.0, -0.2))), now_s=1.0)
    missing = lock.update(_frame(2.0), now_s=2.5)
    assert missing.state == "locked_occluded"
    assert missing.using_last_known is True
    assert missing.target is not None
    assert missing.target.xy == (2.0, -0.2)


def test_primary_target_lock_returns_lost_after_lost_budget_expires() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", max_lost_s=1.0))
    lock.update(_frame(1.0, _track("a", 0.9, (2.0, -0.2))), now_s=1.0)
    lost = lock.update(_frame(5.0), now_s=5.0)
    assert lost.state == "lost"
    assert lost.target is None


def test_primary_target_lock_reacquires_same_track_after_occlusion() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", max_lost_s=5.0))
    lock.update(_frame(1.0, _track(3, 0.9, (2.0, 0.0))), now_s=1.0)
    lock.update(_frame(2.0), now_s=2.0)
    reacquired = lock.update(_frame(3.0, _track(3, 0.8, (1.8, 0.1)), _track(9, 0.99, (5.0, 0.0))), now_s=3.0)
    assert reacquired.state == "locked_visible"
    assert reacquired.locked_track_id == 3
    assert reacquired.target is not None
    assert reacquired.target.xy == (1.8, 0.1)


def test_primary_target_lock_rejects_large_jumps_and_keeps_last_known_temporarily() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", max_target_jump_m=0.5, max_lost_s=3.0))
    lock.update(_frame(1.0, _track(1, 0.9, (2.0, 0.0))), now_s=1.0)
    jumped = lock.update(_frame(1.5, _track(1, 0.9, (5.0, 0.0))), now_s=1.5)
    assert jumped.state == "locked_occluded"
    assert jumped.reason == "locked_track_jump_rejected"
    assert jumped.target is not None
    assert jumped.target.xy == (2.0, 0.0)


def test_primary_target_lock_requires_track_id_when_configured() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", require_track_id=True))
    state = lock.update(_frame(1.0, _track(None, 0.9, (2.0, 0.0))), now_s=1.0)
    assert state.state == "unlocked"
    assert state.target is None


def test_primary_target_lock_update_read_result_uses_stale_status() -> None:
    lock = PrimaryTargetLock(PrimaryTargetLockConfig(target_class="bottle", max_lost_s=2.0))
    fresh = DetectionStreamReadResult(
        frame=_frame(10.0, _track(1, 0.9, (2.0, 0.0))),
        status=DetectionStreamStatus("stream", DetectionStreamHealth.OK, 10.0),
    )
    assert lock.update_read_result(fresh).state == "locked_visible"
    stale = DetectionStreamReadResult(
        frame=_frame(10.0),
        status=DetectionStreamStatus("stream", DetectionStreamHealth.STALE, 10.5),
    )
    state = lock.update_read_result(stale, allow_stale=False)
    assert state.state == "stale"
    assert state.using_last_known is True
