from types import SimpleNamespace

from g1_nav_rl.navigation.simple_nav import Target2D
from g1_nav_rl.perception.target_from_bbox import Detection2D
from g1_nav_rl.perception.target_provider import BBoxTargetProvider, TargetProviderConfig, TargetProviderStatus
from g1_nav_rl.perception.target_tracker import TargetTracker, TargetTrackerConfig, TargetTrackingState


def test_target_provider_selects_largest_person_and_projects_target():
    detections = [
        Detection2D("chair", (0, 0, 80, 80), 0.9),
        Detection2D("person", (100, 100, 140, 180), 0.35),
        Detection2D("person", (150, 80, 260, 260), 0.80),
    ]

    def projector(det):
        cx, cy = det.center_xy
        return SimpleNamespace(target=Target2D(cx / 100.0, cy / 100.0), detection=det, object_xy_w=(cx / 100.0, cy / 100.0))

    provider = BBoxTargetProvider(
        TargetProviderConfig(target_source="auto", target_class="person", min_confidence=0.5, selection_policy="largest")
    )
    result = provider.acquire(detections=detections, target_projector=projector)

    assert result.status == TargetProviderStatus.TARGET_READY
    assert result.selected.source == "yolo"
    assert result.target_xy == (2.05, 1.7)


def test_target_provider_can_disable_fallback_for_strict_yolo_validation():
    provider = BBoxTargetProvider(
        TargetProviderConfig(target_source="yolo", target_class="person", min_confidence=0.95, fallback_enabled=False)
    )
    result = provider.acquire(
        detections=[Detection2D("person", (0, 0, 10, 10), 0.2)],
        target_projector=lambda det: SimpleNamespace(target=Target2D(0.0, 0.0)),
    )
    assert result.status == TargetProviderStatus.NO_DETECTION
    assert result.target is None


def test_target_tracker_smooths_occludes_rejects_jump_and_loses_target():
    tracker = TargetTracker(
        TargetTrackerConfig(refresh_period_steps=5, occlusion_grace_steps=10, smoothing_alpha=0.5, max_jump_m=1.0)
    )

    first = tracker.update(step=0, measurement=(1.0, 0.0))
    assert first.state == TargetTrackingState.TRACKING
    assert first.active_target.xy == (1.0, 0.0)

    smoothed = tracker.update(step=5, measurement=(1.4, 0.0))
    assert smoothed.state == TargetTrackingState.TRACKING
    assert smoothed.active_target.xy == (1.2, 0.0)

    rejected = tracker.update(step=10, measurement=(4.0, 0.0))
    assert rejected.state == TargetTrackingState.OCCLUDED_RECENT
    assert rejected.rejected_measurement is True
    assert rejected.active_target.xy == (1.2, 0.0)

    assert tracker.update(step=14, measurement=None).state == TargetTrackingState.OCCLUDED_RECENT
    lost = tracker.update(step=20, measurement=None)
    assert lost.state == TargetTrackingState.LOST
    assert lost.active_target is None
