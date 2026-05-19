#!/usr/bin/env python3
"""Offline smoke test for reusable target provider and target tracker.

No Isaac, Torch, YOLO, or camera dependency. This validates the phase transition
from one-shot target acquisition toward periodic re-detection and target-state
tracking.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

try:
    from g1_nav_rl.navigation.simple_nav import Target2D
    from g1_nav_rl.perception.target_from_bbox import Detection2D
    from g1_nav_rl.perception.target_provider import BBoxTargetProvider, TargetProviderConfig, TargetProviderStatus
    from g1_nav_rl.perception.target_tracker import TargetTracker, TargetTrackerConfig, TargetTrackingState
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script use
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from g1_nav_rl.navigation.simple_nav import Target2D
    from g1_nav_rl.perception.target_from_bbox import Detection2D
    from g1_nav_rl.perception.target_provider import BBoxTargetProvider, TargetProviderConfig, TargetProviderStatus
    from g1_nav_rl.perception.target_tracker import TargetTracker, TargetTrackerConfig, TargetTrackingState


def assert_equal(name: str, got, expected) -> None:
    if got != expected:
        raise AssertionError(f"{name}: got {got!r}, expected {expected!r}")


def assert_close(name: str, got: float, expected: float, tol: float = 1e-6) -> None:
    if abs(float(got) - float(expected)) > float(tol):
        raise AssertionError(f"{name}: got {got}, expected {expected} ± {tol}")


def run_provider_checks() -> None:
    detections = [
        Detection2D("chair", (0, 0, 80, 80), 0.9),
        Detection2D("person", (100, 100, 140, 180), 0.35),
        Detection2D("person", (150, 80, 260, 260), 0.80),
    ]

    def fallback_detection() -> Detection2D:
        return Detection2D("person", (300, 90, 420, 320), 1.0)

    def projector(det: Detection2D):
        # Deterministic fake projection: target is bbox center scaled into meters.
        cx, cy = det.center_xy
        return SimpleNamespace(
            target=Target2D(float(cx) / 100.0, float(cy) / 100.0),
            object_xy_w=(float(cx) / 100.0, float(cy) / 100.0),
            detection=det,
            depth_m=3.0,
            used_stand_off=False,
            already_within_standoff=False,
            valid_depth_pixels=42,
            depth_crop_xyxy=(1, 2, 3, 4),
            reason="ok_test",
        )

    provider = BBoxTargetProvider(
        TargetProviderConfig(target_source="auto", target_class="person", min_confidence=0.5, selection_policy="largest")
    )
    result = provider.acquire(detections=detections, fallback_detection_factory=fallback_detection, target_projector=projector)
    assert_equal("provider status", result.status, TargetProviderStatus.TARGET_READY)
    assert_equal("provider source", result.selected.source, "yolo")
    assert_close("selected target x", result.target.x, 2.05)
    assert_close("selected target y", result.target.y, 1.70)

    fallback_provider = BBoxTargetProvider(TargetProviderConfig(target_source="fallback", target_class="person"))
    fallback_result = fallback_provider.acquire(
        detections=detections,
        fallback_detection_factory=fallback_detection,
        target_projector=projector,
    )
    assert_equal("fallback source", fallback_result.selected.source, "synthetic_projection")
    assert_close("fallback target x", fallback_result.target.x, 3.60)
    assert_close("fallback target y", fallback_result.target.y, 2.05)

    strict_provider = BBoxTargetProvider(
        TargetProviderConfig(target_source="yolo", target_class="person", min_confidence=0.95, fallback_enabled=False)
    )
    strict_result = strict_provider.acquire(detections=detections, target_projector=projector)
    assert_equal("strict status", strict_result.status, TargetProviderStatus.NO_DETECTION)

    meta = result.to_metadata(extra={"check": "provider"})
    assert_equal("metadata selected class", meta["selected_detection"]["class_name"], "person")
    assert_equal("metadata extra", meta["check"], "provider")


def run_tracker_checks() -> None:
    tracker = TargetTracker(
        TargetTrackerConfig(
            refresh_period_steps=5,
            occlusion_grace_steps=10,
            smoothing_alpha=0.5,
            max_jump_m=1.0,
            min_update_distance_m=0.0,
        )
    )
    first = tracker.update(step=0, measurement=(1.0, 0.0))
    assert_equal("first state", first.state, TargetTrackingState.TRACKING)
    assert_close("first x", first.active_target.x, 1.0)

    smoothed = tracker.update(step=5, measurement=(1.4, 0.0))
    assert_equal("smooth state", smoothed.state, TargetTrackingState.TRACKING)
    assert_close("smoothed x", smoothed.active_target.x, 1.2)

    rejected = tracker.update(step=10, measurement=(4.0, 0.0))
    assert_equal("jump rejection state", rejected.state, TargetTrackingState.OCCLUDED_RECENT)
    assert_equal("jump rejected", rejected.rejected_measurement, True)
    assert_close("keep previous target", rejected.active_target.x, 1.2)

    occluded = tracker.update(step=14, measurement=None)
    assert_equal("occluded state", occluded.state, TargetTrackingState.OCCLUDED_RECENT)
    assert_close("occluded target x", occluded.active_target.x, 1.2)

    lost = tracker.update(step=20, measurement=None)
    assert_equal("lost state", lost.state, TargetTrackingState.LOST)
    assert_equal("lost target", lost.active_target, None)


def main() -> None:
    run_provider_checks()
    run_tracker_checks()
    print("[INFO]: target provider/tracker smoke test passed")


if __name__ == "__main__":
    main()
