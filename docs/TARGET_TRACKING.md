# Target Provider and Target Tracker

This document describes the first cleanup/refactor pass for moving from one-shot person-target navigation to periodic autonomous target following.

## Why this refactor exists

`smoke_test_yolo_person_nav.py` reached the current full-pipeline milestone, but it is now too large to absorb the next phase safely. It mixes Isaac scene setup, RGB-D/YOLO, bbox projection, LiDAR occupancy, A* planning, replanning, safety filters, logging, and visualization.

The refactor extracts two reusable perception components:

```text
src/g1_nav_rl/perception/target_provider.py
src/g1_nav_rl/perception/target_tracker.py
```

The Isaac script still owns Isaac-specific camera reads and the tensor-depth workaround. The reusable modules own target-selection policy and target-tracking state.

## `target_provider.py`

`BBoxTargetProvider` handles deterministic target acquisition policy:

- `target_source='auto'`: use detector detections first, then fallback if enabled;
- `target_source='yolo'`: prefer detector detections; fallback is controlled by `fallback_enabled`;
- `target_source='fallback'`: bypass detector detections and use the fallback factory;
- class/confidence filtering and selection policy are centralized;
- projection is delegated to a caller-supplied function, so the provider does not depend on Isaac, Torch, YOLO, or OpenCV.

This keeps the existing Isaac-safe behavior intact: select a bbox first, then compute only the scalar depth needed for that bbox rather than copying the full depth image to CPU.

## `target_tracker.py`

`TargetTracker` is a deterministic state machine for periodic re-detection:

```text
TRACKING          fresh target measurement accepted
OCCLUDED_RECENT   no valid measurement, but last target is still usable
LOST              no usable target remains
```

It provides:

- refresh-period scheduling via `should_refresh(step)`;
- low-pass filtering through `smoothing_alpha`;
- jump rejection through `max_jump_m`;
- short occlusion grace through `occlusion_grace_steps`;
- a stable `active_target` that navigation can consume.

## Offline check

```bash
PYTHONPATH=src python scripts/smoke_test_target_provider_tracker.py
PYTHONPATH=src pytest -q tests
```

## How to use in the next Isaac phase

The next Isaac iteration should not run YOLO every simulator step. A safe first version is:

```python
tracker = TargetTracker(TargetTrackerConfig(refresh_period_steps=30, occlusion_grace_steps=90))

for step in range(max_steps):
    if tracker.should_refresh(step):
        maybe_target_xy, meta = try_extract_target_snapshot()
        measurement = None if maybe_target_xy is None else TargetMeasurement.from_xy(
            maybe_target_xy,
            step=step,
            source=meta.get("detection_source", "unknown"),
            metadata=meta,
        )
        tracker_update = tracker.update(step=step, measurement=measurement)
    else:
        tracker_update = tracker.update(step=step, measurement=None)

    if tracker_update.active_target is None:
        command = stop_command()
    else:
        nav_target = tracker_update.active_target
        command = nav_runtime.step(..., target=nav_target, ...)
```

The first periodic version should update target only at low frequency and should replan only when the accepted target moves beyond a threshold. Continuous camera/depth readback should remain avoided until the periodic version is stable.
