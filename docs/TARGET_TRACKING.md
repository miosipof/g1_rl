# Target Provider and Target Tracker

This document describes the perception-target layer used to move from one-shot person-target navigation toward simple target following.

## Current status

`smoke_test_yolo_person_nav.py` now has two target modes:

```text
initial snapshot target
periodic re-detection target refresh
```

At startup, the script still captures one RGB-D snapshot, runs YOLO/fallback bbox selection, projects the bbox to a `Target2D`, and creates the initial navigation target.

During navigation, the script now performs a lightweight re-detection attempt every 50 steps by default:

```bash
--redetect_every 50
```

Each refresh reads the latest RGB-D camera output, runs YOLO/fallback bbox selection, estimates a new `Target2D`, passes it through `TargetTracker`, and replans if the accepted active target changed.

To recover the previous one-shot behavior:

```bash
--redetect_every 0
```

## Reusable modules

The refactor extracts two perception components:

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

This keeps the Isaac-safe behavior intact: select a bbox first, then compute only the scalar depth needed for that bbox rather than copying the full depth image to CPU.

## `target_tracker.py`

`TargetTracker` is a deterministic state machine for periodic re-detection:

```text
TRACKING           fresh target measurement accepted
OCCLUDED_RECENT    no valid measurement, but last target is still usable
LOST               no usable target remains
```

It provides:

- refresh-period scheduling via `should_refresh(step)`;
- low-pass filtering through `smoothing_alpha`;
- jump rejection through `max_jump_m`;
- short occlusion grace through `occlusion_grace_steps`;
- a stable `active_target` that navigation can consume.

## Isaac script behavior

The navigation loop now does this:

```python
if step > 0 and tracker.should_refresh(step):
    maybe_target_xy, meta = extract_periodic_redetection_target(scene, proxy, step_idx=step)
    measurement = None if maybe_target_xy is None else TargetMeasurement.from_xy(...)
    tracker_update = tracker.update(step=step, measurement=measurement)

    if tracker_update.active_target is None:
        stop_robot()
    elif tracker_update.accepted_measurement:
        nav_target = tracker_update.active_target
        nav_runtime.plan(current_pose, nav_target, live_grid)
```

So the current behavior is no longer strictly one-shot. It is now:

```text
initial detection -> navigate -> re-detect every 50 steps -> update/replan if target is valid -> keep last target through short occlusions -> stop if target becomes LOST
```

## Important CLI options

```bash
--redetect_every 50
--target_occlusion_grace_steps 150
--target_smoothing_alpha 0.35
--target_max_jump_m 1.25
```

For strict YOLO validation without synthetic fallback:

```bash
--target_source yolo --no_fallback_synthetic_bbox
```

For deterministic geometry/fallback validation:

```bash
--target_source fallback --target_depth_source geometric
```

## Offline check

```bash
PYTHONPATH=src python scripts/smoke_test_target_provider_tracker.py
PYTHONPATH=src pytest -q tests
```

The full periodic re-detection path still requires Isaac/RTX because it reads the live camera sensor from `smoke_test_yolo_person_nav.py`.
