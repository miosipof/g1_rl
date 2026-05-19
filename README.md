# Unitree G1 Autonomy Prototype

Production-oriented autonomy prototype for Unitree G1 navigation and perception.

The current stack is intentionally deterministic and inspectable:

1. perception or a user-defined command produces a `Target2D(x, y)`;
2. the navigation runtime converts the target into planar motion commands `(vx, vy, wz)`;
3. LiDAR occupancy, A* planning, replanning, and safety filters constrain motion around obstacles;
4. RGB-D/YOLO target extraction can produce a person standoff target for navigation.

## Current milestone

The repository has an Isaac Sim full-pipeline smoke test that performs one-shot RGB-D/YOLO person-target acquisition, freezes the target, and then runs live LiDAR obstacle-avoiding navigation toward that target.

The one-shot freeze is deliberate: continuous RGB-D tensor handling inside the Isaac navigation loop previously caused CUDA/RTX synchronization instability. The next phase adds periodic target refresh using a small target-tracking state machine rather than embedding more logic into the large Isaac smoke script.

## Important entry points

```text
src/g1_nav_rl/navigation/
  simple_nav.py       deterministic 2D waypoint controller
  runtime.py          replanning wrapper around SimpleNavigator
  safety.py           LiDAR/RGB-D command safety filters

src/g1_nav_rl/perception/
  target_from_bbox.py bbox + depth + intrinsics + camera transform -> Target2D
  yolo_detector.py    optional Ultralytics adapter
  target_provider.py  reusable detection/fallback/selection target acquisition
  target_tracker.py   TRACKING / OCCLUDED_RECENT / LOST target state machine

scripts/
  smoke_test_nav.py                     offline deterministic A* navigation
  smoke_test_nav_lidar.py               Isaac LiDAR navigation smoke test
  smoke_test_yolo_person_target.py      Isaac YOLO/RGB-D target extraction
  smoke_test_yolo_person_nav.py         one-shot target + live LiDAR navigation
  smoke_test_target_provider_tracker.py offline provider/tracker smoke test
```

## Offline validation

From repo root:

```bash
PYTHONPATH=src python -m compileall -q src scripts tests
PYTHONPATH=src python scripts/smoke_test_target_from_bbox.py --stand_off 0.8
PYTHONPATH=src python scripts/smoke_test_target_provider_tracker.py
PYTHONPATH=src pytest -q tests
```

Isaac scripts must be run inside the Isaac Lab / Omniverse environment.

## Next implementation phase

Use `TargetTracker` for periodic target refresh:

- every `N` simulator steps, capture a CPU RGB snapshot;
- run YOLO or fallback bbox acquisition through `BBoxTargetProvider`;
- update `TargetTracker` with the new target measurement;
- pass `tracker.active_target` to `NavRuntime` and replan only when the target has moved enough;
- during short occlusion, keep moving toward the last known target;
- when state becomes `LOST`, stop or switch to a search behavior.

