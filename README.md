# Unitree G1 Autonomy Prototype

Production-oriented autonomy prototype for Unitree G1 navigation and perception.

The current stack is intentionally deterministic and inspectable:

1. perception or a user-defined command produces a `Target2D(x, y)`;
2. the navigation runtime converts the target into planar motion commands `(vx, vy, wz)`;
3. LiDAR occupancy, A* planning, replanning, and safety filters constrain motion around obstacles;
4. RGB-D/YOLO target extraction can produce a person standoff target for navigation.

## Current milestone

The repository has an Isaac Sim full-pipeline smoke test that performs initial RGB-D/YOLO person-target acquisition and then runs live LiDAR obstacle-avoiding navigation toward that target. The script now also supports simple periodic target re-detection inside the navigation loop.

By default, `smoke_test_yolo_person_nav.py` refreshes the target every 50 navigation steps with `--redetect_every 50`. Each refresh runs YOLO/fallback bbox selection, updates `TargetTracker`, and replans if a valid target update is accepted. Use `--redetect_every 0` to recover the older one-shot behavior.

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
  smoke_test_yolo_person_nav.py         initial target + periodic re-detection + live LiDAR navigation
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

## Isaac stepwise odometry simulation

See [`docs/ODOMETRY.md`](docs/ODOMETRY.md) for the complete simulation and real
G1 testing procedure.

The stepwise Isaac runner now completes motion from simulated odometry instead
of assuming that command duration equals traveled distance. It uses the same
completion, stop, and settling executor as the real G1 path. With no impairment
flags, odometry is derived directly from the command proxy.

```bash
python scripts/smoke_test_yolo_stepwise_nav.py -- --scenario front_box --max_steps 80
```

Artificial sensor imperfections are deterministic for a fixed seed:

```bash
python scripts/smoke_test_yolo_stepwise_nav.py \
  --odom-lag-s 0.08 \
  --odom-position-noise-std-m 0.01 \
  --odom-yaw-noise-std-deg 0.5 \
  --odom-dropout-probability 0.02 \
  --odom-seed 7 \
  -- --scenario front_box --max_steps 80
```

Use `--legacy-duration-execution` to compare against the earlier timed-action
behavior. Per-action measured and ground-truth results are written to
`stepwise_trace.jsonl`.

## Next implementation phase

The current periodic re-detection is intentionally simple. Next steps:

- validate `--redetect_every 50` in Isaac on static and slowly moving person targets;
- tune `--target_occlusion_grace_steps`, `--target_smoothing_alpha`, and `--target_max_jump_m`;
- add moving-person scenarios and scripted person trajectories;
- add search behavior for the `LOST` state instead of only stopping;
- only after this baseline is stable, evaluate velocity prediction or recurrent filtering.
