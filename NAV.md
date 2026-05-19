# NAV.md — Deterministic 2D Navigation Baseline for Unitree G1

## Purpose

This document summarizes the current navigation baseline developed for the Unitree G1 autonomy stack. The goal is not yet a full production planner. The goal is a clean, testable deterministic navigation layer that can be used as a reliable baseline before adding persistent mapping, replanning, RGB-D safety, dynamic obstacle handling, and learned policies.

The current work has two complementary test paths:

1. **Non-Isaac baseline:** `scripts/smoke_test_nav.py`
   - Pure 2D occupancy-grid navigation test.
   - Uses analytic scenario geometry.
   - No Isaac Sim, no RayCaster, no USD scene.
   - Best for validating A*, inflation margins, waypoint tracking logic, scenario definitions, and pass/fail criteria.

2. **Isaac/LiDAR baseline:** `scripts/smoke_test_nav_lidar.py`
   - Isaac Lab / Isaac Sim scene with G1 proxy, target marker, obstacles, and RayCaster/LiDAR access.
   - Uses the same planning philosophy as `smoke_test_nav.py`.
   - Adds sensor-backed occupancy input and visualization in a fixed world frame.
   - Best for validating the bridge from Isaac scene geometry and LiDAR observations into the deterministic navigation stack.

---

## Current status

We started with a more complex LiDAR/A* smoke test that mixed several behaviors:

- global A* planning,
- periodic replanning,
- path shortcutting,
- command smoothing,
- local clearance safety,
- escape behavior,
- live LiDAR observations,
- and Isaac/G1 proxy dynamics.

That test was useful, but too many mechanisms interacted at once. Several failures were not caused by A* itself. They came from control-layer complications: shortcutting, insufficient margins, smoothing, command clipping, and local safety overrides fighting the planned trajectory.

To isolate the core stack, we built `smoke_test_nav.py`, a minimal deterministic 2D navigation test. It uses a one-shot inflated occupancy grid, one A* plan, and direct waypoint tracking. No replanning, no smoothing, no shortcutting, and no local escape behavior. This baseline succeeded on `front_box` with `--safety_margin 0.35`, reaching the target in roughly 966 steps while preserving positive obstacle clearance.

We then created `smoke_test_nav_lidar.py` to bring the same minimal approach into Isaac Lab with G1 proxy motion and RayCaster/LiDAR access. The Isaac/LiDAR script now supports stable world-frame occupancy visualization, setup-derived obstacle contours, executed trajectory, swept safety zone, and an aggregate RayCaster mesh mode for multi-obstacle scenes.

---

## Core navigation algorithm

The canonical baseline algorithm is deliberately simple:

1. Build a 2D occupancy grid.
2. Inflate obstacles by:

   ```text
   inflation_radius = robot_body_radius + safety_margin
   ```

3. Convert the robot start pose and target position into grid cells.
4. Run A* once on the inflated grid.
5. Convert the A* grid-cell path back into world-frame waypoints.
6. Track the path monotonically with a direct waypoint controller.
7. Stop when the target is within the success radius.
8. Fail if collision or safety-margin violation is detected when assertions are enabled.

Important baseline exclusions:

- no replanning,
- no path shortcutting,
- no command smoothing,
- no local escape controller,
- no learned policy,
- no dynamic obstacle handling.

This gives a clean reference for later complexity.

---

## Controller behavior

The controller tracks the next path waypoint in the robot body frame and maps it to a velocity command:

```text
command = [vx, vy, wz]
```

Typical defaults:

```text
max_vx = 0.30 m/s
max_vy = 0.25 m/s
max_wz = 0.60 rad/s
waypoint_reached_radius = 0.12 m
success radius ≈ hand_length = 0.75 m
safety_margin = 0.35 m
```

By default, the Isaac/LiDAR script disables extra proxy command dynamics so that it stays close to the non-Isaac baseline. Use `--use_config_proxy_dynamics` only when explicitly testing configured proxy smoothing / acceleration behavior.

---

## Occupancy sources in Isaac/LiDAR mode

`smoke_test_nav_lidar.py` supports three practical planning-grid sources:

```text
--grid_source scenario
--grid_source lidar
--grid_source hybrid
```

### `--grid_source scenario`

Uses the known scenario obstacle footprints, equivalent in spirit to `smoke_test_nav.py`. This is not pure perception. It is useful for validating that the Isaac/G1 proxy controller and planner behave correctly when the 2D map is known.

### `--grid_source lidar`

Uses RayCaster/LiDAR hit points. This is closer to perception-based planning, but a single 2D scan only gives first-return surfaces. It does **not** know the full hidden obstacle volume behind a visible surface.

### `--grid_source hybrid`

Combines LiDAR-derived cells with known scenario footprints. This is useful as an integration test while the pure LiDAR map is still being developed.

---

## Important limitation: invisible obstacle volume

Pure LiDAR mode currently observes only visible obstacle surfaces. For example, a wall may appear as a line of first returns, not as a full rectangle. A* therefore cannot infer the complete occupied region unless we add a conservative mapping model.

The next mapping step should introduce:

```text
unknown / observed_free / observed_occupied / conservatively_occupied
```

and a persistent world-frame map with default obstacle thickness or surface extrusion. A future replanning layer should update this map periodically and replan when the current path intersects newly observed occupied space.

Recommended future sequence:

1. Persistent world-frame occupancy map, no replanning.
2. Path-invalidity detection, no replanning.
3. Event-driven replanning with cooldown.
4. Later: shortcutting, smoothing, and local safety.

---

## Isaac/LiDAR visualization

The current visualization saves one image per saved step:

```text
outputs/nav_lidar/<scenario>/grid/occupancy_map_000000.png
outputs/nav_lidar/<scenario>/grid/occupancy_map_000100.png
...
```

Each image is fixed in the world/start frame and shows:

```text
gray       = unknown
white      = observed free
dark       = occupied
red dashed = real setup obstacle contour, not from LiDAR
green      = start point
blue       = target
yellow     = current robot position
dark gray  = covered trajectory
light gray = swept safety zone around executed trajectory
orange     = current remaining A* plan
```

This fixed-frame visualization replaced the earlier robot/body-frame-like debug images where obstacle shapes appeared to rotate or morph from step to step.

---

## RayCaster multi-obstacle fix

`multi` initially crashed during Isaac `sim.reset()` when RayCaster received multiple mesh prim paths. The scene was valid before reset, but Isaac exited during RayCaster initialization.

The script now supports:

```text
--raycast_mesh_mode auto|actual|aggregate
```

Default behavior:

```text
--raycast_mesh_mode auto
```

In `auto`, multi-obstacle scenes use an aggregate raycast-only mesh. The real visual/collision obstacle meshes remain unchanged, but RayCaster targets a single combined mesh. This fixed `multi`.

For debugging:

```text
--raycast_mesh_mode actual
```

reproduces direct multi-mesh targeting.

---

## Built-in scenarios

Typical scenarios include:

```text
clear
front_box
offset_left
offset_right
wide_box
long_box
rotated_box
round
multi
blocked_wall
```

List scenarios:

```bash
python scripts/smoke_test_nav_lidar.py --list_scenarios
```

For non-Isaac tests, use the corresponding scenario list from `smoke_test_nav.py` if available:

```bash
python scripts/smoke_test_nav.py --list_scenarios
```

---

# Test commands without Isaac

These commands use the pure 2D deterministic baseline. Run from the repository root.

## Front-box known-good baseline

```bash
python scripts/smoke_test_nav.py \
  --scenario front_box \
  --safety_margin 0.35 \
  --max_steps 2000 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision \
  --assert_margin
```

Expected behavior:

```text
A* finds a path around the front box.
The target is reached.
Minimum clearance remains positive.
The run writes a trace CSV and grid images.
```

## Scenario sweep, standard margin

```bash
for SCENARIO in clear front_box offset_left offset_right wide_box long_box rotated_box round multi; do
  echo "=== $SCENARIO ==="
  python scripts/smoke_test_nav.py \
    --scenario "$SCENARIO" \
    --safety_margin 0.35 \
    --max_steps 2500 \
    --print_every 100 \
    --save_every 200 \
    --assert_planner \
    --assert_success \
    --assert_no_collision \
    --assert_margin
done
```

## Margin sensitivity on `front_box`

```bash
for MARGIN in 0.25 0.30 0.35 0.40 0.45; do
  echo "=== safety_margin=$MARGIN ==="
  python scripts/smoke_test_nav.py \
    --scenario front_box \
    --safety_margin "$MARGIN" \
    --max_steps 2500 \
    --print_every 100 \
    --save_every 200 \
    --assert_planner \
    --assert_success \
    --assert_no_collision \
    --assert_margin
done
```

## Blocked-wall expected failure

Use this to validate that the planner can correctly report no valid route when the map is blocked.

```bash
python scripts/smoke_test_nav.py \
  --scenario blocked_wall \
  --safety_margin 0.35 \
  --max_steps 1000 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner
```

Expected behavior:

```text
For blocked_wall, the correct outcome is no valid path or no target success.
Do not combine this scenario with --assert_success.
```

---

# Test commands with Isaac / Isaac Lab

Activate the Isaac Lab environment first:

```bash
conda activate isaaclab
```

Run from the repository root.

## List Isaac/LiDAR scenarios

```bash
python scripts/smoke_test_nav_lidar.py --list_scenarios
```

## Known-geometry mode inside Isaac

This validates Isaac/G1 proxy motion with scenario-derived occupancy, not pure LiDAR mapping.

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source scenario \
  --safety_margin 0.35 \
  --max_steps 2000 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision \
  --assert_margin
```

Use this when checking whether the Isaac proxy/controller matches the non-Isaac baseline.

## LiDAR-only mode

This validates the RayCaster/LiDAR bridge and LiDAR-derived occupancy.

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source lidar \
  --safety_margin 0.35 \
  --max_steps 2000 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

Do not over-interpret `--assert_margin` in pure LiDAR mode until conservative obstacle thickness and persistent mapping are implemented. A single first-return scan does not fully represent hidden obstacle volume.

## Hybrid mode

This combines LiDAR hits and scenario footprints. It is useful as an integration mode while pure LiDAR mapping is under development.

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source hybrid \
  --safety_margin 0.35 \
  --max_steps 2000 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision \
  --assert_margin
```

## Multi-obstacle scene

The current default should use aggregate RayCaster mesh mode automatically.

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario multi \
  --grid_source lidar \
  --raycast_mesh_mode auto \
  --safety_margin 0.35 \
  --max_steps 2500 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

If debugging RayCaster multi-mesh behavior:

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario multi \
  --grid_source lidar \
  --raycast_mesh_mode actual \
  --safety_margin 0.35 \
  --max_steps 20 \
  --print_every 1
```

If `actual` crashes but `auto` or `aggregate` works, the issue is Isaac/RayCaster multi-mesh initialization, not scenario creation.

## Blocked wall, LiDAR-only

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario blocked_wall \
  --grid_source lidar \
  --safety_margin 0.35 \
  --max_steps 200 \
  --print_every 20 \
  --save_every 20 \
  --assert_planner
```

Expected behavior:

```text
The wall is visible to RayCaster.
A* should not find a valid path if the inflated wall blocks the planning map.
Do not use --assert_success for blocked_wall.
```

## Faster Isaac runs

The script can become slow when saving images frequently. For faster smoke tests:

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source lidar \
  --safety_margin 0.35 \
  --max_steps 2000 \
  --print_every 100 \
  --save_every 250 \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

Or disable map image writing:

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source lidar \
  --safety_margin 0.35 \
  --max_steps 2000 \
  --print_every 100 \
  --no_save_grid \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

---

## Output files

Non-Isaac baseline usually writes under:

```text
outputs/nav/<scenario>/
```

Isaac/LiDAR baseline writes under:

```text
outputs/nav_lidar/<scenario>/
```

Important files:

```text
nav_lidar_trace.csv
occupancy_map_000000.png
occupancy_map_000100.png
...
```

The trace CSV records pose, waypoint index, target distance, clearance metrics, command values, LiDAR validity, and occupied-cell counts.

---

## Pass/fail criteria

For normal reachable scenarios:

```text
A* found = True
target reached = True
collision detected = False
minimum scenario edge clearance > 0
minimum safety-margin slack >= -margin_tolerance
```

For blocked scenarios:

```text
A* found = False
or target reached = False
```

Do not require `--assert_success` on intentionally blocked scenarios.

---

## Known caveats

1. Pure LiDAR mode currently sees first-return surfaces, not full obstacle volumes.
2. Persistent mapping is not yet part of the canonical baseline.
3. Replanning is intentionally disabled in the current minimal script.
4. Hybrid mode is an integration aid, not a pure perception benchmark.
5. Frequent image saving slows Isaac runs significantly.
6. Multi-obstacle RayCaster scenes should use aggregate mesh mode unless debugging direct multi-mesh targeting.

---

## Next planned navigation work

Recommended sequence:

1. Add persistent world-frame LiDAR occupancy map.
2. Add conservative obstacle thickness for LiDAR hits.
3. Track unknown / observed-free / occupied / conservatively occupied states explicitly.
4. Add path invalidation checks against the updated map.
5. Add event-driven replanning when the current path becomes inconsistent.
6. Only after that reintroduce shortcutting, smoothing, local safety, or learned policies.

This keeps the deterministic baseline interpretable while moving toward a production-oriented autonomy stack for Unitree G1.
