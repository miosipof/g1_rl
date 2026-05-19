# NAV.md — Deterministic 2D Navigation Baseline for Unitree G1

## Purpose

This document summarizes the current deterministic navigation stack for the Unitree G1 autonomy project.

The current goal is intentionally limited:

```text
Target2D(x, y) + robot pose + current occupancy grid
    -> A* path
    -> waypoint-following controller
    -> MotionCommand(vx, vy, wz)
```

The implementation is based on the behavior already validated in the smoke tests, especially `scripts/smoke_test_nav.py` and `scripts/smoke_test_nav_lidar.py`. The extracted modules are meant to be equivalent to the smoke-test navigation logic, not a new architecture.

---

## Current status

We now have three levels of validation.

### 1. Non-Isaac baseline

```text
scripts/smoke_test_nav.py
```

This is the original clean deterministic baseline:

- analytic 2D scenarios,
- scenario-derived occupancy grid,
- A* planning,
- obstacle inflation,
- direct waypoint following,
- no Isaac Sim,
- no RayCaster,
- no LiDAR.

Known-good result:

```text
scenario: front_box
safety_margin: 0.35
target reached: True
step: ~966
```

### 2. Isaac/LiDAR smoke test

```text
scripts/smoke_test_nav_lidar.py
```

This runs the same deterministic navigation logic in Isaac Lab / Isaac Sim with:

- G1 command proxy,
- RayCaster/LiDAR access,
- target marker,
- USD obstacle scene,
- occupancy-grid construction from LiDAR / scenario / hybrid sources,
- fixed world-frame visualization,
- optional path viability checks and replanning.

Current script version should report something like:

```text
smoke_test_nav_lidar_v11_phase2_nav_runtime
```

### 3. Extracted navigation modules

The reusable navigation logic now lives under:

```text
src/g1_nav_rl/navigation/
```

The important files are:

```text
src/g1_nav_rl/navigation/simple_nav.py
src/g1_nav_rl/navigation/runtime.py
src/g1_nav_rl/navigation/__init__.py
```

The smoke tests still own Isaac, LiDAR, scene setup, visualization, and file output. The navigation modules own only target-to-command logic and runtime replanning decisions.

---

## Current module responsibilities

### `simple_nav.py`

This is the Phase 1 extraction.

It provides the minimal deterministic navigation core:

```python
Target2D
Pose2D
MotionCommand
SimpleNavConfig
SimpleNavigator
```

Core behavior:

```text
Target2D + Pose2D + body-frame occupancy grid
    -> A* path in body frame
    -> path converted to world frame
    -> monotonic waypoint following
    -> MotionCommand(vx, vy, wz)
```

It intentionally does **not** implement:

- Isaac,
- LiDAR access,
- RGB-D,
- YOLO,
- persistent mapping,
- local safety overrides,
- shortcutting,
- smoothing,
- learned policy.

Main API pattern:

```python
from g1_nav_rl.navigation import Pose2D, Target2D, SimpleNavigator

navigator = SimpleNavigator(grid_cfg, astar_cfg, nav_cfg)

plan = navigator.plan(
    pose=Pose2D(x=0.0, y=0.0, yaw=0.0),
    target=Target2D(x=4.0, y=0.0),
    occupancy_grid=grid,
)

nav_cmd = navigator.command(current_pose, target)
cmd = nav_cmd.command       # MotionCommand(vx, vy, wz)
```

Or one-step use:

```python
nav_cmd = navigator.step(
    pose=current_pose,
    target=target,
    occupancy_grid=grid,
)
```

`SimpleNavigator.step()` plans only when no current plan exists. It does not automatically replan.

---

### `runtime.py`

This is the corrected Phase 2 extraction.

It provides a small online wrapper around `SimpleNavigator`:

```python
NavRuntime
NavRuntimeConfig
NavRuntimeStep
PathConflictResult
```

Core behavior:

```text
Target2D + Pose2D + current occupancy grid
    -> keep current path
    -> check whether the remaining path intersects current inflated occupancy
    -> optionally replan if path is blocked or periodic replanning is enabled
    -> emit MotionCommand(vx, vy, wz)
```

It intentionally does **not** own:

- LiDAR acquisition,
- occupancy-grid construction,
- Isaac scene state,
- RGB-D,
- visualization,
- real G1 command transport.

Main API pattern:

```python
from g1_nav_rl.navigation import NavRuntime, NavRuntimeConfig

runtime = NavRuntime(
    navigator,
    NavRuntimeConfig(
        enable_replanning=True,
        replan_every=25,
        path_check_ignore_distance=0.25,
        path_check_lookahead=2.0,
        stop_if_replan_fails=False,
    ),
)

step = runtime.step(
    pose=current_pose,
    target=target,
    occupancy_grid=current_grid,
    step_index=i,
)

cmd = step.command.command   # MotionCommand(vx, vy, wz)
```

---

## Core algorithm

The deterministic navigation algorithm is:

1. Receive a world-frame target:

   ```text
   Target2D(x, y)
   ```

2. Receive the current world-frame planar robot pose:

   ```text
   Pose2D(x, y, yaw)
   ```

3. Receive a current body-frame occupancy grid.

4. Inflate occupied cells by:

   ```text
   inflation_radius = robot_body_radius + safety_margin
   ```

5. Convert the target from world frame into robot/body frame.

6. Run A* from the robot origin `(0, 0)` to the target cell.

7. Convert the A* body-frame path back into world-frame waypoints.

8. Follow waypoints monotonically.

9. Convert the active waypoint into body-frame command:

   ```text
   vx = clipped(k_xy * waypoint_x_body)
   vy = clipped(k_xy * waypoint_y_body)
   wz = clipped(k_yaw * yaw_error_to_path_tangent)
   ```

10. Return:

   ```text
   MotionCommand(vx, vy, wz)
   ```

The baseline still avoids shortcutting, smoothing, and local escape behavior by default.

---

## Controller defaults

The default command behavior matches the smoke-test baseline:

```text
waypoint_reached_radius = 0.12 m
success_radius          = 0.75 m
k_xy                    = 1.5
k_yaw                   = 2.0
max_vx                  = 0.30 m/s
max_vy                  = 0.25 m/s
max_wz                  = 0.60 rad/s
allow_backward          = False
```

Typical obstacle inflation:

```text
robot_body_radius = 0.34 m
safety_margin     = 0.35 m
inflation_radius  = 0.69 m
```

---

## Occupancy sources in Isaac/LiDAR mode

`smoke_test_nav_lidar.py` supports:

```text
--grid_source scenario
--grid_source lidar
--grid_source hybrid
```

### `scenario`

Uses known scenario footprints. This is closest to the original non-Isaac `smoke_test_nav.py` logic.

Use this to verify the planner/controller inside Isaac when the 2D map is known.

### `lidar`

Uses RayCaster/LiDAR returns.

This is closer to perception-based navigation, but pure 2D LiDAR gives first-return surfaces, not complete hidden obstacle volume.

### `hybrid`

Combines LiDAR-derived occupancy and scenario-derived footprints.

This is currently the most stable full integration test mode.

---

## Visualization in Isaac/LiDAR mode

The current script saves one normal occupancy image per saved step:

```text
outputs/nav_lidar/<scenario>/grid/occupancy_map_000000.png
outputs/nav_lidar/<scenario>/grid/occupancy_map_000100.png
...
```

The image is fixed in the world/start frame and shows:

```text
gray       = unknown
white      = observed free
dark       = occupied
red dashed = real obstacle contour from scenario setup, not LiDAR
green      = start point
blue       = target
yellow     = current robot position
dark gray  = executed trajectory
light gray = swept safety zone around executed trajectory
orange     = current remaining plan
```

The red dashed contour is deliberately drawn from the setup geometry. It is used to compare real obstacle footprint against LiDAR-derived occupancy.

---

## Multi-obstacle RayCaster handling

`multi` originally crashed during Isaac `sim.reset()` when RayCaster received multiple mesh prim paths.

Current option:

```text
--raycast_mesh_mode auto|actual|aggregate
```

Default:

```text
--raycast_mesh_mode auto
```

Behavior:

```text
actual    = RayCaster targets real obstacle meshes directly
aggregate = script creates one raycast-only aggregate mesh
auto      = uses aggregate when there is more than one obstacle
```

For `multi`, use `auto` or `aggregate` unless specifically debugging RayCaster multi-mesh behavior.

---

## Test commands without Isaac

Run from the repository root.

Set Python path:

```bash
export PYTHONPATH=src
```

### 1. Check the extracted `SimpleNavigator`

This validates:

```text
Target2D + Pose2D + occupancy grid -> A* path -> MotionCommand(vx, vy, wz)
```

Command:

```bash
PYTHONPATH=src python scripts/smoke_test_nav_module.py
```

Expected output:

```text
plan found=True
target_reached=True
commands=...
```

### 2. Check the extracted `NavRuntime`

This validates:

```text
Target2D + Pose2D + updated occupancy grid
    -> keep path
    -> detect blocked path
    -> replan
    -> MotionCommand(vx, vy, wz)
```

Command:

```bash
PYTHONPATH=src python scripts/smoke_test_nav_runtime.py
```

Expected output:

```text
initial plan found=True
replan accepted count=1 reason=path_blocked
runtime smoke test passed
```

### 3. Original non-Isaac navigation baseline

Command:

```bash
PYTHONPATH=src python scripts/smoke_test_nav.py \
  --scenario front_box \
  --safety_margin 0.35 \
  --max_steps 1000 \
  --print_every 100 \
  --no_save_grid \
  --assert_planner \
  --assert_success \
  --assert_no_collision \
  --assert_margin
```

Expected output:

```text
A* found=True
target reached=True
min scenario safety-margin slack > 0
```

### 4. Non-Isaac scenario sweep

```bash
for SCENARIO in clear front_box offset_left offset_right wide_box long_box rotated_box round multi; do
  echo "=== $SCENARIO ==="
  PYTHONPATH=src python scripts/smoke_test_nav.py \
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

### 5. Blocked-wall expected failure without Isaac

Do not use `--assert_success` for blocked scenarios.

```bash
PYTHONPATH=src python scripts/smoke_test_nav.py \
  --scenario blocked_wall \
  --safety_margin 0.35 \
  --max_steps 1000 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner
```

Expected behavior:

```text
A* should report no valid path, or the target should not be reached.
```

---

## Test commands with Isaac / Isaac Lab

Activate Isaac Lab first:

```bash
conda activate isaaclab
cd ~/g1_rl
export PYTHONPATH=src
```

### 1. List Isaac/LiDAR scenarios

```bash
python scripts/smoke_test_nav_lidar.py --list_scenarios
```

### 2. Full pipeline, stable hybrid mode

This is the recommended regression command for the current full pipeline:

```text
Target2D + Isaac G1 proxy pose + RayCaster LiDAR + occupancy grid
    -> NavRuntime
    -> SimpleNavigator
    -> MotionCommand(vx, vy, wz)
    -> G1 proxy motion
```

Command:

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source hybrid \
  --enable_replanning \
  --replan_every 25 \
  --path_check_lookahead 2.0 \
  --path_check_ignore_distance 0.25 \
  --safety_margin 0.35 \
  --max_steps 1200 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision \
  --assert_margin
```

Expected behavior:

```text
script version: smoke_test_nav_lidar_v11_phase2_nav_runtime
A* found=True
target reached=True
no collision
min safety-margin slack > 0
```

### 3. Full pipeline, LiDAR-only mode

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source lidar \
  --enable_replanning \
  --replan_every 25 \
  --path_check_lookahead 2.0 \
  --path_check_ignore_distance 0.25 \
  --safety_margin 0.35 \
  --max_steps 1200 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

Expected behavior:

```text
lidar valid hits > 0
A* found=True
target reached=True
```

Do not treat this as a complete proof of obstacle-volume reasoning. Pure LiDAR mode still sees first-return surfaces only.

### 4. Known-geometry mode inside Isaac

This is closest to the original non-Isaac baseline, but executed through the Isaac/G1 proxy loop.

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source scenario \
  --safety_margin 0.35 \
  --max_steps 1200 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision \
  --assert_margin
```

### 5. Multi-obstacle scene

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario multi \
  --grid_source lidar \
  --raycast_mesh_mode auto \
  --enable_replanning \
  --replan_every 25 \
  --path_check_lookahead 2.0 \
  --safety_margin 0.35 \
  --max_steps 1800 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

Expected early log:

```text
RayCaster mesh mode: auto
using aggregate=True
RayCaster mesh prim paths: ['/World/envs/env_0/_RaycastAggregate/geometry/mesh']
```

Expected final behavior:

```text
target reached=True
no collision
```

### 6. Blocked-wall expected failure with Isaac

Use this to check that the planner does not claim a valid route through a map-blocking wall.

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario blocked_wall \
  --grid_source hybrid \
  --enable_replanning \
  --replan_every 25 \
  --safety_margin 0.35 \
  --max_steps 200 \
  --print_every 25 \
  --save_every 50 \
  --assert_planner
```

Expected behavior:

```text
A* found=False
target reached=False
```

Do not use `--assert_success` for `blocked_wall`.

### 7. Faster full-pipeline check

Saving PNGs is one of the slowest parts of the Isaac run. For speed:

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source hybrid \
  --enable_replanning \
  --replan_every 50 \
  --safety_margin 0.35 \
  --max_steps 1200 \
  --print_every 100 \
  --save_every 100000 \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

Or disable grid images if the script supports it:

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario front_box \
  --grid_source hybrid \
  --enable_replanning \
  --replan_every 50 \
  --safety_margin 0.35 \
  --max_steps 1200 \
  --print_every 100 \
  --no_save_grid \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

---

## Scenario batch command with Isaac

Reachable scenarios:

```bash
for SCENE in front_box offset_box multi gap_corridor; do
  echo "=== $SCENE ==="
  python scripts/smoke_test_nav_lidar.py \
    --scenario "$SCENE" \
    --grid_source hybrid \
    --raycast_mesh_mode auto \
    --enable_replanning \
    --replan_every 25 \
    --safety_margin 0.35 \
    --max_steps 1800 \
    --print_every 100 \
    --save_every 300 \
    --assert_planner \
    --assert_success \
    --assert_no_collision \
    --assert_margin
done
```

Keep expected-failure scenarios separate:

```bash
python scripts/smoke_test_nav_lidar.py \
  --scenario blocked_wall \
  --grid_source hybrid \
  --raycast_mesh_mode auto \
  --enable_replanning \
  --replan_every 25 \
  --safety_margin 0.35 \
  --max_steps 200 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner
```

If a scenario name is not available in the current script, list available scenarios first:

```bash
python scripts/smoke_test_nav_lidar.py --list_scenarios
```

---

## Output files

Non-Isaac baseline:

```text
outputs/nav/<scenario>/
```

Isaac/LiDAR baseline:

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

The trace CSV records pose, target distance, waypoint index, clearance metrics, command values, LiDAR validity, occupied-cell counts, and replanning-related information when enabled.

---

## Pass/fail criteria

For reachable scenarios:

```text
A* found = True
target reached = True
collision detected = False
minimum scenario edge clearance > 0
minimum safety-margin slack >= 0, or within explicit tolerance
commands remain finite
```

For blocked scenarios:

```text
A* found = False
or target reached = False
```

Do not combine blocked scenarios with `--assert_success`.

---

## Known limitations

1. Pure LiDAR mode observes first-return surfaces, not full hidden obstacle volumes.
2. There is not yet a persistent semantic/occupancy map with explicit unknown/free/occupied/conservative states.
3. Replanning checks the current path against the current occupancy grid; it does not yet solve long-term mapping uncertainty.
4. Hybrid mode is an integration aid, not a pure perception benchmark.
5. Frequent image saving significantly slows Isaac runs.
6. `multi` should use `--raycast_mesh_mode auto` or `aggregate` unless debugging direct RayCaster multi-mesh behavior.
7. No RGB-D safety, YOLO target acquisition, or learned policy is part of the current deterministic baseline.

---

## Recommended next step

The next useful step is not another abstraction layer. The next useful step is a small persistent LiDAR map:

```text
instantaneous LiDAR observations
    -> persistent world-frame occupancy state
    -> observed free / observed occupied / unknown
    -> conservative obstacle thickness around hits
    -> path invalidation against persistent inflated occupancy
```

This directly addresses the remaining real limitation: a single LiDAR scan exposes surfaces, not full obstacle volume.

After that, add safety/stuck heuristics as a separate layer:

```text
nominal MotionCommand(vx, vy, wz)
    -> LiDAR/RGB-D safety filter
    -> final command
```

The deterministic planner should remain the reference backend even if an RL policy is later introduced behind the same `Target2D -> MotionCommand` interface.
