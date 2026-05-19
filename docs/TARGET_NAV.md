# TARGET_NAV.md

## Purpose

This document summarizes the current **target-driven navigation smoke test** for the Unitree G1 autonomy stack in Isaac Sim / Isaac Lab.

The goal of this phase was to connect perception-derived target selection to the existing deterministic LiDAR navigation pipeline:

```text
human/person target in Isaac Sim
    -> RGB-D camera snapshot
    -> YOLO person detection or fallback projection
    -> Target2D(x, y) with stand-off
    -> LiDAR occupancy grid
    -> A* / NavRuntime planning and replanning
    -> G1 proxy motion command sequence: vx, vy, wz
```

The resulting script is:

```text
scripts/smoke_test_yolo_person_nav.py
```

The validated script version in the last successful run was:

```text
smoke_test_yolo_person_nav_v10_numpy_import_fallback_debug
```

---

## What We Built

### 1. Human/person target in Isaac Sim

We added support for placing a person-like target in the scene using a downloaded Isaac Sim People/Characters USD asset:

```text
~/isaacsim_assets_partial/Assets/Isaac/5.1/Isaac/People/Characters/F_Business_02/F_Business_02.usd
```

The character asset is not inserted directly as the transform root. Instead, the script uses a lightweight wrapper prim:

```text
/World/envs/env_0/PersonAsset        # wrapper Xform, transform applied here
/World/envs/env_0/PersonAsset/Model  # referenced/payload character USD
```

This avoids composition/transform problems when applying transforms directly to a complex referenced character root.

### 2. RGB-D snapshot target extraction

The script captures a perception snapshot before navigation starts:

```text
camera warmup
    -> read RGB tensor
    -> copy RGB to CPU NumPy
    -> keep depth tensor on device
    -> run YOLO or fallback bbox
    -> estimate target position
    -> freeze Target2D for navigation
```

The important design decision was to separate **snapshot capture** from the navigation control loop. Perception no longer tries to read live camera/proxy tensors inside every navigation step.

For the last successful run, YOLO detected the person and the script used bbox-local depth scalar extraction:

```text
YOLO detections: 1
class: person
confidence: 0.756
object_xy_w: approximately (3.854, -0.013)
target_xy_w: approximately (3.054, -0.010)
object_error_xy_m: 0.146
```

The target is intentionally closer than the person root because `--stand_off 0.8` asks the robot to stop about 0.8 m before the person.

### 3. Navigation toward perception target

The perception target is passed into the same navigation pipeline used in earlier LiDAR smoke tests:

```text
Target2D(x, y)
    -> occupancy grid from LiDAR / conservative LiDAR memory
    -> inflated grid using body radius + safety margin
    -> A* path
    -> NavRuntime waypoint tracking
    -> command output: vx, vy, wz
```

The navigation controller remains deliberately simple:

```text
direct waypoint tracking
no smoothing
no shortcutting
optional replanning
optional LiDAR safety filter
```

The last successful test used the `offset_left` obstacle scenario and reached the perception-generated target.

---

## Problems Encountered and Resolved

### Problem 1: Person USD asset was incomplete or crashed during loading

Initial attempts failed because the asset was not fully downloaded or because transforms were being applied directly to the referenced character root.

**Fix:**

Use a wrapper prim and load the character as a child payload/reference:

```text
/World/envs/env_0/PersonAsset
/World/envs/env_0/PersonAsset/Model
```

Apply translation/rotation/scale to the wrapper, not to the referenced model root.

---

### Problem 2: Isaac appeared stuck during startup or scene setup

Several early runs stopped before our script checkpoints. Some cases were caused by stale Kit/Isaac processes and asset/cache locks.

**Fix:**

Use diagnostics/checkpoints and kill stale processes when necessary:

```bash
ps -ef | grep -E "isaac|Isaac|kit|smoke_test" | grep -v grep
nvidia-smi
pkill -f smoke_test_yolo_person_nav.py
pkill -f isaaclab.python.rendering.kit
pkill -f Isaac-Sim
pkill -f isaacsim
```

---

### Problem 3: Full depth image readback blocked

The perception-only script could copy RGB-D in a simple context, but the full navigation script blocked when copying the full depth tensor from CUDA to CPU.

RGB copy worked:

```text
rgb tensor -> CPU NumPy: OK
```

Depth copy could block:

```text
depth tensor -> CPU NumPy: blocked / unstable
```

This is likely due to Isaac/RTX/Fabric synchronization. RGB and depth are produced by different render paths, and depth readback is a stronger synchronization point.

**Fix:**

Do not copy the full depth image by default. The script now does:

```text
copy RGB image to CPU
run YOLO / fallback bbox on RGB
keep depth on CUDA
crop only bbox region on CUDA
compute robust bbox depth scalar
copy only one scalar depth value to CPU
```

For fallback/static-person tests, we also added geometric depth/target mode so the test can avoid depth tensor synchronization entirely when desired.

---

### Problem 4: Perception was mixed into the live navigation loop

The first full-pipeline script tried to read live camera tensors and live proxy pose while the navigation loop was active. This caused fragile GPU synchronization behavior.

**Fix:**

Use a snapshot-first architecture:

```text
one RGB/depth/pose snapshot before navigation
    -> Target2D
    -> frozen target used by navigation
```

For later moving-person tests, this should become periodic snapshot refresh:

```text
control loop: high rate
perception refresh: lower rate, e.g. every N steps
```

but perception code should still consume a CPU/snapshot object rather than directly touching live Isaac tensors throughout the control loop.

---

### Problem 5: Missing NumPy import in the fallback projection path

The script used `np.array`, `np.linalg.inv`, and other NumPy utilities but did not import NumPy.

**Fix:**

Add:

```python
import numpy as np
```

and add debug checkpoints around fallback projection.

---

### Problem 6: Planner crash during replanning

Once perception worked and navigation started, the planner crashed with:

```text
RuntimeError: deque mutated during iteration
```

The traceback pointed to:

```text
src/g1_nav_rl/planning/grid_astar.py
_nearest_free_cell()
```

The function was iterating over a deque while appending to it.

**Fix:**

Replace the loop with standard BFS:

```python
while queue:
    row, col = queue.popleft()
    ...
    queue.append(...)
```

After this fix, the full pipeline ran to completion.

---

## Last Successful Command

The pasted shell command from the log had a malformed tail, but the run itself succeeded with the effective defaults/options shown by the script output. Use this clean explicit version for reproduction:

```bash
python scripts/smoke_test_yolo_person_nav.py \
  --scenario offset_left \
  --person_asset_usd ~/isaacsim_assets_partial/Assets/Isaac/5.1/Isaac/People/Characters/F_Business_02/F_Business_02.usd \
  --person_asset_reference_mode payload \
  --person_asset_scale 1.0 \
  --person_x 4.0 \
  --person_y 0.0 \
  --stand_off 0.8 \
  --target_source auto \
  --target_depth_source auto \
  --grid_source lidar \
  --enable_replanning \
  --enable_safety \
  --replan_every 50 \
  --path_check_ignore_distance 0.75 \
  --path_check_lookahead 1.5 \
  --safety_margin 0.35 \
  --max_steps 1600 \
  --print_every 50 \
  --save_every 100 \
  --assert_planner \
  --assert_success \
  --assert_no_collision
```

---

## Command Explanation

### Scenario and target asset

```bash
--scenario offset_left
```

Uses the offset-left obstacle scenario. The obstacle is front-left, so the path should usually pass to the right/center while preserving clearance.

```bash
--person_asset_usd ~/isaacsim_assets_partial/Assets/Isaac/5.1/Isaac/People/Characters/F_Business_02/F_Business_02.usd
```

Path to the Isaac Sim human/person USD asset.

```bash
--person_asset_reference_mode payload
```

Loads the character as a payload under the wrapper prim. This is safer for complex character USDs.

```bash
--person_asset_scale 1.0
```

Keeps the asset at original scale.

```bash
--person_x 4.0
--person_y 0.0
```

Places the person target root at world position approximately `(4.0, 0.0)`.

```bash
--stand_off 0.8
```

The robot does not navigate to the person center. It navigates to a target about 0.8 m before the detected person/object location.

---

### Perception target options

```bash
--target_source auto
```

Try YOLO first. If detection is unavailable or insufficient, use fallback synthetic projection when allowed by the script.

```bash
--target_depth_source auto
```

Use bbox-local depth scalar for YOLO detections. For explicit fallback/static tests, the script can use geometric known-person target projection to avoid depth synchronization.

The last successful run used YOLO detection and bbox-local depth scalar:

```text
source: yolo
class: person
confidence: 0.756
valid depth pixels: 4443
target_xy_w: approximately (3.054, -0.010)
```

---

### Occupancy and navigation

```bash
--grid_source lidar
```

Build the planner occupancy grid from LiDAR observations and conservative LiDAR obstacle memory.

```bash
--enable_replanning
```

Allow the planner to replan when the current path is considered blocked or at periodic intervals.

```bash
--replan_every 50
```

Periodic replanning interval in simulation steps.

```bash
--path_check_ignore_distance 0.75
```

Ignore path-blocking checks very close to the robot. This avoids immediately rejecting path cells near the robot footprint.

```bash
--path_check_lookahead 1.5
```

Check path viability over the next 1.5 m of the path.

```bash
--safety_margin 0.35
```

Extra clearance margin used for obstacle inflation and assertions.

```bash
--enable_safety
```

Enable LiDAR-based safety filtering. In the successful run, the filter did not need to override commands:

```text
safety override count: 0
```

---

### Runtime and assertions

```bash
--max_steps 1600
```

Maximum simulation/control steps.

```bash
--print_every 50
```

Print status every 50 steps.

```bash
--save_every 100
```

Save occupancy visualization every 100 steps.

```bash
--assert_planner
```

Fail if no planner path is produced.

```bash
--assert_success
```

Fail if the target is not reached.

```bash
--assert_no_collision
```

Fail if collision/negative clearance is detected.

---

## Last Successful Result

The successful `offset_left` run produced:

```text
target reached: True step=693
target distance first/last/min: 3.054 / 0.748 / 0.748
min scenario edge clearance: 0.455
min scenario safety-margin slack: 0.105
min grid edge clearance: 0.385
replan count: 267
safety override count: 0
RGB-D safety override count: 0
```

The navigation stopped when target distance reached approximately `0.748 m`, which is inside the target-reached radius:

```text
target reached radius: 0.750
```

So the test reached the perception-derived stand-off target successfully.

---

## Remaining Issues / Next Tuning

### 1. Replanning is too frequent

The last successful run had:

```text
replan count: 267
```

This is correct but inefficient. It means the conservative path-check logic often decided the path was blocked and accepted a new plan, even while the robot was safely progressing.

Suggested tuning experiment:

```bash
--path_check_ignore_distance 1.0 \
--path_check_lookahead 1.0 \
--replan_every 100
```

Compare:

```text
target reached
min clearance
replan count
trajectory smoothness
```

### 2. RGB-D safety remains disabled in the successful run

The successful run had:

```text
RGB-D/depth safety filter: disabled
```

This is intentional for the current baseline because depth readback is still sync-sensitive. Re-enable RGB-D safety only after the base perception-target navigation remains stable.

### 3. Moving target support is not implemented yet

The current implementation freezes the perception target before navigation. This is correct for a static target smoke test.

For moving people, use periodic snapshot updates:

```text
capture RGB snapshot every N steps
run YOLO on snapshot
update Target2D
NavRuntime replans if target moved enough
```

Do not directly process live Isaac tensors inside every control step.

### 4. Keep snapshot/perception code separated from navigation code

Perception should output:

```text
Target2D(x, y)
metadata
confidence
source
```

Navigation should consume only:

```text
Target2D + LiDAR occupancy + robot pose -> MotionCommand(vx, vy, wz)
```

This keeps the deterministic navigation pipeline testable independently from detector/sensor issues.
