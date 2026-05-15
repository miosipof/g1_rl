# G1 Navigation RL — Project Introduction

## Purpose

This repository is an Isaac Sim / Isaac Lab prototype for building a **high-level navigation stack for Unitree G1-like operation**.

The key design decision is that we are **not trying to train or simulate full low-level humanoid locomotion**. On the real G1, the intended deployment interface is expected to be high-level movement commands such as:

```text
move(vx, vy, wz)
```

where:

- `vx` = forward/backward velocity in the robot/body frame
- `vy` = lateral velocity in the robot/body frame
- `wz` = yaw rate

Therefore, the simulation stack is being shaped around the same abstraction:

```text
G1-like perception + command-level 2D motion interface
```

instead of:

```text
full G1 joint-level balance and locomotion policy
```

This keeps the simulation transferable to the real robot: policies trained or tested in simulation should output the same type of command that the physical robot will receive.

---

## Current repository snapshot

The uploaded archive contains the following meaningful files:

```text
configs/
  robot/
    unitree_g1.yaml
    g1_command_proxy.yaml
  env/
    nav_env.yaml          # placeholder, currently empty
  train/
    ppo.yaml              # placeholder, currently empty

scripts/
  smoke_test_scene.py
  smoke_test_command_proxy.py
  smoke_test_go_to_target.py
  train.py                # placeholder, currently empty
  eval.py                 # placeholder, currently empty

src/g1_nav_rl/
  observations/
    rgbd_obs.py
  robots/
    g1_command_proxy.py
  envs/
    nav_env.py            # placeholder, currently empty
  rewards/
    navigation_reward.py  # placeholder, currently empty
```

`README.md` is currently empty, so this file is intended to document the current state and rationale of the work.

---

## What has been implemented

### 1. Built-in Isaac Lab G1 smoke test

File:

```text
scripts/smoke_test_scene.py
configs/robot/unitree_g1.yaml
src/g1_nav_rl/observations/rgbd_obs.py
```

This was the first validation step.

It uses Isaac Lab’s built-in `G1_CFG` asset and attaches a front RGB-D camera to the G1 torso link:

```text
{ENV_REGEX_NS}/Robot/torso_link/front_cam
```

The purpose was to confirm that Isaac Sim / Isaac Lab can:

- load the G1 asset,
- run headless rendering,
- attach an RGB-D camera,
- produce RGB tensors,
- produce depth tensors,
- expose proprioceptive robot state,
- save camera screenshots.

The observation helper returns:

```python
{
    "rgb": rgb,                         # [N, H, W, 3]
    "depth": depth,                     # [N, H, W, 1]
    "proprio": proprio,                 # base + joint state
    "base_lin_vel_b": base_lin_vel_b,
    "base_ang_vel_b": base_ang_vel_b,
    "gravity_b": gravity_b,
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
}
```

This confirmed that Isaac Lab can provide the perception signals we need. However, it also showed that using the full G1 articulation directly is not the right path for the high-level navigation objective, because we do not want to solve low-level humanoid balance and gait control in this project.

---

### 2. Command-level G1 proxy

Files:

```text
src/g1_nav_rl/robots/g1_command_proxy.py
configs/robot/g1_command_proxy.yaml
scripts/smoke_test_command_proxy.py
```

This is the main architectural pivot.

`G1CommandProxy` models the real robot interface we intend to use:

```python
proxy.set_command([vx, vy, wz])
proxy.step(dt)
```

It keeps internal state:

```text
x, y, z, yaw, vx, vy, wz
```

and returns command-level proprioception:

```text
[x, y, yaw, vx, vy, wz]
```

The proxy applies:

- velocity limits,
- acceleration limits,
- yaw-rate limits,
- smoothing,
- body-frame to world-frame velocity integration.

Important implementation detail:

```text
The proxy deliberately avoids PhysX rigid-body tensor writes.
```

Instead, it moves a clean USD `Xform` prim directly. This was done because the earlier PhysX rigid-body version produced CUDA illegal memory access errors in the Isaac Sim / Isaac Lab environment.

The current USD structure is:

```text
/World/envs/env_0/G1Proxy             # clean moving Xform
/World/envs/env_0/G1Proxy/body        # visual proxy body
/World/envs/env_0/G1Proxy/front_cam   # RGB-D camera
```

This avoids transform-operation warnings and keeps the proxy stable.

The command-proxy smoke test executes a scripted command sequence:

```text
0-299     move forward
300-599   rotate
600-899   move laterally
900+      stop
```

This validated:

- `[vx, vy, wz]` commands move the proxy,
- yaw updates correctly,
- the RGB-D camera moves with the proxy,
- screenshots are saved,
- the proprio vector is reduced to the command-level state.

---

### 3. Deterministic go-to-target controller

File:

```text
scripts/smoke_test_go_to_target.py
```

This test adds a simple target marker and computes a deterministic command:

```text
current pose + target pose -> [vx, vy, wz]
```

The target is configured in:

```text
configs/robot/g1_command_proxy.yaml
```

Current target example:

```yaml
target:
  pos: [4.0, 0.0, 0.08]
  success_radius: 0.20
```

The controller works in the robot/body frame. It computes:

- vector from proxy to target,
- desired yaw,
- yaw error,
- target-relative position in the body frame,
- velocity command,
- slow-down behavior near the target,
- stop behavior inside the success radius.

This validated that the proxy can move from `A` to `B` using a reusable high-level navigation command.

Example validated behavior:

```text
start:  [0.0, 0.0]
target: [3.0, 1.0]

step 0     distance ≈ 3.16
step 100   distance ≈ 1.85
step 200   distance ≈ 0.54
step 300   distance ≈ 0.28
success    distance ≈ 0.20
```

This gives us a deterministic baseline before introducing RL.

---

### 4. Depth-based obstacle safety layer

File:

```text
scripts/smoke_test_go_to_target.py
```

The go-to-target smoke test now includes a simple front-depth safety filter:

```text
raw A->B command
    -> depth safety filter
    -> final command sent to proxy
```

The safety layer examines a configurable crop of the depth image:

```yaml
obstacle_safety:
  vertical_band: [0.25, 0.90]
  center_width_frac: 0.60
  safety_distance: 2.0
  hard_stop_distance: 1.1
  forward_scale_at_safety: 0.0
  sidestep_speed: 0.30
  yaw_away_rate: 0.50
```

If an obstacle is too close in the center crop, it:

- suppresses forward velocity,
- chooses a sidestep direction based on which side has more free space,
- adds a yaw-away command.

This was tested with a debug cube placed in the path:

```yaml
debug_cube:
  enabled: true
  size: [0.8, 0.8, 0.8]
  pos: [2.2, 0.0, 0.4]
```

Validated behavior:

```text
raw command        : [3.2, 0.0, 0.0]
target command     : [0.0, -0.3, -0.5]
safety depth       : center=1.450, ... active=True
```

This means:

```text
The target controller wanted to drive forward.
The camera saw an obstacle ahead.
The safety layer blocked forward motion and commanded a sidestep/yaw maneuver.
```

The proxy then successfully moved around the cube and reached the target.

---

## Why we chose this architecture

### We match the real robot command interface

The physical G1 will not be controlled through joint torques for this project. We expect to send high-level motion commands. Therefore, the simulation should also operate at the high-level command interface:

```text
policy output -> [vx, vy, wz]
```

This is the main sim-to-real transfer boundary.

### We avoid solving low-level humanoid locomotion

Training or stabilizing a full humanoid locomotion policy is a separate, much harder problem. It would introduce unnecessary risk and obscure the actual project objective: high-level navigation, target following, obstacle avoidance, and perception-conditioned motion.

### Deterministic baselines come before RL

Before any RL experiment, deterministic policies give us:

- expected behavior,
- measurable baselines,
- debugging tools,
- fallback behavior,
- sanity checks for observations and actions.

The current deterministic stack is intentionally simple but useful:

```text
target position -> go-to-target controller -> depth safety -> [vx, vy, wz]
```

### Safety is outside the policy

Even when we later add learned policies, raw policy output should not go directly to the robot. The safe command path should be:

```text
policy command
    -> command limiter
    -> safety filter
    -> watchdog
    -> sim or real G1 adapter
```

The current depth filter is the first prototype of that safety layer.

---

## Current limitations

The current repository is still a prototype. The following components are not production-ready yet.

### No reusable environment class yet

The main logic still lives in smoke-test scripts.

Current placeholder:

```text
src/g1_nav_rl/envs/nav_env.py
```

Expected next step:

```python
env.reset()
obs, reward, done, info = env.step(action)
```

### Lidar is not implemented yet

The current perception stack has RGB and depth. Lidar is planned but not present in the uploaded repo.

Target production perception:

```text
RGB + depth + lidar/ray-scan + command-level proprioception
```

### No map / costmap yet

The current obstacle handling is direct depth-threshold reflex behavior. It is not a planner.

Target production mapping stack:

```text
lidar scan
    -> 2D occupancy grid
    -> cleanup/filtering
    -> obstacle inflation
    -> local costmap
```

### No trajectory planner yet

The current controller points directly toward the target. It can sidestep around a simple obstacle, but it does not compute a global or local path.

Target deterministic stack:

```text
target_xy
    -> occupancy/costmap
    -> path planner
    -> local waypoint
    -> trajectory tracker
    -> [vx, vy, wz]
```

### No RL training pipeline yet

The placeholders exist:

```text
configs/train/ppo.yaml
scripts/train.py
scripts/eval.py
```

but they are currently empty.

RL should come after the deterministic environment and metrics are stable.

### No YOLO target perception yet

The final target pipeline is planned but not implemented:

```text
RGB frame
    -> YOLO detector
    -> target class selection
    -> depth lookup
    -> camera-to-base transform
    -> temporal tracker
    -> target_xy
```

---

## Recommended next production step

The next step should be to convert the current smoke-test logic into reusable modules.

Recommended target structure:

```text
src/g1_nav_rl/
  robots/
    g1_command_proxy.py

  sensors/
    rgbd.py
    lidar.py

  safety/
    depth_safety.py
    command_limiter.py

  planning/
    go_to_target.py
    local_planner.py
    costmap.py

  envs/
    g1_nav_env.py

  policies/
    deterministic.py
```

Minimum production environment API:

```python
class G1NavEnv:
    def reset(self) -> dict:
        ...

    def step(self, action) -> tuple[dict, float, bool, dict]:
        ...
```

Recommended observation:

```python
{
    "rgb": rgb,
    "depth": depth,
    "lidar": lidar_scan,
    "proprio": [x, y, yaw, vx, vy, wz],
    "target": [dx_body, dy_body, distance, bearing],
}
```

Recommended action:

```python
[vx, vy, wz]
```

Recommended metrics:

```text
success rate
time to target
path length
minimum obstacle distance
number of safety interventions
mean command magnitude
oscillation count
collisions / near-collisions
```

---

## Suggested near-term implementation plan

### Step 1 — Refactor current scripts into modules

Move the current logic out of `smoke_test_go_to_target.py`:

```text
compute_go_to_target_command() -> planning/go_to_target.py
apply_depth_safety_filter()    -> safety/depth_safety.py
build_camera_observation()     -> sensors/rgbd.py
```

Keep the smoke scripts, but make them call the reusable modules.

### Step 2 — Implement `G1NavEnv`

Create:

```text
src/g1_nav_rl/envs/g1_nav_env.py
```

The environment should own:

- scene setup,
- proxy setup,
- observation building,
- action application,
- reward calculation,
- done logic,
- metrics.

### Step 3 — Add lidar

Add a simulated planar lidar or ray-caster sensor. Use it to build a local 2D obstacle representation.

### Step 4 — Add deterministic map-based navigation

Use the lidar-derived grid to build:

```text
occupancy grid -> inflated costmap -> local waypoint -> velocity command
```

### Step 5 — Add real/sim command adapter boundary

Define one command interface:

```python
send_velocity(vx, vy, wz)
```

Implement:

```text
SimCommandAdapter  -> G1CommandProxy
RealG1Adapter      -> real G1 high-level command API
```

### Step 6 — Add RL experiments only after deterministic baseline is stable

Start with low-dimensional observations:

```text
lidar scan + target vector + current velocity
```

Do not start with raw RGB-D end-to-end RL.

### Step 7 — Add YOLO-based target perception

Once navigation is stable, add:

```text
YOLO detection -> RGB-D localization -> temporal tracking -> target_xy
```

Then connect that target estimate to the same deterministic or learned navigation policy.

---

## Current status summary

Working now:

- Isaac Lab G1 RGB-D smoke test
- command-level G1 proxy
- RGB-D camera attached to proxy
- command-level proprioception `[x, y, yaw, vx, vy, wz]`
- scripted command smoke test
- deterministic A-to-B controller
- depth-based front-obstacle safety reflex
- screenshots for RGB and depth

Not implemented yet:

- lidar
- reusable `G1NavEnv`
- map/costmap
- trajectory planner
- Gym/Gymnasium wrapper
- PPO training
- YOLO target detector
- real G1 command adapter
- robustness tests for moving target, sudden obstacles, and occlusion

The repository is therefore at the transition point between **validated smoke-test prototype** and **production navigation environment**.
