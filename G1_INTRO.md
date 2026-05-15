# G1 Navigation RL — Project Introduction

## 1. Purpose

This repository is an Isaac Sim / Isaac Lab prototype for building a **command-level navigation stack for Unitree G1-like operation**.

The central design decision is that this project does **not** attempt to train or simulate full low-level humanoid locomotion. The intended real-robot deployment boundary is expected to be a high-level velocity command interface, conceptually:

```text
move(vx, vy, wz)
```

where:

- `vx` is forward/backward velocity in the robot/body frame,
- `vy` is lateral velocity in the robot/body frame,
- `wz` is yaw rate.

Therefore, the simulation is shaped around the same abstraction:

```text
G1-like perception + command-level 2D motion interface
```

rather than:

```text
full G1 joint-level balance and locomotion policy
```

This aligns the simulated policy boundary with the intended real Unitree G1 command boundary. It improves transferability at the command/API level, but real deployment will still require a real G1 command adapter, command-rate validation, latency handling, sensor-frame calibration, and hard safety limits.

---

## 2. Current repository snapshot

The uploaded archive is a smoke-test prototype. It contains a small set of working scripts and modules plus several placeholders for the future production stack.

```text
g1_rl/
  README.md                         # currently empty
  G1_INTRO.md                       # project overview
  ROADMAP.md                        # implementation roadmap

  configs/
    robot/
      unitree_g1.yaml               # Isaac Lab built-in G1 RGB-D smoke-test config
      g1_command_proxy.yaml         # command-proxy, RGB-D, target, cube, depth-safety config
    env/
      nav_env.yaml                  # placeholder, currently empty
    train/
      ppo.yaml                      # placeholder, currently empty

  scripts/
    smoke_test_scene.py             # built-in G1 articulation + RGB-D smoke test
    smoke_test_command_proxy.py     # command proxy movement smoke test
    smoke_test_go_to_target.py      # go-to-target + depth safety smoke test
    train.py                        # placeholder, currently empty
    eval.py                         # placeholder, currently empty

  src/g1_nav_rl/
    observations/
      rgbd_obs.py                   # RGB-D + proprio observation builder for built-in G1
    robots/
      g1_command_proxy.py           # kinematic command-level proxy
      g1_asset.py                   # placeholder, currently empty
    envs/
      nav_env.py                    # placeholder, currently empty
    rewards/
      navigation_reward.py          # placeholder, currently empty
    utils/
      __init__.py                   # placeholder
```

The current repository should be treated as a validated prototype direction, not as a production navigation environment.

---

## 3. What has been implemented

### 3.1 Built-in Isaac Lab G1 RGB-D smoke test

Relevant files:

```text
scripts/smoke_test_scene.py
configs/robot/unitree_g1.yaml
src/g1_nav_rl/observations/rgbd_obs.py
```

This test uses Isaac Lab's built-in `G1_CFG` articulation and attaches a front RGB-D camera to the G1 torso link:

```text
{ENV_REGEX_NS}/Robot/torso_link/front_cam
```

Its purpose is to check that Isaac Sim / Isaac Lab can:

- load the G1 asset,
- run headless rendering,
- attach an RGB-D camera,
- produce RGB tensors,
- produce depth tensors,
- expose proprioceptive robot state,
- save camera screenshots.

The observation helper returns a dictionary of camera and robot-state tensors:

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

This test is useful for sensor placement, asset sanity checks, and later comparison with real G1 frame layout. It is **not** the main navigation architecture, because this project does not currently solve full humanoid balance or gait control.

---

### 3.2 Command-level G1 proxy

Relevant files:

```text
src/g1_nav_rl/robots/g1_command_proxy.py
configs/robot/g1_command_proxy.yaml
scripts/smoke_test_command_proxy.py
```

`G1CommandProxy` models the high-level command interface we intend to use:

```python
proxy.set_command([vx, vy, wz])
proxy.step(dt)
```

The proxy maintains a compact command-level state:

```text
x, y, z, yaw, vx, vy, wz
```

and exposes command-level proprioception:

```text
[x, y, yaw, vx, vy, wz]
```

The proxy applies:

- velocity limits,
- acceleration limits,
- yaw-rate limits,
- command smoothing,
- body-frame to world-frame velocity integration.

Important implementation detail:

```text
The proxy deliberately avoids PhysX rigid-body tensor writes.
```

Instead, it moves a clean USD `Xform` prim directly. This was chosen because earlier PhysX rigid-body tensor writes caused CUDA illegal memory access errors in the tested Isaac Sim / Isaac Lab environment.

The current USD hierarchy is:

```text
/World/envs/env_0/G1Proxy             # clean moving Xform root
/World/envs/env_0/G1Proxy/body        # visual proxy body
/World/envs/env_0/G1Proxy/front_cam   # RGB-D camera
```

The proxy is **kinematic**. It does not model physical balance, foot contacts, whole-body dynamics, or collision response. Obstacle handling is currently performed through perception and command filtering, not through humanoid physics.

The command-proxy smoke test executes a scripted sequence:

```text
0-299     move forward
300-599   rotate
600-899   move laterally
900+      stop
```

This checks that:

- `[vx, vy, wz]` commands move the proxy,
- yaw updates correctly,
- the RGB-D camera moves with the proxy,
- screenshots can be saved,
- the proprio vector is reduced to command-level state.

---

### 3.3 Deterministic go-to-target controller

Relevant file:

```text
scripts/smoke_test_go_to_target.py
```

This script adds a target marker and computes a deterministic command:

```text
current proxy pose + target pose -> [vx, vy, wz]
```

The target is configured in:

```text
configs/robot/g1_command_proxy.yaml
```

Example target configuration:

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
- slowdown behavior near the target,
- stop behavior inside the success radius.

This provides a deterministic A-to-B baseline before introducing map-based planning or reinforcement learning.

Recommended production destination:

```text
scripts/smoke_test_go_to_target.py
  compute_go_to_target_command()
    -> src/g1_nav_rl/policies/deterministic_go_to_target.py
```

The `policies/` package should contain modules that map observations or target state directly to `[vx, vy, wz]`. The `planning/` package should be reserved for occupancy grids, costmaps, path planning, local waypoint selection, and trajectory planning.

---

### 3.4 Depth-based obstacle safety layer

Relevant file:

```text
scripts/smoke_test_go_to_target.py
```

The go-to-target smoke test includes a simple front-depth safety filter:

```text
raw go-to-target command
    -> depth safety filter
    -> safe command sent to proxy
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

- suppresses or reduces forward velocity,
- chooses a sidestep direction based on which side appears more open,
- adds a yaw-away command.

Example debug-cube configuration:

```yaml
debug_cube:
  enabled: true
  size: [0.8, 0.8, 0.8]
  pos: [2.2, 0.0, 0.4]
```

Example diagnostic interpretation:

```text
raw command        : [3.2, 0.0, 0.0]
safe command       : [0.0, -0.3, -0.5]
safety depth       : center=1.450, ... active=True
```

This means:

```text
The target controller wanted to drive forward.
The camera saw an obstacle ahead.
The safety layer blocked forward motion and commanded a sidestep/yaw maneuver.
```

This is not a full obstacle-avoidance planner. It is a near-field safety reflex.

Recommended production destination:

```text
scripts/smoke_test_go_to_target.py
  apply_depth_safety_filter()
    -> src/g1_nav_rl/safety/depth_safety.py
```

The current filter uses minimum depth in image crops. That is simple but noise-sensitive. A production version should ignore invalid pixels and use a robust statistic such as a low percentile instead of a raw minimum.

---

## 4. Why this architecture is appropriate

### 4.1 It matches the intended real robot control boundary

The physical G1 is expected to be controlled through a high-level motion interface for this project, not through joint torques. Therefore, the simulated policy output should also be:

```text
[vx, vy, wz]
```

This avoids training a policy against a simulation boundary that will not exist during deployment.

---

### 4.2 It avoids turning navigation into a locomotion project

Full humanoid locomotion training would require balance, contact dynamics, gait stabilization, fall recovery, torque limits, and extensive sim-to-real work. That is a separate research problem.

This project is instead focused on:

```text
perception -> target/map reasoning -> safe high-level command generation
```

Low-level walking should remain handled by the robot's built-in controller or SDK-level high-level motion interface.

---

### 4.3 Deterministic baselines come before RL

The current deterministic stack is intentionally simple:

```text
target position -> go-to-target controller -> depth safety -> [vx, vy, wz]
```

This gives the project:

- expected behavior,
- measurable baselines,
- debugging tools,
- fallback behavior,
- sanity checks for observations and actions.

Reinforcement learning should be added only after the deterministic environment, observation API, metrics, and safety layer are stable.

---

### 4.4 Safety must stay outside the learned policy

Even if learned policies are added later, raw policy output should not be sent directly to the simulator or real robot.

The safe command path should be:

```text
policy command
    -> command limiter
    -> safety filter
    -> watchdog
    -> sim or real G1 adapter
```

The current depth filter is the first prototype of this external safety layer.

---

## 5. Current limitations

### 5.1 No reusable environment class yet

The main logic still lives in smoke-test scripts.

Current placeholder:

```text
src/g1_nav_rl/envs/nav_env.py
```

Expected production API:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

---

### 5.2 Lidar is not implemented yet

The current perception stack has RGB and depth. Lidar or ray-scan sensing is planned but not present in the uploaded implementation.

Target production perception:

```text
RGB + depth + lidar/ray-scan + command-level proprioception
```

Lidar should be prioritized next because 2D navigation should primarily use a lidar-derived occupancy grid or costmap. RGB-D should act as near-field safety and, later, semantic perception.

---

### 5.3 No map, costmap, or trajectory planner yet

Current obstacle handling is direct depth-threshold reflex behavior. It is not map-based planning.

Target deterministic navigation stack:

```text
lidar scan
    -> 2D occupancy grid
    -> cleanup/filtering
    -> obstacle inflation
    -> local costmap
    -> path/local waypoint planner
    -> velocity command
    -> command limiter + safety filter
```

---

### 5.4 No RL training pipeline yet

The placeholders exist:

```text
configs/train/ppo.yaml
scripts/train.py
scripts/eval.py
```

but they are currently empty.

When RL is introduced, it should start with low-dimensional observations such as:

```text
lidar scan + target vector + current command/proprio state
```

Raw RGB-D end-to-end RL should not be the first RL experiment.

---

### 5.5 No YOLO or semantic target perception yet

The future target-perception pipeline is planned but not implemented:

```text
RGB frame
    -> YOLO detector
    -> target class selection
    -> depth lookup
    -> camera-to-base transform
    -> temporal tracker
    -> target_xy
```

This should be added after deterministic navigation is stable.

---

### 5.6 No real G1 adapter yet

The repo currently contains only simulation-side proxy logic. Real deployment requires a separate adapter that sends safe velocity commands to the G1 through the selected Unitree SDK/API path.

The production policy should not know whether it is controlling simulation or hardware. That difference should live behind adapters:

```text
SimCommandAdapter  -> G1CommandProxy
RealG1Adapter      -> physical Unitree G1 high-level command API
```

---

## 6. Recommended next production step

The next step is to move reusable logic out of the smoke-test scripts into modules.

Recommended near-term structure:

```text
src/g1_nav_rl/
  robots/
    g1_command_proxy.py
    command_types.py
    adapters.py

  observations/
    rgbd_articulation_obs.py
    rgbd_proxy_obs.py
    nav_obs.py

  sensors/
    lidar.py

  safety/
    command_limiter.py
    depth_safety.py
    lidar_safety.py
    watchdog.py

  policies/
    deterministic_go_to_target.py
    deterministic_target_following.py

  planning/
    occupancy_grid.py
    costmap.py
    local_planner.py
    trajectory_tracker.py

  envs/
    g1_nav_env.py

  metrics/
    navigation_metrics.py
```

Consistent migration targets:

| Current script/function | Production destination |
|---|---|
| `G1CommandProxy` | `src/g1_nav_rl/robots/g1_command_proxy.py` |
| `compute_go_to_target_command()` | `src/g1_nav_rl/policies/deterministic_go_to_target.py` |
| `apply_depth_safety_filter()` | `src/g1_nav_rl/safety/depth_safety.py` |
| proxy RGB-D observation builder | `src/g1_nav_rl/observations/rgbd_proxy_obs.py` |
| built-in G1 RGB-D observation builder | `src/g1_nav_rl/observations/rgbd_articulation_obs.py` |
| target/debug object spawning | `src/g1_nav_rl/scenes/objects.py` |
| smoke-test scene assembly | `src/g1_nav_rl/envs/g1_nav_env.py` or `src/g1_nav_rl/scenes/g1_nav_scene.py` |

---

## 7. Minimum target environment API

The production environment should expose:

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
    "costmap": optional_local_costmap,
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

## 8. Immediate cleanup items

Before further development, clean the repository/archive hygiene:

```text
remove .git/ from shared zip archives unless intentionally sharing history
remove __MACOSX/
remove __pycache__/
remove *.pyc
fill README.md with status and run instructions
add pyproject.toml or equivalent install metadata
```

Recommended `.gitignore` additions:

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.DS_Store
__MACOSX/
outputs/
logs/
.venv/
```

Also fix the proxy-body initialization inconsistency between `smoke_test_command_proxy.py` and `smoke_test_go_to_target.py`: the moving root should own global pose, and the body should be local under the root.

---

## 9. Current status summary

Working or implemented in prototype form:

- Isaac Lab built-in G1 RGB-D smoke test,
- command-level kinematic G1 proxy,
- RGB-D camera attached to proxy,
- command-level proprioception `[x, y, yaw, vx, vy, wz]`,
- scripted command smoke test,
- deterministic A-to-B controller,
- simple depth-based safety reflex.

Not implemented yet:

- reusable `G1NavEnv`,
- lidar/ray-scan sensor,
- occupancy grid,
- costmap,
- planner,
- real G1 adapter,
- RL training loop,
- semantic target detection/tracking.

The correct next milestone is therefore not “train RL.” It is:

```text
refactor smoke tests -> reusable command-level navigation environment -> add lidar/costmap -> add deterministic planner -> then consider RL
```
