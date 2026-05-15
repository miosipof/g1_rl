# ROADMAP.md — G1 Command-Level Navigation Stack

## 0. Purpose

This roadmap describes how to move the current `g1_rl` repository from smoke tests into a production-oriented navigation stack for Unitree G1 high-level motion control.

The central design decision is:

> Simulate and train the layer we intend to deploy on the real robot: high-level velocity commands, not low-level joint control.

The real robot interface is expected to look like:

```text
move(vx, vy, wz)
```

where:

- `vx`: forward/backward velocity in the robot/body frame
- `vy`: lateral velocity in the robot/body frame
- `wz`: yaw rate

Therefore the simulation should expose the same policy boundary:

```text
observation -> high-level policy -> [vx, vy, wz] -> command adapter
```

The current repository has already validated this idea in smoke tests. The next step is to consolidate those tests into reusable modules, add lidar/local mapping, define a clean environment API, then layer deterministic and optionally RL policies on top.

---

## 1. Current repository status

The attached archive contains the following relevant files:

```text
g1_rl/
  README.md                         # currently empty
  configs/
    robot/
      unitree_g1.yaml               # Isaac Lab built-in G1 RGB-D smoke-test config
      g1_command_proxy.yaml         # command-proxy config with RGB-D, target, debug cube, depth safety
    env/
      nav_env.yaml                  # currently empty
    train/
      ppo.yaml                      # currently empty

  scripts/
    smoke_test_scene.py             # built-in G1 articulation + RGB-D camera smoke test
    smoke_test_command_proxy.py     # command proxy movement smoke test
    smoke_test_go_to_target.py      # deterministic target reaching + depth safety smoke test
    train.py                        # currently empty
    eval.py                         # currently empty

  src/g1_nav_rl/
    observations/
      rgbd_obs.py                   # RGB-D + G1 proprio observation builder for built-in G1
    robots/
      g1_command_proxy.py           # command-level proxy: [vx, vy, wz] -> USD transform motion
      g1_asset.py                   # currently empty
    envs/
      nav_env.py                    # currently empty
    rewards/
      navigation_reward.py          # currently empty
    utils/
      __init__.py                   # currently empty
```

---

## 2. What has already been proven

### 2.1 Isaac Sim / Isaac Lab installation and rendering

We confirmed Isaac Sim / Isaac Lab can run in the remote Linux container with GPU rendering in headless mode. RGB and depth camera tensors can be generated and screenshots can be saved.

Relevant files:

```text
scripts/smoke_test_scene.py
configs/robot/unitree_g1.yaml
src/g1_nav_rl/observations/rgbd_obs.py
```

What this proved:

```text
Isaac Lab scene -> built-in G1 asset -> mounted RGB-D camera -> RGB/depth/proprio tensors
```

Important caveat:

- This built-in G1 smoke test uses the Isaac Lab `G1_CFG` articulation.
- It does not solve locomotion.
- It is useful for sensor placement, robot asset sanity checks, and possible future comparison to a real G1 body layout.

### 2.2 Command-level G1 proxy

We implemented a command-level proxy that moves a clean USD transform directly instead of using PhysX rigid-body tensor writes.

Relevant files:

```text
src/g1_nav_rl/robots/g1_command_proxy.py
scripts/smoke_test_command_proxy.py
configs/robot/g1_command_proxy.yaml
```

What this proved:

```text
[vx, vy, wz] -> smoothed/clamped body-frame command -> world-frame proxy motion
```

The proxy currently returns command-level proprioception:

```text
[x, y, yaw, vx, vy, wz]
```

The proxy body is represented by a simple cuboid under:

```text
/World/envs/env_0/G1Proxy/body
```

The moving root is:

```text
/World/envs/env_0/G1Proxy
```

The camera is attached under:

```text
/World/envs/env_0/G1Proxy/front_cam
```

This parent/body split fixed USD transform warnings caused by trying to directly overwrite xform operations on a spawned cuboid prim.

### 2.3 Deterministic A-to-B controller

We implemented a simple deterministic controller that maps current proxy pose and target position to `[vx, vy, wz]`.

Relevant file:

```text
scripts/smoke_test_go_to_target.py
```

Validated behavior:

```text
start:  [0, 0]
target: [3, 1] or [4, 0]
controller decreases distance-to-target
proxy stops when inside success radius
```

### 2.4 Depth-based obstacle safety layer

We added a simple RGB-D safety reflex:

```text
if center depth < safety distance:
    stop/reduce forward velocity
    sidestep toward clearer side
    add small yaw-away command
```

Validated behavior:

```text
raw A-to-B command wants to move forward
camera sees cube in front
safety layer modifies command to sidestep/yaw
proxy goes around obstacle
proxy reaches target
```

This is not a full planner yet. It is a near-field safety reflex.

---

## 3. Design principle for the production stack

The production stack should model the real deployment boundary:

```text
policy output = high-level velocity command
```

not:

```text
policy output = G1 joint positions / torques
```

This avoids turning the project into a low-level humanoid locomotion research effort. Low-level locomotion should remain handled by the real robot’s built-in controller or SDK-level high-level motion interface.

In simulation, the G1 abstraction should therefore consist of:

```text
G1-like perception
G1-like body footprint
G1-like velocity limits
G1-like latency / smoothing / command-rate limits
G1-like safety envelope
```

rather than full joint-level physics fidelity.

---

## 4. Target architecture

### 4.1 High-level data flow

```text
RGB camera ─────┐
Depth camera ───┼──────────────┐
Lidar scan ─────┤              │
Proprio ────────┘              │
                               ▼
                     Observation builder
                               │
                               ▼
                 Target / map / safety modules
                               │
                               ▼
                  Deterministic or learned policy
                               │
                               ▼
                         raw [vx, vy, wz]
                               │
                               ▼
                 command limiter + safety filter
                               │
                               ▼
                     sim adapter / real adapter
                               │
                               ▼
                    G1 proxy / real Unitree G1
```

### 4.2 Runtime abstraction

Both simulation and real-robot execution should use the same high-level loop:

```python
obs = adapter.get_observation()
cmd_raw = policy(obs)
cmd_safe = safety_filter(cmd_raw, obs)
adapter.send_velocity(cmd_safe)
```

The policy should not know whether it is controlling:

- the Isaac Sim command proxy, or
- the physical G1.

The adapter owns that difference.

---

## 5. What to keep from the current version

### 5.1 Keep and harden `G1CommandProxy`

Current file:

```text
src/g1_nav_rl/robots/g1_command_proxy.py
```

Keep:

- body-frame command interpretation
- `[vx, vy, wz]` command interface
- clamping of `max_vx`, `max_vy`, `max_wz`
- acceleration limits
- smoothing
- command-level observation `[x, y, yaw, vx, vy, wz]`
- direct USD transform updates

Improve:

- rename `G1CommandProxy` to a clearer production name if useful, e.g. `G1CommandKinematicProxy`
- add command latency simulation
- add command watchdog timeout
- add optional command dropout / noise for robustness testing
- add footprint radius / body box metadata
- make `write_to_usd()` robust to repeated xform op definitions
- expose body-frame and world-frame pose helpers

### 5.2 Keep `g1_command_proxy.yaml`, but split it

Current file:

```text
configs/robot/g1_command_proxy.yaml
```

It currently contains robot proxy config, scene config, camera config, debug cube config, target config, and obstacle safety config.

For production, split this into:

```text
configs/robot/g1_proxy.yaml
configs/sensors/g1_rgbd.yaml
configs/sensors/g1_lidar.yaml
configs/env/nav_env.yaml
configs/policies/deterministic_nav.yaml
configs/safety/depth_safety.yaml
configs/safety/command_limits.yaml
```

Keep the values, but separate concerns.

### 5.3 Keep `smoke_test_command_proxy.py`

Current file:

```text
scripts/smoke_test_command_proxy.py
```

Keep it as a regression test, not as production logic.

Rename later to:

```text
scripts/tests/smoke_command_proxy.py
```

### 5.4 Keep `smoke_test_go_to_target.py` as source material

Current file:

```text
scripts/smoke_test_go_to_target.py
```

Move reusable logic out of it into production modules:

| Current function | Production destination |
|---|---|
| `compute_go_to_target_command()` | `src/g1_nav_rl/policies/deterministic_go_to_target.py` |
| `apply_depth_safety_filter()` | `src/g1_nav_rl/safety/depth_safety.py` |
| `build_camera_observation()` | `src/g1_nav_rl/observations/rgbd_proxy_obs.py` |
| scene config class | `src/g1_nav_rl/scenes/g1_nav_scene.py` or `envs/g1_nav_env.py` |
| target/debug cube spawning | `src/g1_nav_rl/scenes/objects.py` |

After refactoring, keep the script as:

```text
scripts/tests/smoke_go_to_target.py
```

### 5.5 Keep `smoke_test_scene.py` for built-in G1 sensor checks

Current file:

```text
scripts/smoke_test_scene.py
```

Keep it as a separate sensor/body sanity check for the Isaac Lab built-in G1 asset.

Do not use it as the main navigation environment until we intentionally decide to simulate full G1 articulation again.

### 5.6 Keep `rgbd_obs.py`, but separate built-in G1 from proxy observations

Current file:

```text
src/g1_nav_rl/observations/rgbd_obs.py
```

It currently assumes a full robot articulation and builds:

```text
RGB
Depth
base velocities
projected gravity
joint positions
joint velocities
```

That is useful for the built-in G1 asset, but the command proxy observation should be simpler:

```text
RGB
Depth
Lidar
[x, y, yaw, vx, vy, wz]
target-relative vector
local map/costmap
```

Recommended split:

```text
src/g1_nav_rl/observations/rgbd_articulation_obs.py
src/g1_nav_rl/observations/rgbd_proxy_obs.py
src/g1_nav_rl/observations/nav_obs.py
```

---

## 6. Proposed repository structure

Recommended production structure:

```text
g1_rl/
  README.md
  G1_INTRO.md
  ROADMAP.md

  configs/
    robot/
      g1_proxy.yaml
      unitree_g1_asset.yaml

    sensors/
      g1_rgbd.yaml
      g1_lidar.yaml

    env/
      nav_env.yaml
      scenarios.yaml

    safety/
      command_limits.yaml
      depth_safety.yaml
      lidar_safety.yaml

    policies/
      deterministic_nav.yaml
      deterministic_target_following.yaml
      rl_nav.yaml

    train/
      ppo.yaml

  src/g1_nav_rl/
    robots/
      g1_command_proxy.py
      command_types.py
      adapters.py

    scenes/
      g1_nav_scene.py
      objects.py
      scenario_loader.py

    sensors/
      rgbd.py
      lidar.py
      sensor_utils.py

    observations/
      nav_obs.py
      rgbd_proxy_obs.py
      rgbd_articulation_obs.py

    mapping/
      lidar_to_points.py
      occupancy_grid.py
      costmap.py
      map_cleanup.py

    planning/
      grid_planner.py
      local_planner.py
      trajectory_tracker.py

    policies/
      deterministic_go_to_target.py
      deterministic_obstacle_avoidance.py
      policy_interface.py
      learned_policy_adapter.py

    safety/
      command_limiter.py
      depth_safety.py
      lidar_safety.py
      watchdog.py

    perception/
      detector.py
      target_tracker.py
      rgbd_target_localizer.py

    envs/
      g1_nav_env.py

    rewards/
      navigation_reward.py
      safety_reward.py

    evaluation/
      metrics.py
      scenarios.py
      rollout_logger.py

    utils/
      config.py
      geometry.py
      io.py
      transforms.py

  scripts/
    tests/
      smoke_scene_g1_asset.py
      smoke_command_proxy.py
      smoke_go_to_target.py
      smoke_lidar.py
      smoke_nav_env.py

    run_deterministic_nav.py
    train.py
    eval.py
    export_policy.py
```

---

## 7. Production modules to implement

### 7.1 Command types

Create:

```text
src/g1_nav_rl/robots/command_types.py
```

Purpose:

```python
@dataclass
class VelocityCommand:
    vx: float
    vy: float
    wz: float
    frame: str = "body"
    stamp: float | None = None
```

Why:

- makes simulation and real command adapters consistent
- avoids passing anonymous tensors through production code
- allows metadata: frame, timestamp, source policy, safety status

### 7.2 Sim and real adapters

Create:

```text
src/g1_nav_rl/robots/adapters.py
```

Interfaces:

```python
class MotionCommandAdapter:
    def get_observation(self) -> dict: ...
    def send_velocity(self, cmd: VelocityCommand) -> None: ...
    def stop(self) -> None: ...
```

Implement:

```text
SimCommandAdapter -> wraps G1CommandProxy
RealG1CommandAdapter -> later wraps Unitree SDK / high-level command interface
```

### 7.3 Navigation environment

Create:

```text
src/g1_nav_rl/envs/g1_nav_env.py
```

Minimal API:

```python
obs = env.reset(seed=...)
obs, reward, done, info = env.step(action)
```

Action:

```text
[vx, vy, wz]
```

Observation:

```python
{
    "rgb": RGB tensor,
    "depth": depth tensor,
    "lidar": lidar scan tensor,
    "proprio": [x, y, yaw, vx, vy, wz],
    "target": [dx_body, dy_body, distance, bearing],
    "costmap": optional local grid,
}
```

Done conditions:

```text
target reached
timeout
collision / unsafe distance
left allowed map bounds
manual termination
```

Info dictionary:

```python
{
    "distance_to_target": ...,
    "success": ...,
    "collision": ...,
    "min_depth": ...,
    "min_lidar": ...,
    "safety_active": ...,
    "path_length": ...,
    "num_safety_interventions": ...,
}
```

### 7.4 Lidar module

Create:

```text
src/g1_nav_rl/sensors/lidar.py
```

Goal:

- add a planar simulated lidar / ray-cast sensor
- output a fixed-size 1D range vector
- convert ranges to body-frame points

Suggested output:

```python
{
    "ranges": [N],
    "angles": [N],
    "points_body": [N, 2],
    "valid": [N],
}
```

### 7.5 Occupancy grid and costmap

Create:

```text
src/g1_nav_rl/mapping/occupancy_grid.py
src/g1_nav_rl/mapping/costmap.py
src/g1_nav_rl/mapping/map_cleanup.py
```

Pipeline:

```text
lidar ranges
  -> 2D points in body frame
  -> occupancy grid
  -> morphological cleanup
  -> obstacle inflation
  -> local costmap
```

Costmap values should represent:

```text
free
unknown
occupied
inflated risk near obstacle
```

This becomes the primary navigation representation.

### 7.6 Deterministic planning stack

Create:

```text
src/g1_nav_rl/planning/grid_planner.py
src/g1_nav_rl/planning/local_planner.py
src/g1_nav_rl/planning/trajectory_tracker.py
```

Initial deterministic stack:

```text
target_xy
  -> local costmap
  -> path / local waypoint
  -> velocity command [vx, vy, wz]
  -> safety filter
```

Start simple:

1. Use target-relative vector when no obstacle is present.
2. Use depth/lidar safety to stop or sidestep.
3. Add A* or Dijkstra over a small local grid.
4. Add a trajectory follower to convert path into `[vx, vy, wz]`.

### 7.7 Safety modules

Create:

```text
src/g1_nav_rl/safety/command_limiter.py
src/g1_nav_rl/safety/depth_safety.py
src/g1_nav_rl/safety/lidar_safety.py
src/g1_nav_rl/safety/watchdog.py
```

Safety layers:

```text
raw policy command
  -> command limiter
  -> depth safety
  -> lidar safety
  -> watchdog / deadman
  -> adapter
```

Required production safety behaviors:

- clamp max velocity and yaw rate
- clamp acceleration and jerk
- stop if command is stale
- stop if obstacle is inside emergency radius
- stop if depth/lidar is invalid for too long
- support manual override later

### 7.8 Perception and target tracking

Create later:

```text
src/g1_nav_rl/perception/detector.py
src/g1_nav_rl/perception/rgbd_target_localizer.py
src/g1_nav_rl/perception/target_tracker.py
```

Pipeline:

```text
RGB frame
  -> YOLO detector
  -> chosen class bbox
  -> depth lookup inside bbox
  -> camera-to-body transform
  -> target xy in body/world frame
  -> temporal tracker
```

Target tracker output:

```python
{
    "visible": bool,
    "class_name": str,
    "xy_body": [x, y] | None,
    "xy_world": [x, y] | None,
    "confidence": float,
    "age_sec": float,
    "lost_sec": float,
}
```

Occlusion behavior:

```text
short occlusion: keep predicted target position
medium occlusion: slow down and continue cautiously
long occlusion: stop or enter search behavior
```

---

## 8. Deterministic policy roadmap

### 8.1 Baseline 1 — Go to target without obstacles

Already implemented in smoke test.

Production destination:

```text
src/g1_nav_rl/policies/deterministic_go_to_target.py
```

Inputs:

```text
current pose
target xy
```

Output:

```text
[vx, vy, wz]
```

### 8.2 Baseline 2 — Depth safety reflex

Already implemented in smoke test.

Production destination:

```text
src/g1_nav_rl/safety/depth_safety.py
```

Inputs:

```text
depth image
raw command
```

Output:

```text
safe command
safety info
```

Improvements:

- add hysteresis to prevent threshold flicker
- use percentile depth rather than minimum depth to reduce noise sensitivity
- define separate center/left/right regions explicitly
- track safety interventions over time

### 8.3 Baseline 3 — Lidar local planner

New implementation.

Pipeline:

```text
lidar -> local occupancy grid -> inflated costmap -> local waypoint -> command
```

Policy behavior:

- drive toward target when path is clear
- stop when frontal lidar distance is unsafe
- choose side with larger free-space corridor
- optionally use A* over a local grid

### 8.4 Baseline 4 — Moving target following

Later deterministic behavior:

```text
YOLO/RGB-D target estimate -> target tracker -> moving target xy -> local planner
```

Add constraints:

- maintain follow distance
- avoid overshooting
- slow down when target is uncertain
- stop when target is lost for too long

---

## 9. RL roadmap

RL should be optional and introduced after deterministic baselines work.

Do not start with raw RGB-D-to-command RL. It is harder to debug and less likely to transfer.

Recommended progression:

### Stage RL-0 — Environment validation

Implement `G1NavEnv` and test:

```text
random actions
fixed scripted actions
deterministic go-to-target policy
deterministic obstacle-avoidance policy
```

No PPO yet.

### Stage RL-1 — Low-dimensional local navigation

Observation:

```text
lidar ranges
target vector in body frame
current command
last action
```

Action:

```text
[vx, vy, wz]
```

Reward:

```text
+ progress toward target
+ target reached bonus
- collision / near collision
- time penalty
- excessive action magnitude
- action jerk
- safety intervention
```

### Stage RL-2 — Costmap-based policy

Observation:

```text
local costmap
target vector
current velocity
```

This is usually easier to train and debug than raw sensor images.

### Stage RL-3 — Vision-assisted policy

Only after earlier stages work:

```text
RGB-D features + lidar/costmap + target tracker state -> [vx, vy, wz]
```

Avoid relying on RGB-D alone for obstacle avoidance.

---

## 10. Scenario and evaluation roadmap

Create a scenario config file:

```text
configs/env/scenarios.yaml
```

Initial scenarios:

```yaml
scenarios:
  empty_target:
    target: [3.0, 0.0]
    obstacles: []

  single_cube_blocking:
    target: [4.0, 0.0]
    obstacles:
      - type: cube
        pos: [2.2, 0.0, 0.4]
        size: [0.8, 0.8, 0.8]

  side_obstacle:
    target: [4.0, 0.0]
    obstacles:
      - type: cube
        pos: [2.0, 0.6, 0.4]
        size: [0.6, 0.6, 0.8]

  corridor:
    target: [5.0, 0.0]
    obstacles:
      - type: wall
        pos: [2.5, 0.8, 0.5]
      - type: wall
        pos: [2.5, -0.8, 0.5]

  sudden_obstacle:
    target: [4.0, 0.0]
    dynamic_obstacles:
      - spawn_step: 100
        pos: [1.8, 0.0, 0.4]
```

Evaluation metrics:

```text
success rate
time to target
path length
final distance
minimum obstacle clearance
number of safety interventions
collision count
timeout count
average speed
average command jerk
occlusion recovery time
```

---

## 11. Immediate next steps

### Step 1 — Freeze smoke tests as regression tests

Move current scripts:

```text
scripts/smoke_test_scene.py          -> scripts/tests/smoke_scene_g1_asset.py
scripts/smoke_test_command_proxy.py  -> scripts/tests/smoke_command_proxy.py
scripts/smoke_test_go_to_target.py   -> scripts/tests/smoke_go_to_target.py
```

Do not delete them. They are useful regression checks.

### Step 2 — Refactor reusable code out of smoke scripts

Create:

```text
src/g1_nav_rl/policies/deterministic_go_to_target.py
src/g1_nav_rl/safety/depth_safety.py
src/g1_nav_rl/observations/rgbd_proxy_obs.py
src/g1_nav_rl/scenes/g1_nav_scene.py
src/g1_nav_rl/scenes/objects.py
```

Move logic from `smoke_test_go_to_target.py` into these modules.

### Step 3 — Implement `G1NavEnv`

Populate:

```text
src/g1_nav_rl/envs/g1_nav_env.py
configs/env/nav_env.yaml
```

Minimum environment features:

- reset scene
- spawn target
- spawn obstacles
- step action `[vx, vy, wz]`
- collect RGB-D and proprio observation
- compute target-relative vector
- compute reward
- compute done
- log metrics

### Step 4 — Add lidar

Create:

```text
src/g1_nav_rl/sensors/lidar.py
configs/sensors/g1_lidar.yaml
scripts/tests/smoke_lidar.py
```

First test only:

```text
spawn proxy + obstacle
read lidar scan
print min/left/right/front ranges
save optional visualization
```

### Step 5 — Add local occupancy grid

Create:

```text
src/g1_nav_rl/mapping/occupancy_grid.py
src/g1_nav_rl/mapping/costmap.py
scripts/tests/smoke_costmap.py
```

Goal:

```text
lidar scan -> occupancy grid -> inflated costmap
```

### Step 6 — Implement deterministic navigation runner

Create:

```text
scripts/run_deterministic_nav.py
```

It should run:

```text
G1NavEnv + deterministic policy + safety layer + metrics
```

This becomes the baseline before training.

### Step 7 — Only then implement training

Populate:

```text
scripts/train.py
scripts/eval.py
configs/train/ppo.yaml
src/g1_nav_rl/rewards/navigation_reward.py
```

First PPO task:

```text
input: lidar + target vector + proprio
action: [vx, vy, wz]
goal: reach target while avoiding static obstacles
```

---

## 12. Priority checklist

### Highest priority

- [ ] Refactor smoke-test logic into modules
- [ ] Implement `G1NavEnv`
- [ ] Add lidar sensor
- [ ] Add local occupancy/costmap
- [ ] Add deterministic local planner
- [ ] Add metrics and scenario configs

### Medium priority

- [ ] Add RL training loop
- [ ] Add PPO config
- [ ] Add policy evaluation script
- [ ] Add run logging
- [ ] Add domain randomization: noise, latency, obstacle placement

### Later priority

- [ ] Add YOLO detector
- [ ] Add RGB-D target localization
- [ ] Add target tracker
- [ ] Add moving-target scenarios
- [ ] Add occlusion and reacquisition behavior
- [ ] Add real G1 command adapter

---

## 13. Main risks

### Risk 1 — Overfitting to the kinematic proxy

The proxy is intentionally simple. To improve transfer, add:

- command latency
- acceleration limits
- command noise
- imperfect stopping
- body footprint
- sensor noise
- sensor dropout

### Risk 2 — RGB-D safety is brittle

Depth can fail on reflective, transparent, distant, or poorly lit surfaces. Lidar/costmap should become the primary obstacle source. RGB-D should remain a safety/perception supplement.

### Risk 3 — RL before deterministic baselines

Training before the deterministic baseline is stable will make debugging difficult. Deterministic policies should be treated as the oracle/fallback.

### Risk 4 — YOLO target position uncertainty

Single-frame detection is not enough. Production target following requires temporal filtering, confidence tracking, and occlusion handling.

### Risk 5 — Real robot safety

The real robot must never execute raw policy output directly. Always use:

```text
policy -> command limiter -> safety filters -> watchdog -> real robot adapter
```

---

## 14. Definition of done for the next milestone

The next milestone is complete when the repo has a reusable production-style environment:

```text
src/g1_nav_rl/envs/g1_nav_env.py
```

and the following command works:

```bash
PYTHONUNBUFFERED=1 python scripts/run_deterministic_nav.py --headless --scenario single_cube_blocking
```

Expected behavior:

- proxy starts at the origin
- target is spawned
- obstacle is spawned
- RGB-D and lidar observations are produced
- local costmap is created
- deterministic policy commands `[vx, vy, wz]`
- safety layer intervenes when needed
- proxy reaches target or reports failure
- metrics are printed at the end

Minimum printed metrics:

```text
success: true/false
time_to_target
path_length
min_obstacle_distance
num_safety_interventions
final_distance
```

---

## 15. Long-term target

The long-term stack should support:

```text
predefined target point
  -> deterministic / RL navigation
  -> safe G1 velocity commands

YOLO-detected object
  -> RGB-D localization
  -> target tracking
  -> deterministic / RL navigation
  -> safe G1 velocity commands

moving target / sudden obstacle / temporary occlusion
  -> robust tracker and safety behavior
  -> safe high-level motion
```

The final sim-to-real contract remains:

```text
same observations where possible
same high-level command interface
same safety filters
sim adapter swapped for real G1 adapter
```
