# ROADMAP — G1 Command-Level Navigation Stack

## 0. Objective

Move the current `g1_rl` repository from smoke tests into a production-oriented navigation stack for Unitree G1 high-level motion control.

The guiding design decision is:

> Simulate and train the layer intended for real deployment: high-level velocity-command navigation, not low-level humanoid joint control.

The expected command boundary is:

```text
move(vx, vy, wz)
```

where:

- `vx` is forward/backward velocity in the robot/body frame,
- `vy` is lateral velocity in the robot/body frame,
- `wz` is yaw rate.

The target policy boundary is therefore:

```text
observation -> policy/planner -> raw [vx, vy, wz] -> safety/limits -> command adapter
```

The same policy interface should work in simulation and on the real robot. The difference between simulation and hardware should be hidden behind adapters.

---

## 1. Current repository status

The current archive is a smoke-test prototype.

```text
g1_rl/
  README.md                         # currently empty
  G1_INTRO.md                       # project overview
  ROADMAP.md                        # roadmap

  configs/
    robot/
      unitree_g1.yaml               # built-in G1 RGB-D smoke-test config
      g1_command_proxy.yaml         # proxy, camera, target, cube, safety config
    env/
      nav_env.yaml                  # placeholder, currently empty
    train/
      ppo.yaml                      # placeholder, currently empty

  scripts/
    smoke_test_scene.py             # built-in G1 articulation + RGB-D camera smoke test
    smoke_test_command_proxy.py     # command proxy movement smoke test
    smoke_test_go_to_target.py      # deterministic target reaching + depth safety smoke test
    train.py                        # placeholder, currently empty
    eval.py                         # placeholder, currently empty

  src/g1_nav_rl/
    observations/
      rgbd_obs.py                   # RGB-D + built-in G1 proprio observation builder
    robots/
      g1_command_proxy.py           # command-level proxy: [vx, vy, wz] -> USD transform motion
      g1_asset.py                   # placeholder, currently empty
    envs/
      nav_env.py                    # placeholder, currently empty
    rewards/
      navigation_reward.py          # placeholder, currently empty
    utils/
      __init__.py                   # placeholder
```

Implemented now:

- built-in G1 RGB-D smoke test,
- command-level kinematic proxy,
- RGB-D camera attached to the proxy,
- command-level proprioception `[x, y, yaw, vx, vy, wz]`,
- scripted command smoke test,
- deterministic A-to-B target controller,
- simple depth-based obstacle safety reflex.

Not implemented yet:

- reusable navigation environment API,
- lidar/ray-scan sensor,
- occupancy grid,
- costmap,
- global or local planner,
- real G1 adapter,
- RL training loop,
- YOLO/semantic target perception,
- tracking under occlusion.

---

## 2. What the current prototype has established

### 2.1 Isaac Sim / Isaac Lab RGB-D smoke test

Relevant files:

```text
scripts/smoke_test_scene.py
configs/robot/unitree_g1.yaml
src/g1_nav_rl/observations/rgbd_obs.py
```

Purpose:

```text
Isaac Lab scene -> built-in G1 asset -> mounted RGB-D camera -> RGB/depth/proprio tensors
```

This is useful for asset and sensor sanity checks. It should not be treated as the main navigation environment, because it uses the full G1 articulation and does not solve low-level locomotion.

---

### 2.2 Command-level G1 proxy

Relevant files:

```text
src/g1_nav_rl/robots/g1_command_proxy.py
scripts/smoke_test_command_proxy.py
configs/robot/g1_command_proxy.yaml
```

Purpose:

```text
[vx, vy, wz] -> smoothed/clamped body-frame command -> world-frame kinematic proxy motion
```

The proxy returns:

```text
[x, y, yaw, vx, vy, wz]
```

Current USD hierarchy:

```text
/World/envs/env_0/G1Proxy             # moving Xform root
/World/envs/env_0/G1Proxy/body        # visual body
/World/envs/env_0/G1Proxy/front_cam   # camera attached to proxy
```

Important caveat:

```text
The proxy is kinematic. It does not simulate balance, foot contacts, whole-body dynamics, or physical collision response.
```

Obstacle handling currently comes from perception and command filtering, not from physics.

---

### 2.3 Deterministic A-to-B controller

Relevant file:

```text
scripts/smoke_test_go_to_target.py
```

Purpose:

```text
current proxy pose + target pose -> [vx, vy, wz]
```

This provides a deterministic target-reaching baseline and proves the command-level interface is sufficient for simple navigation behavior.

Production destination:

```text
src/g1_nav_rl/policies/deterministic_go_to_target.py
```

Reasoning: this function maps target-relative state directly to a command, so it belongs in `policies/`, not `planning/`.

---

### 2.4 Depth-based obstacle safety reflex

Relevant file:

```text
scripts/smoke_test_go_to_target.py
```

Purpose:

```text
raw [vx, vy, wz] + depth image -> safe [vx, vy, wz]
```

Current behavior:

```text
if center depth < safety distance:
    suppress forward command
    sidestep toward clearer side
    add yaw-away command
```

Production destination:

```text
src/g1_nav_rl/safety/depth_safety.py
```

Near-term improvements:

- replace crop minimum with robust low-percentile depth,
- ignore invalid pixels explicitly,
- log safety intervention counts,
- test image-left/body-left sign conventions,
- expose tunable margins and hysteresis.

---

## 3. Production architecture

### 3.1 High-level data flow

```text
RGB camera ─────┐
Depth camera ───┼──────────────┐
Lidar scan ─────┤              │
Proprio ────────┘              │
                               ▼
                     Observation builder
                               │
                               ▼
                  Target / map / safety state
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

### 3.2 Runtime abstraction

Both simulation and real-robot execution should use the same high-level loop:

```python
obs = adapter.get_observation()
cmd_raw = policy(obs)
cmd_safe = safety_stack(cmd_raw, obs)
adapter.send_velocity(cmd_safe)
```

The policy should not know whether it controls:

- the Isaac Sim command proxy, or
- the physical Unitree G1.

The adapter owns that difference.

---

## 4. Target repository structure

Recommended production layout:

```text
g1_rl/
  README.md
  G1_INTRO.md
  ROADMAP.md
  pyproject.toml

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
      watchdog.yaml

    policies/
      deterministic_go_to_target.yaml
      deterministic_nav.yaml
      rl_nav.yaml

    train/
      ppo.yaml

  scripts/
    tests/
      smoke_scene_rgbd.py
      smoke_command_proxy.py
      smoke_go_to_target.py
      smoke_lidar.py
      smoke_nav_env.py
    train.py
    eval.py

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
      camera.py

    scenes/
      g1_nav_scene.py
      objects.py
      scenarios.py

    safety/
      command_limiter.py
      depth_safety.py
      lidar_safety.py
      watchdog.py

    policies/
      deterministic_go_to_target.py
      deterministic_target_following.py
      rl_policy_wrappers.py

    planning/
      occupancy_grid.py
      costmap.py
      path_planner.py
      local_planner.py
      trajectory_tracker.py

    envs/
      g1_nav_env.py

    rewards/
      navigation_reward.py

    metrics/
      navigation_metrics.py

    real/
      unitree_g1_adapter.py

    utils/
      geometry.py
      transforms.py
      config.py
```

---

## 5. Refactor plan for current files

### 5.1 Keep and harden `G1CommandProxy`

Current file:

```text
src/g1_nav_rl/robots/g1_command_proxy.py
```

Keep:

- body-frame command interpretation,
- `[vx, vy, wz]` interface,
- `max_vx`, `max_vy`, `max_wz` limits,
- acceleration limits,
- command smoothing,
- command-level observation `[x, y, yaw, vx, vy, wz]`,
- direct USD transform updates.

Improve:

- add command latency simulation,
- add command watchdog timeout,
- add optional command dropout/noise,
- add footprint radius/body-box metadata,
- expose body-frame and world-frame pose helpers,
- ensure repeated USD transform writes are robust,
- standardize proxy root/body local transform handling.

Immediate issue to fix:

```text
smoke_test_command_proxy.py and smoke_test_go_to_target.py initialize the proxy body differently.
```

Recommended rule:

```text
The moving root owns global pose. The body is local under the root.
```

---

### 5.2 Split `g1_command_proxy.yaml`

Current file:

```text
configs/robot/g1_command_proxy.yaml
```

It currently mixes robot proxy config, scene config, camera config, debug cube config, target config, and obstacle safety config.

Split into:

```text
configs/robot/g1_proxy.yaml
configs/sensors/g1_rgbd.yaml
configs/sensors/g1_lidar.yaml
configs/env/nav_env.yaml
configs/policies/deterministic_go_to_target.yaml
configs/safety/depth_safety.yaml
configs/safety/command_limits.yaml
```

Keep the current values, but separate concerns.

---

### 5.3 Keep smoke scripts as regression tests

Current scripts:

```text
scripts/smoke_test_scene.py
scripts/smoke_test_command_proxy.py
scripts/smoke_test_go_to_target.py
```

Future names:

```text
scripts/tests/smoke_scene_rgbd.py
scripts/tests/smoke_command_proxy.py
scripts/tests/smoke_go_to_target.py
```

These scripts should not contain core logic. They should import production modules and verify them.

---

### 5.4 Move reusable logic out of `smoke_test_go_to_target.py`

Consistent migration table:

| Current function / responsibility | Production destination |
|---|---|
| `compute_go_to_target_command()` | `src/g1_nav_rl/policies/deterministic_go_to_target.py` |
| `apply_depth_safety_filter()` | `src/g1_nav_rl/safety/depth_safety.py` |
| proxy RGB-D observation builder | `src/g1_nav_rl/observations/rgbd_proxy_obs.py` |
| built-in G1 observation builder | `src/g1_nav_rl/observations/rgbd_articulation_obs.py` |
| scene config class | `src/g1_nav_rl/scenes/g1_nav_scene.py` or `src/g1_nav_rl/envs/g1_nav_env.py` |
| target/debug cube spawning | `src/g1_nav_rl/scenes/objects.py` |
| metrics/logging | `src/g1_nav_rl/metrics/navigation_metrics.py` |

---

## 6. Environment API

Create:

```text
src/g1_nav_rl/envs/g1_nav_env.py
```

Minimum interface:

```python
class G1NavEnv:
    def reset(self) -> dict:
        ...

    def step(self, action) -> tuple[dict, float, bool, dict]:
        ...
```

The environment should own:

- scene setup,
- proxy setup,
- sensor setup,
- observation building,
- action application,
- safety filtering,
- reward calculation,
- done logic,
- metrics.

Recommended observation schema:

```python
{
    "rgb": rgb,                                      # optional for RL baseline
    "depth": depth,                                  # near-field safety / perception
    "lidar": lidar_scan,                             # primary 2D navigation input
    "proprio": [x, y, yaw, vx, vy, wz],
    "target": [dx_body, dy_body, distance, bearing],
    "costmap": optional_local_costmap,
}
```

Recommended action:

```python
[vx, vy, wz]
```

Recommended `info` fields:

```python
{
    "distance_to_target": float,
    "success": bool,
    "timeout": bool,
    "safety_active": bool,
    "min_obstacle_distance": float,
    "raw_command": [vx, vy, wz],
    "safe_command": [vx, vy, wz],
}
```

---

## 7. Safety architecture

Safety should remain outside the policy.

Recommended command path:

```text
policy/planner command
    -> command limiter
    -> depth safety
    -> lidar safety
    -> watchdog
    -> sim or real adapter
```

### 7.1 Command limiter

Production module:

```text
src/g1_nav_rl/safety/command_limiter.py
```

Responsibilities:

- clamp `vx`, `vy`, `wz`,
- enforce acceleration limits,
- enforce jerk/smoothing if needed,
- prevent stale commands from persisting.

### 7.2 Depth safety

Production module:

```text
src/g1_nav_rl/safety/depth_safety.py
```

Responsibilities:

- detect near-field frontal obstacles,
- reduce forward velocity,
- optionally sidestep/yaw away,
- log intervention state,
- use robust depth statistics.

### 7.3 Lidar safety

Production module:

```text
src/g1_nav_rl/safety/lidar_safety.py
```

Responsibilities:

- enforce minimum obstacle distance in 2D,
- prevent commands into occupied cells,
- provide local free-space checks,
- act as the primary navigation safety source once lidar is added.

### 7.4 Watchdog

Production module:

```text
src/g1_nav_rl/safety/watchdog.py
```

Responsibilities:

- stop if observations become stale,
- stop if command stream becomes stale,
- stop on invalid tensors or NaNs,
- stop on adapter errors.

---

## 8. Lidar and deterministic navigation

Lidar should be the next major addition.

### 8.1 Add lidar/ray-scan sensor

Production module:

```text
src/g1_nav_rl/sensors/lidar.py
```

Recommended first target:

```text
planar 2D ray scan attached to the command proxy
```

The initial interface can be:

```python
{
    "ranges": tensor[num_envs, num_rays],
    "angles": tensor[num_rays],
    "range_min": float,
    "range_max": float,
}
```

### 8.2 Build occupancy grid and costmap

Production modules:

```text
src/g1_nav_rl/planning/occupancy_grid.py
src/g1_nav_rl/planning/costmap.py
```

Pipeline:

```text
lidar scan
    -> local occupancy grid
    -> filtering / cleanup
    -> obstacle inflation
    -> local costmap
```

### 8.3 Add deterministic planner

Production modules:

```text
src/g1_nav_rl/planning/path_planner.py
src/g1_nav_rl/planning/local_planner.py
src/g1_nav_rl/planning/trajectory_tracker.py
```

Initial deterministic pipeline:

```text
target_xy
    -> occupancy/costmap
    -> path or local waypoint
    -> trajectory tracker
    -> [vx, vy, wz]
    -> safety stack
```

This deterministic stack is the required baseline before RL.

---

## 9. RL plan

RL should be optional and staged. It should not be the next immediate step.

### 9.1 First RL experiment

Use low-dimensional inputs:

```text
lidar scan + target vector + current command/proprio
```

Avoid raw RGB-D end-to-end RL initially.

Possible action space:

```text
continuous [vx, vy, wz]
```

or a small discrete action set:

```text
forward, slow_forward, sidestep_left, sidestep_right, turn_left, turn_right, stop
```

### 9.2 Reward components

Production module:

```text
src/g1_nav_rl/rewards/navigation_reward.py
```

Initial reward terms:

```text
+ progress toward target
+ success bonus
- collision / near-collision penalty
- timeout penalty
- excessive command magnitude
- oscillation / command reversal penalty
- safety intervention penalty or diagnostic term
```

### 9.3 Metrics

Production module:

```text
src/g1_nav_rl/metrics/navigation_metrics.py
```

Track:

```text
success rate
time to target
path length
minimum obstacle distance
number of safety interventions
mean command magnitude
oscillation count
collision / near-collision count
```

---

## 10. Perception-based targets

After deterministic navigation is stable, add target perception.

Planned pipeline:

```text
RGB frame
    -> YOLO detector
    -> target class selection
    -> depth lookup
    -> camera-to-base transform
    -> temporal tracker
    -> target_xy
    -> navigation policy/planner
```

Recommended modules:

```text
src/g1_nav_rl/perception/yolo_detector.py
src/g1_nav_rl/perception/rgbd_localizer.py
src/g1_nav_rl/perception/target_tracker.py
```

Key requirements:

- filter low-confidence detections,
- reject invalid depth,
- transform detections from camera frame to robot/base frame,
- maintain a target estimate through short occlusions,
- expose target uncertainty to planner/policy if possible.

---

## 11. Sim/real adapter boundary

Define a common command adapter interface:

```python
class CommandAdapter:
    def get_observation(self) -> dict:
        ...

    def send_velocity(self, vx: float, vy: float, wz: float) -> None:
        ...

    def stop(self) -> None:
        ...
```

Implement:

```text
SimCommandAdapter
  -> wraps G1CommandProxy and Isaac sensors

RealG1Adapter
  -> wraps selected Unitree G1 SDK/API path
```

The policy/planner should depend only on the adapter interface, not on Isaac-specific or Unitree-specific internals.

Real deployment must include:

- command-rate validation,
- command scaling and unit validation,
- emergency stop path,
- watchdog stop behavior,
- sensor timestamp checks,
- frame calibration,
- conservative speed limits,
- field testing with staged risk.

---

## 12. Implementation milestones

### Milestone 1 — Repository cleanup and documentation

Deliverables:

- clean archive without `.git/`, `__MACOSX/`, `__pycache__/`, or `*.pyc`,
- non-empty `README.md`,
- consistent `G1_INTRO.md` and `ROADMAP.md`,
- `pyproject.toml` or equivalent package metadata,
- smoke-test run instructions.

### Milestone 2 — Refactor smoke-test logic into modules

Deliverables:

- `policies/deterministic_go_to_target.py`,
- `safety/depth_safety.py`,
- `observations/rgbd_proxy_obs.py`,
- cleaned smoke scripts importing these modules,
- consistent log names: `raw_command`, `safe_command`, `cmd_current`.

### Milestone 3 — Implement `G1NavEnv`

Deliverables:

- `envs/g1_nav_env.py`,
- `reset()` and `step(action)` API,
- observation dictionary,
- reward/done/info outputs,
- target success and timeout logic,
- metric collection.

### Milestone 4 — Add lidar/ray-scan

Deliverables:

- simulated lidar/ray-scan attached to proxy,
- lidar observation schema,
- smoke test proving obstacle detection,
- frame/sign convention tests.

### Milestone 5 — Add costmap and deterministic planner

Deliverables:

- local occupancy grid,
- obstacle inflation,
- costmap,
- local waypoint generation,
- trajectory tracking to `[vx, vy, wz]`,
- deterministic benchmark scenarios.

### Milestone 6 — Add sim/real adapter interface

Deliverables:

- common adapter interface,
- `SimCommandAdapter`,
- stub or initial `RealG1Adapter`,
- command limits shared between sim and real,
- watchdog safety path.

### Milestone 7 — Optional RL experiments

Deliverables:

- low-dimensional RL observation space,
- PPO or equivalent training config,
- training/evaluation scripts,
- deterministic baseline comparison,
- ablations with and without safety filters.

### Milestone 8 — Perception-based target following

Deliverables:

- YOLO detector integration,
- RGB-D target localization,
- temporal target tracker,
- target-following scenarios,
- safety-gated real/sim adapter compatibility.

---

## 13. First refactor PR checklist

The first implementation PR should be narrow and mechanical.

Recommended scope:

```text
1. Create src/g1_nav_rl/policies/deterministic_go_to_target.py
2. Create src/g1_nav_rl/safety/depth_safety.py
3. Create src/g1_nav_rl/observations/rgbd_proxy_obs.py
4. Update smoke_test_go_to_target.py to import those modules
5. Rename log field "target command" to "safe command" or "proxy cmd_target"
6. Fix proxy body local/global initialization consistency
7. Add README.md with current status and smoke-test commands
8. Add pyproject.toml
9. Update .gitignore
```

Do not add lidar, RL, YOLO, and real-robot adapters in the same PR. Those are separate milestones.

---

## 14. Near-term technical risks

### 14.1 Proxy-body transform ambiguity

The proxy root and body child should have a single clear ownership rule:

```text
root = global pose
body = local visual geometry
camera = child of root or body with fixed local transform
```

Avoid applying the same initial offset at both root and body levels.

### 14.2 Depth safety noise

Raw minimum depth is too sensitive to single-pixel artifacts. Use valid-pixel filtering and a robust statistic.

### 14.3 Left/right sign conventions

The sidestep direction assumes image-left and robot-left are aligned through the configured camera transform. Add explicit tests:

```text
obstacle on robot-left  -> safety chooses right or correct avoidance behavior
obstacle on robot-right -> safety chooses left or correct avoidance behavior
```

### 14.4 Overstated sim-to-real transfer

The current proxy improves command-interface alignment, but it does not by itself guarantee real-world transfer. Real deployment still requires adapter validation, safety gating, latency handling, calibration, and staged hardware tests.

---

## 15. Summary

The current repo has already moved beyond installation checks: it contains RGB-D smoke tests, a command-level G1 proxy, deterministic target reaching, and a first depth-based safety reflex.

The next phase is not low-level humanoid locomotion and not immediate RL. The next phase is to convert the smoke-test code into a production-style command-level navigation repository:

```text
reusable modules
+ clean environment API
+ lidar/costmap navigation
+ deterministic planner
+ sim/real adapter boundary
+ external safety stack
```

Only after that baseline is stable should RL and semantic target-following experiments be added.
