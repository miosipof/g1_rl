# ADAPTER_PLAN.md

# G1 sim-to-real adapter plan

This document defines the step-by-step plan for connecting the current `g1_rl` simulation autonomy stack with the real Unitree G1 hardware utilities from `unitree_g1_vibes`.

The target architecture is:

```text
real sensors / Isaac sensors
        ↓
ObservationAdapter
        ↓
RobotObservation
        ↓
g1_rl perception + target tracking + navigation runtime
        ↓
Command2D(vx, vy, wz)
        ↓
SafetySupervisor
        ↓
CommandSink
        ↓
Isaac proxy / Unitree locomotion client
```

The goal is not to merge the two repositories into one monolith. The goal is to create a narrow, explicit interface so that observations can come from either Isaac Sim or the real G1, and commands can be consumed either by Isaac Sim or by the real robot.

---

## 1. Design principles

### 1.1 Keep `g1_rl` as the autonomy stack

`g1_rl` should remain responsible for:

- target detection and target tracking;
- RGB-D target projection;
- LiDAR-based local occupancy;
- A* planning;
- local command generation;
- navigation safety filters;
- simulation validation.

### 1.2 Keep `unitree_g1_vibes` as the hardware reference layer

`unitree_g1_vibes` should be treated as the donor/reference repo for:

- Unitree SDK2 locomotion command examples;
- RealSense RGB-D streaming;
- Livox / MID-360 capture;
- KISS-ICP / SLAM capture;
- safe robot probing scripts;
- known quirks of the real robot APIs.

The real bridge should use this repo for implementation patterns, but `g1_rl` should not become hard-dependent on all of its runtime dependencies during normal simulation runs.

### 1.3 Use explicit interface contracts

Do not directly hack real sensor calls into `scripts/smoke_test_yolo_person_nav.py`.

Instead, create stable types:

```text
RobotObservation
RGBDFrame
LidarFrame
RobotState
Command2D
AdapterStatus
```

and make both sim and real adapters produce/consume these types.

### 1.4 Safety overrides must be outside the policy

The command path must be:

```text
g1_rl proposed command
    ↓
SafetySupervisor
    ↓
CommandSink
```

Never:

```text
g1_rl proposed command
    ↓
Unitree Move()
```

The real robot must always have independent command clipping, watchdogs, stop handling, and operator override.

---

## 2. Proposed package layout

Create the adapter package inside the existing `g1_nav_rl` package:

```text
g1_rl/
  src/
    g1_nav_rl/
      adapters/
        __init__.py
        types.py
        config.py
        safety_supervisor.py

        sim_isaac.py
        real_realsense.py
        real_livox.py
        real_unitree_loco.py
        real_bridge.py

  configs/
    real_g1.yaml

  scripts/
    real_g1_probe_adapters.py
    real_g1_follow_dryrun.py
    real_g1_follow_low_speed.py
```

Do **not** create a loose top-level package such as:

```text
g1_rl/src/adapter
```

Use:

```text
g1_rl/src/g1_nav_rl/adapters
```

so imports stay inside the project namespace.

---

## 3. Phase 1 — shared interface types

Create:

```text
src/g1_nav_rl/adapters/__init__.py
src/g1_nav_rl/adapters/types.py
```

### 3.1 Core dataclasses

`types.py` should define at least:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str | None = None
    distortion_coeffs: tuple[float, ...] = ()


@dataclass
class RGBDFrame:
    timestamp_s: float
    rgb: np.ndarray          # HxWx3, RGB, uint8
    depth_m: np.ndarray      # HxW, float32, meters
    intrinsics: CameraIntrinsics
    T_body_camera: np.ndarray
    frame_id: str = "camera_color_optical"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LidarFrame:
    timestamp_s: float
    points_body_xyz: np.ndarray       # Nx3, meters, body frame
    points_body_xy: np.ndarray | None # Nx2, meters, body frame
    T_body_lidar: np.ndarray
    frame_id: str = "lidar"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose2DReal:
    timestamp_s: float
    x: float
    y: float
    yaw: float
    frame_id: str = "map"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotState:
    timestamp_s: float
    mode: str | None = None
    fsm_state: str | None = None
    battery: float | None = None
    is_estopped: bool = False
    is_faulted: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotObservation:
    timestamp_s: float
    pose: Pose2DReal | None = None
    rgbd: RGBDFrame | None = None
    lidar: LidarFrame | None = None
    robot_state: RobotState | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Command2D:
    timestamp_s: float
    vx: float
    vy: float
    wz: float
    duration_s: float | None = None
    source: str = "policy"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterStatus:
    ok: bool
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
```

### 3.2 Required conventions

The interface should document these conventions explicitly:

```text
rgb:
  HxWx3 uint8 RGB, not BGR.

depth_m:
  HxW float32, meters.
  Depth is aligned to RGB image pixels.
  Depth is optical-axis z-depth unless explicitly documented otherwise.

points_body_xyz:
  Nx3 float32 meters in robot body frame.

body frame:
  +x forward
  +y left
  +z up

Command2D:
  vx: body-frame forward velocity, m/s
  vy: body-frame lateral-left velocity, m/s
  wz: yaw rate, rad/s
```

---

## 4. Phase 2 — configuration schema

Create:

```text
src/g1_nav_rl/adapters/config.py
configs/real_g1.yaml
```

### 4.1 Required real-robot config fields

`configs/real_g1.yaml` should include:

```yaml
real_g1:
  robot:
    iface: eth0
    command_rate_hz: 10.0

  camera:
    provider: realsense
    width: 640
    height: 480
    fps: 30
    color_format: rgb8
    depth_format: z16
    align_depth_to_color: true

    intrinsics:
      width: null
      height: null
      fx: null
      fy: null
      cx: null
      cy: null
      distortion_model: null
      distortion_coeffs: []

    # Placeholder until calibrated.
    T_body_camera:
      translation_xyz: [0.35, 0.0, 0.35]
      rotation_wxyz: [0.5, -0.5, 0.5, -0.5]

  lidar:
    provider: livox_mid360
    mount: normal
    min_range_m: 0.20
    max_range_m: 8.00

    # Project 3D LiDAR to 2D local navigation points.
    z_filter_body_m: [-0.10, 1.50]
    self_filter_box_body:
      x: [-0.50, 0.50]
      y: [-0.40, 0.40]
      z: [-0.20, 1.50]

    # Placeholder until calibrated.
    T_body_lidar:
      translation_xyz: [0.0, 0.0, 0.55]
      rotation_wxyz: [1.0, 0.0, 0.0, 0.0]

  command_limits:
    vx_min: -0.05
    vx_max: 0.10
    vy_min: -0.05
    vy_max: 0.05
    wz_min: -0.20
    wz_max: 0.20

    ax_max: 0.20
    ay_max: 0.20
    awz_max: 0.50

  watchdog:
    max_rgbd_age_s: 0.50
    max_lidar_age_s: 0.30
    max_pose_age_s: 0.50
    max_policy_age_s: 0.20
    stop_on_stale_lidar: true
    stop_on_stale_robot_state: true

  target_tracking:
    redetect_every_steps: 50
    occlusion_grace_steps: 600
    smoothing_alpha: 0.35
    max_jump_m: 1.25

  yolo:
    model: yolov8n.pt
    device: auto
    min_confidence: 0.25
    allowed_classes: ["person"]
```

### 4.2 Config adapter responsibilities

`config.py` should:

- load YAML;
- validate required fields;
- convert transforms into 4x4 matrices;
- expose command limits;
- expose sensor freshness limits;
- avoid importing RealSense, Livox, or Unitree packages.

---

## 5. Phase 3 — Isaac simulation adapter

Create:

```text
src/g1_nav_rl/adapters/sim_isaac.py
```

This should wrap the Isaac-specific parts currently embedded in:

```text
scripts/smoke_test_yolo_person_nav.py
```

### 5.1 Responsibilities

The Isaac adapter should provide:

```python
obs = sim_adapter.read_observation()
sim_adapter.send_command(cmd)
sim_adapter.stop()
status = sim_adapter.status()
```

It should translate Isaac data into `RobotObservation`:

```text
Isaac camera RGB tensor
  -> RGBDFrame.rgb

Isaac depth tensor
  -> RGBDFrame.depth_m

Isaac RayCaster hits
  -> LidarFrame.points_body_xyz / points_body_xy

G1CommandProxy pose
  -> Pose2DReal

G1CommandProxy command
  <- Command2D
```

### 5.2 Why implement sim adapter before real adapter

This validates the abstraction without moving hardware.

The same policy loop should run in sim as:

```python
obs = adapter.read_observation()
cmd = runtime.step(obs)
cmd_safe = safety.filter(obs, cmd)
adapter.send_command(cmd_safe)
```

---

## 6. Phase 4 — RealSense RGB-D adapter

Create:

```text
src/g1_nav_rl/adapters/real_realsense.py
```

Use these `unitree_g1_vibes` files as donors/reference:

```text
snapshot_sensors_v2.py
stream_realsense.py
```

### 6.1 Responsibilities

The RealSense adapter should:

- start the RealSense pipeline;
- configure color and depth streams;
- align depth to color;
- read synchronized color/depth frames;
- convert color BGR -> RGB;
- convert raw depth `uint16` -> `float32` meters;
- retrieve camera intrinsics;
- return `RGBDFrame`;
- attach `T_body_camera` from config;
- expose timestamps and freshness.

### 6.2 Required conversion

RealSense color often arrives as BGR:

```python
rgb = color_bgr[..., ::-1].copy()
```

Raw depth must become meters:

```python
depth_m = depth_raw.astype(np.float32) * depth_scale
```

### 6.3 Acceptance tests

A real RGB-D probe should print:

```text
rgb shape/dtype/range
depth shape/dtype/min/max/valid count
intrinsics fx/fy/cx/cy
depth scale
timestamp age
```

and save debug images:

```text
outputs/real_g1_probe/rgb.png
outputs/real_g1_probe/depth_preview.png
```

---

## 7. Phase 5 — Livox / MID-360 LiDAR adapter

Create:

```text
src/g1_nav_rl/adapters/real_livox.py
```

Use these `unitree_g1_vibes` files as donors/reference:

```text
live_slam.py
live_slam_headless_capture.py
snapshot_sensors_v2.py
```

### 7.1 Responsibilities

The LiDAR adapter should:

- start the Livox stream;
- read the latest point cloud;
- apply mount correction;
- transform points into robot body frame using `T_body_lidar`;
- filter robot self-points;
- filter floor/ceiling if needed;
- produce `LidarFrame`;
- optionally expose SLAM pose separately.

### 7.2 Required processing

The real MID-360 produces 3D points. The current local planner uses 2D occupancy. The adapter must project 3D to 2D safely:

```text
raw Livox points
  -> mount correction
  -> T_body_lidar
  -> height filter in body z
  -> self filter
  -> points_body_xy
  -> local occupancy
```

### 7.3 Acceptance tests

Probe script should print:

```text
raw points count
filtered points count
body-frame xyz min/max
2D xy min/max
timestamp age
```

and save:

```text
outputs/real_g1_probe/lidar_xy.png
```

---

## 8. Phase 6 — Unitree locomotion adapter

Create:

```text
src/g1_nav_rl/adapters/real_unitree_loco.py
```

Use this `unitree_g1_vibes` file as the main donor:

```text
g1_loco_nudge_test.py
```

### 8.1 Responsibilities

The locomotion adapter should:

- initialize Unitree SDK communication;
- probe robot state / FSM state;
- expose `send_command(Command2D)`;
- expose `stop()`;
- handle `continuous_move` versus `continous_move` spelling difference;
- enforce command rate;
- log all sent commands;
- never send motion unless the real bridge is explicitly armed.

### 8.2 Important command convention to verify

Before using autonomy commands, run manual sign tests:

```text
vx > 0  -> robot moves forward
vy > 0  -> robot moves left
wz > 0  -> robot yaws counterclockwise / left
```

If any sign differs, fix it in `real_unitree_loco.py`, not in the planner.

### 8.3 Initial real limits

Start with conservative limits:

```text
vx: -0.05 to 0.10 m/s
vy: -0.05 to 0.05 m/s
wz: -0.20 to 0.20 rad/s
```

Do not use the larger simulation limits for first hardware tests.

---

## 9. Phase 7 — Safety supervisor

Create:

```text
src/g1_nav_rl/adapters/safety_supervisor.py
```

### 9.1 Responsibilities

`SafetySupervisor` should:

- clamp velocities;
- clamp acceleration;
- stop on stale sensor data;
- stop on stale policy data;
- stop on robot fault;
- stop on operator stop;
- optionally stop if LiDAR obstacle is too close;
- log every override.

### 9.2 Suggested API

```python
class SafetySupervisor:
    def __init__(self, config):
        ...

    def filter_command(
        self,
        observation: RobotObservation,
        command: Command2D | None,
        now_s: float,
    ) -> tuple[Command2D, dict]:
        ...
```

### 9.3 Required default behavior

If anything important is missing, default to stop:

```text
missing command       -> stop
stale observation     -> stop
robot fault           -> stop
operator stop         -> stop
adapter exception     -> stop
```

---

## 10. Phase 8 — high-level real bridge

Create:

```text
src/g1_nav_rl/adapters/real_bridge.py
```

### 10.1 External commands

The bridge should accept high-level commands:

```text
follow
stop
pause
resume
status
dry_run
```

### 10.2 Internal states

Use explicit supervisor states:

```text
IDLE
ARMED
FOLLOWING
OCCLUDED_RECENT
LOST_GO_TO_LAST_KNOWN
STOPPING
FAULT
E_STOP
```

### 10.3 Responsibilities

`RealBridge` should:

- initialize adapters;
- collect latest RGB-D, LiDAR, pose, robot state;
- build `RobotObservation`;
- run target provider/tracker and navigation runtime;
- apply `SafetySupervisor`;
- either send command or log command in dry-run mode;
- handle operator stop immediately;
- provide structured status logs.

### 10.4 Important behavior for target loss

For the current project, losing visual contact should not immediately abort motion.

Desired behavior:

```text
TRACKING
  -> use current detection and update target

OCCLUDED_RECENT
  -> continue to last known target

LOST_GO_TO_LAST_KNOWN
  -> continue cautiously to last known target
  -> then stop/search/reacquire depending on next policy
```

Abort only when:

```text
no target was ever acquired
robot reaches last known target and search/reacquisition fails
operator stops
safety supervisor faults
```

---

## 11. Phase 9 — scripts

Create scripts in this order.

### 11.1 Probe adapters

```text
scripts/real_g1_probe_adapters.py
```

No motion.

Responsibilities:

```text
load real_g1.yaml
probe RealSense
probe LiDAR
probe robot state
print sensor stats
save debug artifacts
exit
```

Example command:

```bash
PYTHONPATH=src python scripts/real_g1_probe_adapters.py   --config configs/real_g1.yaml   --iface eth0   --duration_s 5
```

### 11.2 Dry-run follow

```text
scripts/real_g1_follow_dryrun.py
```

No motion.

Responsibilities:

```text
live sensors
run YOLO/target projection
run LiDAR occupancy
run policy
run safety supervisor
print/log proposed commands
do not call Move()
```

Example command:

```bash
PYTHONPATH=src python scripts/real_g1_follow_dryrun.py   --config configs/real_g1.yaml   --iface eth0   --duration_s 30   --target_source yolo
```

### 11.3 Low-speed live follow

```text
scripts/real_g1_follow_low_speed.py
```

Motion only with explicit flag.

Example command:

```bash
PYTHONPATH=src python scripts/real_g1_follow_low_speed.py   --config configs/real_g1.yaml   --iface eth0   --duration_s 20   --target_source yolo   --yes-i-understand-this-moves-the-robot
```

This script must refuse to move unless the explicit flag is present.

---

## 12. Validation sequence

### 12.1 Offline validation

Run:

```bash
PYTHONPATH=src python -m compileall -q src scripts tests
PYTHONPATH=src pytest -q tests
```

### 12.2 Sim adapter validation

Goal:

```text
same behavior as smoke_test_yolo_person_nav.py,
but through ObservationAdapter + CommandSink.
```

Test:

```bash
PYTHONPATH=src python scripts/smoke_test_yolo_person_nav.py   --scenario offset_left   --person_asset_usd /PATH/TO/HUMAN.usd   --person_asset_reference_mode reference   --grid_source hybrid   --enable_safety   --target_source yolo   --no_fallback_synthetic_bbox   --target_depth_source bbox   --redetect_every 50   --target_occlusion_grace_steps 600   --max_steps 1600   --assert_detection   --assert_target   --assert_planner   --assert_success   --assert_no_collision   --assert_margin
```

Expected:

```text
YOLO initial detection succeeds.
Periodic re-detection may temporarily fail.
Tracker does not enter terminal LOST too early.
Robot reaches target.
```

### 12.3 Real sensor probe

No motion.

Acceptance:

```text
RealSense RGB visible and RGB-ordered.
Depth is metric and aligned to RGB.
LiDAR points are in plausible body-frame coordinates.
Robot state is readable.
Timestamps are fresh.
```

### 12.4 Real dry-run

No motion.

Acceptance:

```text
YOLO detects person.
Target2D is plausible.
LiDAR occupancy looks plausible.
Command2D values are bounded.
SafetySupervisor logs allow command.
No Move() call is made.
```

### 12.5 Real low-speed motion

Motion enabled only after dry-run.

Acceptance:

```text
stop command works immediately.
watchdog stops on sensor loss.
commands remain within low-speed limits.
robot moves in expected directions.
target following works only in open area first.
```

---

## 13. Known compatibility gaps to close

### 13.1 RGB/BGR mismatch

RealSense donor scripts commonly produce BGR images.

`g1_rl` expects RGB images.

Required:

```python
rgb = color_bgr[..., ::-1].copy()
```

### 13.2 Depth units mismatch

Sim depth is float meters.

RealSense depth is usually raw `uint16`.

Required:

```python
depth_m = depth_raw.astype(np.float32) * depth_scale
```

### 13.3 Camera intrinsics missing from snapshots

Real adapter must retrieve and store:

```text
fx
fy
cx
cy
width
height
distortion model
distortion coefficients
```

### 13.4 Camera extrinsics missing

Need calibrated or estimated:

```text
T_body_camera
```

This is required for:

```text
bbox + depth -> Target2D
```

### 13.5 LiDAR extrinsics missing

Need calibrated or estimated:

```text
T_body_lidar
```

This is required for:

```text
Livox points -> body-frame occupancy
```

### 13.6 3D LiDAR to 2D occupancy

Real MID-360 returns 3D points. Current planner uses 2D occupancy.

Required:

```text
height filtering
self filtering
range filtering
projection to x-y
```

### 13.7 Timing and synchronization

Need timestamps for:

```text
RGB-D
LiDAR
pose
robot state
policy command
sent command
```

Need freshness checks before motion.

### 13.8 Command scale and sign

Need manual validation for:

```text
vx
vy
wz
```

Do not assume sim and real command signs are identical until tested.

### 13.9 Watchdog

Need automatic stop when:

```text
sensor stream stalls
policy loop stalls
robot state becomes invalid
operator requests stop
adapter throws exception
```

---

## 14. Dependency policy

Normal simulation should not require real-robot packages.

Good:

```bash
PYTHONPATH=src python scripts/smoke_test_yolo_person_nav.py
```

should work without:

```text
pyrealsense2
Livox SDK bindings
Unitree SDK2 Python
```

Real scripts may require them.

Use optional imports inside real adapter modules:

```python
try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise RuntimeError("RealSense adapter requires pyrealsense2") from exc
```

Do not import hardware dependencies in `g1_nav_rl/adapters/__init__.py`.

---

## 15. Recommended implementation order

Implement in this order:

1. `types.py`
2. `config.py`
3. `safety_supervisor.py`
4. `configs/real_g1.yaml`
5. `real_realsense.py`
6. `real_livox.py`
7. `real_unitree_loco.py`
8. `real_bridge.py`
9. `real_g1_probe_adapters.py`
10. `real_g1_follow_dryrun.py`
11. `real_g1_follow_low_speed.py`
12. Optional: `sim_isaac.py` extraction from current smoke script

If time is limited, prioritize:

```text
types.py
config.py
safety_supervisor.py
real_g1_probe_adapters.py
real_g1_follow_dryrun.py
```

before any script that can move the robot.

---

## 16. First milestone definition

The first milestone is **not** autonomous walking.

The first milestone is:

```text
Real sensors live
    ↓
RobotObservation built
    ↓
g1_rl computes target and command
    ↓
SafetySupervisor filters command
    ↓
Command is printed/logged only
    ↓
Robot does not move
```

Acceptance criteria:

```text
RGB-D frame valid.
YOLO person detection works.
Depth projection gives plausible Target2D.
LiDAR occupancy is plausible.
Command2D is bounded.
Stop/fault paths are tested.
No motion command is sent.
```

Only after that should we enable low-speed real motion.

---

## 17. Second milestone definition

The second milestone is **low-speed real command validation**.

Acceptance criteria:

```text
Robot accepts Move(vx, vy, wz).
Command signs are verified.
Stop command works.
Watchdog stop works.
Safety limits are enforced.
Open-area low-speed motion behaves as expected.
```

This should be done before target following with obstacles.

---

## 18. Third milestone definition

The third milestone is **static-target following in open space**.

Acceptance criteria:

```text
Person is detected.
Target2D remains plausible.
Robot moves slowly toward standoff target.
Temporary visual loss does not immediately abort.
Robot stops at safe distance.
Operator can stop immediately.
```

No obstacle test yet.

---

## 19. Fourth milestone definition

The fourth milestone is **static-target following with obstacle safety**.

Acceptance criteria:

```text
LiDAR obstacle detection works.
Local occupancy is stable.
Robot slows/stops near obstacles.
Planner avoids simple obstacle layouts.
Safety watchdog remains active.
```

This is the first real-world analogue of the current Isaac `offset_left` scenario.

---

## 20. Summary

The current `g1_rl` stack already has the right high-level abstractions in concept:

```text
observation -> target/navigation runtime -> vx/vy/wz command
```

The missing piece is an explicit adapter layer that normalizes real and simulated data into the same contract.

The most important rule for the next phase:

```text
Do not connect the real robot directly to the current Isaac smoke script.
```

Instead:

```text
Create adapter types.
Create config and safety supervisor.
Probe sensors without motion.
Run dry-run policy without motion.
Only then enable low-speed command output.
```
