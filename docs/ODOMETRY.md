# Odometry in Isaac Sim and on the Unitree G1

This document describes the odometry work implemented in this repository and
how to test it in headless tests, Isaac Sim, and on a real Unitree G1.

## What was implemented

The stepwise navigator no longer needs to assume:

```text
commanded velocity × planned duration = actual movement
```

Both simulation and the real robot can now use the same closed-loop action
executor:

```text
stop and confirm stationary
        ↓
read fresh target, LiDAR, and odometry
        ↓
plan one stepwise action
        ↓
send/refresh the velocity command
        ↓
measure translation or rotation from odometry
        ↓
checked stop and wait for stationary samples
        ↓
invalidate the old body-frame plan and target
        ↓
observe and replan
```

The relevant files are:

- `src/adapters/real_unitree_odometry.py`: subscribes to real G1 DDS odometry.
- `src/adapters/sim_isaac_odometry.py`: converts Isaac proxy ground truth into
  the same snapshot interface and can inject sensor imperfections.
- `src/adapters/stepwise_executor.py`: shared measured-motion executor.
- `scripts/smoke_test_yolo_person_nav.py`: Isaac scene and navigation loop.
- `scripts/smoke_test_yolo_stepwise_nav.py`: convenient Isaac launcher.
- `scripts/real_g1_odom_probe.py`: read-only real odometry probe.
- `scripts/real_g1_odom_nudge.py`: guarded single-motion real test.
- `scripts/real_g1_follow_stepwise.py`: full real stepwise navigation runner.

## Shared odometry contract

The simulation and hardware adapters provide the executor with the same core
operations:

- `snapshot()`: return the latest atomic pose/velocity sample;
- `wait_for_snapshot(...)`: wait for a newer sequence number;
- `status()`: report healthy, stale, unavailable, or faulted state;
- `capture_origin()`: make the current pose the run-relative origin.

Each sample contains:

```text
x, y, z                 position in meters
yaw                     heading in radians
vx, vy, vz              linear velocity in m/s
yaw_speed               angular velocity in rad/s
sequence                 monotonically increasing sample number
epoch                    changes after a detected real-odometry discontinuity
receive_time_s           local monotonic receive time
```

The executor uses position/yaw differences to decide when an action has
completed. Velocity is used to confirm that the robot has settled after the
stop command. Planned `duration_s` is only a timeout estimate.

## Run the automated tests

The tests do not require Isaac Sim or Unitree hardware:

```bash
PYTHONPATH=src pytest -q \
  tests/test_sim_isaac_odometry.py \
  tests/test_unitree_odometry.py \
  tests/test_stepwise_odometry_executor.py \
  tests/test_stepwise_odometry_integration.py
```

Run the complete repository suite with:

```bash
PYTHONPATH=src pytest -q
```

The focused tests cover parsing, freshness, discontinuities, lag, deterministic
noise, sample loss, yaw wrapping, stop/settle behavior, and action completion.

## Isaac Sim

### What the simulated adapter measures

The Isaac adapter reads the G1 command proxy's world position, yaw, body-frame
velocity, and yaw rate after each physics step. It packages those values like a
real Unitree odometry sample.

This is simulated odometry, not a separate native Isaac IMU. With all error
options set to zero, its input is simulator ground truth. The imperfection model
can then add:

- publication lag;
- a lower sample rate;
- position and yaw noise;
- velocity and yaw-rate noise;
- fixed position/yaw bias;
- distance and yaw scale error;
- random sample loss.

Random errors are repeatable when the same seed is used.

### Baseline run

Run this inside the Isaac Lab/Isaac Sim Python environment, from the repository
root:

```bash
python scripts/smoke_test_yolo_stepwise_nav.py \
  -- \
  --scenario front_box \
  --max_steps 80
```

Measured odometry execution is enabled by default. With the default odometry
settings there is no added lag or noise.

### Run with artificial odometry errors

This example adds 80 ms of lag, position/yaw noise, and 2% sample loss:

```bash
python scripts/smoke_test_yolo_stepwise_nav.py \
  --odom-lag-s 0.08 \
  --odom-position-noise-std-m 0.01 \
  --odom-yaw-noise-std-deg 0.5 \
  --odom-velocity-noise-std-mps 0.01 \
  --odom-yaw-rate-noise-std-radps 0.01 \
  --odom-dropout-probability 0.02 \
  --odom-seed 7 \
  -- \
  --scenario front_box \
  --max_steps 80
```

Useful launcher options:

| Option | Default | Meaning |
|---|---:|---|
| `--odom-sample-period-s` | `0.02` | Time between generated samples. |
| `--odom-lag-s` | `0.0` | Delay before a ground-truth sample becomes visible. |
| `--odom-position-noise-std-m` | `0.0` | Gaussian position noise standard deviation. |
| `--odom-yaw-noise-std-deg` | `0.0` | Gaussian yaw noise standard deviation. |
| `--odom-velocity-noise-std-mps` | `0.0` | Linear-velocity noise standard deviation. |
| `--odom-yaw-rate-noise-std-radps` | `0.0` | Yaw-rate noise standard deviation. |
| `--odom-position-bias-x-m` | `0.0` | Fixed world-x bias. |
| `--odom-position-bias-y-m` | `0.0` | Fixed world-y bias. |
| `--odom-yaw-bias-deg` | `0.0` | Fixed heading bias. |
| `--odom-distance-scale` | `1.0` | Scale applied to position and linear velocity. |
| `--odom-yaw-scale` | `1.0` | Scale applied to yaw and yaw rate. |
| `--odom-dropout-probability` | `0.0` | Independent probability of dropping each sample. |
| `--odom-stale-s` | `0.10` | Maximum accepted age of the latest sample. |
| `--odom-seed` | `0` | Random seed for repeatable experiments. |

Keep the sample period below `--odom-stale-s`; otherwise the executor may
correctly reject the stream as stale. Large velocity noise may also prevent the
settling thresholds from being satisfied.

### Compare with the previous time-based behavior

```bash
python scripts/smoke_test_yolo_stepwise_nav.py \
  --legacy-duration-execution \
  -- \
  --scenario front_box \
  --max_steps 80
```

This disables the simulated odometry executor and runs each action for its
planned duration, which is the behavior used before this change.

### Isaac outputs

The default run directory is:

```text
outputs/isaac_stepwise_stream/<run-id>/
```

Important files include:

```text
stepwise_summary.json
child_stdout.log
child_stderr.log
yolo_stepwise_nav/nav_lidar_trace.csv
yolo_stepwise_nav/stepwise_trace.jsonl
```

Each `stepwise_trace.jsonl` action row includes:

- the planned action and command;
- odometry before execution;
- `execution_result`, including requested and achieved distance/angle;
- odometry after stop/settling;
- simulator ground truth after execution;
- completion/failure reason and sample counts.

Compare `execution_result` with `ground_truth_after_execution` when evaluating
lag, noise, scale error, or overshoot.

## Real Unitree G1

### Real odometry source

The real adapter subscribes through `unitree_sdk2py` to:

```text
rt/odommodestate
```

using:

```python
unitree_sdk2py.idl.unitree_go.msg.dds_.SportModeState_
```

The adapter reads `position`, `velocity`, `yaw_speed`, and
`imu_state.rpy[2]`. This is the G1 firmware's integrated base odometry. It is
not LiDAR SLAM and should not be treated as globally drift-free localization.

### Hardware prerequisites

Before testing:

- run on the computer connected to the G1 control network;
- install this repository's real-side dependencies and `unitree_sdk2py`;
- identify the correct network interface, such as `eth0`;
- confirm that `rt/odommodestate` is present;
- keep the robot in a clear area with an operator ready to stop it;
- for the full follower, configure the RealSense and MID-360/Livox setup;
- check the mount transforms in `configs/real_g1.yaml` for the actual robot.

The locomotion state gate expects FSM ID `200` by default. Do not bypass that
check unless the active firmware/mode has been independently verified.

### 1. Read-only odometry probe

This command never sends a motion command:

```bash
PYTHONPATH=src python scripts/real_g1_odom_probe.py \
  --iface eth0 \
  --topic rt/odommodestate \
  --duration-s 10 \
  --rate-hz 10 \
  --jsonl outputs/real_g1_odom_probe.jsonl
```

While the robot is still, verify:

- `sequence` increases;
- `age_s` stays small, normally below the configured 0.10 s stale limit;
- `velocity` and `yaw_speed` are close to zero;
- values are finite and `epoch` remains constant.

Use the guarded nudge below—not manual handling—to verify motion direction and
measured displacement.

### 2. Guarded single forward nudge

This test moves the robot. Start with a clear floor, a nearby operator, and the
small default 0.10 m request:

```bash
PYTHONPATH=src python scripts/real_g1_odom_nudge.py \
  --iface eth0 \
  --distance-m 0.10 \
  --speed-mps 0.15 \
  --expected-fsm-id 200 \
  --yes-i-understand-this-moves-the-robot
```

The script checks the locomotion state, confirms stationary odometry, refreshes
the move command, stops when measured progress reaches the tolerance, and waits
for stationary samples. It always attempts a stop during cleanup.

A successful JSON result has `completed: true`, `reason: "completed"`, and
`stop_accepted: true`. Also inspect `requested_distance_m`,
`achieved_distance_m`, `lateral_error_m`, and `settle_time_s`.

Do not continue to autonomous navigation if this test reports stale odometry,
wrong-direction motion, excessive lateral deviation, no progress, a timeout,
or a rejected stop.

### 3. Full follower dry run

The full runner requires live perception and LiDAR even when motion is disabled.
Omit the motion-confirmation flag for a dry run:

```bash
PYTHONPATH=src python scripts/real_g1_follow_stepwise.py \
  --config configs/real_g1.yaml \
  --iface eth0 \
  --unitree-vibes-dir /path/to/unitree_g1_vibes \
  --livox-config /path/to/mid360_config.json \
  --target-source yolo \
  --max-steps 5 \
  --save-lidar-debug outputs/real_stepwise/lidar \
  --save-rgbd-debug outputs/real_stepwise/rgbd \
  --json-lines
```

This validates the camera, detector, LiDAR, occupancy grid, planner, and command
generation without sending real movement. Because real motion is disabled, it
does not exercise odometry-based physical completion; use the read-only probe
and nudge for that validation.

### 4. Full real odometry-driven follower

Only after the probe, nudge, sensor checks, and dry run succeed:

```bash
PYTHONPATH=src python scripts/real_g1_follow_stepwise.py \
  --config configs/real_g1.yaml \
  --iface eth0 \
  --unitree-vibes-dir /path/to/unitree_g1_vibes \
  --livox-config /path/to/mid360_config.json \
  --target-source yolo \
  --max-steps 10 \
  --forward-segment-m 0.20 \
  --forward-speed-mps 0.15 \
  --turn-speed-radps 0.25 \
  --save-lidar-debug outputs/real_stepwise/lidar \
  --save-rgbd-debug outputs/real_stepwise/rgbd \
  --json-lines \
  --yes-i-understand-this-moves-the-robot
```

With real motion enabled, the runner:

1. connects to DDS odometry and captures an origin;
2. checks the locomotion FSM/state gate;
3. issues a checked stop and confirms stationary samples before observation;
4. requires a fresh target after every physical action;
5. executes translation/rotation from measured odometry;
6. issues another checked stop and confirms settling;
7. invalidates the old body-frame plan and target before replanning.

When `--save-lidar-debug` is enabled, the timestamped LiDAR output directory
contains `stepwise_real_trace.jsonl`. Each row includes the decision, command,
odometry execution result, start/stop/settled poses, and current odometry
snapshot. `--json-lines` prints the same run information to standard output.

For an external detection stream, replace `--target-source yolo` with:

```text
--target-source stream --detection-stream-path /path/to/latest_tracks.json
```

The detection producer must keep the file fresh and provide body-frame target
coordinates using the repository's detection-stream schema.

### Important real executor options

| Option | Default | Purpose |
|---|---:|---|
| `--odom-topic` | `rt/odommodestate` | DDS odometry topic. |
| `--odom-stale-s` | `0.10` | Reject older samples. |
| `--odom-max-position-jump-m` | `0.50` | Start a new epoch after a large position jump. |
| `--odom-max-yaw-jump-deg` | `45` | Start a new epoch after a large yaw jump. |
| `--odom-distance-tolerance-m` | `0.02` | Absolute translation completion tolerance. |
| `--odom-yaw-tolerance-deg` | `2.0` | Absolute turn completion tolerance. |
| `--odom-max-lateral-error-m` | `0.12` | Abort excessive sideways deviation. |
| `--odom-no-progress-timeout-s` | `0.75` | Abort when measured progress stops. |
| `--odom-settle-linear-mps` | `0.03` | Maximum linear speed considered stationary. |
| `--odom-settle-yaw-radps` | `0.05` | Maximum yaw speed considered stationary. |
| `--odom-settle-samples` | `3` | Consecutive stationary samples required. |
| `--odom-settle-timeout-s` | `1.0` | Maximum settling wait. |
| `--action-timeout-scale` | `2.0` | Multiplier on planned action duration. |
| `--action-timeout-margin-s` | `0.50` | Extra action timeout margin. |

Change these only from recorded evidence. In particular, do not loosen stale,
lateral-error, settling, or timeout gates merely to make a failed run continue.

## Completion and failure reasons

Common `execution_result.reason` values are:

| Reason | Meaning |
|---|---|
| `completed` | Measured target progress was reached and settling succeeded. |
| `odom_stale` | No sufficiently fresh odometry sample was available. |
| `odom_discontinuity` | The real stream jumped to a new odometry epoch. |
| `wrong_direction` | Measured motion opposed the requested direction. |
| `lateral_deviation` | Translation moved too far sideways. |
| `no_odometric_progress` | Commands were accepted but progress stopped. |
| `action_timeout` | The action exceeded its duration-based watchdog. |
| `command_rejected` | The locomotion/simulation command sink rejected a move. |
| `settle_timeout` | The robot did not produce enough stationary samples. |

A rejected stop raises a terminal locomotion fault. Autonomous motion must not
continue after that condition.

## Current limitations

- Isaac uses proxy ground truth plus an artificial error model; it does not yet
  simulate the complete G1 firmware estimator or a separate physical IMU.
- Real G1 odometry is local integrated odometry, not global SLAM.
- Stage A deliberately replans after every completed or partial physical action.
- Cached body-frame targets are invalidated after motion rather than propagated.
- Real camera and LiDAR mount transforms still require robot-specific
  calibration in `configs/real_g1.yaml`.
