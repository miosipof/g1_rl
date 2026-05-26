# STREAMING.md — YOLO Streaming Detection Plan

## Purpose

The original live follow script runs YOLO inside the control loop. This caused two major problems:

```text
1. CPU YOLO makes the control loop slow, often around 1 Hz instead of the requested 10 Hz
2. repeated independent detections do not preserve object identity
```

The goal of streaming detection is to separate perception from control:

```text
detector/tracker process:
  RealSense RGB-D → YOLO/tracker → tracked detections → latest_tracks.json

controller process:
  latest_tracks.json + Livox → planning/control/safety
```

This architecture is valid for both real G1 and Isaac Sim. It should not live only as a real-robot adapter; the detection stream schema is a shared perception contract.

---

## Problem with the old integrated YOLO path

The old live loop did this:

```text
capture RGB-D
run YOLO
select target
project depth
read LiDAR
plan
safety-filter
send command
```

This creates several problems:

```text
YOLO blocks the control loop
redetect-every N means N slow control iterations, not real-time N*dt
detections can jump between visually similar objects
no persistent track identity
debugging mixes perception latency, planning, and control faults
```

In real runs, the detector often found the bottle intermittently and selected different bounding boxes across time. The target estimate could jump dramatically, and the robot would then replan toward a changing target.

---

## Streaming architecture

The proposed architecture is:

```text
Process A: detection stream producer
  RealSense or Isaac RGB-D
  → YOLO detector
  → tracker
  → depth projection
  → publish DetectionStreamFrame

Process B: follow controller
  read latest DetectionStreamFrame
  → select/lock target track
  → convert to target hint
  → LiDAR planning
  → command generation
```

The controller no longer launches YOLO.

---

## Shared schema

The stream message is represented by a shared schema:

```text
g1_nav_rl.perception.detection_stream
```

Core types:

```text
DetectionStreamFrame
TrackDetection
DetectionStreamClient
DetectionStreamReadResult
write_detection_stream_frame_atomic
```

Example JSON:

```json
{
  "schema_version": 1,
  "timestamp_s": 12345.678,
  "frame_id": 1842,
  "source": "real_g1_yolo_track_stream",
  "image_shape": [480, 640, 3],
  "tracks": [
    {
      "track_id": 7,
      "class_name": "bottle",
      "confidence": 0.72,
      "bbox_xyxy": [310.0, 155.0, 365.0, 310.0],
      "bbox_center_uv": [337.5, 232.5],
      "bbox_area_px": 8525.0,
      "target_body_xy": [1.63, -0.18],
      "depth_m": 2.14,
      "depth_valid_ratio": 0.63,
      "status": "tracked",
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

The controller only needs:

```text
track_id
class_name
confidence
target_body_xy
depth_m
depth_valid_ratio
timestamp_s
```

---

## Producer responsibilities

A stream producer should:

```text
1. capture RGB-D
2. run YOLO or another detector
3. optionally run tracker to assign stable track IDs
4. estimate depth for each bbox
5. project bbox/depth to body-frame target coordinates
6. apply target standoff
7. publish latest_tracks.json atomically
8. save detector-side debug overlays
```

Initial producer:

```text
scripts/real_g1_yolo_track_stream.py
```

Initial backend:

```text
Ultralytics YOLO predict/track
```

Later backends:

```text
TensorRT YOLO on Jetson
DeepStream
Isaac synthetic detector
offline replay
```

The stream schema should not change when the backend changes.

---

## Consumer responsibilities

The controller should:

```text
1. read latest_tracks.json
2. reject stale stream frames
3. select or maintain a locked track
4. convert selected TrackDetection into RobotObservation.target_hint
5. use LiDAR for map/planning
6. avoid running internal YOLO
```

Controller flags should look like:

```bash
--target-source stream
--detection-stream-path outputs/runtime/latest_tracks.json
--detection-stream-max-age-s 1.0
--target-class bottle
--stream-min-confidence 0.10
--stream-min-depth-valid-ratio 0.10
```

For the simplified stepwise controller, stream mode should be optional. The simplest version can still run one YOLO observation per stop, but the streaming path remains useful once detector stability is verified.

---

## Target lock

Streaming alone is not enough. The controller should lock onto a target track.

Recommended state machine:

```text
UNLOCKED:
  choose candidate by class, confidence, depth quality, bbox size, and maybe image-center proximity

LOCKED(track_id):
  follow the same track_id if fresh and plausible

OCCLUDED_RECENT:
  track temporarily missing; hold previous plan for a limited number of actions

LOST:
  stop or scan; reacquire only after a stable candidate appears
```

Target lock should not blindly trust track IDs. It should also validate:

```text
same class
track freshness
target_body_xy plausibility
depth quality
confidence
jump distance from previous target
bbox area
```

This avoids switching between two bottles/people or latching onto small false positives.

---

## File-based stream first

The first transport is an atomic JSON file:

```text
outputs/runtime/latest_tracks.json
```

Producer writes:

```text
latest_tracks.tmp
rename → latest_tracks.json
```

Consumer polls every control step.

Advantages:

```text
simple
debuggable
works without extra services
easy to replay
valid in Isaac and real robot
```

Later we can replace the transport with:

```text
ZeroMQ PUB/SUB
UDP localhost
Unix socket
shared memory
```

without changing the schema.

---

## Isaac integration

The same stream contract is valid in Isaac.

Implemented direction:

```text
scripts/smoke_test_yolo_person_nav_stream_compare.py
```

Comparison modes:

```text
direct:
  old architecture with direct/fallback target extraction

stream:
  target pulled from DetectionStreamFrame JSON
```

The stream smoke uses a static synthetic `latest_tracks.json` to validate the consumer path:

```text
DetectionStreamFrame → target_xy_w → A* navigation
```

This validates the controller interface before live detector integration.

---

## Real integration

The intended real test uses two terminals.

Terminal A — producer:

```bash
python scripts/real_g1_yolo_track_stream.py \
  --config configs/real_g1.yaml \
  --output outputs/runtime/latest_tracks.json \
  --mode track \
  --yolo-model yolov8n.pt \
  --yolo-device cpu \
  --yolo-confidence 0.10 \
  --target-class bottle \
  --stand-off-m 0.8 \
  --save-rgbd-debug outputs/stream_detector/rgbd_debug \
  --json-lines
```

Terminal B — controller:

```bash
python scripts/real_g1_follow_detect_live.py \
  --target-source stream \
  --detection-stream-path outputs/runtime/latest_tracks.json \
  ...
```

The current priority is to validate the file stream on CPU first. Jetson GPU/TensorRT is a performance optimization after the architecture is proven.

---

## Jetson GPU path

After the file-stream producer/consumer path works, switch the producer backend.

Preferred order:

```text
1. verify installed Torch/CUDA/Ultralytics on Jetson
2. try Ultralytics on CUDA if versions are compatible
3. export model to TensorRT engine
4. run producer with TensorRT backend
5. keep controller unchanged
```

The controller should not care whether tracks were produced by CPU YOLO, CUDA YOLO, TensorRT, Isaac, or replay.

---

## Relation to simplified controller

The simplified `Stop → Observe → Plan → Execute → Stop` loop reduces the need for continuous streaming, but streaming still helps with detection stability and target identity.

Recommended practical split:

```text
Stepwise v1:
  run one detector observation at each stop
  no continuous stream required

Stepwise v2:
  consume latest stream frame at each stop
  detector can run continuously in parallel
  controller still executes discrete actions
```

This keeps the simple control structure while allowing the detector to run independently and maintain track IDs.

---

## Acceptance criteria

Streaming detection is working when:

```text
1. latest_tracks.json updates at detector rate
2. track_id remains stable for a visible fixed object
3. target_body_xy is physically plausible
4. depth_m and depth_valid_ratio are plausible
5. controller logs target_source=stream
6. internal YOLO is not called by controller
7. controller loop timing no longer includes YOLO inference time
8. debug overlays match selected track
```

It is not enough for YOLO to produce occasional detections. The key improvement is stable target identity and decoupled timing.

---

## Current status

Implemented or planned components:

```text
shared detection stream schema/client          done
Isaac stream consumer smoke                    done
first RealSense + YOLO file producer           started
real follow stream consumer                    next/partial depending on branch state
Jetson GPU/TensorRT backend                    later
target lock manager                            needed
```

For the simplified near-term experiment, streaming should not delay the stop-observe-plan-execute script. It should remain a parallel improvement path.
