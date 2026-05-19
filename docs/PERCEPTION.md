# PERCEPTION.md

## Purpose

This document describes the current perception-target pipeline for the Unitree G1 autonomy project.

The goal of this layer is deliberately narrow:

```text
RGB-D camera frame
+ 2D object detector output
+ camera intrinsics
+ camera-to-world transform
        ↓
Target2D(x, y)
        ↓
NavRuntime / SimpleNavigator
        ↓
MotionCommand(vx, vy, wz)
```

The perception layer does **not** command the robot directly. It only produces a navigation target. Planning, replanning, and safety filtering remain separate modules.

---

## Current status

Implemented and tested:

```text
bbox + depth + intrinsics + camera transform -> Target2D        ✅
Isaac RGB-D camera snapshot -> depth projection                 ✅
Static human USD asset loaded in Isaac Sim                      ✅
YOLO person detection on rendered human                         ✅
YOLO bbox + depth -> Target2D                                   ✅
```

The current working smoke test is:

```text
scripts/smoke_test_yolo_person_target.py
```

The main reusable modules are:

```text
src/g1_nav_rl/perception/target_from_bbox.py
src/g1_nav_rl/perception/yolo_detector.py
```

---

## Data flow

### 1. RGB-D input

The RGB-D input consists of:

```text
rgb_image      : H x W x 3 uint8 image
depth_image    : H x W float image, meters preferred
intrinsics     : fx, fy, cx, cy, width, height
T_world_camera : 4x4 camera pose in the world/map frame
```

For Isaac Sim, the camera is created in the scene and read after warmup steps.

For the real robot, we already verified that a RealSense D435I stream is available. The observed color intrinsics were approximately:

```text
width  = 640
height = 480
fx     = 604.2
fy     = 603.8
cx     = 320.7
cy     = 256.5
```

Important requirement for real robot use:

```text
The depth image must be aligned to the RGB/color image.
```

That means a YOLO bbox in RGB pixel coordinates must index into the correct depth pixels.

For RealSense, the expected approach is conceptually:

```python
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
aligned_depth = aligned_frames.get_depth_frame()
color = aligned_frames.get_color_frame()
```

---

## 2D detection format

The perception stack uses a generic detection format:

```python
Detection2D(
    class_name="person",
    bbox_xyxy=(x1, y1, x2, y2),
    confidence=0.81,
)
```

The detector backend is intentionally interchangeable. Current sources:

```text
synthetic projected bbox  -> deterministic geometry/debug path
YOLO bbox                 -> actual visual detector path
```

For YOLO, the first target class is:

```text
person
```

The detector module is:

```text
src/g1_nav_rl/perception/yolo_detector.py
```

The adapter returns the same `Detection2D` object regardless of detector backend.

---

## Depth extraction from a bbox

A raw bbox may include background, clothing, limbs, and invalid depth values. The target adapter therefore does not use a single pixel by default.

Given a bbox:

```text
x1, y1, x2, y2
```

we select a center crop inside the bbox:

```text
bbox_center_crop_frac = 0.5
```

Then we collect valid depth pixels:

```text
finite depth
and depth_min_m <= depth <= depth_max_m
```

Default values:

```text
depth_min_m = 0.20
depth_max_m = 8.00
```

The representative depth is a robust percentile, usually median:

```text
depth_percentile = 0.50
```

This gives:

```text
depth_m
```

For the tested Isaac human asset at nominal position `(3.0, 0.0)`, YOLO detected a person with confidence around `0.81`, and bbox-depth projection estimated the visible surface around:

```text
object_xy_w ≈ (2.858, 0.010)
```

This is expected: depth measures the visible body surface, not necessarily the USD root position.

---

## Pixel + depth -> 3D camera point

The module assumes the standard pinhole RGB-D camera convention:

```text
camera x = image right
camera y = image down
camera z = optical forward
```

For a bbox center pixel:

```text
u = (x1 + x2) / 2
v = (y1 + y2) / 2
z = depth_m
```

The camera-frame 3D point is:

```text
X_cam = (u - cx) * z / fx
Y_cam = (v - cy) * z / fy
Z_cam = z
```

The result is:

```text
point_camera = [X_cam, Y_cam, Z_cam]
```

Example from the working YOLO human test:

```text
depth_m      ≈ 2.858
point_camera ≈ [-0.0097, -0.0354, 2.858]
```

---

## Camera point -> world point

The camera point is converted to world coordinates with:

```text
point_world_h = T_world_camera @ [X_cam, Y_cam, Z_cam, 1]
```

Then:

```text
object_xy_w = (point_world.x, point_world.y)
```

For a simple forward-facing camera mounted on the robot body, we use this body-frame convention:

```text
body x = forward
body y = left
body z = up
```

The helper:

```python
make_forward_camera_T_body_camera()
```

uses this optical-camera to body-frame mapping:

```text
body_x = camera_z
body_y = -camera_x
body_z = -camera_y
```

For the current Isaac camera setup:

```text
camera_xyz = (0.0, 0.0, 0.75)
```

and the world point from the successful YOLO run was approximately:

```text
point_world = [2.858, 0.010, 0.785]
object_xy_w = [2.858, 0.010]
```

---

## Standoff target

For human targets, the robot should not navigate into the person. Therefore, the target provider can apply a standoff distance.

Given:

```text
robot_xy
object_xy_w
stand_off_distance_m
```

we compute a target point on the line from the robot to the object, stopping short of the detected object:

```text
direction = normalize(object_xy_w - robot_xy)
target_xy_w = object_xy_w - stand_off_distance_m * direction
```

Example:

```text
object_xy_w           ≈ (2.858, 0.010)
stand_off_distance_m  = 0.8
target_xy_w           ≈ (2.058, 0.007)
```

This is why target error relative to the actual person root is intentionally large. The relevant perception geometry metric is `object_error_xy_m`, not `target_error_xy_m`, when standoff is enabled.

---

## Main API

The core function is:

```python
bbox_depth_to_target2d(
    detection,
    depth_image,
    intrinsics,
    T_world_camera,
    robot_pose=None,
    config=TargetFromBBoxConfig(...),
)
```

Typical usage:

```python
from g1_nav_rl.perception.target_from_bbox import (
    CameraIntrinsics,
    Detection2D,
    TargetFromBBoxConfig,
    bbox_depth_to_target2d,
    make_forward_camera_T_body_camera,
    pose2d_to_T_world_body,
)
from g1_nav_rl.navigation.simple_nav import Pose2D

intr = CameraIntrinsics(
    fx=604.2,
    fy=603.8,
    cx=320.7,
    cy=256.5,
    width=640,
    height=480,
)

det = Detection2D(
    class_name="person",
    bbox_xyxy=(274.4, 30.9, 359.7, 429.9),
    confidence=0.812,
)

robot_pose = Pose2D(0.0, 0.0, 0.0)

T_world_body = pose2d_to_T_world_body(robot_pose)
T_body_camera = make_forward_camera_T_body_camera(
    translation_body=(0.0, 0.0, 0.75)
)
T_world_camera = T_world_body @ T_body_camera

result = bbox_depth_to_target2d(
    detection=det,
    depth_image=depth_m,
    intrinsics=intr,
    T_world_camera=T_world_camera,
    robot_pose=robot_pose,
    config=TargetFromBBoxConfig(
        depth_min_m=0.20,
        depth_max_m=8.0,
        bbox_center_crop_frac=0.50,
        depth_percentile=0.50,
        stand_off_distance_m=0.8,
    ),
)

target = result.target
```

The output includes:

```text
target          : Target2D(x, y)
depth_m         : selected depth estimate
point_camera    : 3D camera-frame point
point_world     : 3D world-frame point
object_xy_w     : detected object XY position
used_stand_off  : whether standoff was applied
reason          : status/debug string
```

---

## YOLO detector adapter

The optional detector wrapper is:

```text
src/g1_nav_rl/perception/yolo_detector.py
```

It loads an Ultralytics YOLO model and converts YOLO outputs to `Detection2D`.

The current smoke test uses the default lightweight model:

```text
yolov8n.pt
```

If the model is not found locally, Ultralytics may download it automatically on first run.

The detector is optional. The rest of the perception-target geometry does not depend on YOLO.

---

## Isaac human asset loading

The static human target test uses:

```text
scripts/smoke_test_yolo_person_target.py
```

The tested human asset path was:

```text
~/isaacsim_assets_partial/Assets/Isaac/5.1/Isaac/People/Characters/F_Business_02/F_Business_02.usd
```

A direct transform on the referenced character root caused problems. The robust pattern is:

```text
/World/envs/env_0/PersonAsset        # wrapper Xform, transform ops here
/World/envs/env_0/PersonAsset/Model  # payload/reference to human USD here
```

Use payload mode for large character assets:

```bash
--person_asset_reference_mode payload
```

This avoids transform-op conflicts on the referenced USD root and worked in headless Isaac Sim.

---

## Smoke tests

### 1. Offline bbox-depth geometry test

```bash
PYTHONPATH=src python scripts/smoke_test_target_from_bbox.py
```

With standoff:

```bash
PYTHONPATH=src python scripts/smoke_test_target_from_bbox.py --stand_off 0.8
```

Expected behavior:

```text
synthetic RGB-D bbox -> Target2D
target-from-bbox smoke test passed
```

---

### 2. Isaac RGB-D + synthetic projected bbox

This checks camera/depth plumbing without YOLO:

```bash
python scripts/smoke_test_yolo_person_target.py \
  --person_asset_usd ~/isaacsim_assets_partial/Assets/Isaac/5.1/Isaac/People/Characters/F_Business_02/F_Business_02.usd \
  --person_asset_reference_mode payload \
  --person_asset_scale 1.0 \
  --person_x 3.0 \
  --person_y 0.0 \
  --fallback_synthetic_bbox \
  --skip_yolo \
  --stand_off 0.8 \
  --warmup_steps 30 \
  --assert_detection \
  --assert_target
```

---

### 3. Isaac RGB-D + YOLO person target

```bash
python scripts/smoke_test_yolo_person_target.py \
  --person_asset_usd ~/isaacsim_assets_partial/Assets/Isaac/5.1/Isaac/People/Characters/F_Business_02/F_Business_02.usd \
  --person_asset_reference_mode payload \
  --person_asset_scale 1.0 \
  --person_x 3.0 \
  --person_y 0.0 \
  --fallback_synthetic_bbox \
  --stand_off 0.8 \
  --warmup_steps 30 \
  --assert_target
```

Successful output should contain:

```text
YOLO detections: 1
det class='person'
selected source/class/conf : yolo / person / ...
object_xy_w ≈ (...)
target_xy_w ≈ (...)
yolo/person target smoke test complete
```

The latest successful result had:

```text
YOLO confidence      ≈ 0.812
object_error_xy_m    ≈ 0.142
target_xy_w          ≈ (2.058, 0.007)
```

---

## Outputs

The Isaac YOLO/person smoke test saves:

```text
outputs/yolo_person_target/rgb.png
outputs/yolo_person_target/depth_vis.png
outputs/yolo_person_target/rgb.npy
outputs/yolo_person_target/depth_m.npy
outputs/yolo_person_target/result.json
```

Use these to debug:

```text
whether the person is visible
whether YOLO bbox is reasonable
whether depth image contains valid values inside the bbox
whether the target estimate is plausible
```

---

## Integration with navigation

The perception output is just:

```python
Target2D(x, y)
```

It should feed the existing runtime:

```text
YOLO person bbox
+ aligned depth
+ intrinsics
+ camera pose
        ↓
Target2D(x, y)
        ↓
NavRuntime.step(pose, target, occupancy_grid)
        ↓
MotionCommand(vx, vy, wz)
        ↓
minimal safety guard
```

Do not couple YOLO directly to motion commands.

The next recommended smoke test is:

```text
scripts/smoke_test_yolo_person_nav.py
```

Initial scope:

```text
static human target
no obstacle or simple obstacle
YOLO target estimate
LiDAR occupancy grid
NavRuntime
minimal safety guard
motion toward standoff target
```

---

## Current limitations

### 1. Static camera transform assumption

The current Isaac test uses an assumed forward-facing camera transform. On the real robot, this must be replaced by a measured or calibrated `T_body_camera`.

### 2. Depth measures visible surface

The object estimate is based on visible depth, not object/root center. For people, this is acceptable if we use a standoff distance.

### 3. RGB-D alignment required

For the real RealSense stream, depth must be aligned to color before using YOLO bbox pixels.

### 4. No target tracking yet

Each frame is treated independently. There is no temporal smoothing, identity tracking, or multi-person target lock.

### 5. YOLO dependency is optional

The geometry module works without YOLO. The YOLO wrapper requires `ultralytics` to be installed in the active environment.

### 6. No navigation coupling yet

Phase 5 currently validates target generation. The next step is connecting the generated `Target2D` to `NavRuntime` in a perception-navigation smoke test.

---

## Real robot checklist

Before using this on the real G1:

```text
1. Start RealSense RGB-D stream headlessly, without cv2.imshow.
2. Align depth to color.
3. Save RGB, aligned depth in meters, and intrinsics.
4. Confirm camera-to-body transform.
5. Run YOLO on RGB.
6. Convert bbox + depth to Target2D in robot/body or map frame.
7. Feed Target2D to NavRuntime.
8. Use LiDAR/SLAM occupancy for planning.
9. Keep safety filter minimal: emergency collision guard only.
```

Suggested snapshot files:

```text
snapshots/realsense_rgb.npy
snapshots/realsense_depth_m.npy
snapshots/realsense_intrinsics.json
snapshots/realsense_meta.json
```

---

## Summary

The perception layer now provides a tested bridge from visual detection to navigation target:

```text
RGB-D + YOLO person bbox -> robust depth estimate -> 3D camera point -> world XY -> Target2D
```

This keeps the autonomy stack modular:

```text
Perception decides what/where the target is.
Navigation decides how to reach it.
Safety prevents immediate collisions.
```
