# LiDAR / RayCaster Notes

This document describes the current LiDAR integration in the G1 command-level navigation prototype. The implementation is intentionally modest: it validates raycast-based obstacle sensing, converts ray hits into the G1 proxy/body frame, rasterizes a local occupancy grid, and saves debug visualizations. It is not yet a physically faithful moving LiDAR simulation.

## Current status

Implemented pieces:

```text
src/g1_nav_rl/sensors/lidar.py
src/g1_nav_rl/sensors/lidar_mount.py
src/g1_nav_rl/mapping/occupancy_grid.py
scripts/smoke_test_lidar.py
scripts/smoke_test_lidar_geometry.py
configs/robot/g1_command_proxy.yaml
```

Validated behavior:

```text
spawn Isaac Lab RayCaster sensor              OK
cast rays against a mesh cube target          OK
read ray hit points / derived ranges          OK
convert world hit points to proxy/body frame  OK
rasterize local occupancy grid                OK
save occupancy-grid debug images              OK
validate front / left / right convention      OK
validate proxy-relative obstacle distance     OK
```

Important limitation:

```text
The current RayCaster origin is static.
The proxy moves.
Obstacle hit points are transformed into the moving proxy frame.
This gives useful proxy-relative direction and distance for static obstacles,
but it is not a true moving LiDAR scan stream.
```

---

## Configuration

The LiDAR configuration lives in:

```text
configs/robot/g1_command_proxy.yaml
```

The important fields are:

```yaml
lidar:
  prim_path: "{ENV_REGEX_NS}/G1LidarMount"

  mesh_prim_paths:
    - "{ENV_REGEX_NS}/DebugCube/geometry/mesh"

  max_distance: 8.0
  update_period: 0.0

  # Pattern parameters are supplied through Isaac Lab LidarPatternCfg.
  pattern:
    channels: 1
    vertical_fov_range: [0.0, 0.0]
    horizontal_fov_range: [-60.0, 60.0]
    horizontal_res: 0.5

occupancy_grid:
  resolution: 0.05
  size_x: 8.0
  size_y: 8.0
```

The mesh target path matters. The RayCaster target should point to the actual USD mesh prim:

```text
/World/envs/env_0/DebugCube/geometry/mesh
```

not just:

```text
/World/envs/env_0/DebugCube
```

The smoke test uses `MeshCuboidCfg` for the debug cube so that the actual mesh target exists under:

```text
DebugCube/geometry/mesh
```

---

## Modules

### `sensors/lidar.py`

This module contains the sensor-side processing helpers.

Typical responsibilities:

```text
RayCaster output
→ finite-hit filtering
→ range computation
→ world-hit to body-frame projection
→ scan/point dictionary for downstream code
```

Conceptually:

```python
from g1_nav_rl.sensors.lidar import build_lidar_observation

lidar_obs = build_lidar_observation(
    raycaster=scene["lidar"],
    proxy_pos_w=proxy.pos_w,
    proxy_yaw=proxy.yaw,
    cfg=lidar_cfg,
)
```

Expected outputs include:

```text
ranges        # [N, num_rays], derived from ray_hits_w - ray_origin_w
points_b      # [N, num_rays, 2], obstacle points in proxy/body frame
valid_mask    # [N, num_rays], finite and in-range hits
min_range     # derived range summary
```

The important frame convention is:

```text
+x = forward
+y = left
-y = right
```

### `mapping/occupancy_grid.py`

This module converts local 2D points into a local occupancy grid.

Conceptually:

```python
from g1_nav_rl.mapping.occupancy_grid import rasterize_points_to_occupancy_grid

grid = rasterize_points_to_occupancy_grid(
    points_b=lidar_obs["points_b"],
    valid_mask=lidar_obs["valid_mask"],
    cfg=grid_cfg,
)
```

The grid is robot-local. With the current convention:

```text
front of robot  → positive x cells
left of robot   → positive y cells
right of robot  → negative y cells
```

Debug images are saved under:

```text
outputs/lidar_grid/
```

or, for geometry tests:

```text
outputs/lidar_grid/front/
outputs/lidar_grid/left/
outputs/lidar_grid/right/
```

### `sensors/lidar_mount.py`

This module contains helpers related to the LiDAR mount prim.

Current role:

```text
provide a stable RayCaster attachment prim
avoid attaching RayCaster directly to the manually moved G1Proxy Xform
preserve canonical USD transform handling where needed
```

A previous experiment tried making this mount a `RigidObject` and moving it with `write_root_pose_to_sim(...)`. That caused a low-level PhysX CUDA illegal-memory-access error in this environment, so the current safe path keeps the RayCaster mount static.

---

## Smoke tests

Run all commands from the repo root:

```bash
cd /home/osipov/g1_rl
```

The scripts should be launched the same way as the existing Isaac Lab smoke tests:

```bash
PYTHONUNBUFFERED=1 python scripts/<script_name>.py --headless
```

### 1. Basic LiDAR smoke test

```bash
PYTHONUNBUFFERED=1 python scripts/smoke_test_lidar.py --headless
```

Expected behavior:

```text
RayCaster initializes
valid ranges are reported
local 2D points are produced
occupancy-grid images are saved
```

Example successful output:

```text
ranges=(1, 241)
points_b=(1, 241, 2)
valid_env0=25
min_range_env0=1.806
occupied cells=16
Saved occupancy grid: outputs/lidar_grid/occupancy_grid_000000.png
```

### 2. Proxy-relative moving check

```bash
PYTHONUNBUFFERED=1 python scripts/smoke_test_lidar.py \
  --headless \
  --command 0.2 0.0 0.0 \
  --assert_proxy_relative_distance_decrease
```

Expected behavior:

```text
proxy x position increases
raw RayCaster min range stays approximately constant
proxy-relative min distance decreases
```

Example successful output:

```text
raw RayCaster min range first/last: 1.806 / 1.806
proxy-relative min distance first/last: 1.800 / 1.378
```

Interpretation:

```text
The RayCaster itself did not move.
The proxy moved toward the static cube.
The cube's world hit points became closer in the proxy/body frame.
```

Do not use the older moving-origin assertion:

```bash
--assert_range_decrease
```

That assertion assumes a true moving RayCaster origin, which is not the current architecture.

### 3. Front / left / right geometry tests

```bash
PYTHONUNBUFFERED=1 python scripts/smoke_test_lidar_geometry.py --headless --case front --assert_geometry
PYTHONUNBUFFERED=1 python scripts/smoke_test_lidar_geometry.py --headless --case left  --assert_geometry
PYTHONUNBUFFERED=1 python scripts/smoke_test_lidar_geometry.py --headless --case right --assert_geometry
```

Expected centroid signs:

```text
front → centroid y approximately 0
left  → centroid y positive
right → centroid y negative
```

Example successful outputs:

```text
front centroid xy_b: (1.800, -0.000)
left  centroid xy_b: (1.657,  0.705)
right centroid xy_b: (1.657, -0.705)
```

These tests validate the local frame convention used by the planner:

```text
+x forward
+y left
-y right
```

---

## The RayCaster moving-sensor issue

In simple terms, the problem is this:

```text
Our G1 proxy is moved manually by writing USD transforms.
Isaac Lab RayCaster expects to track a stable scene/physics object.
Those two systems do not mix cleanly for a moving LiDAR origin.
```

What was tried:

### 1. Attach RayCaster directly to `G1Proxy`

This produced warnings because `G1Proxy` is a manually moved USD `Xform`, not a physics object. Isaac Lab warned that pose updates may not be correct in headless mode.

### 2. Attach RayCaster to a child LiDAR prim

This failed until the prim existed and had the expected canonical USD transform structure.

### 3. Attach RayCaster to a kinematic rigid-body mount

This was closer to the expected physics-style approach, but updating that mount caused a low-level PhysX CUDA illegal-memory-access error in the current environment.

Therefore the current safe design is:

```text
static RayCaster origin
→ static world hit points
→ moving G1 proxy pose
→ transform world hit points into proxy/body frame
→ local occupancy grid
```

This is enough to validate local mapping and planner-side geometry for static obstacles. It is not a physically complete moving LiDAR simulation.

---

## What is valid now?

| Quantity | Valid now? | Meaning |
|---|---:|---|
| Direction in proxy/body frame | Yes | Front / left / right relative to the G1 proxy. |
| Proxy-relative obstacle distance | Yes, for static obstacles | Distance from current proxy pose to static world hit points. |
| Occupancy grid in proxy frame | Yes, for static obstacles | Local grid built from transformed hit points. |
| Raw moving-lidar range stream | No | Raw ranges come from the static RayCaster origin. |
| Moving-sensor occlusion changes | No | Rays are not recast from the robot's changing pose. |
| Dynamic obstacle sensing | No | Current RayCaster setup targets static mesh geometry. |

---

## How to make it more realistic later

A more physically correct architecture would be:

```text
at scan time t:
  get current proxy pose
  cast rays from that pose
  compute current hits
  build local occupancy grid
repeat periodically
```

Possible backends:

```text
1. StaticRayCasterBackend
   Current working method. Static world hits re-expressed in proxy frame.

2. SnapshotRayCasterBackend
   Periodically recast rays from the current proxy pose.
   This may be implemented with a custom analytic raycaster rather than resetting Isaac Lab RayCaster.

3. RtxLidarBackend
   More realistic Isaac sensor pipeline for moving LiDAR and dynamic scenes.

4. Analytic2DRaycasterBackend
   Fast 2D raycasting against known obstacle geometry or a global occupancy map.
   Likely practical for command-level navigation and local planner development.
```

For the near-term deterministic navigation stack, the recommended path is:

```text
keep current static RayCaster smoke tests
use proxy-relative local occupancy grids for planner development
avoid further PhysX moving-mount experiments in this branch
consider custom 2D snapshot raycasting or RTX LiDAR later
```

---

## Recommended next integration step

Add LiDAR outputs to `G1NavEnv` observations behind config flags:

```python
obs = {
    "rgb": ...,
    "depth": ...,
    "target_vector": ...,
    "proxy_pose": ...,
    "last_command": ...,

    "lidar_ranges": ...,
    "lidar_points_b": ...,
    "occupancy_grid": ...,
}
```

Add metrics:

```python
info["metrics"].update({
    "lidar_valid_count": ...,
    "lidar_min_range": ...,
    "proxy_relative_lidar_min_distance": ...,
    "occupied_cell_count": ...,
})
```

Then the deterministic planner can consume:

```text
target_vector + occupancy_grid
→ local command [vx, vy, wz]
→ safety filter
→ command limiter
→ proxy / real adapter
```
