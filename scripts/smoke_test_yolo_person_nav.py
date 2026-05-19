#!/usr/bin/env python3
"""
Full-pipeline smoke test: YOLO/RGB-D person target + LiDAR/A* navigation in Isaac Lab.

This script is the bridge between:

  - scripts/smoke_test_nav.py
      One-shot inflated occupancy grid, one A* plan, direct waypoint tracking,
      no replanning, no shortcutting, no command smoothing, no local escape.

  - scripts/smoke_test_lidar_astar_planner.py
      Isaac Lab scene setup, G1 command proxy, RayCaster/LiDAR access, target
      marker, obstacle scenario assets, and debug grid image output.

Design constraints:

  - Plan exactly once from the initial LiDAR-derived occupancy grid.
  - Follow the raw A* path monotonically in world frame.
  - Do not run periodic replanning.
  - Do not shortcut the path.
  - Do not apply clearance-safety overrides or escape behavior.
  - By default, remove the proxy's extra command smoothing/acceleration dynamics
    so this test remains close to smoke_test_nav.py. Pass
    --use_config_proxy_dynamics to test the configured proxy dynamics.

This is therefore a sensor-backed baseline, not the full production planner.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_VERSION = "smoke_test_yolo_person_nav_v10_numpy_import_fallback_debug"

from isaaclab.app import AppLauncher  # noqa: E402


# -----------------------------------------------------------------------------
# Scenario library
# -----------------------------------------------------------------------------
# Obstacles are specified in the env-local/world frame used by the smoke tests:
# +x forward from the proxy start, +y left, z is height.
SCENARIOS: dict[str, dict[str, Any]] = {
    "clear": {
        "description": "No blocking obstacle; a small marker is far off the direct start-to-target line.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": False,
        "expect_success": True,
        "allow_empty_lidar": True,
        "obstacles": [
            {
                "name": "ClearSideMarker",
                "shape": "box",
                "pos": [4.5, 3.0, 0.35],
                "size": [0.35, 0.35, 0.70],
                "yaw": 0.0,
                "color": [0.25, 0.65, 0.25],
            }
        ],
    },
    "front_box": {
        "description": "Single rectangular box centered on the direct start-to-target line.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "FrontBox",
                "shape": "box",
                "pos": [2.2, 0.0, 0.4],
                "size": [0.8, 0.8, 0.8],
                "yaw": 0.0,
                "color": [0.2, 0.8, 0.2],
            }
        ],
    },
    "offset_left": {
        "description": "Obstacle is front-left; baseline should usually keep right/center.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "OffsetLeft",
                "shape": "box",
                "pos": [2.0, 0.55, 0.35],
                "size": [0.75, 0.75, 0.70],
                "yaw": 0.0,
                "color": [0.2, 0.75, 0.35],
            }
        ],
    },
    "offset_right": {
        "description": "Obstacle is front-right; baseline should usually keep left/center.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "OffsetRight",
                "shape": "box",
                "pos": [2.0, -0.55, 0.35],
                "size": [0.75, 0.75, 0.70],
                "yaw": 0.0,
                "color": [0.2, 0.75, 0.35],
            }
        ],
    },
    "wide_box": {
        "description": "Wide obstacle across y; the path should choose a side route.",
        "target_pos": [4.5, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "WideBox",
                "shape": "box",
                "pos": [2.2, 0.0, 0.35],
                "size": [0.55, 1.60, 0.70],
                "yaw": 0.0,
                "color": [0.2, 0.75, 0.35],
            }
        ],
    },
    "long_box": {
        "description": "Long obstacle along x; useful for checking sustained side following.",
        "target_pos": [5.0, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "LongBox",
                "shape": "box",
                "pos": [2.6, 0.0, 0.35],
                "size": [1.60, 0.55, 0.70],
                "yaw": 0.0,
                "color": [0.2, 0.75, 0.35],
            }
        ],
    },
    "rotated_box": {
        "description": "Rotated rectangular obstacle centered near the start-to-target line.",
        "target_pos": [4.5, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "RotatedBox",
                "shape": "box",
                "pos": [2.25, 0.0, 0.35],
                "size": [1.20, 0.55, 0.70],
                "yaw": 0.65,
                "color": [0.15, 0.70, 0.50],
            }
        ],
    },
    "round": {
        "description": "Circular/oval-like obstacle using a cylinder mesh.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "RoundObstacle",
                "shape": "cylinder",
                "pos": [2.1, 0.0, 0.40],
                "radius": 0.45,
                "height": 0.80,
                "yaw": 0.0,
                "color": [0.2, 0.8, 0.2],
            }
        ],
    },
    "multi": {
        "description": "Two separated rectangular obstacles leaving a navigable side/gap route.",
        "target_pos": [5.0, 0.0, 0.08],
        "expect_avoidance": True,
        "expect_success": True,
        "obstacles": [
            {
                "name": "MultiLeft",
                "shape": "box",
                "pos": [2.0, 0.65, 0.35],
                "size": [0.80, 0.70, 0.70],
                "yaw": 0.15,
                "color": [0.2, 0.75, 0.35],
            },
            {
                "name": "MultiRight",
                "shape": "box",
                "pos": [3.0, -0.65, 0.35],
                "size": [0.80, 0.70, 0.70],
                "yaw": -0.15,
                "color": [0.2, 0.55, 0.80],
            },
        ],
    },
    "blocked_wall": {
        "description": "A wide wall blocks the map; A* should report no path or fail to reach target.",
        "target_pos": [4.5, 0.0, 0.08],
        "expect_avoidance": False,
        "expect_success": False,
        "obstacles": [
            {
                "name": "BlockingWall",
                "shape": "box",
                "pos": [2.2, 0.0, 0.35],
                # v5: this is a logical map-blocking wall.  The exact y-size is
                # expanded after CLI parsing so it spans the configured planning map.
                "size": [0.45, 5.0, 0.70],
                "blocks_map_y": True,
                "yaw": 0.0,
                "color": [0.8, 0.25, 0.25],
            }
        ],
    },
}


# -----------------------------------------------------------------------------
# CLI + Isaac Sim launcher
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Sensor-backed minimal nav smoke test: Isaac Lab scene + RayCaster/LiDAR "
        "occupancy snapshot + one A* plan + direct waypoint tracking."
    )
)
parser.add_argument(
    "--config",
    type=str,
    default=str(PROJECT_ROOT / "configs" / "robot" / "g1_command_proxy.yaml"),
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="front_box")
parser.add_argument("--list_scenarios", action="store_true", help="List built-in scenarios and exit before launching Isaac Sim.")

parser.add_argument("--output_dir", default=None, help="Default: outputs/nav_lidar/<scenario>.")
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--print_every", type=int, default=50)
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--no_save_grid", action="store_true")
parser.add_argument(
    "--save_raw_grid",
    action="store_true",
    help=(
        "Save raw, uninflated occupancy in the normal per-step grid images. "
        "Default saves the inflated occupancy actually used by A*."
    ),
)
parser.add_argument(
    "--viz_free_update_every",
    type=int,
    default=0,
    help=(
        "How often to ray-carve free/unknown cells for the saved world occupancy visualization. "
        "Default 0 updates only on saved frames, which avoids expensive per-frame image-map updates. "
        "Set to a positive step interval to accumulate free space more densely."
    ),
)

parser.add_argument("--lidar_only", action="store_true", help="Backward-compatible alias for --grid_source lidar.")
parser.add_argument(
    "--raycast_mesh_mode",
    choices=("auto", "actual", "aggregate"),
    default="auto",
    help=(
        "RayCaster mesh targeting strategy. 'actual' targets each obstacle mesh directly; "
        "'aggregate' creates one raycast-only USD mesh containing all obstacle geometry; "
        "'auto' uses aggregate when more than one obstacle is present to avoid Isaac/RayCaster "
        "multi-mesh reset instability."
    ),
)
parser.add_argument(
    "--grid_source",
    choices=("lidar", "scenario", "hybrid"),
    default="hybrid",
    help=(
        "Occupancy source for A*: 'lidar' uses only RayCaster hit cells; "
        "'scenario' uses the same known 2D footprints as smoke_test_nav.py; "
        "'hybrid' uses max(lidar, scenario). --lidar_only overrides this to 'lidar'."
    ),
)
parser.add_argument(
    "--lidar_obstacle_thickness",
    type=float,
    default=0.45,
    help=(
        "Conservative unknown depth, in meters, added behind each first-return LiDAR point. "
        "This converts a visible obstacle surface into a slab-like occupied region. Set 0 for point-only LiDAR."
    ),
)
parser.add_argument(
    "--lidar_obstacle_lateral_padding",
    type=float,
    default=0.08,
    help=(
        "Extra lateral padding, in meters, around each LiDAR return before A* inflation. "
        "This stabilizes sparse scans and prevents sawtooth/aliasing in raw maps."
    ),
)
parser.add_argument(
    "--lidar_map_mode",
    choices=("instant", "memory"),
    default="memory",
    help=(
        "'instant' uses only the current scan; 'memory' accumulates conservative LiDAR obstacle points "
        "in world XY and reprojects them into the current body-frame grid."
    ),
)
parser.add_argument(
    "--map_update_every",
    type=int,
    default=10,
    help="Append a new conservative LiDAR scan to map memory every N control steps. Initial scan is always inserted.",
)
parser.add_argument(
    "--max_lidar_memory_points",
    type=int,
    default=120000,
    help="Maximum conservative world-frame LiDAR obstacle points retained per environment in memory mode.",
)
parser.add_argument(
    "--enable_replanning",
    action="store_true",
    help="Periodically update the LiDAR map and replan if the remaining path conflicts with the current inflated occupancy.",
)
parser.add_argument(
    "--replan_every",
    type=int,
    default=25,
    help="When --enable_replanning is set, force a replan every N control steps. Set <=0 to rely only on path-block checks.",
)
parser.add_argument(
    "--path_check_lookahead",
    type=float,
    default=3.0,
    help="Meters of the remaining path checked against the current inflated occupancy map before triggering replanning.",
)
parser.add_argument(
    "--path_check_ignore_distance",
    type=float,
    default=0.30,
    help="Ignore the first N meters of the remaining path during path/occupancy conflict checks to avoid self-near noise.",
)
parser.add_argument(
    "--stop_if_replan_fails",
    action="store_true",
    help="Stop safely if a triggered replan cannot find a path. Default keeps the previous path for diagnostics.",
)

parser.add_argument(
    "--lidar_z_offset",
    type=float,
    default=0.10,
    help=(
        "Override CFG lidar.offset.pos[2]. Default 0.10 puts the horizontal scan at world z≈0.65 "
        "for the default proxy root z=0.55. This avoids grazing 0.70 m-tall obstacle tops."
    ),
)
parser.add_argument(
    "--lidar_warmup_steps",
    type=int,
    default=3,
    help="Number of sim frames to step before taking the one-shot LiDAR occupancy snapshot.",
)
parser.add_argument(
    "--force_lidar_visible_obstacle_height",
    type=float,
    default=1.10,
    help=(
        "For box/cylinder smoke-test obstacles, force vertical extent high enough for the horizontal "
        "RayCaster scan to cut through the mesh. Set <=0 to keep scenario heights unchanged. "
        "Default 1.10 makes all 2D nav obstacles unambiguously LiDAR-visible."
    ),
)
parser.add_argument(
    "--allow_empty_lidar",
    action="store_true",
    help="Do not fail early if --lidar_only produces zero valid hits in a scenario that should expose obstacles.",
)
parser.add_argument("--resolution", type=float, default=0.05)
parser.add_argument("--x_min", type=float, default=-2.0)
parser.add_argument("--x_max", type=float, default=6.0)
parser.add_argument("--y_min", type=float, default=-4.0)
parser.add_argument("--y_max", type=float, default=4.0)
parser.add_argument("--disable_diagonal", action="store_true")
parser.add_argument("--max_expansions", type=int, default=0)

parser.add_argument("--safety_margin", type=float, default=0.35, help="Extra planning margin added to the proxy body radius.")
parser.add_argument("--hand_length", type=float, default=0.75, help="Default target success radius.")
parser.add_argument("--success_radius", type=float, default=None)

parser.add_argument("--waypoint_reached_radius", type=float, default=0.12)
parser.add_argument("--k_xy", type=float, default=1.5)
parser.add_argument("--k_yaw", type=float, default=2.0)
parser.add_argument("--max_vx", type=float, default=0.30)
parser.add_argument("--max_vy", type=float, default=0.25)
parser.add_argument("--max_wz", type=float, default=0.60)
parser.add_argument("--allow_backward", action="store_true")

# Phase-4 LiDAR-only safety filter. Disabled by default; when enabled it runs
# after NavRuntime emits the nominal command and before the proxy receives it.
parser.add_argument("--enable_safety", action="store_true", help="Enable LiDAR-based slow/stop/backup/sidestep/stuck command filter.")
parser.add_argument("--safety_front_slow_distance", type=float, default=0.90)
parser.add_argument("--safety_front_stop_distance", type=float, default=0.55)
parser.add_argument("--safety_front_backup_distance", type=float, default=0.35)
parser.add_argument("--safety_side_open_distance", type=float, default=0.80)
parser.add_argument("--safety_side_slow_distance", type=float, default=0.0)
parser.add_argument("--safety_side_stop_distance", type=float, default=0.45)
parser.add_argument("--safety_side_critical_distance", type=float, default=0.35)
parser.add_argument("--safety_clearance_percentile", type=float, default=0.10)
parser.add_argument("--safety_backup_speed", type=float, default=0.08)
parser.add_argument("--safety_backup_lateral_speed", type=float, default=0.03)
parser.add_argument("--safety_sidestep_speed", type=float, default=0.08)
parser.add_argument("--safety_slow_min_scale", type=float, default=0.50)
parser.add_argument("--safety_stuck_window_steps", type=int, default=0)
parser.add_argument("--safety_stuck_recovery_steps", type=int, default=30)

# Optional RGB-D/depth safety filter. This runs after LiDAR safety, using the
# camera depth image only. RGB is reserved for later semantic target detection.
parser.add_argument("--enable_rgbd_safety", action="store_true", help="Enable RGB-D/depth command safety filter after LiDAR safety.")
parser.add_argument("--rgbd_safety_front_slow_distance", type=float, default=0.90)
parser.add_argument("--rgbd_safety_front_stop_distance", type=float, default=0.55)
parser.add_argument("--rgbd_safety_front_backup_distance", type=float, default=0.35)
parser.add_argument("--rgbd_safety_side_open_distance", type=float, default=0.80)
parser.add_argument("--rgbd_safety_side_slow_distance", type=float, default=0.0)
parser.add_argument("--rgbd_safety_side_stop_distance", type=float, default=0.45)
parser.add_argument("--rgbd_safety_side_critical_distance", type=float, default=0.35)
parser.add_argument("--rgbd_safety_vertical_band", type=float, nargs=2, default=(0.25, 0.90))
parser.add_argument("--rgbd_safety_center_width_frac", type=float, default=0.60)
parser.add_argument("--rgbd_safety_clearance_percentile", type=float, default=0.05)
parser.add_argument("--rgbd_safety_slow_min_scale", type=float, default=0.50)
parser.add_argument("--rgbd_safety_backup_speed", type=float, default=0.08)
parser.add_argument("--rgbd_safety_backup_lateral_speed", type=float, default=0.03)
parser.add_argument("--rgbd_safety_sidestep_speed", type=float, default=0.08)

# Phase-5 perception target provider. This script uses RGB-D + YOLO/fallback bbox
# to produce Target2D, then passes that target to NavRuntime.
parser.add_argument("--person_x", type=float, default=None, help="World/env x for the static person target. Default: scenario target x.")
parser.add_argument("--person_y", type=float, default=None, help="World/env y for the static person target. Default: scenario target y.")
parser.add_argument("--person_yaw", type=float, default=3.141592653589793)
parser.add_argument("--person_asset_usd", type=str, default="", help="Optional realistic human USD asset. If omitted, use a simple proxy humanoid.")
parser.add_argument("--person_asset_reference_mode", type=str, default="payload", choices=("reference", "payload"))
parser.add_argument("--person_asset_scale", type=float, default=1.0)
parser.add_argument("--skip_person_asset_reference", action="store_true")
parser.add_argument("--skip_yolo", action="store_true", help="Do not run YOLO; use fallback synthetic bbox if enabled.")
parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
parser.add_argument("--yolo_conf", type=float, default=0.25)
parser.add_argument("--yolo_iou", type=float, default=0.45)
parser.add_argument("--yolo_device", type=str, default="")
parser.add_argument("--target_source", choices=["auto", "yolo", "fallback"], default="auto", help="Perception target source. auto runs YOLO first then fallback; fallback skips YOLO and uses projected known-person bbox; yolo requires YOLO detection unless assertions are disabled.")
parser.add_argument("--target_class", type=str, default="person")
parser.add_argument("--min_confidence", type=float, default=0.25)
parser.add_argument("--selection_policy", type=str, default="nearest", choices=("nearest", "largest", "highest_confidence"))
parser.add_argument("--fallback_synthetic_bbox", action="store_true", default=True, help="Use projected known person bbox if YOLO is unavailable/no detection.")
parser.add_argument("--synthetic_bbox_w", type=float, default=120.0)
parser.add_argument("--synthetic_bbox_h", type=float, default=260.0)
parser.add_argument("--synthetic_point_z", type=float, default=1.0)
parser.add_argument("--target_depth_min", type=float, default=0.20)
parser.add_argument("--target_depth_max", type=float, default=8.0)
parser.add_argument("--target_depth_source", choices=("auto", "bbox", "geometric"), default="auto", help="Depth source for bbox->Target2D. auto uses geometric depth for synthetic fallback and bbox depth for YOLO. geometric bypasses Isaac depth read/sync and uses known static person XY; useful for full-pipeline nav isolation.")
parser.add_argument("--target_crop_frac", type=float, default=0.50)
parser.add_argument("--target_depth_percentile", type=float, default=0.50)
parser.add_argument("--stand_off", type=float, default=0.8, help="Meters to stop short of the detected person/object.")
parser.add_argument("--perception_warmup_steps", type=int, default=5, help="Extra camera warmup frames before extracting the target.")
parser.add_argument("--perception_camera_pose_source", choices=("fixed", "proxy"), default="fixed", help="Camera pose source for the initial perception target. fixed avoids CUDA proxy-pose reads; proxy tracks the live proxy pose but can synchronize GPU tensors.")
parser.add_argument("--skip_second_scene_reset", action="store_true", help="Do not call scene.reset() again inside run_simulator after main sim.reset(); useful for debugging heavy external assets.")
parser.add_argument("--assert_detection", action="store_true")
parser.add_argument("--assert_target", action="store_true")
parser.add_argument("--assert_target_error", type=float, default=-1.0)
parser.add_argument(
    "--use_config_proxy_dynamics",
    action="store_true",
    help="Keep proxy smoothing/acceleration limits from YAML. Default removes these extra dynamics for baseline equivalence.",
)

parser.add_argument("--assert_planner", action="store_true")
parser.add_argument("--assert_success", action="store_true")
parser.add_argument("--assert_no_collision", action="store_true")
parser.add_argument("--assert_margin", action="store_true")
parser.add_argument("--margin_tolerance", type=float, default=0.03)

AppLauncher.add_app_launcher_args(parser)

# Parse early enough to support --list_scenarios without launching Isaac.
if "--list_scenarios" in sys.argv:
    tmp = argparse.ArgumentParser(add_help=False)
    tmp.add_argument("--list_scenarios", action="store_true")
    tmp.parse_known_args()
    print("Available minimal nav + LiDAR scenarios:")
    for name, spec in sorted(SCENARIOS.items()):
        print(f"  {name:14s} {spec['description']}")
    raise SystemExit(0)

args_cli = parser.parse_args()
# Isaac/RTX cameras must be enabled before AppLauncher starts the app.
args_cli.enable_cameras = True
GRID_SOURCE = "lidar" if bool(args_cli.lidar_only) else str(args_cli.grid_source)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Isaac Lab imports after app launch
# -----------------------------------------------------------------------------
import torch  # noqa: E402
import numpy as np  # noqa: E402
import traceback  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg, RayCasterCfg, patterns  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from g1_nav_rl.mapping.occupancy_grid import (  # noqa: E402
    OccupancyGridCfg,
    occupancy_grid_to_uint8_image,
    rasterize_points_to_occupancy_grid,
)
from g1_nav_rl.planning.grid_astar import (  # noqa: E402
    GridAStarCfg,
    GridAStarResult,
    inflate_occupancy_grid,
    plan_grid_path_xy,
    xy_to_grid_cell,
)
from g1_nav_rl.navigation.simple_nav import (  # noqa: E402
    Pose2D as NavPose2D,
    SimpleNavConfig,
    SimpleNavigator,
    Target2D as NavTarget2D,
)
from g1_nav_rl.navigation.runtime import NavRuntime, NavRuntimeConfig  # noqa: E402
from g1_nav_rl.navigation.safety import LidarSafetyConfig, LidarSafetyFilter, RgbdSafetyConfig, RgbdSafetyFilter  # noqa: E402
from g1_nav_rl.perception.target_from_bbox import (  # noqa: E402
    CameraIntrinsics,
    Detection2D,
    TargetFromBBoxConfig,
    bbox_depth_to_target2d,
    center_crop_bbox_xyxy,
    target_xy_with_standoff,
    transform_point,
    unproject_pixel_to_camera,
    make_forward_camera_T_body_camera,
    pose2d_to_T_world_body,
    select_detection_for_target,
)
from g1_nav_rl.robots.g1_command_proxy import (  # noqa: E402
    G1CommandProxy,
    G1CommandProxyLimits,
    ensure_canonical_xform_ops,
)
from g1_nav_rl.sensors.lidar import (  # noqa: E402
    LidarObservationCfg,
    build_lidar_proxy_observation,
    summarize_lidar_observation,
)
from g1_nav_rl.sensors.lidar_mount import (  # noqa: E402
    LidarMountSyncCfg,
    sync_lidar_mounts_to_proxy_usd,
)
from g1_nav_rl.utils.geometry import rotate_world_to_body_2d  # noqa: E402


PointXY = tuple[float, float]
Pose2D = tuple[float, float, float]


@dataclass
class StepRecord:
    step: int
    x: float
    y: float
    yaw: float
    waypoint_index: int
    waypoint_x: float
    waypoint_y: float
    target_distance: float
    scenario_edge_clearance: float
    scenario_margin_slack: float
    grid_edge_clearance: float
    vx: float
    vy: float
    wz: float
    nominal_vx: float
    nominal_vy: float
    nominal_wz: float
    lidar_valid: int
    occupied_cells: int
    safety_action: str = "disabled"
    safety_front_clearance: float = math.inf
    safety_left_clearance: float = math.inf
    safety_right_clearance: float = math.inf
    safety_override: bool = False
    safety_scale: float = 1.0
    rgbd_safety_action: str = "disabled"
    rgbd_safety_front_depth: float = math.inf
    rgbd_safety_left_depth: float = math.inf
    rgbd_safety_right_depth: float = math.inf
    rgbd_safety_override: bool = False
    rgbd_safety_scale: float = 1.0


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    return data


CFG = load_yaml(args_cli.config)
if args_cli.num_envs is not None:
    CFG["scene"]["num_envs"] = int(args_cli.num_envs)

# Patch the config for this specific smoke test.
CFG.setdefault("grid_astar", {})["safety_margin"] = float(args_cli.safety_margin)
CFG["grid_astar"]["allow_diagonal"] = not bool(args_cli.disable_diagonal)
CFG["grid_astar"]["max_expansions"] = int(args_cli.max_expansions)
occ_cfg_map = CFG.setdefault("lidar", {}).setdefault("occupancy_grid", {})
occ_cfg_map["resolution"] = float(args_cli.resolution)
occ_cfg_map["x_min"] = float(args_cli.x_min)
occ_cfg_map["x_max"] = float(args_cli.x_max)
occ_cfg_map["y_min"] = float(args_cli.y_min)
occ_cfg_map["y_max"] = float(args_cli.y_max)

# Avoid horizontal RayCaster rays exactly grazing short obstacle top faces.
# The YAML default is z=0.15, i.e. world z≈0.70 with the default proxy root,
# which is exactly the top of several 0.70 m-tall smoke-test boxes/walls.
CFG.setdefault("lidar", {}).setdefault("offset", {}).setdefault("pos", [0.0, 0.0, 0.15])
CFG["lidar"]["offset"]["pos"][2] = float(args_cli.lidar_z_offset)


def make_obstacles_lidar_visible(obstacles: list[dict[str, Any]], *, min_height: float) -> None:
    """Make 2D smoke-test obstacles vertically visible to the horizontal RayCaster.

    smoke_test_nav.py has no z dimension: a rectangle always exists in the 2D
    grid. In Isaac/RayCaster tests, the same XY rectangle is only observed if
    the horizontal scan plane actually intersects the USD mesh. Several older
    scenarios used height=0.70 with top z=0.70, which can be grazed/missed by a
    scan near z=0.65-0.70. This helper keeps the XY footprint identical and only
    lifts the vertical extent for sensor robustness.
    """
    if min_height <= 0.0:
        return
    min_height = float(min_height)
    for obs in obstacles:
        shape = str(obs.get("shape", "box")).lower()
        if shape in ("box", "rect", "rectangle", "cuboid"):
            size = list(obs.get("size", [0.8, 0.8, 0.8]))
            while len(size) < 3:
                size.append(0.8)
            if float(size[2]) < min_height:
                size[2] = min_height
                obs["size"] = size
                pos = list(obs.get("pos", [0.0, 0.0, 0.5 * min_height]))
                while len(pos) < 3:
                    pos.append(0.0)
                pos[2] = 0.5 * min_height
                obs["pos"] = pos
        elif shape in ("cylinder", "round", "circle", "oval"):
            if float(obs.get("height", 0.8)) < min_height:
                obs["height"] = min_height
                pos = list(obs.get("pos", [0.0, 0.0, 0.5 * min_height]))
                while len(pos) < 3:
                    pos.append(0.0)
                pos[2] = 0.5 * min_height
                obs["pos"] = pos


def expand_map_blocking_obstacles_y(obstacles: list[dict[str, Any]], *, y_min: float, y_max: float, padding: float) -> None:
    """Expand scenario walls that are intended to block the whole planning map.

    A physical wall with size_y=5.0 blocks the direct path, but it does not block
    an 8 m-wide planning grid: A* can legally route around the top/bottom ends.
    smoke_test_nav.py-style blocked-map scenarios need the obstacle footprint to
    touch or exceed the configured map boundary, otherwise the planner is correct
    to find a side route.
    """
    map_center_y = 0.5 * (float(y_min) + float(y_max))
    map_width_y = float(y_max) - float(y_min)
    requested_size_y = map_width_y + 2.0 * max(0.0, float(padding))
    for obs in obstacles:
        if not bool(obs.get("blocks_map_y", False)):
            continue
        shape = str(obs.get("shape", "box")).lower()
        if shape not in ("box", "rect", "rectangle", "cuboid"):
            continue
        size = list(obs.get("size", [0.8, 0.8, 0.8]))
        while len(size) < 3:
            size.append(0.8)
        if float(size[1]) < requested_size_y:
            size[1] = requested_size_y
            obs["size"] = size
        pos = list(obs.get("pos", [0.0, map_center_y, 0.0]))
        while len(pos) < 3:
            pos.append(0.0)
        pos[1] = map_center_y
        obs["pos"] = pos


SCENARIO = SCENARIOS[args_cli.scenario]
OBSTACLES = [dict(obs) for obs in SCENARIO.get("obstacles", [])]
expand_map_blocking_obstacles_y(
    OBSTACLES,
    y_min=float(args_cli.y_min),
    y_max=float(args_cli.y_max),
    padding=2.0 * float(args_cli.resolution),
)
make_obstacles_lidar_visible(OBSTACLES, min_height=float(args_cli.force_lidar_visible_obstacle_height))
MAX_OBSTACLES = 6
if len(OBSTACLES) > MAX_OBSTACLES:
    raise ValueError(f"Scenario {args_cli.scenario!r} has {len(OBSTACLES)} obstacles; max supported is {MAX_OBSTACLES}.")

CFG.setdefault("target", {})["pos"] = list(SCENARIO.get("target_pos", CFG.get("target", {}).get("pos", [4.0, 0.0, 0.08])))
PERSON_X = float(args_cli.person_x) if args_cli.person_x is not None else float(CFG["target"]["pos"][0])
PERSON_Y = float(args_cli.person_y) if args_cli.person_y is not None else float(CFG["target"]["pos"][1])
# The visible red target marker starts at the person root; the actual navigation
# target is replaced after RGB-D/YOLO extraction, usually with stand-off applied.
CFG["target"]["pos"] = [PERSON_X, PERSON_Y, float(CFG["target"].get("pos", [0.0, 0.0, 0.08])[2])]
for obs_idx, obs in enumerate(OBSTACLES):
    obs.setdefault("name", f"Obstacle{obs_idx}")
    # Keep prim names simple and deterministic.  Scenario labels are currently
    # already safe, but this protects future multi-object scenarios from USD
    # path surprises caused by spaces or punctuation.
    safe_name = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(obs["name"]))
    if not safe_name:
        safe_name = f"Obstacle{obs_idx}"
    obs["name"] = safe_name
    obs["prim_path"] = f"{{ENV_REGEX_NS}}/{obs['name']}"
    obs["mesh_prim_path"] = f"{{ENV_REGEX_NS}}/{obs['name']}/geometry/mesh"

RAYCAST_AGGREGATE_PRIM_PATH = "{ENV_REGEX_NS}/_RaycastAggregate"
RAYCAST_AGGREGATE_MESH_PRIM_PATH = f"{RAYCAST_AGGREGATE_PRIM_PATH}/geometry/mesh"
RAYCAST_MESH_MODE = str(args_cli.raycast_mesh_mode)
USE_RAYCAST_AGGREGATE_MESH = RAYCAST_MESH_MODE == "aggregate" or (RAYCAST_MESH_MODE == "auto" and len(OBSTACLES) > 1)

CFG.setdefault("lidar", {})["mesh_prim_paths"] = (
    [RAYCAST_AGGREGATE_MESH_PRIM_PATH] if USE_RAYCAST_AGGREGATE_MESH else [obs["mesh_prim_path"] for obs in OBSTACLES]
)
if not CFG["lidar"]["mesh_prim_paths"]:
    raise ValueError("RayCaster smoke tests require at least one static mesh obstacle/marker.")


def resolve_env_prim_paths(template_path: str, num_envs: int) -> list[str]:
    return [template_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}") for env_id in range(num_envs)]


def resolve_template_path(template_path: str, env_id: int = 0) -> str:
    return template_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")


def resolve_mesh_prim_paths(mesh_paths: list[str], num_envs: int) -> list[str]:
    resolved: list[str] = []
    for path in mesh_paths:
        if "{ENV_REGEX_NS}" in path:
            if num_envs != 1:
                print("[WARN]: this smoke test is intended for num_envs=1.", flush=True)
            resolved.append(resolve_template_path(path, env_id=0))
        else:
            resolved.append(path)
    return resolved


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def obstacle_asset_cfg(obs: dict[str, Any]) -> AssetBaseCfg:
    shape = str(obs.get("shape", "box")).lower()
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(obs.get("color", [0.2, 0.8, 0.2])))
    common_kwargs = dict(
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=material,
    )
    if shape in ("box", "rect", "rectangle", "cuboid"):
        spawn_cfg = sim_utils.MeshCuboidCfg(size=tuple(obs.get("size", [0.8, 0.8, 0.8])), **common_kwargs)
    elif shape in ("cylinder", "round", "circle", "oval"):
        if hasattr(sim_utils, "MeshCylinderCfg"):
            spawn_cfg = sim_utils.MeshCylinderCfg(
                radius=float(obs.get("radius", 0.45)),
                height=float(obs.get("height", 0.8)),
                **common_kwargs,
            )
        else:
            diameter = 2.0 * float(obs.get("radius", 0.45))
            spawn_cfg = sim_utils.MeshCuboidCfg(
                size=(diameter, diameter, float(obs.get("height", 0.8))),
                **common_kwargs,
            )
    else:
        raise ValueError(f"Unsupported obstacle shape {shape!r} for obstacle {obs.get('name')!r}")

    return AssetBaseCfg(
        prim_path=obs["prim_path"],
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(obs.get("pos", [2.0, 0.0, 0.4])),
            rot=yaw_to_quat_wxyz(float(obs.get("yaw", 0.0))),
        ),
        spawn=spawn_cfg,
    )




def inactive_obstacle_asset_cfg(index: int) -> AssetBaseCfg:
    """Off-map placeholder obstacle used to keep InteractiveScene schema fixed.

    Isaac Lab's configclass/InteractiveScene machinery is more robust when the
    scene schema is stable across scenario choices.  The real RayCaster target
    list still includes only actual scenario obstacle meshes.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/_InactiveObstacle{index}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1000.0 + float(index), 1000.0, 0.5)),
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.02, 0.02)),
        ),
    )

def compute_body_radius() -> float:
    body_size = CFG["proxy"]["body"]["size"]
    return 0.5 * math.hypot(float(body_size[0]), float(body_size[1]))


def compute_astar_cfg() -> tuple[GridAStarCfg, float, float, float]:
    base_cfg = GridAStarCfg.from_mapping(CFG.get("grid_astar", {}))
    body_radius = compute_body_radius()
    safety_margin = float(CFG.get("grid_astar", {}).get("safety_margin", args_cli.safety_margin))
    inflation_radius = body_radius + safety_margin
    # Use body radius + margin exactly, matching smoke_test_nav.py semantics.
    astar_cfg = GridAStarCfg(
        occupancy_threshold=base_cfg.occupancy_threshold,
        inflation_radius=inflation_radius,
        allow_diagonal=not bool(args_cli.disable_diagonal),
        heuristic_weight=base_cfg.heuristic_weight,
        max_expansions=int(args_cli.max_expansions),
    )
    return astar_cfg, body_radius, safety_margin, inflation_radius


def ensure_proxy_roots_exist(num_envs: int) -> None:
    stage = omni.usd.get_context().get_stage()
    for path in resolve_env_prim_paths(CFG["proxy"]["prim_path"], num_envs):
        proxy_xform = UsdGeom.Xform.Define(stage, path)
        ensure_canonical_xform_ops(proxy_xform.GetPrim())
        print(f"[INFO]: Ensured canonical proxy root exists: {path}", flush=True)


def _append_box_to_mesh_buffers(
    points: list[tuple[float, float, float]],
    face_counts: list[int],
    face_indices: list[int],
    *,
    center: Sequence[float],
    size: Sequence[float],
    yaw: float,
) -> None:
    """Append one yawed cuboid to USD mesh buffers in world/env coordinates."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    local = [
        (-0.5 * sx, -0.5 * sy, -0.5 * sz),
        ( 0.5 * sx, -0.5 * sy, -0.5 * sz),
        ( 0.5 * sx,  0.5 * sy, -0.5 * sz),
        (-0.5 * sx,  0.5 * sy, -0.5 * sz),
        (-0.5 * sx, -0.5 * sy,  0.5 * sz),
        ( 0.5 * sx, -0.5 * sy,  0.5 * sz),
        ( 0.5 * sx,  0.5 * sy,  0.5 * sz),
        (-0.5 * sx,  0.5 * sy,  0.5 * sz),
    ]
    base = len(points)
    for lx, ly, lz in local:
        x = cx + c * lx - s * ly
        y = cy + s * lx + c * ly
        z = cz + lz
        points.append((x, y, z))
    # Quad faces.  RayCaster only needs surface triangles/quads; winding is not
    # relied on, and the mesh is marked double-sided below.
    faces = [
        (0, 1, 2, 3),  # bottom
        (4, 7, 6, 5),  # top
        (0, 4, 5, 1),  # -y side
        (1, 5, 6, 2),  # +x side
        (2, 6, 7, 3),  # +y side
        (3, 7, 4, 0),  # -x side
    ]
    for face in faces:
        face_counts.append(4)
        face_indices.extend([base + idx for idx in face])


def _append_cylinder_to_mesh_buffers(
    points: list[tuple[float, float, float]],
    face_counts: list[int],
    face_indices: list[int],
    *,
    center: Sequence[float],
    radius: float,
    height: float,
    yaw: float,
    segments: int = 32,
) -> None:
    """Append a simple vertical cylinder prism to USD mesh buffers."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    radius = float(radius)
    height = float(height)
    c_yaw, s_yaw = math.cos(float(yaw)), math.sin(float(yaw))
    segments = max(8, int(segments))
    base = len(points)
    z0, z1 = cz - 0.5 * height, cz + 0.5 * height
    for z in (z0, z1):
        for i in range(segments):
            a = 2.0 * math.pi * float(i) / float(segments)
            lx, ly = radius * math.cos(a), radius * math.sin(a)
            x = cx + c_yaw * lx - s_yaw * ly
            y = cy + s_yaw * lx + c_yaw * ly
            points.append((x, y, z))
    # bottom and top ngon faces
    face_counts.append(segments)
    face_indices.extend([base + i for i in range(segments)])
    face_counts.append(segments)
    face_indices.extend([base + segments + i for i in reversed(range(segments))])
    # side quads
    for i in range(segments):
        j = (i + 1) % segments
        face_counts.append(4)
        face_indices.extend([base + i, base + j, base + segments + j, base + segments + i])


def create_raycast_aggregate_mesh(num_envs: int) -> None:
    """Create one raycast-only USD mesh containing all scenario obstacle surfaces.

    Isaac/RayCaster has shown reset-time instability when targeting multiple
    mesh prims in this smoke script.  The physical/visual obstacles remain as
    normal scene assets; this aggregate is only a sensor target and has no
    collision body.  It keeps multi-obstacle RayCaster input equivalent but
    reduces mesh_prim_paths to one mesh.
    """
    if not USE_RAYCAST_AGGREGATE_MESH:
        return
    if int(num_envs) != 1:
        print("[WARN]: aggregate raycast mesh currently assumes num_envs=1.", flush=True)

    stage = omni.usd.get_context().get_stage()
    for env_id in range(int(num_envs)):
        root_path = resolve_template_path(RAYCAST_AGGREGATE_PRIM_PATH, env_id=env_id)
        mesh_path = resolve_template_path(RAYCAST_AGGREGATE_MESH_PRIM_PATH, env_id=env_id)
        UsdGeom.Xform.Define(stage, root_path)
        UsdGeom.Xform.Define(stage, f"{root_path}/geometry")
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        points: list[tuple[float, float, float]] = []
        face_counts: list[int] = []
        face_indices: list[int] = []
        for obs in OBSTACLES:
            shape = str(obs.get("shape", "box")).lower()
            if shape in ("box", "rect", "rectangle", "cuboid"):
                _append_box_to_mesh_buffers(
                    points,
                    face_counts,
                    face_indices,
                    center=obs.get("pos", [0.0, 0.0, 0.5]),
                    size=obs.get("size", [0.8, 0.8, 0.8]),
                    yaw=float(obs.get("yaw", 0.0)),
                )
            elif shape in ("cylinder", "round", "circle", "oval"):
                _append_cylinder_to_mesh_buffers(
                    points,
                    face_counts,
                    face_indices,
                    center=obs.get("pos", [0.0, 0.0, 0.5]),
                    radius=float(obs.get("radius", 0.45)),
                    height=float(obs.get("height", 0.8)),
                    yaw=float(obs.get("yaw", 0.0)),
                )
            else:
                raise ValueError(f"Unsupported obstacle shape {shape!r} for aggregate raycast mesh")

        if not points:
            raise ValueError("Cannot create empty aggregate raycast mesh")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateDoubleSidedAttr(True)
        try:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            zs = [p[2] for p in points]
            mesh.CreateExtentAttr([(min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))])
        except Exception:
            pass
        print(
            f"[INFO]: Created aggregate raycast mesh: {mesh_path} "
            f"obstacles={len(OBSTACLES)} vertices={len(points)} faces={len(face_counts)}",
            flush=True,
        )


def _world_bbox_string(prim: Any) -> str:
    if not prim.IsValid():
        return "bbox=INVALID"
    try:
        purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = aligned.GetMin()
        mx = aligned.GetMax()
        if not all(math.isfinite(float(v)) for v in (mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])):
            return "bbox=nonfinite"
        return (
            f"bbox_min=({float(mn[0]):.3f},{float(mn[1]):.3f},{float(mn[2]):.3f}) "
            f"bbox_max=({float(mx[0]):.3f},{float(mx[1]):.3f},{float(mx[2]):.3f})"
        )
    except Exception as exc:  # pragma: no cover - Isaac/PXR runtime diagnostic only.
        return f"bbox_error={type(exc).__name__}:{exc}"


def print_stage_prim_diagnostics(num_envs: int, *, label: str) -> None:
    stage = omni.usd.get_context().get_stage()
    paths = [
        resolve_template_path(CFG["lidar"]["prim_path"], env_id=0),
        resolve_template_path(CFG["target"]["prim_path"], env_id=0),
    ]
    if USE_RAYCAST_AGGREGATE_MESH:
        paths.append(resolve_template_path(RAYCAST_AGGREGATE_PRIM_PATH, env_id=0))
        paths.append(resolve_template_path(RAYCAST_AGGREGATE_MESH_PRIM_PATH, env_id=0))
    for obs in OBSTACLES:
        paths.append(resolve_template_path(obs["prim_path"], env_id=0))
        paths.append(resolve_template_path(obs["mesh_prim_path"], env_id=0))
    print(f"[INFO]: Stage prim diagnostics {label}:", flush=True)
    for path in dict.fromkeys(paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"  - {path}: INVALID", flush=True)
            continue
        is_mesh = bool(UsdGeom.Mesh(prim))
        is_xform = bool(UsdGeom.Xformable(prim))
        print(
            f"  - {path}: type={prim.GetTypeName()} is_mesh={is_mesh} "
            f"is_xformable={is_xform} {_world_bbox_string(prim)}",
            flush=True,
        )


def _obstacle_footprint_points_xy(obs: dict[str, Any], spacing: float) -> torch.Tensor:
    shape = str(obs.get("shape", "box")).lower()
    spacing = max(0.02, float(spacing))
    pos = obs.get("pos", [0.0, 0.0, 0.0])
    pos_x, pos_y = float(pos[0]), float(pos[1])
    yaw = float(obs.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)

    if shape in ("box", "rect", "rectangle", "cuboid"):
        size = obs.get("size", [0.8, 0.8, 0.8])
        sx, sy = float(size[0]), float(size[1])
        xs = torch.arange(-0.5 * sx, 0.5 * sx + spacing, spacing, dtype=torch.float32)
        ys = torch.arange(-0.5 * sy, 0.5 * sy + spacing, spacing, dtype=torch.float32)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        local = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
    elif shape in ("cylinder", "round", "circle", "oval"):
        radius = float(obs.get("radius", 0.45))
        xs = torch.arange(-radius, radius + spacing, spacing, dtype=torch.float32)
        ys = torch.arange(-radius, radius + spacing, spacing, dtype=torch.float32)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        keep = (gx * gx + gy * gy) <= radius * radius
        local = torch.stack((gx[keep], gy[keep]), dim=-1)
    else:
        raise ValueError(f"Unsupported obstacle shape {shape!r} for footprint rasterization.")

    rot_x = c * local[:, 0] - s * local[:, 1] + pos_x
    rot_y = s * local[:, 0] + c * local[:, 1] + pos_y
    return torch.stack((rot_x, rot_y), dim=-1)


def rasterize_scenario_obstacle_footprints(
    obstacles: list[dict[str, Any]],
    *,
    proxy: G1CommandProxy,
    scene: InteractiveScene,
    grid_cfg: OccupancyGridCfg,
    device: torch.device,
) -> torch.Tensor:
    num_envs = proxy.num_envs
    if not obstacles:
        return torch.zeros((num_envs, grid_cfg.height, grid_cfg.width), dtype=torch.float32, device=device)

    spacing = max(float(grid_cfg.resolution), 0.05)
    env_local_points = [_obstacle_footprint_points_xy(obs, spacing=spacing) for obs in obstacles]
    points_env_xy = torch.cat(env_local_points, dim=0).to(device=device)
    points_w_xy = scene.env_origins[:, None, 0:2].to(device=device) + points_env_xy[None, :, :]
    delta_xy_w = points_w_xy - proxy.pos_w[:, None, 0:2]
    points_b = rotate_world_to_body_2d(delta_xy_w, proxy.yaw[:, None])
    valid = torch.ones(points_b.shape[:-1], dtype=torch.bool, device=device)
    return rasterize_points_to_occupancy_grid(points_b, valid, grid_cfg)


def expand_lidar_points_to_conservative_obstacle_points(
    points_xy_b: torch.Tensor,
    valid: torch.Tensor,
    *,
    thickness: float,
    lateral_padding: float,
    resolution: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert first-return LiDAR points into conservative obstacle slabs.

    A 2D LiDAR return is only the first visible surface.  For navigation we do
    not know how deep the obstacle extends behind that surface, so this expands
    each hit away from the sensor by ``thickness`` and sideways by
    ``lateral_padding``.  A* then inflates this slab further by body radius +
    safety margin.
    """
    if not torch.is_tensor(points_xy_b):
        raise TypeError("points_xy_b must be a tensor")
    if points_xy_b.ndim != 3 or points_xy_b.shape[-1] != 2:
        raise ValueError(f"points_xy_b must have shape [N,B,2], got {tuple(points_xy_b.shape)}")
    if valid.shape != points_xy_b.shape[:-1]:
        raise ValueError(f"valid must have shape {tuple(points_xy_b.shape[:-1])}, got {tuple(valid.shape)}")

    thickness = max(0.0, float(thickness))
    lateral_padding = max(0.0, float(lateral_padding))
    step = max(1.0e-3, float(resolution))
    valid = valid.to(device=points_xy_b.device, dtype=torch.bool) & torch.isfinite(points_xy_b).all(dim=-1)

    if thickness <= 1.0e-9 and lateral_padding <= 1.0e-9:
        return points_xy_b, valid

    depth_offsets = torch.arange(0.0, thickness + 0.5 * step, step, dtype=torch.float32, device=points_xy_b.device)
    lateral_offsets = torch.arange(-lateral_padding, lateral_padding + 0.5 * step, step, dtype=torch.float32, device=points_xy_b.device)
    if depth_offsets.numel() <= 0:
        depth_offsets = torch.zeros((1,), dtype=torch.float32, device=points_xy_b.device)
    if lateral_offsets.numel() <= 0:
        lateral_offsets = torch.zeros((1,), dtype=torch.float32, device=points_xy_b.device)

    expanded_per_env: list[torch.Tensor] = []
    max_count = 0
    for env_id in range(points_xy_b.shape[0]):
        pts = points_xy_b[env_id][valid[env_id]]
        if pts.numel() == 0:
            env_pts = torch.empty((0, 2), dtype=torch.float32, device=points_xy_b.device)
        else:
            norms = torch.linalg.norm(pts, dim=-1, keepdim=True).clamp_min(1.0e-6)
            radial = pts / norms
            tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=-1)
            env_pts = (
                pts[:, None, None, :]
                + radial[:, None, None, :] * depth_offsets[None, :, None, None]
                + tangent[:, None, None, :] * lateral_offsets[None, None, :, None]
            ).reshape(-1, 2)
        expanded_per_env.append(env_pts)
        max_count = max(max_count, int(env_pts.shape[0]))

    if max_count == 0:
        empty = torch.empty((points_xy_b.shape[0], 0, 2), dtype=torch.float32, device=points_xy_b.device)
        empty_valid = torch.empty((points_xy_b.shape[0], 0), dtype=torch.bool, device=points_xy_b.device)
        return empty, empty_valid

    expanded = torch.zeros((points_xy_b.shape[0], max_count, 2), dtype=torch.float32, device=points_xy_b.device)
    expanded_valid = torch.zeros((points_xy_b.shape[0], max_count), dtype=torch.bool, device=points_xy_b.device)
    for env_id, env_pts in enumerate(expanded_per_env):
        n = int(env_pts.shape[0])
        if n > 0:
            expanded[env_id, :n] = env_pts
            expanded_valid[env_id, :n] = True
    return expanded.contiguous(), expanded_valid.contiguous()


def body_points_to_world_xy(points_b: torch.Tensor, valid: torch.Tensor, proxy: G1CommandProxy) -> tuple[torch.Tensor, torch.Tensor]:
    if points_b.numel() == 0:
        return points_b.clone(), valid.clone()
    points_w = proxy.pos_w[:, None, 0:2] + _rotate_body_to_world_xy(points_b, proxy.yaw[:, None])
    valid_w = valid.to(device=points_b.device, dtype=torch.bool) & torch.isfinite(points_w).all(dim=-1)
    return points_w.contiguous(), valid_w.contiguous()


def append_lidar_memory_points(
    memory_points_w: torch.Tensor | None,
    points_b: torch.Tensor,
    valid: torch.Tensor,
    proxy: G1CommandProxy,
    *,
    max_points: int,
) -> torch.Tensor | None:
    points_w, valid_w = body_points_to_world_xy(points_b, valid, proxy)
    if points_w.numel() == 0 or valid_w.sum().item() <= 0:
        return memory_points_w

    max_points = max(1, int(max_points))
    per_env: list[torch.Tensor] = []
    max_count = 0
    for env_id in range(points_w.shape[0]):
        new_pts = points_w[env_id][valid_w[env_id]]
        if memory_points_w is not None and memory_points_w.numel() > 0:
            old_pts = memory_points_w[env_id]
            old_pts = old_pts[torch.isfinite(old_pts).all(dim=-1)]
            pts = torch.cat((old_pts, new_pts), dim=0)
        else:
            pts = new_pts
        if pts.shape[0] > max_points:
            pts = pts[-max_points:]
        per_env.append(pts)
        max_count = max(max_count, int(pts.shape[0]))

    if max_count == 0:
        return memory_points_w

    out = torch.full((points_w.shape[0], max_count, 2), float("nan"), dtype=torch.float32, device=points_w.device)
    for env_id, pts in enumerate(per_env):
        n = int(pts.shape[0])
        if n > 0:
            out[env_id, :n] = pts
    return out.contiguous()


def rasterize_lidar_memory_points(
    memory_points_w: torch.Tensor | None,
    *,
    proxy: G1CommandProxy,
    grid_cfg: OccupancyGridCfg,
) -> torch.Tensor:
    if memory_points_w is None or memory_points_w.numel() == 0:
        return torch.zeros((proxy.num_envs, grid_cfg.height, grid_cfg.width), dtype=torch.float32, device=proxy.pos_w.device)
    delta_xy_w = memory_points_w - proxy.pos_w[:, None, 0:2]
    points_b = rotate_world_to_body_2d(delta_xy_w, proxy.yaw[:, None])
    valid = torch.isfinite(points_b).all(dim=-1)
    return rasterize_points_to_occupancy_grid(points_b, valid, grid_cfg)


def build_planning_grid(
    *,
    scene: InteractiveScene,
    proxy: G1CommandProxy,
    lidar_obs_cfg: LidarObservationCfg,
    grid_cfg: OccupancyGridCfg,
    device: torch.device,
    lidar_memory_points_w: torch.Tensor | None = None,
    update_lidar_memory: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    lidar_obs = build_lidar_proxy_observation(scene, proxy, lidar_name="lidar", cfg=lidar_obs_cfg)
    raw_lidar_grid = rasterize_points_to_occupancy_grid(lidar_obs["lidar_points_b"], lidar_obs["lidar_valid"], grid_cfg)
    conservative_points_b, conservative_valid = expand_lidar_points_to_conservative_obstacle_points(
        lidar_obs["lidar_points_b"],
        lidar_obs["lidar_valid"],
        thickness=float(args_cli.lidar_obstacle_thickness),
        lateral_padding=float(args_cli.lidar_obstacle_lateral_padding),
        resolution=float(grid_cfg.resolution),
    )
    instant_lidar_grid = rasterize_points_to_occupancy_grid(conservative_points_b, conservative_valid, grid_cfg)

    new_memory_points_w = lidar_memory_points_w
    if str(args_cli.lidar_map_mode) == "memory" and update_lidar_memory:
        new_memory_points_w = append_lidar_memory_points(
            lidar_memory_points_w,
            conservative_points_b,
            conservative_valid,
            proxy,
            max_points=int(args_cli.max_lidar_memory_points),
        )

    if str(args_cli.lidar_map_mode) == "memory":
        memory_grid = rasterize_lidar_memory_points(new_memory_points_w, proxy=proxy, grid_cfg=grid_cfg)
        lidar_grid = torch.maximum(instant_lidar_grid, memory_grid)
    else:
        lidar_grid = instant_lidar_grid

    # Always build the known scenario-footprint grid for diagnostics and image
    # overlays. In --lidar_only mode it is NOT used for planning.
    footprint_grid = rasterize_scenario_obstacle_footprints(
        OBSTACLES,
        proxy=proxy,
        scene=scene,
        grid_cfg=grid_cfg,
        device=device,
    )
    if GRID_SOURCE == "lidar":
        grid = lidar_grid
    elif GRID_SOURCE == "scenario":
        grid = footprint_grid
    elif GRID_SOURCE == "hybrid":
        grid = torch.maximum(lidar_grid, footprint_grid)
    else:  # Defensive guard against future argparse changes.
        raise ValueError(f"Unsupported grid source: {GRID_SOURCE!r}")
    return grid, lidar_grid, footprint_grid, lidar_obs, raw_lidar_grid, instant_lidar_grid, new_memory_points_w


def print_lidar_raw_diagnostics(lidar_obs: dict[str, torch.Tensor], *, label: str) -> None:
    hits = lidar_obs["lidar_ray_hits_w"]
    sensor_pos = lidar_obs["lidar_sensor_pos_w"]
    valid = lidar_obs["lidar_valid"]
    finite = torch.isfinite(hits).all(dim=-1)
    finite_count = int(finite[0].detach().cpu().sum().item()) if finite.numel() else 0
    valid_count = int(valid[0].detach().cpu().sum().item()) if valid.numel() else 0
    spos = sensor_pos[0].detach().cpu().tolist() if sensor_pos.numel() else [math.nan, math.nan, math.nan]
    print(
        f"[INFO]: LiDAR raw diagnostics {label}: sensor_pos_w=({spos[0]:.3f},{spos[1]:.3f},{spos[2]:.3f}), "
        f"finite_hits={finite_count}, valid_hits={valid_count}",
        flush=True,
    )
    if finite_count > 0:
        pts = hits[0][finite[0]].detach().cpu()
        mn = pts.min(dim=0).values.tolist()
        mx = pts.max(dim=0).values.tolist()
        print(
            f"[INFO]: LiDAR hit bbox {label}: "
            f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
            f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Geometry and direct waypoint control copied from smoke_test_nav.py semantics
# -----------------------------------------------------------------------------
def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def _rotate_body_to_world_xy(points_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    x = points_b[..., 0]
    y = points_b[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def path_body_to_world(path_xy_b: Sequence[PointXY], proxy: G1CommandProxy) -> list[PointXY]:
    if not path_xy_b:
        return []
    device = proxy.pos_w.device
    pts_b = torch.tensor(path_xy_b, dtype=torch.float32, device=device)
    pts_w = proxy.pos_w[0, 0:2] + _rotate_body_to_world_xy(pts_b, proxy.yaw[0])
    return [(float(x), float(y)) for x, y in pts_w.detach().cpu().tolist()]


def path_world_to_body(path_xy_w: Sequence[PointXY], proxy: G1CommandProxy) -> list[PointXY]:
    if not path_xy_w:
        return []
    device = proxy.pos_w.device
    pts_w = torch.tensor(path_xy_w, dtype=torch.float32, device=device)
    delta = pts_w[None, :, :] - proxy.pos_w[0:1, None, 0:2]
    pts_b = rotate_world_to_body_2d(delta, proxy.yaw[0:1, None])[0]
    return [(float(x), float(y)) for x, y in pts_b.detach().cpu().tolist()]


def select_waypoint(
    path_xy: Sequence[PointXY],
    current_xy: PointXY,
    waypoint_index: int,
    waypoint_reached_radius: float,
) -> int:
    """Advance monotonically through the existing path; never shortcut."""
    i = waypoint_index
    while i < len(path_xy) - 1:
        wx, wy = path_xy[i]
        if math.hypot(wx - current_xy[0], wy - current_xy[1]) > waypoint_reached_radius:
            break
        i += 1
    return i


def path_tangent_yaw(path_xy: Sequence[PointXY], waypoint_index: int, fallback_yaw: float) -> float:
    if waypoint_index <= 0 or waypoint_index >= len(path_xy):
        return fallback_yaw
    prev_xy = path_xy[max(0, waypoint_index - 1)]
    next_xy = path_xy[waypoint_index]
    dx = next_xy[0] - prev_xy[0]
    dy = next_xy[1] - prev_xy[1]
    if abs(dx) + abs(dy) < 1e-9:
        return fallback_yaw
    return math.atan2(dy, dx)


def command_to_waypoint(
    pose: Pose2D,
    waypoint_xy: PointXY,
    path_yaw: float,
    *,
    k_xy: float,
    k_yaw: float,
    max_vx: float,
    max_vy: float,
    max_wz: float,
    allow_backward: bool,
) -> tuple[float, float, float, PointXY]:
    x, y, yaw = pose
    dx_w = waypoint_xy[0] - x
    dy_w = waypoint_xy[1] - y

    c = math.cos(yaw)
    s = math.sin(yaw)
    x_b = c * dx_w + s * dy_w
    y_b = -s * dx_w + c * dy_w

    min_vx = -max_vx if allow_backward else 0.0
    vx = clip(k_xy * x_b, min_vx, max_vx)
    vy = clip(k_xy * y_b, -max_vy, max_vy)

    yaw_error = angle_wrap(path_yaw - yaw)
    wz = clip(k_yaw * yaw_error, -max_wz, max_wz)

    return vx, vy, wz, (x_b, y_b)


def path_length_xy(path: Sequence[PointXY]) -> float:
    if len(path) <= 1:
        return 0.0
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]))


def iter_sampled_path_points(
    path_xy: Sequence[PointXY],
    *,
    spacing: float,
    ignore_distance: float,
    max_distance: float,
) -> list[PointXY]:
    """Sample a polyline after an initial ignore band and up to a lookahead distance."""
    if len(path_xy) < 2:
        return []
    spacing = max(1.0e-3, float(spacing))
    ignore_distance = max(0.0, float(ignore_distance))
    max_distance = max(0.0, float(max_distance))
    out: list[PointXY] = []
    accumulated = 0.0
    for (x0, y0), (x1, y1) in zip(path_xy[:-1], path_xy[1:]):
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len <= 1.0e-9:
            continue
        n = max(1, int(math.ceil(seg_len / spacing)))
        for i in range(1, n + 1):
            t = float(i) / float(n)
            d = accumulated + t * seg_len
            if d < ignore_distance:
                continue
            if max_distance > 0.0 and d > max_distance:
                return out
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        accumulated += seg_len
        if max_distance > 0.0 and accumulated > max_distance:
            break
    return out


def path_conflicts_with_inflated_grid(
    path_xy_b: Sequence[PointXY],
    occupancy_grid: torch.Tensor,
    *,
    grid_cfg: OccupancyGridCfg,
    astar_cfg: GridAStarCfg,
    ignore_distance: float,
    lookahead_distance: float,
) -> tuple[bool, PointXY | None]:
    """Check whether the remaining path intersects current inflated occupancy."""
    inflated = inflate_occupancy_grid(
        occupancy_grid,
        cfg=grid_cfg,
        inflation_radius=float(astar_cfg.inflation_radius),
        occupancy_threshold=float(astar_cfg.occupancy_threshold),
    )
    samples = iter_sampled_path_points(
        path_xy_b,
        spacing=0.5 * float(grid_cfg.resolution),
        ignore_distance=float(ignore_distance),
        max_distance=float(lookahead_distance),
    )
    for xy in samples:
        row, col = xy_to_grid_cell(xy, grid_cfg)
        if 0 <= row < grid_cfg.height and 0 <= col < grid_cfg.width and bool(inflated[row, col]):
            return True, xy
    return False, None


# -----------------------------------------------------------------------------
# Clearance diagnostics
# -----------------------------------------------------------------------------
def signed_distance_to_rotated_box(x: float, y: float, obs: Mapping[str, Any]) -> float:
    pos = obs.get("pos", [0.0, 0.0, 0.0])
    size = obs.get("size", [0.8, 0.8, 0.8])
    cx, cy = float(pos[0]), float(pos[1])
    sx, sy = float(size[0]), float(size[1])
    yaw = float(obs.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)

    dx = x - cx
    dy = y - cy
    # World -> obstacle-local rotation by -yaw.
    lx = c * dx + s * dy
    ly = -s * dx + c * dy

    qx = abs(lx) - 0.5 * sx
    qy = abs(ly) - 0.5 * sy
    outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
    inside = min(max(qx, qy), 0.0)
    return outside + inside


def signed_distance_to_cylinder_xy(x: float, y: float, obs: Mapping[str, Any]) -> float:
    pos = obs.get("pos", [0.0, 0.0, 0.0])
    radius = float(obs.get("radius", 0.45))
    return math.hypot(x - float(pos[0]), y - float(pos[1])) - radius


def min_scenario_edge_clearance(x: float, y: float, obstacles: Sequence[Mapping[str, Any]], body_radius: float) -> float:
    if not obstacles:
        return math.inf
    distances: list[float] = []
    for obs in obstacles:
        shape = str(obs.get("shape", "box")).lower()
        if shape in ("box", "rect", "rectangle", "cuboid"):
            d = signed_distance_to_rotated_box(x, y, obs)
        elif shape in ("cylinder", "round", "circle", "oval"):
            d = signed_distance_to_cylinder_xy(x, y, obs)
        else:
            continue
        distances.append(d - float(body_radius))
    return min(distances) if distances else math.inf


def compute_obstacle_clearance_from_grid(
    occupancy_grid: torch.Tensor,
    grid_cfg: OccupancyGridCfg,
    *,
    body_radius: float,
    occupancy_threshold: float = 0.5,
) -> dict[str, Any]:
    occupied = occupancy_grid >= float(occupancy_threshold)
    cells = torch.nonzero(occupied, as_tuple=False)
    if cells.numel() == 0:
        return {
            "has_obstacle": False,
            "min_center_distance": math.inf,
            "edge_clearance": math.inf,
            "closest_xy_b": None,
        }

    rows = cells[:, 0].to(dtype=torch.float32)
    cols = cells[:, 1].to(dtype=torch.float32)
    x = float(grid_cfg.x_max) - (rows + 0.5) * float(grid_cfg.resolution)
    y = float(grid_cfg.y_min) + (cols + 0.5) * float(grid_cfg.resolution)
    dist = torch.sqrt(x * x + y * y)
    min_idx = int(torch.argmin(dist).detach().cpu().item())
    min_dist = float(dist[min_idx].detach().cpu().item())
    closest_xy = (float(x[min_idx].detach().cpu().item()), float(y[min_idx].detach().cpu().item()))
    return {
        "has_obstacle": True,
        "min_center_distance": min_dist,
        "edge_clearance": min_dist - float(body_radius),
        "closest_xy_b": closest_xy,
    }


# -----------------------------------------------------------------------------
# Debug images + CSV
# -----------------------------------------------------------------------------
def _xy_to_debug_pixel(xy_b: PointXY, grid_cfg: OccupancyGridCfg) -> tuple[int, int] | None:
    row, col = xy_to_grid_cell(xy_b, grid_cfg)
    if 0 <= row < grid_cfg.height and 0 <= col < grid_cfg.width:
        return int(col), int(row)
    return None


def _draw_polyline_xy(
    draw: Any,
    points_xy_b: Sequence[PointXY],
    grid_cfg: OccupancyGridCfg,
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    if len(points_xy_b) < 2:
        return
    prev = _xy_to_debug_pixel(points_xy_b[0], grid_cfg)
    for xy in points_xy_b[1:]:
        cur = _xy_to_debug_pixel(xy, grid_cfg)
        if prev is not None and cur is not None:
            draw.line([prev, cur], fill=fill, width=width)
        prev = cur


def _draw_marker(
    draw: Any,
    xy_b: PointXY,
    grid_cfg: OccupancyGridCfg,
    *,
    fill: tuple[int, int, int],
    radius: int = 4,
    outline: tuple[int, int, int] | None = None,
    cross: bool = False,
) -> None:
    pix = _xy_to_debug_pixel(xy_b, grid_cfg)
    if pix is None:
        return
    col, row = pix
    draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill=fill, outline=outline)
    if cross:
        draw.line((col - radius - 2, row, col + radius + 2, row), fill=outline or (0, 0, 0), width=1)
        draw.line((col, row - radius - 2, col, row + radius + 2), fill=outline or (0, 0, 0), width=1)


def quat_wxyz_to_yaw(quat_wxyz: Sequence[float]) -> float:
    if len(quat_wxyz) != 4:
        raise ValueError(f"quat_wxyz must have length 4, got {quat_wxyz!r}")
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def make_fixed_world_grid_cfg(local_grid_cfg: OccupancyGridCfg, anchor_xy_w: PointXY) -> OccupancyGridCfg:
    ax, ay = float(anchor_xy_w[0]), float(anchor_xy_w[1])
    return OccupancyGridCfg(
        x_min=ax + float(local_grid_cfg.x_min),
        x_max=ax + float(local_grid_cfg.x_max),
        y_min=ay + float(local_grid_cfg.y_min),
        y_max=ay + float(local_grid_cfg.y_max),
        resolution=float(local_grid_cfg.resolution),
        occupied_value=float(local_grid_cfg.occupied_value),
    )


def _xy_to_world_debug_pixel(xy_w: PointXY, grid_cfg: OccupancyGridCfg) -> tuple[int, int] | None:
    row, col = xy_to_grid_cell(xy_w, grid_cfg)
    if 0 <= row < grid_cfg.height and 0 <= col < grid_cfg.width:
        return int(col), int(row)
    return None


def _draw_polyline_world_xy(
    draw: Any,
    points_xy_w: Sequence[PointXY],
    grid_cfg: OccupancyGridCfg,
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    if len(points_xy_w) < 2:
        return
    prev = _xy_to_world_debug_pixel(points_xy_w[0], grid_cfg)
    for xy in points_xy_w[1:]:
        cur = _xy_to_world_debug_pixel(xy, grid_cfg)
        if prev is not None and cur is not None:
            draw.line([prev, cur], fill=fill, width=width)
        prev = cur


def _draw_marker_world(
    draw: Any,
    xy_w: PointXY,
    grid_cfg: OccupancyGridCfg,
    *,
    fill: tuple[int, int, int],
    radius: int = 4,
    outline: tuple[int, int, int] | None = None,
    cross: bool = False,
) -> None:
    pix = _xy_to_world_debug_pixel(xy_w, grid_cfg)
    if pix is None:
        return
    col, row = pix
    draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill=fill, outline=outline)
    if cross:
        draw.line((col - radius - 2, row, col + radius + 2, row), fill=outline or (0, 0, 0), width=1)
        draw.line((col, row - radius - 2, col, row + radius + 2), fill=outline or (0, 0, 0), width=1)


def _draw_dashed_polyline_world_xy(
    draw: Any,
    points_xy_w: Sequence[PointXY],
    grid_cfg: OccupancyGridCfg,
    *,
    fill: tuple[int, int, int],
    width: int = 1,
    dash_px: float = 8.0,
    gap_px: float = 5.0,
) -> None:
    """Draw a dashed world-frame polyline using pixel-space dash lengths."""
    if len(points_xy_w) < 2:
        return
    pixels = [_xy_to_world_debug_pixel(xy, grid_cfg) for xy in points_xy_w]
    for p0, p1 in zip(pixels[:-1], pixels[1:]):
        if p0 is None or p1 is None:
            continue
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len <= 1.0e-9:
            continue
        ux, uy = dx / seg_len, dy / seg_len
        cursor = 0.0
        while cursor < seg_len:
            dash_end = min(cursor + dash_px, seg_len)
            sx = x0 + ux * cursor
            sy = y0 + uy * cursor
            ex = x0 + ux * dash_end
            ey = y0 + uy * dash_end
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
            cursor += dash_px + gap_px


def obstacle_contour_world_xy(
    obs: Mapping[str, Any],
    *,
    env_origin_xy_w: PointXY,
    circle_segments: int = 64,
) -> list[PointXY]:
    """Return a closed 2D setup-derived obstacle contour in world coordinates."""
    shape = str(obs.get("shape", "box")).lower()
    pos = obs.get("pos", [0.0, 0.0, 0.0])
    cx = float(env_origin_xy_w[0]) + float(pos[0])
    cy = float(env_origin_xy_w[1]) + float(pos[1])
    yaw = float(obs.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)

    if shape in ("box", "rect", "rectangle", "cuboid"):
        size = obs.get("size", [0.8, 0.8, 0.8])
        hx, hy = 0.5 * float(size[0]), 0.5 * float(size[1])
        local = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        points: list[PointXY] = []
        for lx, ly in local:
            wx = cx + c * lx - s * ly
            wy = cy + s * lx + c * ly
            points.append((wx, wy))
        points.append(points[0])
        return points

    if shape in ("cylinder", "round", "circle", "oval"):
        radius = float(obs.get("radius", 0.45))
        n = max(12, int(circle_segments))
        points = []
        for i in range(n + 1):
            a = 2.0 * math.pi * float(i) / float(n)
            points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
        return points

    return []


def _draw_trajectory_safety_zone_world(
    base_rgb: Any,
    trajectory_xy_w: Sequence[PointXY],
    grid_cfg: OccupancyGridCfg,
    *,
    safety_zone_radius_m: float,
) -> Any:
    """Draw a translucent swept safety zone around the executed trajectory."""
    if safety_zone_radius_m <= 0.0 or not trajectory_xy_w:
        return base_rgb
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return base_rgb

    # Diameter in pixels. Clamp to at least 3 px so the safety trail is visible.
    radius_px = max(2, int(round(float(safety_zone_radius_m) / float(grid_cfg.resolution))))
    width_px = max(3, 2 * radius_px)
    overlay = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    pix = [_xy_to_world_debug_pixel(xy, grid_cfg) for xy in trajectory_xy_w]
    pix = [p for p in pix if p is not None]
    if len(pix) >= 2:
        draw_overlay.line(pix, fill=(185, 185, 185, 95), width=width_px, joint="curve")
    elif len(pix) == 1:
        col, row = pix[0]
        draw_overlay.ellipse((col - radius_px, row - radius_px, col + radius_px, row + radius_px), fill=(185, 185, 185, 95))
    return Image.alpha_composite(base_rgb.convert("RGBA"), overlay).convert("RGB")


def occupied_cells_world_from_local_grid(
    local_grid: torch.Tensor,
    *,
    proxy: G1CommandProxy,
    local_grid_cfg: OccupancyGridCfg,
    occupancy_threshold: float,
) -> list[PointXY]:
    occupied = local_grid[0] >= float(occupancy_threshold)
    cells = torch.nonzero(occupied, as_tuple=False)
    if cells.numel() == 0:
        return []
    rows = cells[:, 0].to(dtype=torch.float32)
    cols = cells[:, 1].to(dtype=torch.float32)
    x_b = float(local_grid_cfg.x_max) - (rows + 0.5) * float(local_grid_cfg.resolution)
    y_b = float(local_grid_cfg.y_min) + (cols + 0.5) * float(local_grid_cfg.resolution)
    pts_b = torch.stack((x_b, y_b), dim=-1).to(device=proxy.pos_w.device)
    pts_w = proxy.pos_w[0, 0:2] + _rotate_body_to_world_xy(pts_b, proxy.yaw[0])
    pts_w_cpu = pts_w.detach().cpu().tolist()
    return [(float(x), float(y)) for x, y in pts_w_cpu]


def lidar_scan_endpoints_world(
    lidar_obs: dict[str, torch.Tensor],
    *,
    proxy: G1CommandProxy,
    max_range: float,
) -> tuple[list[PointXY], list[bool], PointXY]:
    sensor_pos_xy = lidar_obs["lidar_sensor_pos_w"][0, 0:2].detach().cpu()
    sensor_xy_w = (float(sensor_pos_xy[0].item()), float(sensor_pos_xy[1].item()))
    ranges = lidar_obs["lidar_ranges"][0].detach().cpu()
    valid = lidar_obs["lidar_valid"][0].detach().cpu().to(dtype=torch.bool)
    num_rays = int(ranges.shape[0])
    if num_rays <= 0:
        return [], [], sensor_xy_w
    h0, h1 = [float(v) for v in CFG["lidar"]["pattern"]["horizontal_fov_range"]]
    if num_rays == 1:
        angles = torch.tensor([0.5 * (h0 + h1)], dtype=torch.float32)
    else:
        angles = torch.linspace(h0, h1, steps=num_rays, dtype=torch.float32)
    angles = torch.deg2rad(angles)
    yaw_sensor = float(proxy.yaw[0].detach().cpu().item()) + quat_wxyz_to_yaw(CFG["lidar"]["offset"]["rot_wxyz"])
    cy = math.cos(yaw_sensor)
    sy = math.sin(yaw_sensor)

    endpoints: list[PointXY] = []
    valid_list: list[bool] = []
    for i in range(num_rays):
        rng = float(ranges[i].item()) if bool(valid[i].item()) else float(max_range)
        a = float(angles[i].item())
        lx = rng * math.cos(a)
        ly = rng * math.sin(a)
        wx = sensor_xy_w[0] + cy * lx - sy * ly
        wy = sensor_xy_w[1] + sy * lx + cy * ly
        endpoints.append((wx, wy))
        valid_list.append(bool(valid[i].item()))
    return endpoints, valid_list, sensor_xy_w


def build_world_free_state_from_lidar(
    lidar_obs: dict[str, torch.Tensor],
    *,
    proxy: G1CommandProxy,
    world_grid_cfg: OccupancyGridCfg,
    max_range: float,
) -> torch.Tensor:
    """Ray-carve free cells from the current LiDAR scan in a fixed world frame.

    The returned tensor is ``[H, W]`` uint8 with 1 for observed free cells and
    0 elsewhere.  Occupancy is applied later and always overrides free.
    """
    free = torch.zeros((world_grid_cfg.height, world_grid_cfg.width), dtype=torch.uint8)
    endpoints, _valid_list, sensor_xy_w = lidar_scan_endpoints_world(
        lidar_obs,
        proxy=proxy,
        max_range=max_range,
    )
    step = max(0.5 * float(world_grid_cfg.resolution), 1.0e-3)
    for end_xy_w in endpoints:
        dx = float(end_xy_w[0] - sensor_xy_w[0])
        dy = float(end_xy_w[1] - sensor_xy_w[1])
        dist = math.hypot(dx, dy)
        if dist <= 1.0e-9:
            continue
        num_steps = max(1, int(math.ceil(dist / step)))
        prev_cell: tuple[int, int] | None = None
        # Use range(num_steps), not range(num_steps + 1), so hit cells are not
        # marked free. Invalid/no-hit rays terminate at max_range and carve all
        # sampled cells up to that max-range endpoint.
        for k in range(num_steps):
            t = float(k) / float(num_steps)
            px = sensor_xy_w[0] + t * dx
            py = sensor_xy_w[1] + t * dy
            row, col = xy_to_grid_cell((px, py), world_grid_cfg)
            if not (0 <= row < world_grid_cfg.height and 0 <= col < world_grid_cfg.width):
                continue
            cell = (row, col)
            if prev_cell == cell:
                continue
            free[row, col] = 1
            prev_cell = cell
    return free


def build_world_occupancy_state_map(
    planning_grid_local: torch.Tensor,
    *,
    free_state_w: torch.Tensor | None,
    proxy: G1CommandProxy,
    local_grid_cfg: OccupancyGridCfg,
    world_grid_cfg: OccupancyGridCfg,
    occupancy_threshold: float,
) -> torch.Tensor:
    """Build a fixed world-frame occupancy visualization map.

    Cell states:
      0 = unknown
      1 = free (accumulated LiDAR ray-carved free cells)
      2 = occupied (from the selected planning grid projected into world frame)

    This is a visualization map only.  A* still uses the script's body-frame
    planning grid.  The fixed world frame prevents obstacle geometry from
    appearing to rotate or jump as the robot turns.
    """
    state = torch.zeros((world_grid_cfg.height, world_grid_cfg.width), dtype=torch.uint8)
    if free_state_w is not None:
        if tuple(free_state_w.shape) != (world_grid_cfg.height, world_grid_cfg.width):
            raise ValueError(
                f"free_state_w must have shape {(world_grid_cfg.height, world_grid_cfg.width)}, "
                f"got {tuple(free_state_w.shape)}"
            )
        state[free_state_w.detach().cpu().to(dtype=torch.uint8) > 0] = 1

    for occ_xy_w in occupied_cells_world_from_local_grid(
        planning_grid_local,
        proxy=proxy,
        local_grid_cfg=local_grid_cfg,
        occupancy_threshold=occupancy_threshold,
    ):
        row, col = xy_to_grid_cell(occ_xy_w, world_grid_cfg)
        if 0 <= row < world_grid_cfg.height and 0 <= col < world_grid_cfg.width:
            state[row, col] = 2

    return state


def save_world_occupancy_debug_image(
    planning_grid_local: torch.Tensor,
    *,
    free_state_w: torch.Tensor | None,
    proxy: G1CommandProxy,
    local_grid_cfg: OccupancyGridCfg,
    world_grid_cfg: OccupancyGridCfg,
    start_xy_w: PointXY,
    target_xy_w: PointXY | None,
    robot_xy_w: PointXY,
    covered_trajectory_xy_w: Sequence[PointXY],
    active_path_xy_w: Sequence[PointXY],
    obstacles: Sequence[Mapping[str, Any]],
    env_origin_xy_w: PointXY,
    safety_zone_radius_m: float,
    out_dir: Path,
    step_idx: int,
    occupancy_threshold: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN]: Pillow is not installed. Skipping world occupancy map save.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    state = build_world_occupancy_state_map(
        planning_grid_local,
        free_state_w=free_state_w,
        proxy=proxy,
        local_grid_cfg=local_grid_cfg,
        world_grid_cfg=world_grid_cfg,
        occupancy_threshold=occupancy_threshold,
    )
    rgb_arr = torch.empty((world_grid_cfg.height, world_grid_cfg.width, 3), dtype=torch.uint8)
    rgb_arr[state == 0] = torch.tensor([150, 150, 150], dtype=torch.uint8)  # unknown
    rgb_arr[state == 1] = torch.tensor([245, 245, 245], dtype=torch.uint8)  # free
    rgb_arr[state == 2] = torch.tensor([35, 35, 35], dtype=torch.uint8)     # occupied

    rgb = Image.fromarray(rgb_arr.numpy(), mode="RGB")

    # Draw the swept safety zone first, behind the thin executed trajectory and plan overlays.
    rgb = _draw_trajectory_safety_zone_world(
        rgb,
        covered_trajectory_xy_w,
        world_grid_cfg,
        safety_zone_radius_m=float(safety_zone_radius_m),
    )
    draw = ImageDraw.Draw(rgb)

    # Red dashed = ground-truth/setup obstacle contour. This is deliberately not derived from LiDAR.
    for obs in obstacles:
        contour = obstacle_contour_world_xy(obs, env_origin_xy_w=env_origin_xy_w)
        if contour:
            _draw_dashed_polyline_world_xy(draw, contour, world_grid_cfg, fill=(255, 0, 0), width=2, dash_px=8.0, gap_px=5.0)

    _draw_polyline_world_xy(draw, covered_trajectory_xy_w, world_grid_cfg, fill=(105, 105, 105), width=2)
    # Orange = current remaining A* plan. Red is reserved for setup obstacle contours.
    _draw_polyline_world_xy(draw, active_path_xy_w, world_grid_cfg, fill=(255, 145, 0), width=2)
    _draw_marker_world(draw, start_xy_w, world_grid_cfg, fill=(0, 200, 0), radius=4, outline=(0, 90, 0))
    if target_xy_w is not None:
        _draw_marker_world(draw, target_xy_w, world_grid_cfg, fill=(0, 70, 255), radius=4, outline=(0, 0, 90))
    _draw_marker_world(draw, robot_xy_w, world_grid_cfg, fill=(255, 220, 0), radius=5, outline=(0, 0, 0), cross=True)

    path = out_dir / f"occupancy_map_{step_idx:06d}.png"
    rgb.save(path)
    print(f"[INFO]: Saved occupancy map: {path}", flush=True)


def save_nav_lidar_debug_image(
    grid: torch.Tensor,
    *,
    active_path_xy_b: Sequence[PointXY],
    covered_path_xy_b: Sequence[PointXY],
    start_xy_b: PointXY | None,
    target_xy_b: PointXY | None,
    robot_xy_b: PointXY,
    scenario_overlay_grid: torch.Tensor | None,
    grid_cfg: OccupancyGridCfg,
    out_dir: Path,
    step_idx: int,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN]: Pillow is not installed. Skipping nav/LiDAR grid save.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image = occupancy_grid_to_uint8_image(grid, env_id=0).numpy()
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    if scenario_overlay_grid is not None:
        overlay_cells = torch.nonzero(scenario_overlay_grid[0] > 0, as_tuple=False).detach().cpu()
        # Cyan = known scenario footprint overlay. In grid_source=lidar runs this is
        # diagnostic only and is not used by A*; white pixels remain planning occupancy.
        for row, col in overlay_cells.tolist():
            draw.point((int(col), int(row)), fill=(0, 180, 180))

    _draw_polyline_xy(draw, covered_path_xy_b, grid_cfg, fill=(130, 130, 130), width=2)
    _draw_polyline_xy(draw, active_path_xy_b, grid_cfg, fill=(255, 0, 0), width=2)
    if start_xy_b is not None:
        _draw_marker(draw, start_xy_b, grid_cfg, fill=(0, 255, 0), radius=4, outline=(0, 90, 0))
    if target_xy_b is not None:
        _draw_marker(draw, target_xy_b, grid_cfg, fill=(0, 0, 255), radius=4, outline=(0, 0, 90))
    _draw_marker(draw, robot_xy_b, grid_cfg, fill=(255, 230, 0), radius=5, outline=(0, 0, 0), cross=True)

    path = out_dir / f"nav_lidar_grid_{step_idx:06d}.png"
    rgb.save(path)
    print(f"[INFO]: Saved nav/LiDAR grid: {path}", flush=True)


def save_simple_grid_image(
    grid: torch.Tensor,
    *,
    grid_cfg: OccupancyGridCfg,
    out_dir: Path,
    name: str,
    active_path_xy_b: Sequence[PointXY] | None = None,
    target_xy_b: PointXY | None = None,
) -> None:
    """Save a single-source occupancy-grid diagnostic image.

    This is intentionally separate from the normal overlay image so we can tell
    whether the wall exists in the scenario footprint grid, the LiDAR hit grid,
    or only in the physics collision scene.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN]: Pillow is not installed. Skipping diagnostic grid save.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image = occupancy_grid_to_uint8_image(grid, env_id=0).numpy()
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    if active_path_xy_b:
        _draw_polyline_xy(draw, active_path_xy_b, grid_cfg, fill=(255, 0, 0), width=2)
    if target_xy_b is not None:
        _draw_marker(draw, target_xy_b, grid_cfg, fill=(0, 0, 255), radius=4, outline=(0, 0, 90))
    _draw_marker(draw, (0.0, 0.0), grid_cfg, fill=(255, 230, 0), radius=5, outline=(0, 0, 0), cross=True)
    path = out_dir / name
    rgb.save(path)
    print(f"[INFO]: Saved diagnostic grid: {path}", flush=True)


def inflated_grid_for_debug(
    grid: torch.Tensor,
    *,
    grid_cfg: OccupancyGridCfg,
    astar_cfg: GridAStarCfg,
) -> torch.Tensor:
    """Return float [N,H,W] occupancy after applying the same inflation A* uses."""
    inflated = inflate_occupancy_grid(
        grid,
        cfg=grid_cfg,
        inflation_radius=float(astar_cfg.inflation_radius),
        occupancy_threshold=float(astar_cfg.occupancy_threshold),
    )
    return inflated.to(dtype=torch.float32)


def write_records_csv(path: Path, records: Sequence[StepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(StepRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)
    print(f"[INFO]: Wrote trace CSV: {path}", flush=True)


def print_step(
    *,
    record: StepRecord,
    total_waypoints: int,
    waypoint_xy_b: PointXY,
    success_radius: float,
    safety_margin: float,
) -> None:
    print("-" * 80, flush=True)
    print(f"step                    : {record.step}", flush=True)
    print(f"target distance         : {record.target_distance:.3f}", flush=True)
    print(f"target reached radius   : {success_radius:.3f}", flush=True)
    print(f"waypoint index          : {record.waypoint_index}/{max(total_waypoints - 1, 0)}", flush=True)
    print(f"waypoint world          : ({record.waypoint_x:.3f}, {record.waypoint_y:.3f})", flush=True)
    print(f"waypoint body           : ({waypoint_xy_b[0]:.3f}, {waypoint_xy_b[1]:.3f})", flush=True)
    print(f"scenario edge clearance : {record.scenario_edge_clearance:.3f}", flush=True)
    print(f"scenario margin slack   : {record.scenario_margin_slack:.3f}  # margin={safety_margin:.3f}", flush=True)
    print(f"grid edge clearance     : {record.grid_edge_clearance:.3f}  # raw LiDAR/footprint grid", flush=True)
    print(f"lidar valid hits        : {record.lidar_valid}", flush=True)
    print(f"occupied cells          : {record.occupied_cells}", flush=True)
    print(
        f"safety action          : {record.safety_action} "
        f"(override={record.safety_override}, scale={record.safety_scale:.2f})",
        flush=True,
    )
    print(
        f"safety front/left/right: {record.safety_front_clearance:.3f} / "
        f"{record.safety_left_clearance:.3f} / {record.safety_right_clearance:.3f}",
        flush=True,
    )
    print(
        f"rgbd safety action     : {record.rgbd_safety_action} "
        f"(override={record.rgbd_safety_override}, scale={record.rgbd_safety_scale:.2f})",
        flush=True,
    )
    print(
        f"rgbd depth front/L/R   : {record.rgbd_safety_front_depth:.3f} / "
        f"{record.rgbd_safety_left_depth:.3f} / {record.rgbd_safety_right_depth:.3f}",
        flush=True,
    )
    print(
        f"nominal [vx, vy, wz]   : [{record.nominal_vx:.3f}, {record.nominal_vy:.3f}, {record.nominal_wz:.3f}]",
        flush=True,
    )
    print(f"command [vx, vy, wz]    : [{record.vx:.3f}, {record.vy:.3f}, {record.wz:.3f}]", flush=True)
    print(f"pose [x, y, yaw]        : [{record.x:.3f}, {record.y:.3f}, {record.yaw:.3f}]", flush=True)


@configclass
class G1NavLidarSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path=CFG["scene"]["ground_prim_path"],
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path=CFG["scene"]["light_prim_path"],
        spawn=sim_utils.DomeLightCfg(
            intensity=float(CFG["scene"]["dome_light"]["intensity"]),
            color=tuple(CFG["scene"]["dome_light"]["color"]),
        ),
    )

    proxy_body = AssetBaseCfg(
        prim_path=CFG["proxy"]["body_prim_path"],
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=tuple(CFG["proxy"]["body"]["size"]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(CFG["proxy"]["body"]["color"])),
        ),
    )

    target = AssetBaseCfg(
        prim_path=CFG["target"]["prim_path"],
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(CFG["target"]["pos"])),
        spawn=sim_utils.CuboidCfg(
            size=tuple(CFG["target"].get("size", [0.25, 0.25, 0.16])),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(CFG["target"].get("color", [1.0, 0.1, 0.1]))),
        ),
    )

    lidar_mount = RigidObjectCfg(
        prim_path=CFG["lidar"]["prim_path"],
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(CFG["proxy"]["init_state"]["pos"])),
        spawn=sim_utils.CuboidCfg(
            size=tuple(CFG["lidar"].get("mount_size", [0.05, 0.05, 0.05])),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(CFG["lidar"].get("mount_color", [1.0, 0.5, 0.0]))),
        ),
    )

    # Keep a fixed schema for all scenarios.  This avoids scenario-dependent
    # configclass/InteractiveScene edge cases when moving from one obstacle to
    # multiple obstacles, while RayCaster still targets only actual obstacle meshes.
    obstacle_0 = obstacle_asset_cfg(OBSTACLES[0]) if len(OBSTACLES) > 0 else inactive_obstacle_asset_cfg(0)
    obstacle_1 = obstacle_asset_cfg(OBSTACLES[1]) if len(OBSTACLES) > 1 else inactive_obstacle_asset_cfg(1)
    obstacle_2 = obstacle_asset_cfg(OBSTACLES[2]) if len(OBSTACLES) > 2 else inactive_obstacle_asset_cfg(2)
    obstacle_3 = obstacle_asset_cfg(OBSTACLES[3]) if len(OBSTACLES) > 3 else inactive_obstacle_asset_cfg(3)
    obstacle_4 = obstacle_asset_cfg(OBSTACLES[4]) if len(OBSTACLES) > 4 else inactive_obstacle_asset_cfg(4)
    obstacle_5 = obstacle_asset_cfg(OBSTACLES[5]) if len(OBSTACLES) > 5 else inactive_obstacle_asset_cfg(5)

    lidar = RayCasterCfg(
        prim_path=CFG["lidar"]["prim_path"],
        update_period=float(CFG["lidar"]["update_period"]),
        offset=RayCasterCfg.OffsetCfg(
            pos=tuple(CFG["lidar"]["offset"]["pos"]),
            rot=tuple(CFG["lidar"]["offset"]["rot_wxyz"]),
        ),
        mesh_prim_paths=resolve_mesh_prim_paths(list(CFG["lidar"]["mesh_prim_paths"]), int(CFG["scene"]["num_envs"])),
        ray_alignment=str(CFG["lidar"].get("ray_alignment", "base")),
        max_distance=float(CFG["lidar"]["max_distance"]),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=int(CFG["lidar"]["pattern"]["channels"]),
            vertical_fov_range=list(CFG["lidar"]["pattern"]["vertical_fov_range"]),
            horizontal_fov_range=list(CFG["lidar"]["pattern"]["horizontal_fov_range"]),
            horizontal_res=float(CFG["lidar"]["pattern"]["horizontal_res"]),
        ),
        debug_vis=bool(CFG["lidar"].get("debug_vis", False)),
    )

    if True:
        camera = CameraCfg(
            prim_path=CFG["camera"]["prim_path"],
            update_period=float(CFG["camera"]["update_period"]),
            height=int(CFG["camera"]["height"]),
            width=int(CFG["camera"]["width"]),
            data_types=list(CFG["camera"].get("data_types", ["rgb", "distance_to_image_plane"])),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(CFG["camera"]["pinhole"]["focal_length"]),
                focus_distance=float(CFG["camera"]["pinhole"]["focus_distance"]),
                horizontal_aperture=float(CFG["camera"]["pinhole"]["horizontal_aperture"]),
                clipping_range=tuple(CFG["camera"]["pinhole"]["clipping_range"]),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(CFG["camera"]["offset"]["pos"]),
                rot=tuple(CFG["camera"]["offset"]["rot_wxyz"]),
                convention=str(CFG["camera"]["offset"].get("convention", "ros")),
            ),
        )


    # Static person-like visual target for Phase-5 perception. If a realistic USD
    # is provided, it is composed manually below as a payload/reference under a
    # lightweight wrapper Xform. Otherwise we spawn a simple proxy humanoid.
    if not bool(str(args_cli.person_asset_usd).strip()):
        person_torso = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ProxyPersonTorso",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(float(PERSON_X), float(PERSON_Y), 1.05),
                rot=yaw_to_quat_wxyz(float(args_cli.person_yaw)),
            ),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.35, 0.22, 0.90),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.20, 0.90)),
            ),
        )
        person_head = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ProxyPersonHead",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(float(PERSON_X), float(PERSON_Y), 1.65),
                rot=yaw_to_quat_wxyz(float(args_cli.person_yaw)),
            ),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.24, 0.24, 0.24),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.70, 0.55)),
            ),
        )
        person_leg_l = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ProxyPersonLegL",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(float(PERSON_X), float(PERSON_Y) + 0.08, 0.45),
                rot=yaw_to_quat_wxyz(float(args_cli.person_yaw)),
            ),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.14, 0.08, 0.75),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
            ),
        )
        person_leg_r = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ProxyPersonLegR",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(float(PERSON_X), float(PERSON_Y) - 0.08, 0.45),
                rot=yaw_to_quat_wxyz(float(args_cli.person_yaw)),
            ),
            spawn=sim_utils.MeshCuboidCfg(
                size=(0.14, 0.08, 0.75),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08)),
            ),
        )


def add_referenced_person_asset() -> None:
    """Add an external person USD under a lightweight wrapper Xform.

    The wrapper owns the transform. The child prim owns the payload/reference.
    This avoids fragile edits on the referenced character root prim.
    """
    if not bool(str(args_cli.person_asset_usd).strip()):
        return
    if bool(args_cli.skip_person_asset_reference):
        print("[WARN]: --skip_person_asset_reference set; external person USD will not be added", flush=True)
        return
    usd_path = str(Path(args_cli.person_asset_usd).expanduser())
    usd_file = Path(usd_path)
    print(f"[INFO]: person USD resolved path: {usd_path}", flush=True)
    print(f"[INFO]: person USD exists={usd_file.exists()} is_file={usd_file.is_file() if usd_file.exists() else False}", flush=True)
    if not usd_file.exists():
        raise FileNotFoundError(f"person USD does not exist: {usd_path}")

    wrapper_path = "/World/envs/env_0/PersonAsset"
    model_path = f"{wrapper_path}/Model"
    stage = omni.usd.get_context().get_stage()
    wrapper_xform = UsdGeom.Xform.Define(stage, wrapper_path)
    wrapper_xform.ClearXformOpOrder()
    wrapper_xform.AddTranslateOp().Set(Gf.Vec3d(float(PERSON_X), float(PERSON_Y), 0.0))
    wrapper_xform.AddRotateZOp().Set(float(math.degrees(float(args_cli.person_yaw))))
    scale = float(args_cli.person_asset_scale)
    wrapper_xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    model_xform = UsdGeom.Xform.Define(stage, model_path)
    model_prim = model_xform.GetPrim()
    mode = str(args_cli.person_asset_reference_mode)
    if mode == "payload":
        model_prim.GetPayloads().AddPayload(usd_path)
    else:
        model_prim.GetReferences().AddReference(usd_path)
    print(f"[INFO]: Added person asset ({mode}): {usd_path} -> {model_path}; wrapper={wrapper_path}", flush=True)


def yolo_tensor_rgb_to_numpy(rgb: torch.Tensor) -> np.ndarray:
    rgb0 = rgb[0].detach().cpu()
    if rgb0.dtype != torch.uint8:
        if float(rgb0.max().item()) <= 1.0:
            rgb0 = rgb0 * 255.0
        rgb0 = rgb0.clamp(0, 255).to(torch.uint8)
    return rgb0.numpy()


def yolo_tensor_depth_to_numpy(depth: torch.Tensor) -> np.ndarray:
    """Convert full depth tensor to numpy.

    This is kept for offline/debug use only. In the full Isaac+RTX navigation loop,
    copying the whole CUDA depth image can block on GPU synchronization. The
    perception path below therefore estimates only the selected bbox depth directly
    from the CUDA tensor and copies a single scalar to CPU.
    """
    depth0 = depth[0].detach().to(dtype=torch.float32)
    if depth0.ndim == 3 and depth0.shape[-1] == 1:
        depth0 = depth0[..., 0]
    return depth0.contiguous().cpu().numpy()


def estimate_bbox_depth_from_torch(
    depth: torch.Tensor,
    detection: Detection2D,
    intrinsics: CameraIntrinsics,
    cfg: TargetFromBBoxConfig,
) -> tuple[float, int, tuple[int, int, int, int]] | None:
    """Estimate robust bbox depth without copying the full depth image to CPU.

    The Isaac camera depth image is a CUDA tensor in this script. Copying the whole
    HxW float image to numpy can block; for target projection we only need a robust
    depth statistic inside the selected bbox crop. This function performs the crop
    and filtering on the tensor device and synchronizes only the final scalar.
    """
    d = depth.detach().to(dtype=torch.float32)
    if d.ndim == 4:
        d = d[0]
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"depth tensor must reduce to HxW, got shape={tuple(depth.shape)} -> {tuple(d.shape)}")

    h, w = int(d.shape[0]), int(d.shape[1])
    if int(intrinsics.width) != w or int(intrinsics.height) != h:
        raise ValueError(
            "intrinsics resolution does not match depth tensor: "
            f"intrinsics=({intrinsics.width},{intrinsics.height}) depth=({w},{h})"
        )

    ix1, iy1, ix2, iy2 = center_crop_bbox_xyxy(
        detection.bbox_xyxy,
        crop_frac=float(cfg.bbox_center_crop_frac),
        width=w,
        height=h,
    )
    crop = d[iy1:iy2, ix1:ix2].reshape(-1)
    valid = crop[torch.isfinite(crop)]
    valid = valid[(valid >= float(cfg.depth_min_m)) & (valid <= float(cfg.depth_max_m))]
    n = int(valid.numel())
    if n <= 0:
        return None

    q = min(max(float(cfg.depth_percentile), 0.0), 1.0)
    # Sorting a bbox crop is cheap and avoids copying all valid depths to CPU.
    valid_sorted = torch.sort(valid).values
    idx = int(round(q * float(n - 1)))
    z = float(valid_sorted[idx].detach().cpu().item())
    return z, n, (ix1, iy1, ix2, iy2)


def bbox_depth_tensor_to_target2d(
    detection: Detection2D,
    depth: torch.Tensor,
    intrinsics: CameraIntrinsics,
    T_world_camera: np.ndarray,
    *,
    robot_pose: NavPose2D,
    config: TargetFromBBoxConfig,
):
    depth_est = estimate_bbox_depth_from_torch(depth, detection, intrinsics, config)
    if depth_est is None:
        return None
    depth_m, valid_count, crop_xyxy = depth_est
    u, v = detection.center_xy
    point_camera = unproject_pixel_to_camera(u, v, depth_m, intrinsics)
    point_world = transform_point(T_world_camera, point_camera)
    object_xy_w = (float(point_world[0]), float(point_world[1]))
    target, used_stand_off, already_within_standoff = target_xy_with_standoff(
        object_xy_w,
        robot_pose=robot_pose,
        stand_off_distance_m=float(config.stand_off_distance_m),
        min_target_distance_m=float(config.min_target_distance_m),
    )
    from types import SimpleNamespace
    return SimpleNamespace(
        target=target,
        object_xy_w=object_xy_w,
        detection=detection,
        depth_m=float(depth_m),
        point_camera=point_camera,
        point_world=point_world,
        used_stand_off=bool(used_stand_off),
        already_within_standoff=bool(already_within_standoff),
        valid_depth_pixels=int(valid_count),
        depth_crop_xyxy=tuple(int(v) for v in crop_xyxy),
        reason="ok",
    )


def known_person_geometry_to_target2d(
    detection: Detection2D,
    intrinsics: CameraIntrinsics,
    T_world_camera: np.ndarray,
    *,
    robot_pose: NavPose2D,
    config: TargetFromBBoxConfig,
):
    """Produce Target2D from the known static person pose without touching depth.

    This is an isolation path for full-pipeline navigation tests. It keeps the
    RGB/YOLO/fallback detection path active, but avoids synchronizing Isaac's
    RTX depth tensor. The first real depth-dependent path remains available via
    --target_depth_source bbox.
    """
    projected = project_world_to_pixel(
        (float(PERSON_X), float(PERSON_Y), float(args_cli.synthetic_point_z)),
        intrinsics,
        T_world_camera,
    )
    depth_m = float(projected[2]) if projected is not None else math.hypot(float(PERSON_X), float(PERSON_Y))
    object_xy_w = (float(PERSON_X), float(PERSON_Y))
    target, used_stand_off, already_within_standoff = target_xy_with_standoff(
        object_xy_w,
        robot_pose=robot_pose,
        stand_off_distance_m=float(config.stand_off_distance_m),
        min_target_distance_m=float(config.min_target_distance_m),
    )
    point_camera = unproject_pixel_to_camera(*detection.center_xy, depth_m, intrinsics)
    point_world = transform_point(T_world_camera, point_camera)
    from types import SimpleNamespace
    return SimpleNamespace(
        target=target,
        object_xy_w=object_xy_w,
        detection=detection,
        depth_m=float(depth_m),
        point_camera=point_camera,
        point_world=point_world,
        used_stand_off=bool(used_stand_off),
        already_within_standoff=bool(already_within_standoff),
        valid_depth_pixels=-1,
        depth_crop_xyxy=(),
        reason="ok_geometric_known_person",
        depth_source="geometric",
    )


def compute_camera_intrinsics_from_cfg() -> CameraIntrinsics:
    cam_cfg = CFG["camera"]
    width = int(cam_cfg["width"])
    height = int(cam_cfg["height"])
    focal = float(cam_cfg["pinhole"]["focal_length"])
    aperture = float(cam_cfg["pinhole"]["horizontal_aperture"])
    fx = focal / aperture * float(width)
    fy = fx
    return CameraIntrinsics(fx=fx, fy=fy, cx=0.5 * (width - 1), cy=0.5 * (height - 1), width=width, height=height)


def T_world_camera_from_proxy(proxy: G1CommandProxy) -> np.ndarray:
    # Live pose source. This reads CUDA tensors and can synchronize/block in headless RTX runs.
    pose = NavPose2D(
        float(proxy.pos_w[0, 0].detach().cpu().item()),
        float(proxy.pos_w[0, 1].detach().cpu().item()),
        float(proxy.yaw_w[0].detach().cpu().item()),
    )
    z = float(proxy.pos_w[0, 2].detach().cpu().item()) if proxy.pos_w.shape[-1] >= 3 else 0.0
    T_world_body = pose2d_to_T_world_body(pose, z=z)
    cam_pos = CFG["camera"]["offset"]["pos"]
    T_body_camera = make_forward_camera_T_body_camera(translation_body=(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
    return T_world_body @ T_body_camera


def T_world_camera_fixed_initial() -> np.ndarray:
    # Same assumption as smoke_test_yolo_person_target.py: initial body/world pose is origin, yaw=0.
    # Use this for the first static target extraction to avoid GPU proxy-pose synchronization.
    pose = NavPose2D(0.0, 0.0, 0.0)
    T_world_body = pose2d_to_T_world_body(pose, z=0.0)
    cam_pos = CFG["camera"]["offset"]["pos"]
    T_body_camera = make_forward_camera_T_body_camera(translation_body=(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
    return T_world_body @ T_body_camera


def project_world_to_pixel(point_w: Sequence[float], intr: CameraIntrinsics, T_world_camera: np.ndarray) -> tuple[float, float, float] | None:
    print(f"[CKPT]: project_world_to_pixel: start point_w={tuple(float(v) for v in point_w)}", flush=True)
    p_w = np.array([float(point_w[0]), float(point_w[1]), float(point_w[2]), 1.0], dtype=np.float64)
    T = np.asarray(T_world_camera, dtype=np.float64)
    print(f"[CKPT]: project_world_to_pixel: T_world_camera shape={T.shape}", flush=True)
    T_camera_world = np.linalg.inv(T)
    p_c = T_camera_world @ p_w
    x, y, z = float(p_c[0]), float(p_c[1]), float(p_c[2])
    print(f"[CKPT]: project_world_to_pixel: point_camera=({x:.3f}, {y:.3f}, {z:.3f})", flush=True)
    if not math.isfinite(z) or z <= 1e-6:
        print("[CKPT]: project_world_to_pixel: point behind camera or nonfinite", flush=True)
        return None
    u = float(intr.fx) * x / z + float(intr.cx)
    v = float(intr.fy) * y / z + float(intr.cy)
    print(f"[CKPT]: project_world_to_pixel: pixel=({u:.1f}, {v:.1f}), z={z:.3f}", flush=True)
    return u, v, z


def synthetic_person_bbox_from_projection(intr: CameraIntrinsics, T_world_camera: np.ndarray) -> Detection2D:
    print("[CKPT]: synthetic_person_bbox_from_projection: start", flush=True)
    projected = project_world_to_pixel((float(PERSON_X), float(PERSON_Y), float(args_cli.synthetic_point_z)), intr, T_world_camera)
    print(f"[CKPT]: synthetic_person_bbox_from_projection: projected={projected}", flush=True)
    if projected is None:
        raise RuntimeError("known person point projects behind the camera; check camera/person pose")
    u, v, _ = projected
    w = float(args_cli.synthetic_bbox_w)
    h = float(args_cli.synthetic_bbox_h)
    det = Detection2D(class_name=str(args_cli.target_class), bbox_xyxy=(u - 0.5 * w, v - 0.5 * h, u + 0.5 * w, v + 0.5 * h), confidence=1.0)
    print(f"[CKPT]: synthetic_person_bbox_from_projection: bbox={tuple(round(v, 1) for v in det.bbox_xyxy)}", flush=True)
    return det


_YOLO_DETECTOR_CACHE = None


def run_yolo_if_requested(rgb_np: np.ndarray) -> list[Detection2D]:
    global _YOLO_DETECTOR_CACHE
    if bool(args_cli.skip_yolo) or str(args_cli.target_source) == "fallback":
        print("[INFO]: YOLO skipped by --skip_yolo or --target_source=fallback", flush=True)
        return []
    try:
        print("[CKPT]: YOLO import/config start", flush=True)
        from g1_nav_rl.perception.yolo_detector import YoloDetector, YoloDetectorConfig
        if _YOLO_DETECTOR_CACHE is None:
            print(f"[CKPT]: YOLO detector init start model={str(args_cli.yolo_model)!r} device={str(args_cli.yolo_device) or 'auto'}", flush=True)
            _YOLO_DETECTOR_CACHE = YoloDetector(YoloDetectorConfig(model=str(args_cli.yolo_model), confidence=float(args_cli.yolo_conf), iou=float(args_cli.yolo_iou), device=str(args_cli.yolo_device) or None, verbose=False))
            print("[CKPT]: YOLO detector init returned", flush=True)
        print(f"[CKPT]: YOLO inference start rgb_shape={tuple(rgb_np.shape)} dtype={rgb_np.dtype}", flush=True)
        detections = _YOLO_DETECTOR_CACHE.detect(rgb_np)
        print("[CKPT]: YOLO inference returned", flush=True)
        print(f"[INFO]: YOLO detections: {len(detections)}", flush=True)
        for det in detections:
            print(f"  det class={det.class_name!r} conf={det.confidence:.3f} bbox={tuple(round(v, 1) for v in det.bbox_xyxy)} area={det.area:.1f}", flush=True)
        return detections
    except Exception as exc:
        print(f"[WARN]: YOLO unavailable or failed: {exc}", flush=True)
        return []


def extract_perception_target(scene: InteractiveScene, proxy: G1CommandProxy) -> tuple[PointXY, dict[str, Any]]:
    """Run YOLO/fallback bbox + depth projection and return a world-frame Target2D."""
    print("[CKPT]: extract_perception_target: reading camera outputs", flush=True)
    camera = scene["camera"]
    rgb = camera.data.output.get("rgb")
    depth = camera.data.output.get("distance_to_image_plane")
    print(f"[CKPT]: extract_perception_target: rgb is None={rgb is None}, depth is None={depth is None}", flush=True)
    if rgb is not None:
        try:
            print(f"[CKPT]: extract_perception_target: rgb tensor shape={tuple(rgb.shape)} dtype={rgb.dtype} device={rgb.device}", flush=True)
        except Exception:
            pass
    if depth is not None:
        try:
            print(f"[CKPT]: extract_perception_target: depth tensor shape={tuple(depth.shape)} dtype={depth.dtype} device={depth.device}", flush=True)
        except Exception:
            pass
    if rgb is None or depth is None:
        raise RuntimeError("camera did not provide both rgb and distance_to_image_plane outputs")
    print("[CKPT]: extract_perception_target: converting rgb tensor to numpy", flush=True)
    rgb_np = yolo_tensor_rgb_to_numpy(rgb)
    print(f"[CKPT]: extract_perception_target: rgb_np shape={tuple(rgb_np.shape)} dtype={rgb_np.dtype}", flush=True)
    print("[CKPT]: extract_perception_target: keeping depth tensor on device for bbox-only depth", flush=True)
    print("[CKPT]: extract_perception_target: computing intrinsics", flush=True)
    intr = compute_camera_intrinsics_from_cfg()
    print(f"[CKPT]: extract_perception_target: intrinsics fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.cx:.3f} cy={intr.cy:.3f}", flush=True)
    print(f"[CKPT]: extract_perception_target: computing T_world_camera source={args_cli.perception_camera_pose_source}", flush=True)
    if str(args_cli.perception_camera_pose_source) == "proxy":
        T_world_camera = T_world_camera_from_proxy(proxy)
    else:
        T_world_camera = T_world_camera_fixed_initial()
    print("[CKPT]: extract_perception_target: T_world_camera computed", flush=True)
    print("[CKPT]: extract_perception_target: running/selecting detection", flush=True)
    detections = run_yolo_if_requested(rgb_np)
    print("[CKPT]: extract_perception_target: selecting detection", flush=True)

    selected = None
    source = "none"

    # Important: when explicitly using fallback, do not call the generic selection helper.
    # This keeps the first full-pipeline test equivalent to the earlier working
    # smoke_test_yolo_person_target.py path: projected bbox + depth -> Target2D.
    if str(args_cli.target_source) == "fallback":
        print("[CKPT]: extract_perception_target: target_source=fallback, computing synthetic bbox directly", flush=True)
        selected = synthetic_person_bbox_from_projection(intr, T_world_camera)
        source = "synthetic_projection"
        print(f"[INFO]: using fallback synthetic bbox: {tuple(round(v, 1) for v in selected.bbox_xyxy)}", flush=True)
    else:
        print(f"[CKPT]: extract_perception_target: selecting from {len(detections)} YOLO detections", flush=True)
        selected = select_detection_for_target(
            detections,
            allowed_classes={str(args_cli.target_class)},
            min_confidence=float(args_cli.min_confidence),
            depth_image=None,
            policy=str(args_cli.selection_policy),
        )
        source = "yolo"
        if selected is None and bool(args_cli.fallback_synthetic_bbox):
            print("[CKPT]: extract_perception_target: no YOLO target, computing fallback synthetic bbox", flush=True)
            selected = synthetic_person_bbox_from_projection(intr, T_world_camera)
            source = "synthetic_projection"
            print(f"[INFO]: using fallback synthetic bbox: {tuple(round(v, 1) for v in selected.bbox_xyxy)}", flush=True)
    if selected is None:
        if bool(args_cli.assert_detection):
            raise RuntimeError("no YOLO/fallback person detection available")
        # Fall back to scenario/static person xy so the nav loop can still run for diagnostics.
        return (float(PERSON_X), float(PERSON_Y)), {"detection_source": "none", "detections": []}
    print("[CKPT]: extract_perception_target: projecting bbox+depth to Target2D", flush=True)
    if str(args_cli.perception_camera_pose_source) == "proxy":
        print("[CKPT]: extract_perception_target: reading proxy pose for target standoff", flush=True)
        robot_pose_for_target = NavPose2D(
            float(proxy.pos_w[0, 0].detach().cpu().item()),
            float(proxy.pos_w[0, 1].detach().cpu().item()),
            float(proxy.yaw_w[0].detach().cpu().item()),
        )
    else:
        # Match the fixed initial camera transform. This avoids a second proxy CUDA
        # synchronization during the initial perception target extraction.
        robot_pose_for_target = NavPose2D(0.0, 0.0, 0.0)
        print("[CKPT]: extract_perception_target: using fixed initial robot pose for target standoff", flush=True)
    target_cfg = TargetFromBBoxConfig(
        depth_min_m=float(args_cli.target_depth_min),
        depth_max_m=float(args_cli.target_depth_max),
        bbox_center_crop_frac=float(args_cli.target_crop_frac),
        depth_percentile=float(args_cli.target_depth_percentile),
        stand_off_distance_m=float(args_cli.stand_off),
    )
    result = bbox_depth_tensor_to_target2d(
        detection=selected,
        depth=depth,
        intrinsics=intr,
        T_world_camera=T_world_camera,
        robot_pose=robot_pose_for_target,
        config=target_cfg,
    )
    print("[CKPT]: extract_perception_target: bbox_depth_tensor_to_target2d returned", flush=True)
    if result is None:
        if bool(args_cli.assert_target):
            raise RuntimeError("bbox+depth failed to produce Target2D")
        return (float(PERSON_X), float(PERSON_Y)), {"detection_source": source, "reason": "target_projection_failed"}
    target_xy = (float(result.target.x), float(result.target.y))
    object_error = math.hypot(float(result.object_xy_w[0]) - float(PERSON_X), float(result.object_xy_w[1]) - float(PERSON_Y))
    if float(args_cli.assert_target_error) >= 0.0 and object_error > float(args_cli.assert_target_error):
        raise RuntimeError(f"perception object error {object_error:.3f} exceeds threshold {float(args_cli.assert_target_error):.3f}")
    meta = {
        "detection_source": source,
        "detections": [{"class_name": d.class_name, "confidence": d.confidence, "bbox_xyxy": list(d.bbox_xyxy)} for d in detections],
        "selected_detection": {"class_name": selected.class_name, "confidence": selected.confidence, "bbox_xyxy": list(selected.bbox_xyxy)},
        "depth_m": float(result.depth_m),
        "object_xy_w": [float(result.object_xy_w[0]), float(result.object_xy_w[1])],
        "target_xy_w": [target_xy[0], target_xy[1]],
        "known_person_xy": [float(PERSON_X), float(PERSON_Y)],
        "object_error_xy_m": float(object_error),
        "used_stand_off": bool(result.used_stand_off),
        "valid_depth_pixels": int(getattr(result, "valid_depth_pixels", -1)),
        "depth_crop_xyxy": list(getattr(result, "depth_crop_xyxy", ())),
    }
    print("[INFO]: perception target result", flush=True)
    print(f"  selected source/class/conf : {source} / {selected.class_name} / {selected.confidence:.3f}", flush=True)
    print(f"  bbox_xyxy                  : {tuple(round(v, 1) for v in selected.bbox_xyxy)}", flush=True)
    print(f"  depth_m                    : {float(result.depth_m):.3f}", flush=True)
    print(f"  object_xy_w                : ({float(result.object_xy_w[0]):.3f}, {float(result.object_xy_w[1]):.3f})", flush=True)
    print(f"  nav target_xy_w            : ({target_xy[0]:.3f}, {target_xy[1]:.3f})", flush=True)
    print(f"  known_person_xy            : ({float(PERSON_X):.3f}, {float(PERSON_Y):.3f})", flush=True)
    print(f"  object_error_xy_m          : {object_error:.3f}", flush=True)
    return target_xy, meta


def make_proxy(device: torch.device, num_envs: int) -> G1CommandProxy:
    limits_cfg = CFG["proxy"]["command_limits"]
    if args_cli.use_config_proxy_dynamics:
        max_ax = float(limits_cfg["max_ax"])
        max_ay = float(limits_cfg["max_ay"])
        max_awz = float(limits_cfg["max_awz"])
        smoothing_alpha = float(limits_cfg["smoothing_alpha"])
    else:
        # Keep velocity clipping, but remove extra command dynamics so this test
        # behaves like smoke_test_nav.py's direct integration baseline.
        max_ax = 1.0e6
        max_ay = 1.0e6
        max_awz = 1.0e6
        smoothing_alpha = 1.0

    limits = G1CommandProxyLimits(
        max_vx=float(limits_cfg["max_vx"]),
        max_vy=float(limits_cfg["max_vy"]),
        max_wz=float(limits_cfg["max_wz"]),
        max_ax=max_ax,
        max_ay=max_ay,
        max_awz=max_awz,
        smoothing_alpha=smoothing_alpha,
    )
    return G1CommandProxy(
        proxy_prim_paths=resolve_env_prim_paths(CFG["proxy"]["prim_path"], num_envs),
        num_envs=num_envs,
        device=device,
        init_pos=tuple(CFG["proxy"]["init_state"]["pos"]),
        init_yaw=float(CFG["proxy"]["init_state"]["yaw"]),
        limits=limits,
    )



def extract_perception_target_from_cpu_snapshot(scene: InteractiveScene, sim: sim_utils.SimulationContext) -> tuple[PointXY, dict[str, Any]]:
    """Capture RGB on CPU, keep depth on-device, and convert bbox to frozen Target2D.

    RGB is copied to CPU because YOLO/fallback operates on a regular image.  The
    depth image is *not* copied in full: Isaac/RTX depth tensors can block during
    full CUDA->CPU readback in the combined LiDAR+navigation scene.  Instead we
    select a bbox first, compute a robust depth scalar inside that bbox on the
    tensor device, and synchronize only that scalar.
    """
    print("[CKPT]: snapshot_target: starting camera warmup before RGB snapshot", flush=True)
    sim_dt = float(sim.get_physics_dt())
    warmup_steps = max(0, int(args_cli.perception_warmup_steps))
    for i in range(warmup_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if i == 0 or i == warmup_steps - 1:
            print(f"[CKPT]: snapshot_target: warmup step {i + 1}/{warmup_steps} complete", flush=True)

    print("[CKPT]: snapshot_target: reading camera outputs", flush=True)
    camera = scene["camera"]
    rgb = camera.data.output.get("rgb")
    depth = camera.data.output.get("distance_to_image_plane")
    print(f"[CKPT]: snapshot_target: rgb is None={rgb is None}, depth is None={depth is None}", flush=True)
    if rgb is None or depth is None:
        raise RuntimeError("camera did not provide both rgb and distance_to_image_plane outputs")
    try:
        print(f"[CKPT]: snapshot_target: rgb tensor shape={tuple(rgb.shape)} dtype={rgb.dtype} device={rgb.device}", flush=True)
        print(f"[CKPT]: snapshot_target: depth tensor shape={tuple(depth.shape)} dtype={depth.dtype} device={depth.device}", flush=True)
    except Exception:
        pass

    print("[CKPT]: snapshot_target: copying RGB tensor to CPU numpy", flush=True)
    rgb_np = yolo_tensor_rgb_to_numpy(rgb)
    print(f"[CKPT]: snapshot_target: rgb_np shape={tuple(rgb_np.shape)} dtype={rgb_np.dtype}", flush=True)
    print("[CKPT]: snapshot_target: keeping full depth tensor on device; no full depth CPU copy", flush=True)

    depth_min = float(args_cli.target_depth_min)
    depth_max = float(args_cli.target_depth_max)

    print("[CKPT]: snapshot_target: computing intrinsics and fixed T_world_camera", flush=True)
    intr = compute_camera_intrinsics_from_cfg()
    if str(args_cli.perception_camera_pose_source) == "proxy":
        print("[WARN]: snapshot target currently uses fixed initial camera pose; ignoring --perception_camera_pose_source=proxy", flush=True)
    T_world_camera = T_world_camera_fixed_initial()

    print("[CKPT]: snapshot_target: running/selecting detection on RGB CPU snapshot", flush=True)
    detections = run_yolo_if_requested(rgb_np)
    print(f"[CKPT]: snapshot_target: run_yolo_if_requested returned {len(detections)} detections", flush=True)
    selected = None
    source = "none"
    if str(args_cli.target_source) == "fallback":
        print("[CKPT]: snapshot_target: target_source=fallback, computing synthetic bbox", flush=True)
        selected = synthetic_person_bbox_from_projection(intr, T_world_camera)
        source = "synthetic_projection"
        print(f"[INFO]: using fallback synthetic bbox: {tuple(round(v, 1) for v in selected.bbox_xyxy)}", flush=True)
    else:
        # Do not pass the full depth image into selection; that would force a CPU
        # depth readback.  For nearest policy, select_detection_for_target falls
        # back to the largest bbox when depth_image is omitted.
        selected = select_detection_for_target(
            detections,
            allowed_classes={str(args_cli.target_class)},
            min_confidence=float(args_cli.min_confidence),
            depth_image=None,
            policy=str(args_cli.selection_policy),
        )
        source = "yolo"
        if selected is None and bool(args_cli.fallback_synthetic_bbox):
            selected = synthetic_person_bbox_from_projection(intr, T_world_camera)
            source = "synthetic_projection"
            print(f"[INFO]: using fallback synthetic bbox: {tuple(round(v, 1) for v in selected.bbox_xyxy)}", flush=True)

    if selected is None:
        if bool(args_cli.assert_detection):
            raise RuntimeError("no YOLO/fallback person detection available")
        return (float(PERSON_X), float(PERSON_Y)), {"detection_source": "none", "detections": []}

    cfg = TargetFromBBoxConfig(
        depth_min_m=depth_min,
        depth_max_m=depth_max,
        bbox_center_crop_frac=float(args_cli.target_crop_frac),
        depth_percentile=float(args_cli.target_depth_percentile),
        stand_off_distance_m=float(args_cli.stand_off),
    )
    depth_source = str(args_cli.target_depth_source)
    if depth_source == "auto":
        depth_source = "geometric" if source == "synthetic_projection" else "bbox"
    if depth_source == "geometric":
        print("[CKPT]: snapshot_target: using geometric known-person depth/XY; no depth tensor sync", flush=True)
        target_result = known_person_geometry_to_target2d(
            detection=selected,
            intrinsics=intr,
            T_world_camera=T_world_camera,
            robot_pose=NavPose2D(0.0, 0.0, 0.0),
            config=cfg,
        )
        print("[CKPT]: snapshot_target: geometric target projection returned", flush=True)
    else:
        print("[CKPT]: snapshot_target: projecting bbox+depth-scalar tensor to Target2D", flush=True)
        target_result = bbox_depth_tensor_to_target2d(
            detection=selected,
            depth=depth,
            intrinsics=intr,
            T_world_camera=T_world_camera,
            robot_pose=NavPose2D(0.0, 0.0, 0.0),
            config=cfg,
        )
        print("[CKPT]: snapshot_target: bbox depth scalar projection returned", flush=True)
    if target_result is None:
        if bool(args_cli.assert_target):
            raise RuntimeError("bbox+depth did not produce a valid Target2D from bbox depth scalar")
        return (float(PERSON_X), float(PERSON_Y)), {"detection_source": source, "detections": [d.__dict__ for d in detections], "target_valid": False}

    object_error = math.hypot(float(target_result.object_xy_w[0]) - float(PERSON_X), float(target_result.object_xy_w[1]) - float(PERSON_Y))
    depth_shape = tuple(int(v) for v in depth.shape)
    meta: dict[str, Any] = {
        "snapshot_mode": "cpu_rgb_bbox_plus_device_depth_scalar_once_before_navigation",
        "detection_source": source,
        "detections": [
            {"class_name": d.class_name, "confidence": float(d.confidence), "bbox_xyxy": [float(v) for v in d.bbox_xyxy]}
            for d in detections
        ],
        "selected_detection": {
            "class_name": selected.class_name,
            "confidence": float(selected.confidence),
            "bbox_xyxy": [float(v) for v in selected.bbox_xyxy],
        },
        "target": {
            "depth_m": float(target_result.depth_m),
            "depth_source": str(getattr(target_result, "depth_source", depth_source)),
            "object_xy_w": [float(target_result.object_xy_w[0]), float(target_result.object_xy_w[1])],
            "target_xy_w": [float(target_result.target.x), float(target_result.target.y)],
            "object_error_xy_m": float(object_error),
            "used_stand_off": bool(target_result.used_stand_off),
            "valid_depth_pixels": int(getattr(target_result, "valid_depth_pixels", -1)),
            "depth_crop_xyxy": list(getattr(target_result, "depth_crop_xyxy", ())),
        },
        "rgb_shape": list(rgb_np.shape),
        "depth_tensor_shape": list(depth_shape),
        "depth_readback": "bbox_scalar_only",
        "intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy, "width": intr.width, "height": intr.height},
    }
    print("[INFO]: snapshot perception target", flush=True)
    print(f"  source/class/conf          : {source} / {selected.class_name} / {selected.confidence:.3f}", flush=True)
    print(f"  bbox_xyxy                  : {tuple(round(v, 1) for v in selected.bbox_xyxy)}", flush=True)
    print(f"  depth_m                    : {float(target_result.depth_m):.3f}", flush=True)
    print(f"  target depth source        : {str(getattr(target_result, 'depth_source', depth_source))}", flush=True)
    print(f"  valid depth pixels         : {int(getattr(target_result, 'valid_depth_pixels', -1))}", flush=True)
    print(f"  object_xy_w                : ({target_result.object_xy_w[0]:.3f}, {target_result.object_xy_w[1]:.3f})", flush=True)
    print(f"  target_xy_w                : ({target_result.target.x:.3f}, {target_result.target.y:.3f})", flush=True)
    print(f"  known_person_xy            : ({PERSON_X:.3f}, {PERSON_Y:.3f})", flush=True)
    print(f"  object_error_xy_m          : {object_error:.3f}", flush=True)
    return (float(target_result.target.x), float(target_result.target.y)), meta


def build_rgbd_depth_observation(scene: InteractiveScene) -> torch.Tensor | None:
    """Return env0 RGB-D camera depth tensor when RGB-D safety is enabled."""
    if not bool(args_cli.enable_rgbd_safety):
        return None
    camera = scene["camera"]
    depth = camera.data.output.get("distance_to_image_plane")
    if depth is None:
        raise RuntimeError("RGB-D safety is enabled but camera output has no distance_to_image_plane depth image.")
    depth = depth.detach().to(dtype=torch.float32).contiguous()
    depth_min = float(CFG.get("observations", {}).get("depth_min", 0.0))
    depth_max = float(CFG.get("observations", {}).get("depth_max", 10.0))
    depth = torch.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=depth_min)
    return depth.clamp(min=depth_min, max=depth_max)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, frozen_target_xy_w: PointXY, perception_meta: dict[str, Any]) -> None:
    print("[CKPT]: run_simulator entered", flush=True)
    sim_dt = sim.get_physics_dt()
    device = torch.device(args_cli.device)
    num_envs = int(CFG["scene"]["num_envs"])

    lidar_obs_cfg = LidarObservationCfg.from_mapping(CFG.get("lidar", {}).get("observations", {}))
    grid_cfg = OccupancyGridCfg.from_mapping(CFG.get("lidar", {}).get("occupancy_grid", {}))
    astar_cfg, body_radius, safety_margin, inflation_radius = compute_astar_cfg()
    success_radius = float(args_cli.success_radius if args_cli.success_radius is not None else args_cli.hand_length)

    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else PROJECT_ROOT / "outputs" / "yolo_person_nav" / args_cli.scenario
    grid_dir = output_dir / "grid"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[CKPT]: creating G1 proxy", flush=True)
    proxy = make_proxy(device=device, num_envs=num_envs)
    print("[CKPT]: resetting G1 proxy", flush=True)
    proxy.reset(scene.env_origins)
    print("[CKPT]: proxy reset returned", flush=True)

    lidar_mount_paths = resolve_env_prim_paths(CFG["lidar"]["prim_path"], num_envs)
    mount_sync_cfg = LidarMountSyncCfg()
    print("[CKPT]: syncing LiDAR mounts before optional scene.reset", flush=True)
    sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)

    if bool(args_cli.skip_second_scene_reset):
        print("[CKPT]: skipping second scene.reset() by request", flush=True)
    else:
        print("[CKPT]: calling second scene.reset() inside run_simulator", flush=True)
        scene.reset()
        print("[CKPT]: second scene.reset() returned", flush=True)
    sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)
    print("[CKPT]: LiDAR mounts synced after scene reset", flush=True)

    target_local_xy = torch.tensor(CFG["target"]["pos"][0:2], dtype=torch.float32, device=device)
    target_xy_w = scene.env_origins[:, 0:2].to(device=device) + target_local_xy

    start_xy_w: PointXY = (
        float(proxy.pos_w[0, 0].detach().cpu().item()),
        float(proxy.pos_w[0, 1].detach().cpu().item()),
    )
    trajectory_w: list[PointXY] = [start_xy_w]

    # Give RayCaster enough frames to populate buffers after reset. The default
    # RayCaster update_period is 0.05 s and sim_dt is 0.02 s, so a single frame
    # can be too early on some runs.
    print(f"[CKPT]: starting LiDAR warmup steps: {max(1, int(args_cli.lidar_warmup_steps))}", flush=True)
    for i in range(max(1, int(args_cli.lidar_warmup_steps))):
        sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        if i == 0 or i == max(1, int(args_cli.lidar_warmup_steps)) - 1:
            print(f"[CKPT]: LiDAR warmup step {i + 1}/{max(1, int(args_cli.lidar_warmup_steps))} complete", flush=True)

    # Perception target is frozen from a CPU RGB-D snapshot captured before the
    # navigation/proxy loop.  Do not touch live camera/proxy tensors here.
    print("[CKPT]: using frozen perception target from pre-navigation CPU snapshot", flush=True)
    target_xy_w = torch.tensor([[float(frozen_target_xy_w[0]), float(frozen_target_xy_w[1])]], dtype=torch.float32, device=device)
    target_local_xy = target_xy_w[0] - scene.env_origins[0, 0:2].to(device=device)

    lidar_memory_points_w: torch.Tensor | None = None
    print("[CKPT]: building initial planning grid", flush=True)
    grid, lidar_grid, footprint_grid, lidar_obs, raw_lidar_grid, instant_lidar_grid, lidar_memory_points_w = build_planning_grid(
        scene=scene,
        proxy=proxy,
        lidar_obs_cfg=lidar_obs_cfg,
        grid_cfg=grid_cfg,
        device=device,
        lidar_memory_points_w=lidar_memory_points_w,
        update_lidar_memory=True,
    )
    print("[CKPT]: initial planning grid built", flush=True)
    print_lidar_raw_diagnostics(lidar_obs, label="initial snapshot")
    # Phase-1 navigation module: same A* + monotonic waypoint controller used by
    # this smoke test, now kept in src/g1_nav_rl/navigation/simple_nav.py.
    nav_cfg = SimpleNavConfig(
        waypoint_reached_radius=float(args_cli.waypoint_reached_radius),
        k_xy=float(args_cli.k_xy),
        k_yaw=float(args_cli.k_yaw),
        max_vx=float(args_cli.max_vx),
        max_vy=float(args_cli.max_vy),
        max_wz=float(args_cli.max_wz),
        allow_backward=bool(args_cli.allow_backward),
        success_radius=success_radius,
    )
    navigator = SimpleNavigator(grid_cfg=grid_cfg, astar_cfg=astar_cfg, config=nav_cfg)
    nav_runtime = NavRuntime(
        navigator,
        NavRuntimeConfig(
            enable_replanning=bool(args_cli.enable_replanning),
            replan_every=int(args_cli.replan_every),
            path_check_ignore_distance=float(args_cli.path_check_ignore_distance),
            path_check_lookahead=float(args_cli.path_check_lookahead),
            allow_goal_relocation=True,
            stop_if_replan_fails=bool(args_cli.stop_if_replan_fails),
        ),
    )
    safety_filter = LidarSafetyFilter(
        LidarSafetyConfig(
            enabled=bool(args_cli.enable_safety),
            clearance_percentile=float(args_cli.safety_clearance_percentile),
            max_clearance=float(lidar_obs_cfg.max_range),
            front_slow_distance=float(args_cli.safety_front_slow_distance),
            front_stop_distance=float(args_cli.safety_front_stop_distance),
            front_backup_distance=float(args_cli.safety_front_backup_distance),
            side_open_distance=float(args_cli.safety_side_open_distance),
            side_slow_distance=float(args_cli.safety_side_slow_distance),
            side_stop_distance=float(args_cli.safety_side_stop_distance),
            side_critical_distance=float(args_cli.safety_side_critical_distance),
            slow_min_scale=float(args_cli.safety_slow_min_scale),
            backup_speed=float(args_cli.safety_backup_speed),
            backup_lateral_speed=float(args_cli.safety_backup_lateral_speed),
            sidestep_speed=float(args_cli.safety_sidestep_speed),
            stuck_window_steps=int(args_cli.safety_stuck_window_steps),
            stuck_recovery_steps=int(args_cli.safety_stuck_recovery_steps),
            max_vx=float(args_cli.max_vx),
            max_vy=float(args_cli.max_vy),
            max_wz=float(args_cli.max_wz),
        )
    )
    rgbd_safety_filter = RgbdSafetyFilter(
        RgbdSafetyConfig(
            enabled=bool(args_cli.enable_rgbd_safety),
            vertical_band=(float(args_cli.rgbd_safety_vertical_band[0]), float(args_cli.rgbd_safety_vertical_band[1])),
            center_width_frac=float(args_cli.rgbd_safety_center_width_frac),
            depth_min=float(CFG.get("observations", {}).get("depth_min", 0.0)),
            depth_max=float(CFG.get("observations", {}).get("depth_max", 10.0)),
            max_clearance=float(CFG.get("observations", {}).get("depth_max", 10.0)),
            clearance_percentile=float(args_cli.rgbd_safety_clearance_percentile),
            front_slow_distance=float(args_cli.rgbd_safety_front_slow_distance),
            front_stop_distance=float(args_cli.rgbd_safety_front_stop_distance),
            front_backup_distance=float(args_cli.rgbd_safety_front_backup_distance),
            side_open_distance=float(args_cli.rgbd_safety_side_open_distance),
            side_slow_distance=float(args_cli.rgbd_safety_side_slow_distance),
            side_stop_distance=float(args_cli.rgbd_safety_side_stop_distance),
            side_critical_distance=float(args_cli.rgbd_safety_side_critical_distance),
            slow_min_scale=float(args_cli.rgbd_safety_slow_min_scale),
            backup_speed=float(args_cli.rgbd_safety_backup_speed),
            backup_lateral_speed=float(args_cli.rgbd_safety_backup_lateral_speed),
            sidestep_speed=float(args_cli.rgbd_safety_sidestep_speed),
            max_vx=float(args_cli.max_vx),
            max_vy=float(args_cli.max_vy),
            max_wz=float(args_cli.max_wz),
        )
    )
    nav_target = NavTarget2D(
        x=float(target_xy_w[0, 0].detach().cpu().item()),
        y=float(target_xy_w[0, 1].detach().cpu().item()),
    )
    nav_start_pose = NavPose2D(
        x=start_xy_w[0],
        y=start_xy_w[1],
        yaw=float(proxy.yaw[0].detach().cpu().item()),
    )
    print("[CKPT]: running initial NavRuntime plan", flush=True)
    nav_plan = nav_runtime.plan(nav_start_pose, nav_target, grid[0]).plan
    print("[CKPT]: initial NavRuntime plan returned", flush=True)
    astar_result: GridAStarResult = nav_plan.astar_result
    path_xy_w = nav_runtime.current_path_xy_w()

    lidar_occupied = int((lidar_grid[0] > 0).detach().cpu().sum().item())
    footprint_occupied = int((footprint_grid[0] > 0).detach().cpu().sum().item())
    occupied = int((grid[0] > 0).detach().cpu().sum().item())
    valid_hits = int(lidar_obs["lidar_valid"][0].detach().cpu().sum().item()) if lidar_obs["lidar_valid"].numel() else 0

    print("[INFO]: Starting YOLO/person-target + LiDAR navigation smoke test...", flush=True)
    print(f"[INFO]: scenario: {args_cli.scenario} — {SCENARIO['description']}", flush=True)
    print(f"[INFO]: target_xy_w env0: {target_xy_w[0].detach().cpu().tolist()}", flush=True)
    print(f"[INFO]: perception target source: {perception_meta.get('detection_source', 'unknown')}", flush=True)
    if perception_meta:
        print(f"[INFO]: perception target meta: {perception_meta}", flush=True)
    print(f"[INFO]: target reached radius: {success_radius:.3f} (hand_length={args_cli.hand_length:.3f})", flush=True)
    print(f"[INFO]: body_radius={body_radius:.3f}, safety_margin={safety_margin:.3f}, inflation={inflation_radius:.3f}", flush=True)
    print(f"[INFO]: forced LiDAR-visible obstacle height: {float(args_cli.force_lidar_visible_obstacle_height):.3f}", flush=True)
    print(f"[INFO]: planning grid source: {GRID_SOURCE}", flush=True)
    print(f"[INFO]: lidar offset pos: {CFG['lidar']['offset']['pos']}, warmup_steps={max(1, int(args_cli.lidar_warmup_steps))}", flush=True)
    print(f"[INFO]: occupancy grid: H={grid_cfg.height}, W={grid_cfg.width}, resolution={grid_cfg.resolution:.3f}", flush=True)
    print(
        "[INFO]: grid images save a single fixed world-frame occupancy map "
        "(unknown/free/occupied + start/target/robot/covered trajectory/current plan)",
        flush=True,
    )
    memory_count = 0 if lidar_memory_points_w is None else int(torch.isfinite(lidar_memory_points_w[0]).all(dim=-1).detach().cpu().sum().item())
    raw_lidar_occupied = int((raw_lidar_grid[0] > 0).detach().cpu().sum().item())
    instant_lidar_occupied = int((instant_lidar_grid[0] > 0).detach().cpu().sum().item())
    print(
        f"[INFO]: occupied cells: {occupied} (lidar={lidar_occupied}, raw_lidar={raw_lidar_occupied}, "
        f"instant_conservative_lidar={instant_lidar_occupied}, footprint={footprint_occupied}), "
        f"lidar valid hits={valid_hits}, memory_points={memory_count}",
        flush=True,
    )
    print(f"[INFO]: lidar: {summarize_lidar_observation(lidar_obs)}", flush=True)
    print(
        f"[INFO]: A*: one-shot plan, diagonal={astar_cfg.allow_diagonal}, "
        f"heuristic_weight={astar_cfg.heuristic_weight}, max_expansions={astar_cfg.max_expansions}",
        flush=True,
    )
    print("[INFO]: controller: direct waypoint mapping to [vx, vy, wz]", flush=True)
    print(
        f"[INFO]: replanning={'enabled' if args_cli.enable_replanning else 'disabled'}, "
        "shortcutting=disabled, command_smoothing=disabled, local_escape=disabled",
        flush=True,
    )
    print(
        f"[INFO]: lidar safety filter: {'enabled' if args_cli.enable_safety else 'disabled'} "
        f"(slow={float(args_cli.safety_front_slow_distance):.2f}, "
        f"stop={float(args_cli.safety_front_stop_distance):.2f}, "
        f"backup={float(args_cli.safety_front_backup_distance):.2f}, "
        f"side_open={float(args_cli.safety_side_open_distance):.2f}, "
        f"side_slow={float(args_cli.safety_side_slow_distance):.2f}, "
        f"side_stop={float(args_cli.safety_side_stop_distance):.2f}, "
        f"side_critical={float(args_cli.safety_side_critical_distance):.2f})",
        flush=True,
    )
    print(
        f"[INFO]: lidar conservative obstacle model: thickness={float(args_cli.lidar_obstacle_thickness):.3f}, "
        f"lateral_padding={float(args_cli.lidar_obstacle_lateral_padding):.3f}, map_mode={args_cli.lidar_map_mode}, "
        f"map_update_every={int(args_cli.map_update_every)}",
        flush=True,
    )
    print(
        f"[INFO]: RGB-D/depth safety filter: {'enabled' if args_cli.enable_rgbd_safety else 'disabled'} "
        f"(slow={float(args_cli.rgbd_safety_front_slow_distance):.2f}, "
        f"stop={float(args_cli.rgbd_safety_front_stop_distance):.2f}, "
        f"backup={float(args_cli.rgbd_safety_front_backup_distance):.2f}, "
        f"side_slow={float(args_cli.rgbd_safety_side_slow_distance):.2f}, "
        f"side_stop={float(args_cli.rgbd_safety_side_stop_distance):.2f}, "
        f"side_critical={float(args_cli.rgbd_safety_side_critical_distance):.2f})",
        flush=True,
    )
    print(f"[INFO]: proxy dynamics: {'YAML config' if args_cli.use_config_proxy_dynamics else 'direct baseline mode'}", flush=True)
    print(f"[INFO]: max_steps={args_cli.max_steps}, print_every={args_cli.print_every}, save_every={args_cli.save_every}", flush=True)
    for i, obs in enumerate(OBSTACLES):
        print(
            f"  obstacle[{i}]: shape={obs.get('shape')} pos={obs.get('pos')} "
            f"size/radius={obs.get('size', obs.get('radius'))} yaw={obs.get('yaw', 0.0)}",
            flush=True,
        )
    print(f"[INFO]: lidar mesh paths: {resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), num_envs)}", flush=True)
    print(
        f"[INFO]: A* found={astar_result.found}, reason={astar_result.reason}, "
        f"expansions={astar_result.expansions}, cost={astar_result.cost:.3f}",
        flush=True,
    )
    print(f"[INFO]: A* path length cells={len(astar_result.path_cells)}, waypoints={len(path_xy_w)}", flush=True)
    print(f"[INFO]: A* body path length={path_length_xy(astar_result.path_xy_b):.3f}, world path length={path_length_xy(path_xy_w):.3f}", flush=True)

    if OBSTACLES and footprint_occupied <= 0:
        raise RuntimeError(
            "Known scenario-footprint rasterization produced zero occupied cells. "
            "This is a script bug: smoke_test_nav.py-style geometry was not converted into the grid."
        )

    target_xy_w_env0 = (
        float(target_xy_w[0, 0].detach().cpu().item()),
        float(target_xy_w[0, 1].detach().cpu().item()),
    )
    env_origin_xy_w: PointXY = (
        float(scene.env_origins[0, 0].detach().cpu().item()),
        float(scene.env_origins[0, 1].detach().cpu().item()),
    )
    safety_zone_radius_m = float(body_radius + safety_margin)
    world_grid_cfg = make_fixed_world_grid_cfg(grid_cfg, start_xy_w)
    print(
        f"[INFO]: fixed world-map extents: x=[{world_grid_cfg.x_min:.3f}, {world_grid_cfg.x_max:.3f}], "
        f"y=[{world_grid_cfg.y_min:.3f}, {world_grid_cfg.y_max:.3f}]",
        flush=True,
    )
    print(
        f"[INFO]: visualization: one world-frame occupancy_map_*.png per saved frame; "
        f"setup obstacle contours=red dashed; safety_zone_radius={safety_zone_radius_m:.3f}; "
        f"viz_free_update_every={int(args_cli.viz_free_update_every)} (0 means saved frames only)",
        flush=True,
    )

    # Accumulated free-space visualization map.  It is deliberately separate
    # from the planner: it only makes the single saved occupancy image easier to
    # interpret by keeping previously ray-carved free cells in the fixed world frame.
    free_state_w = build_world_free_state_from_lidar(
        lidar_obs,
        proxy=proxy,
        world_grid_cfg=world_grid_cfg,
        max_range=float(lidar_obs_cfg.max_range),
    )

    empty_lidar_is_allowed = bool(SCENARIO.get("allow_empty_lidar", False)) or bool(args_cli.allow_empty_lidar)
    if GRID_SOURCE == "lidar" and OBSTACLES and valid_hits == 0 and not empty_lidar_is_allowed:
        raise RuntimeError(
            "LiDAR-only planning produced zero valid hits even though this scenario has obstacles. "
            "This usually means the RayCaster scan plane is missing/grazing the mesh, or the mesh path is wrong. "
            "Try --lidar_z_offset 0.10 or 0.05, keep --force_lidar_visible_obstacle_height at its default, increase --lidar_warmup_steps, or run without --lidar_only "
            "to validate the planner against known scenario footprints."
        )

    expected_planner_found = bool(SCENARIO.get("expect_planner_found", SCENARIO.get("expect_success", True)))
    if args_cli.assert_planner:
        if expected_planner_found and not astar_result.found:
            raise RuntimeError(f"A* planner failed: {astar_result.reason}")
        if not expected_planner_found and astar_result.found:
            raise RuntimeError(
                f"A* planner unexpectedly found a path in scenario {args_cli.scenario!r}: "
                f"reason={astar_result.reason}, cost={astar_result.cost:.3f}"
            )
    if not astar_result.found or len(path_xy_w) < 2:
        if not bool(args_cli.no_save_grid):
            save_world_occupancy_debug_image(
                grid,
                free_state_w=free_state_w,
                proxy=proxy,
                local_grid_cfg=grid_cfg,
                world_grid_cfg=world_grid_cfg,
                start_xy_w=start_xy_w,
                target_xy_w=target_xy_w_env0,
                robot_xy_w=start_xy_w,
                covered_trajectory_xy_w=trajectory_w,
                active_path_xy_w=[],
                obstacles=OBSTACLES,
                env_origin_xy_w=env_origin_xy_w,
                safety_zone_radius_m=safety_zone_radius_m,
                out_dir=grid_dir,
                step_idx=0,
                occupancy_threshold=float(astar_cfg.occupancy_threshold),
            )
        print("[WARN]: No executable path found. Exiting before control loop.", flush=True)
        return

    records: list[StepRecord] = []
    min_scenario_clearance = math.inf
    min_margin_slack = math.inf
    min_grid_edge_clearance = math.inf
    target_reached = False
    reached_step: int | None = None
    nonstraight_command_count = 0
    replan_count = 0
    safety_override_count = 0
    rgbd_safety_override_count = 0

    waypoint_index = 1
    save_grid = not bool(args_cli.no_save_grid)
    print_every = max(1, int(args_cli.print_every))
    save_every = max(1, int(args_cli.save_every))

    print("[CKPT]: entering control loop", flush=True)
    for step_idx in range(int(args_cli.max_steps) + 1):
        if step_idx == 0:
            print("[CKPT]: control loop step 0 entered", flush=True)
        if not simulation_app.is_running():
            break

        # Rebuild LiDAR grid for diagnostics/images only. The A* path remains
        # the initial one-shot path unless replanning is enabled.
        should_save_frame = save_grid and (step_idx == 0 or step_idx % save_every == 0)
        viz_free_update_every = max(0, int(args_cli.viz_free_update_every))
        should_update_free_viz = should_save_frame or (viz_free_update_every > 0 and step_idx % viz_free_update_every == 0)
        should_update_memory = (
            str(args_cli.lidar_map_mode) == "memory"
            and (step_idx == 0 or (int(args_cli.map_update_every) > 0 and step_idx % int(args_cli.map_update_every) == 0))
        )
        live_grid, live_lidar_grid, live_footprint_grid, live_lidar_obs, live_raw_lidar_grid, live_instant_lidar_grid, lidar_memory_points_w = build_planning_grid(
            scene=scene,
            proxy=proxy,
            lidar_obs_cfg=lidar_obs_cfg,
            grid_cfg=grid_cfg,
            device=device,
            lidar_memory_points_w=lidar_memory_points_w,
            update_lidar_memory=should_update_memory,
        )
        if should_update_free_viz:
            current_free_state_w = build_world_free_state_from_lidar(
                live_lidar_obs,
                proxy=proxy,
                world_grid_cfg=world_grid_cfg,
                max_range=float(lidar_obs_cfg.max_range),
            )
            free_state_w = torch.maximum(free_state_w, current_free_state_w)

        x = float(proxy.pos_w[0, 0].detach().cpu().item())
        y = float(proxy.pos_w[0, 1].detach().cpu().item())
        yaw = float(proxy.yaw[0].detach().cpu().item())
        target_distance = float(torch.linalg.norm(target_xy_w[0] - proxy.pos_w[0, 0:2]).detach().cpu().item())

        current_nav_pose = NavPose2D(x=x, y=y, yaw=yaw)
        nav_step = nav_runtime.step(
            current_nav_pose,
            nav_target,
            live_grid[0],
            step_index=step_idx,
        )
        if nav_step.replan_attempted:
            current_plan = nav_step.plan
            blocked_xy_b = nav_step.path_conflict.blocked_xy_b
            blocked_msg = "" if blocked_xy_b is None else f" blocked_xy_b=({blocked_xy_b[0]:.3f},{blocked_xy_b[1]:.3f})"
            if nav_step.replan_accepted and current_plan is not None:
                new_result: GridAStarResult = current_plan.astar_result
                print(
                    f"[INFO]: replan accepted at step={step_idx}: reason={nav_step.replan_reason},{blocked_msg} "
                    f"new_waypoints={len(current_plan.path_xy_w)}, cost={new_result.cost:.3f}, expansions={new_result.expansions}",
                    flush=True,
                )
            elif nav_step.replan_failed:
                reason_text = "unknown"
                current_plan = nav_step.plan
                if current_plan is not None:
                    reason_text = current_plan.reason
                print(
                    f"[WARN]: replan failed at step={step_idx}: reason={nav_step.replan_reason}, planner_reason={reason_text}",
                    flush=True,
                )
        if nav_step.should_stop:
            proxy.set_command(torch.zeros((num_envs, 3), dtype=torch.float32, device=device))
            break

        nav_command = nav_step.command
        path_xy_w = nav_runtime.current_path_xy_w()
        replan_count = nav_runtime.replan_count
        waypoint_index = nav_command.waypoint_index
        waypoint_xy = nav_command.waypoint_xy_w if nav_command.waypoint_xy_w is not None else (x, y)
        tangent_yaw = nav_command.path_yaw
        nominal_vx, nominal_vy, nominal_wz = nav_command.command.tuple
        safety_result = safety_filter.apply_from_lidar_obs(
            nav_command.command,
            live_lidar_obs,
            pose=current_nav_pose,
            target=nav_target,
            step_index=step_idx,
        )
        safety_info = safety_result.info

        rgbd_depth = build_rgbd_depth_observation(scene)
        if rgbd_depth is not None:
            rgbd_safety_result = rgbd_safety_filter.apply(safety_result.command, depth=rgbd_depth)
        else:
            rgbd_safety_result = RgbdSafetyFilter.disabled_result(safety_result.command)
        rgbd_safety_info = rgbd_safety_result.info

        vx, vy, wz = rgbd_safety_result.command.tuple
        waypoint_xy_b = nav_command.waypoint_xy_b if nav_command.waypoint_xy_b is not None else (0.0, 0.0)

        scenario_clearance = min_scenario_edge_clearance(x, y, OBSTACLES, body_radius)
        margin_slack = scenario_clearance - safety_margin if math.isfinite(scenario_clearance) else math.inf
        grid_clearance = compute_obstacle_clearance_from_grid(
            live_grid[0],
            grid_cfg,
            body_radius=body_radius,
            occupancy_threshold=astar_cfg.occupancy_threshold,
        )
        grid_edge_clearance = float(grid_clearance.get("edge_clearance", math.inf))

        min_scenario_clearance = min(min_scenario_clearance, scenario_clearance)
        min_margin_slack = min(min_margin_slack, margin_slack)
        min_grid_edge_clearance = min(min_grid_edge_clearance, grid_edge_clearance)

        live_occupied = int((live_grid[0] > 0).detach().cpu().sum().item())
        live_valid_hits = int(live_lidar_obs["lidar_valid"][0].detach().cpu().sum().item()) if live_lidar_obs["lidar_valid"].numel() else 0
        record = StepRecord(
            step=step_idx,
            x=x,
            y=y,
            yaw=yaw,
            waypoint_index=waypoint_index,
            waypoint_x=waypoint_xy[0],
            waypoint_y=waypoint_xy[1],
            target_distance=target_distance,
            scenario_edge_clearance=scenario_clearance,
            scenario_margin_slack=margin_slack,
            grid_edge_clearance=grid_edge_clearance,
            vx=vx,
            vy=vy,
            wz=wz,
            nominal_vx=nominal_vx,
            nominal_vy=nominal_vy,
            nominal_wz=nominal_wz,
            lidar_valid=live_valid_hits,
            occupied_cells=live_occupied,
            safety_action=safety_info.action,
            safety_front_clearance=safety_info.front_clearance,
            safety_left_clearance=safety_info.left_clearance,
            safety_right_clearance=safety_info.right_clearance,
            safety_override=safety_info.override_active,
            safety_scale=safety_info.scale,
            rgbd_safety_action=rgbd_safety_info.action,
            rgbd_safety_front_depth=rgbd_safety_info.front_clearance,
            rgbd_safety_left_depth=rgbd_safety_info.left_clearance,
            rgbd_safety_right_depth=rgbd_safety_info.right_clearance,
            rgbd_safety_override=rgbd_safety_info.override_active,
            rgbd_safety_scale=rgbd_safety_info.scale,
        )
        records.append(record)

        should_print = step_idx == 0 or step_idx % print_every == 0 or target_distance <= success_radius
        if should_print:
            print_step(
                record=record,
                total_waypoints=len(path_xy_w),
                waypoint_xy_b=waypoint_xy_b,
                success_radius=success_radius,
                safety_margin=safety_margin,
            )

        current_xy_w = (x, y)
        if not trajectory_w or math.hypot(current_xy_w[0] - trajectory_w[-1][0], current_xy_w[1] - trajectory_w[-1][1]) > 1.0e-4:
            trajectory_w.append(current_xy_w)

        if should_save_frame:
            active_path_w = path_xy_w[waypoint_index:] if waypoint_index < len(path_xy_w) else []
            save_world_occupancy_debug_image(
                live_grid,
                free_state_w=free_state_w,
                proxy=proxy,
                local_grid_cfg=grid_cfg,
                world_grid_cfg=world_grid_cfg,
                start_xy_w=start_xy_w,
                target_xy_w=target_xy_w_env0,
                robot_xy_w=current_xy_w,
                covered_trajectory_xy_w=trajectory_w,
                active_path_xy_w=active_path_w,
                obstacles=OBSTACLES,
                env_origin_xy_w=env_origin_xy_w,
                safety_zone_radius_m=safety_zone_radius_m,
                out_dir=grid_dir,
                step_idx=step_idx,
                occupancy_threshold=float(astar_cfg.occupancy_threshold),
            )

        if target_distance <= success_radius:
            target_reached = True
            reached_step = step_idx
            proxy.set_command(torch.zeros((num_envs, 3), dtype=torch.float32, device=device))
            break

        if scenario_clearance < 0.0:
            print(f"[WARN]: collision detected at step={step_idx}: scenario_clearance={scenario_clearance:.3f} m", flush=True)
            break

        if abs(vy) > 0.05 or abs(wz) > 0.05:
            nonstraight_command_count += 1
        if bool(safety_info.override_active):
            safety_override_count += 1
        if bool(rgbd_safety_info.override_active):
            rgbd_safety_override_count += 1

        cmd = torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=device).repeat(num_envs, 1)
        proxy.set_command(cmd)
        proxy.step(sim_dt)
        sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    csv_path = output_dir / "nav_lidar_trace.csv"
    write_records_csv(csv_path, records)

    first_dist = records[0].target_distance if records else math.inf
    last_dist = records[-1].target_distance if records else math.inf
    min_dist = min((r.target_distance for r in records), default=math.inf)

    print("=" * 80, flush=True)
    print(f"[INFO]: target distance first/last/min: {first_dist:.3f} / {last_dist:.3f} / {min_dist:.3f}", flush=True)
    if math.isfinite(min_scenario_clearance):
        print(f"[INFO]: min scenario edge clearance: {min_scenario_clearance:.3f}", flush=True)
        print(f"[INFO]: min scenario safety-margin slack: {min_margin_slack:.3f}", flush=True)
    else:
        print("[INFO]: min scenario edge clearance: inf", flush=True)
        print("[INFO]: min scenario safety-margin slack: inf", flush=True)
    if math.isfinite(min_grid_edge_clearance):
        print(f"[INFO]: min grid edge clearance: {min_grid_edge_clearance:.3f}", flush=True)
    else:
        print("[INFO]: min grid edge clearance: inf", flush=True)
    print(f"[INFO]: target reached: {target_reached} step={reached_step}", flush=True)
    print(f"[INFO]: nonstraight command count: {nonstraight_command_count}", flush=True)
    print(f"[INFO]: replan count: {replan_count}", flush=True)
    print(f"[INFO]: safety override count: {safety_override_count}", flush=True)
    print(f"[INFO]: RGB-D safety override count: {rgbd_safety_override_count}", flush=True)
    print(f"[INFO]: wrote trace: {csv_path}", flush=True)
    print("[INFO]: YOLO/person target + LiDAR navigation smoke test complete.", flush=True)

    if args_cli.assert_planner:
        if len(records) < 2:
            raise RuntimeError("No target distance history was collected.")
        if bool(SCENARIO.get("expect_success", True)) and not (min_dist < first_dist - 0.05):
            raise RuntimeError(
                "Expected one-shot A* path following to reduce target distance by at least 0.05 m. "
                f"first={first_dist:.3f}, min={min_dist:.3f}"
            )
        if bool(SCENARIO.get("expect_avoidance", False)) and nonstraight_command_count <= 0:
            raise RuntimeError(f"Expected at least one lateral/yaw non-straight command for scenario {args_cli.scenario!r}.")

    if args_cli.assert_no_collision and math.isfinite(min_scenario_clearance) and min_scenario_clearance < 0.0:
        raise RuntimeError(f"collision: min_scenario_clearance={min_scenario_clearance:.3f}")

    if args_cli.assert_margin and math.isfinite(min_margin_slack) and min_margin_slack < -float(args_cli.margin_tolerance):
        raise RuntimeError(
            "executed trajectory cut into planning margin: "
            f"min_slack={min_margin_slack:.3f}, tolerance={float(args_cli.margin_tolerance):.3f}"
        )

    if args_cli.assert_success and bool(SCENARIO.get("expect_success", True)) and not target_reached:
        raise RuntimeError(f"target not reached: last_distance={last_dist:.3f}, success_radius={success_radius:.3f}")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=list(CFG["scene"]["viewer"]["eye"]), target=list(CFG["scene"]["viewer"]["target"]))

    ensure_proxy_roots_exist(num_envs=int(CFG["scene"]["num_envs"]))
    create_raycast_aggregate_mesh(num_envs=int(CFG["scene"]["num_envs"]))

    print(f"[INFO]: script version: {SCRIPT_VERSION}", flush=True)
    print("[INFO]: Creating YOLO/person target + LiDAR navigation scene...", flush=True)
    print(f"[INFO]: scenario: {args_cli.scenario}", flush=True)
    print(f"[INFO]: person target root xy: ({PERSON_X:.3f}, {PERSON_Y:.3f})", flush=True)
    print(f"[INFO]: person source: {str(args_cli.person_asset_usd).strip() or 'proxy_humanoid'}", flush=True)
    print(f"[INFO]: requested grid source: {GRID_SOURCE}", flush=True)
    print(
        f"[INFO]: RayCaster mesh mode: {RAYCAST_MESH_MODE} "
        f"(using aggregate={USE_RAYCAST_AGGREGATE_MESH})",
        flush=True,
    )
    if any(bool(obs.get("blocks_map_y", False)) for obs in OBSTACLES):
        print(
            f"[INFO]: map-blocking obstacle y-span expanded to configured map y-range "
            f"[{float(args_cli.y_min):.3f}, {float(args_cli.y_max):.3f}]",
            flush=True,
        )
    print(f"[INFO]: RayCaster attachment prim path: {CFG['lidar']['prim_path']}", flush=True)
    print(f"[INFO]: RayCaster mesh prim paths: {resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), int(CFG['scene']['num_envs']))}", flush=True)
    print(f"[INFO]: effective obstacle specs after visibility patch:", flush=True)
    for i, obs in enumerate(OBSTACLES):
        print(
            f"  obstacle[{i}]: name={obs.get('name')} shape={obs.get('shape')} "
            f"pos={obs.get('pos')} size={obs.get('size', None)} radius={obs.get('radius', None)} "
            f"height={obs.get('height', None)} yaw={obs.get('yaw', 0.0)}",
            flush=True,
        )

    scene_cfg = G1NavLidarSceneCfg(num_envs=int(CFG["scene"]["num_envs"]), env_spacing=float(CFG["scene"]["env_spacing"]))
    scene = InteractiveScene(scene_cfg)
    add_referenced_person_asset()
    print_stage_prim_diagnostics(num_envs=int(CFG["scene"]["num_envs"]), label="before sim.reset()")

    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print_stage_prim_diagnostics(num_envs=int(CFG["scene"]["num_envs"]), label="after sim.reset()")
    print("[INFO]: Setup complete.", flush=True)
    print("[INFO]: Capturing one CPU RGB-D snapshot for initial perception target...", flush=True)
    frozen_target_xy_w, perception_meta = extract_perception_target_from_cpu_snapshot(scene, sim)
    print("[INFO]: Frozen perception target ready; entering LiDAR navigation loop.", flush=True)

    run_simulator(sim, scene, frozen_target_xy_w, perception_meta)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[ERROR]: smoke_test_yolo_person_nav failed with {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
