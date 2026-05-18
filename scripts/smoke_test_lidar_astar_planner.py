#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

# -----------------------------------------------------------------------------
# Built-in scenario library
# -----------------------------------------------------------------------------
# Obstacles are specified in the env-local/world frame used by the smoke tests:
# +x forward from the proxy start, +y left, z is height.
# RayCaster currently consumes static mesh paths; these scenarios are intentionally
# simple and static.
SCENARIOS: dict[str, dict[str, Any]] = {
    "clear": {
        "description": "No blocking obstacle; a small marker is far off the direct A->B line.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": False,
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
        "description": "Single rectangular box centered on the direct A->B line.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": True,
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
    "wide_box": {
        "description": "Wide obstacle across y; the path should choose a side route.",
        "target_pos": [4.5, 0.0, 0.08],
        "expect_avoidance": True,
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
        "description": "Rotated rectangular obstacle centered near the A->B line.",
        "target_pos": [4.5, 0.0, 0.08],
        "expect_avoidance": True,
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
    "offset_left": {
        "description": "Obstacle is front-left; planner should usually keep right/center.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": True,
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
        "description": "Obstacle is front-right; planner should usually keep left/center.",
        "target_pos": [4.0, 0.0, 0.08],
        "expect_avoidance": True,
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
    "multi": {
        "description": "Two separated rectangular obstacles leaving a navigable gap/side route.",
        "target_pos": [5.0, 0.0, 0.08],
        "expect_avoidance": True,
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
        "description": "A wide wall blocks the map; A* may report no path or fail to reach target.",
        "target_pos": [4.5, 0.0, 0.08],
        "expect_avoidance": False,
        "expect_success": False,
        "obstacles": [
            {
                "name": "BlockingWall",
                "shape": "box",
                "pos": [2.2, 0.0, 0.35],
                "size": [0.45, 5.0, 0.70],
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
        "Smoke test for LiDAR occupancy-grid A* planning. Static RayCaster hits "
        "are transformed into the moving proxy frame; A* proposes a body-frame "
        "waypoint path; a simple follower commands the proxy."
    )
)
parser.add_argument(
    "--config",
    type=str,
    default=str(PROJECT_ROOT / "configs" / "robot" / "g1_command_proxy.yaml"),
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum simulation frames to run. Default is intentionally long enough for obstacle-detour tests.")
parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="front_box")
parser.add_argument("--list_scenarios", action="store_true", help="List built-in scenarios and exit before launching Isaac Sim.")
parser.add_argument("--target_success_radius", type=float, default=None)
parser.add_argument("--assert_planner", action="store_true", help="Assert target progress, path availability, and expected avoidance.")
parser.add_argument("--assert_success", action="store_true", help="Assert that the target success radius is reached.")
parser.add_argument(
    "--lidar_only",
    action="store_true",
    help=(
        "Plan from RayCaster hit points only. By default, this smoke test also "
        "rasterizes the known scenario obstacle footprints into the local grid, "
        "so A* sees the full static obstacle rather than only the face visible "
        "to the current static RayCaster."
    ),
)
parser.add_argument(
    "--disable_shortcut",
    action="store_true",
    help="Follow the raw grid A* path instead of the line-of-sight shortcut path. Useful for testing whether shortcutting cuts too close to obstacles.",
)
parser.add_argument(
    "--disable_clearance_safety",
    action="store_true",
    help="Disable the clearance-aware command limiter while keeping path planning unchanged.",
)
parser.add_argument(
    "--disable_command_smoothing",
    action="store_true",
    help="Disable command/momentum smoothing. Useful for testing whether smoothing prevents turns near obstacles.",
)
parser.add_argument("--safety_margin", type=float, default=None, help="Override grid_astar.safety_margin for planning inflation tests.")
parser.add_argument("--replan_period", type=int, default=None, help="Override grid_astar.path_memory.replan_period_steps.")
parser.add_argument("--path_switch_weight", type=float, default=None, help="Override grid_astar.path_memory.path_switch_weight.")
parser.add_argument("--lookahead_distance", type=float, default=None, help="Override grid_astar.path_follower.lookahead_distance.")
parser.add_argument("--smoothing_alpha", type=float, default=None, help="Override grid_astar.command_smoothing.alpha.")
parser.add_argument("--print_every", type=int, default=None, help="Override grid_astar.debug.print_every for less verbose logs.")
parser.add_argument("--save_every", type=int, default=None, help="Override grid_astar.debug.save_every for grid-debug images.")
parser.add_argument("--no_save_grid", action="store_true", help="Disable saving A* grid debug images.")
parser.add_argument("--max_delta_vy", type=float, default=None, help="Override grid_astar.command_smoothing.max_delta_vy.")
parser.add_argument("--max_delta_wz", type=float, default=None, help="Override grid_astar.command_smoothing.max_delta_wz.")
AppLauncher.add_app_launcher_args(parser)

# Parse early enough to support --list_scenarios without launching Isaac.
if "--list_scenarios" in sys.argv:
    tmp = argparse.ArgumentParser(add_help=False)
    tmp.add_argument("--list_scenarios", action="store_true")
    args_tmp, _ = tmp.parse_known_args()
    print("Available LiDAR A* planner scenarios:")
    for name, spec in sorted(SCENARIOS.items()):
        print(f"  {name:14s} {spec['description']}")
    raise SystemExit(0)

args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Isaac Lab imports after app launch
# -----------------------------------------------------------------------------
import torch  # noqa: E402
import omni.usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import RayCasterCfg, patterns  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from g1_nav_rl.mapping.occupancy_grid import (  # noqa: E402
    OccupancyGridCfg,
    occupancy_grid_to_uint8_image,
    rasterize_points_to_occupancy_grid,
)
from g1_nav_rl.planning.grid_astar import (  # noqa: E402
    GridAStarCfg,
    PathFollowerCfg,
    follow_body_frame_path,
    plan_grid_path_xy,
    shortcut_grid_path_result,
    xy_to_grid_cell,
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
from g1_nav_rl.utils.geometry import rotate_world_to_body_2d, target_vector_body_frame  # noqa: E402


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml(args_cli.config)
if args_cli.num_envs is not None:
    CFG["scene"]["num_envs"] = int(args_cli.num_envs)

# Lightweight CLI overrides for ablation tests. These intentionally patch the
# loaded YAML before configs are constructed, so the printed runtime config shows
# exactly what is being tested.
grid_astar_cfg = CFG.setdefault("grid_astar", {})
if args_cli.safety_margin is not None:
    grid_astar_cfg["safety_margin"] = float(args_cli.safety_margin)
if args_cli.replan_period is not None:
    grid_astar_cfg.setdefault("path_memory", {})["replan_period_steps"] = int(args_cli.replan_period)
if args_cli.path_switch_weight is not None:
    grid_astar_cfg.setdefault("path_memory", {})["path_switch_weight"] = float(args_cli.path_switch_weight)
if args_cli.lookahead_distance is not None:
    grid_astar_cfg.setdefault("path_follower", {})["lookahead_distance"] = float(args_cli.lookahead_distance)
if args_cli.smoothing_alpha is not None:
    grid_astar_cfg.setdefault("command_smoothing", {})["alpha"] = float(args_cli.smoothing_alpha)
if args_cli.max_delta_vy is not None:
    grid_astar_cfg.setdefault("command_smoothing", {})["max_delta_vy"] = float(args_cli.max_delta_vy)
if args_cli.max_delta_wz is not None:
    grid_astar_cfg.setdefault("command_smoothing", {})["max_delta_wz"] = float(args_cli.max_delta_wz)
if args_cli.disable_command_smoothing:
    grid_astar_cfg.setdefault("command_smoothing", {})["enabled"] = False
if args_cli.disable_clearance_safety:
    grid_astar_cfg.setdefault("clearance_safety", {})["enabled"] = False

SCENARIO = SCENARIOS[args_cli.scenario]
OBSTACLES = list(SCENARIO.get("obstacles", []))
MAX_OBSTACLES = 6
if len(OBSTACLES) > MAX_OBSTACLES:
    raise ValueError(f"Scenario {args_cli.scenario!r} has {len(OBSTACLES)} obstacles; max supported is {MAX_OBSTACLES}.")

# Scenario overrides target and lidar mesh targets. MeshCuboidCfg/MeshCylinderCfg
# create the raycast target at {prim_path}/geometry/mesh.
CFG.setdefault("target", {})["pos"] = list(SCENARIO.get("target_pos", CFG.get("target", {}).get("pos", [4.0, 0.0, 0.08])))
for obs_idx, obs in enumerate(OBSTACLES):
    obs.setdefault("name", f"Obstacle{obs_idx}")
    obs["prim_path"] = f"{{ENV_REGEX_NS}}/{obs['name']}"
    obs["mesh_prim_path"] = f"{{ENV_REGEX_NS}}/{obs['name']}/geometry/mesh"
CFG.setdefault("lidar", {})["mesh_prim_paths"] = [obs["mesh_prim_path"] for obs in OBSTACLES]

# If a scenario ever has no obstacles, RayCaster has no mesh to initialize against.
# Keep built-in scenarios with at least one off-path marker instead.
if not CFG["lidar"]["mesh_prim_paths"]:
    raise ValueError("LiDAR RayCaster smoke test requires at least one static mesh obstacle/marker.")


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
            # Conservative fallback for older Isaac Lab builds without MeshCylinderCfg.
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


def compute_astar_cfg_with_body_radius(base_cfg: GridAStarCfg) -> tuple[GridAStarCfg, float, float, float]:
    body_size = CFG["proxy"]["body"]["size"]
    body_radius = 0.5 * math.hypot(float(body_size[0]), float(body_size[1]))
    safety_margin = float(CFG.get("grid_astar", {}).get("safety_margin", 0.15))
    effective_radius = body_radius + safety_margin
    # Respect explicit inflation if it is larger, but never allow it to be less
    # than body radius + planning margin.
    inflation_radius = max(float(base_cfg.inflation_radius), effective_radius)
    return (
        GridAStarCfg(
            occupancy_threshold=base_cfg.occupancy_threshold,
            inflation_radius=inflation_radius,
            allow_diagonal=base_cfg.allow_diagonal,
            heuristic_weight=base_cfg.heuristic_weight,
            max_expansions=base_cfg.max_expansions,
        ),
        body_radius,
        safety_margin,
        inflation_radius,
    )


def ensure_proxy_roots_exist(num_envs: int) -> None:
    stage = omni.usd.get_context().get_stage()
    for path in resolve_env_prim_paths(CFG["proxy"]["prim_path"], num_envs):
        proxy_xform = UsdGeom.Xform.Define(stage, path)
        ensure_canonical_xform_ops(proxy_xform.GetPrim())
        print(f"[INFO]: Ensured canonical proxy root exists: {path}", flush=True)


def print_stage_prim_diagnostics(num_envs: int) -> None:
    stage = omni.usd.get_context().get_stage()
    paths = [resolve_template_path(CFG["lidar"]["prim_path"], env_id=0), resolve_template_path(CFG["target"]["prim_path"], env_id=0)]
    for obs in OBSTACLES:
        paths.append(resolve_template_path(obs["prim_path"], env_id=0))
        paths.append(resolve_template_path(obs["mesh_prim_path"], env_id=0))
    print("[INFO]: Stage prim diagnostics before sim.reset():", flush=True)
    for path in dict.fromkeys(paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"  - {path}: INVALID", flush=True)
            continue
        is_mesh = bool(UsdGeom.Mesh(prim))
        is_xform = bool(UsdGeom.Xformable(prim))
        print(f"  - {path}: type={prim.GetTypeName()} is_mesh={is_mesh} is_xformable={is_xform}", flush=True)


def _xy_to_debug_pixel(xy_b: PointXY, grid_cfg: OccupancyGridCfg) -> tuple[int, int] | None:
    """Convert body-frame XY to image pixel coordinates ``(col, row)``."""
    row, col = xy_to_grid_cell(xy_b, grid_cfg)
    if 0 <= row < grid_cfg.height and 0 <= col < grid_cfg.width:
        return int(col), int(row)
    return None


def _draw_polyline_xy(
    draw: Any,
    points_xy_b: list[PointXY],
    grid_cfg: OccupancyGridCfg,
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    """Draw consecutive in-bounds path segments without connecting across clipped gaps."""
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


def save_astar_debug_image(
    grid: torch.Tensor,
    *,
    active_path_xy_b: list[PointXY],
    covered_path_xy_b: list[PointXY],
    start_xy_b: PointXY | None,
    target_xy_b: PointXY | None,
    robot_xy_b: PointXY,
    grid_cfg: OccupancyGridCfg,
    out_dir: Path,
    step_idx: int,
) -> None:
    """Save a local-grid A* debug image with unambiguous markers.

    Color convention:
      - grey: already covered robot trajectory, transformed into the current body frame
      - red: active path still being followed
      - yellow/cross: current robot position, always body-frame origin ``(0, 0)``
      - green: original start point transformed into the current body frame
      - blue: target transformed into the current body frame
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("[WARN]: Pillow is not installed. Skipping A* grid save.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image = occupancy_grid_to_uint8_image(grid, env_id=0).numpy()
    rgb = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(rgb)

    # Draw history first, then active route, then semantic markers.
    _draw_polyline_xy(draw, covered_path_xy_b, grid_cfg, fill=(130, 130, 130), width=2)
    _draw_polyline_xy(draw, active_path_xy_b, grid_cfg, fill=(255, 0, 0), width=2)

    if start_xy_b is not None:
        _draw_marker(draw, start_xy_b, grid_cfg, fill=(0, 255, 0), radius=4, outline=(0, 90, 0))
    if target_xy_b is not None:
        _draw_marker(draw, target_xy_b, grid_cfg, fill=(0, 0, 255), radius=4, outline=(0, 0, 90))
    _draw_marker(draw, robot_xy_b, grid_cfg, fill=(255, 230, 0), radius=5, outline=(0, 0, 0), cross=True)

    path = out_dir / f"astar_grid_{step_idx:06d}.png"
    rgb.save(path)
    print(f"[INFO]: Saved A* planner grid: {path}", flush=True)


def _obstacle_footprint_points_xy(obs: dict[str, Any], spacing: float) -> torch.Tensor:
    """Sample dense env-local XY points inside one scenario obstacle footprint.

    This is used only by the scenario smoke test to give A* the full static
    obstacle footprint. The current RayCaster setup sees only static mesh hit
    points from a fixed origin, so using lidar points alone can underrepresent
    the occupied obstacle volume as the proxy moves.
    """
    shape = str(obs.get("shape", "box")).lower()
    spacing = max(0.02, float(spacing))
    pos_x, pos_y = float(obs.get("pos", [0.0, 0.0, 0.0])[0]), float(obs.get("pos", [0.0, 0.0, 0.0])[1])
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
    """Rasterize known static scenario obstacle footprints into proxy-local grids.

    This is intentionally scenario/test-only. It stabilizes the A* smoke test by
    representing the whole rectangle/cylinder footprint, not merely the sparse
    RayCaster hit surface.
    """
    num_envs = proxy.num_envs
    if not obstacles:
        return torch.zeros((num_envs, grid_cfg.height, grid_cfg.width), dtype=torch.float32, device=device)

    spacing = max(grid_cfg.resolution, 0.05)
    env_local_points = [_obstacle_footprint_points_xy(obs, spacing=spacing) for obs in obstacles]
    points_env_xy = torch.cat(env_local_points, dim=0).to(device=device)
    points_w_xy = scene.env_origins[:, None, 0:2].to(device=device) + points_env_xy[None, :, :]
    delta_xy_w = points_w_xy - proxy.pos_w[:, None, 0:2]
    points_b = rotate_world_to_body_2d(delta_xy_w, proxy.yaw[:, None])
    valid = torch.ones(points_b.shape[:-1], dtype=torch.bool, device=device)
    return rasterize_points_to_occupancy_grid(points_b, valid, grid_cfg)


# -----------------------------------------------------------------------------
# Persistent path + command smoothing helpers
# -----------------------------------------------------------------------------
PointXY = tuple[float, float]


def _rotate_body_to_world_xy(points_b: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    x = points_b[..., 0]
    y = points_b[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def path_body_to_world(path_xy_b: list[PointXY], proxy: G1CommandProxy) -> list[PointXY]:
    if not path_xy_b:
        return []
    device = proxy.pos_w.device
    pts_b = torch.tensor(path_xy_b, dtype=torch.float32, device=device)
    pts_w = proxy.pos_w[0, 0:2] + _rotate_body_to_world_xy(pts_b, proxy.yaw[0])
    return [(float(x), float(y)) for x, y in pts_w.detach().cpu().tolist()]


def path_world_to_body(path_xy_w: list[PointXY], proxy: G1CommandProxy) -> list[PointXY]:
    if not path_xy_w:
        return []
    device = proxy.pos_w.device
    pts_w = torch.tensor(path_xy_w, dtype=torch.float32, device=device)
    delta = pts_w[None, :, :] - proxy.pos_w[0:1, None, 0:2]
    pts_b = rotate_world_to_body_2d(delta, proxy.yaw[0:1, None])[0]
    return [(float(x), float(y)) for x, y in pts_b.detach().cpu().tolist()]


def path_length_xy(path: list[PointXY]) -> float:
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def resample_path_xy(path: list[PointXY], num_samples: int) -> list[PointXY]:
    if not path:
        return []
    if len(path) == 1 or num_samples <= 1:
        return [path[0]]
    total_len = path_length_xy(path)
    if total_len <= 1.0e-8:
        return [path[0] for _ in range(num_samples)]

    samples: list[PointXY] = []
    seg_i = 0
    seg_start = path[0]
    seg_end = path[1]
    seg_len = math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
    traversed_before = 0.0

    for k in range(num_samples):
        d = total_len * float(k) / float(num_samples - 1)
        while seg_i < len(path) - 2 and traversed_before + seg_len < d:
            traversed_before += seg_len
            seg_i += 1
            seg_start = path[seg_i]
            seg_end = path[seg_i + 1]
            seg_len = math.hypot(seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        if seg_len <= 1.0e-8:
            samples.append(seg_end)
        else:
            t = max(0.0, min(1.0, (d - traversed_before) / seg_len))
            samples.append((seg_start[0] + t * (seg_end[0] - seg_start[0]), seg_start[1] + t * (seg_end[1] - seg_start[1])))
    return samples


def path_delta_xy(path_a: list[PointXY], path_b: list[PointXY], num_samples: int) -> float:
    if not path_a or not path_b:
        return math.inf
    aa = resample_path_xy(path_a, num_samples)
    bb = resample_path_xy(path_b, num_samples)
    if len(aa) != len(bb) or not aa:
        return math.inf
    return sum(math.hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(aa, bb)) / float(len(aa))


def prune_active_path_w(
    active_path_w: list[PointXY],
    proxy: G1CommandProxy,
    *,
    reached_distance: float,
    behind_distance: float,
) -> list[PointXY]:
    """Drop active-path waypoints that are already reached or behind the proxy."""
    path = list(active_path_w)
    while len(path) > 1:
        first_b = path_world_to_body([path[0]], proxy)[0]
        dist = math.hypot(first_b[0], first_b[1])
        if dist <= reached_distance or first_b[0] < -behind_distance:
            path.pop(0)
        else:
            break
    return path


def path_cells_from_body_path(path_xy_b: list[PointXY], grid_cfg: OccupancyGridCfg) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for xy in path_xy_b:
        cell = xy_to_grid_cell(xy, grid_cfg)
        row, col = cell
        if 0 <= row < grid_cfg.height and 0 <= col < grid_cfg.width:
            cells.append(cell)
    return cells


def is_hard_front_blocked(lidar_obs: dict[str, torch.Tensor], cfg: dict[str, Any]) -> bool:
    """Return true only for a near obstacle in the forward body-frame corridor."""
    if not cfg.get("enabled", True):
        return False
    points = lidar_obs["lidar_points_b"][0]
    valid = lidar_obs["lidar_valid"][0]
    if not bool(valid.any()):
        return False
    pts = points[valid]
    x = pts[:, 0]
    y = pts[:, 1]
    mask = (
        (x > 0.0)
        & (x < float(cfg.get("front_x", 0.45)))
        & (torch.abs(y) < float(cfg.get("half_width", 0.35)))
    )
    return bool(mask.any().detach().cpu().item())


def compute_obstacle_clearance_from_grid(
    occupancy_grid: torch.Tensor,
    grid_cfg: OccupancyGridCfg,
    *,
    body_radius: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute simple body-frame clearance diagnostics from the raw obstacle grid.

    The smoke-test planning grid contains raw obstacle cells from LiDAR hits plus
    optional known scenario footprints. This function intentionally measures
    clearance against that *raw* obstacle representation, not the A*-inflated
    grid, so the value approximates center-to-obstacle distance. Edge clearance
    is reported as ``center_distance - body_radius``.
    """
    threshold = float(cfg.get("occupancy_threshold", 0.5))
    occupied = occupancy_grid >= threshold
    cells = torch.nonzero(occupied, as_tuple=False)
    if cells.numel() == 0:
        return {
            "has_obstacle": False,
            "min_center_distance": math.inf,
            "min_forward_center_distance": math.inf,
            "edge_clearance": math.inf,
            "closest_xy_b": None,
            "closest_forward_xy_b": None,
        }

    rows = cells[:, 0].to(dtype=torch.float32)
    cols = cells[:, 1].to(dtype=torch.float32)
    x = float(grid_cfg.x_max) - (rows + 0.5) * float(grid_cfg.resolution)
    y = float(grid_cfg.y_min) + (cols + 0.5) * float(grid_cfg.resolution)
    dist = torch.sqrt(x * x + y * y)

    min_idx = int(torch.argmin(dist).detach().cpu().item())
    min_dist = float(dist[min_idx].detach().cpu().item())
    closest_xy = (float(x[min_idx].detach().cpu().item()), float(y[min_idx].detach().cpu().item()))

    forward_min_x = float(cfg.get("forward_min_x", -0.20))
    forward_mask = x >= forward_min_x
    if bool(forward_mask.any().detach().cpu().item()):
        forward_indices = torch.nonzero(forward_mask, as_tuple=False).flatten()
        forward_dist = dist[forward_indices]
        f_rel_idx = int(torch.argmin(forward_dist).detach().cpu().item())
        f_idx = int(forward_indices[f_rel_idx].detach().cpu().item())
        f_dist = float(dist[f_idx].detach().cpu().item())
        f_xy = (float(x[f_idx].detach().cpu().item()), float(y[f_idx].detach().cpu().item()))
    else:
        f_dist = math.inf
        f_xy = None

    return {
        "has_obstacle": True,
        "min_center_distance": min_dist,
        "min_forward_center_distance": f_dist,
        "edge_clearance": f_dist - float(body_radius),
        "closest_xy_b": closest_xy,
        "closest_forward_xy_b": f_xy,
    }


def apply_clearance_safety(
    raw_cmd: torch.Tensor,
    clearance_stats: dict[str, Any],
    cfg: dict[str, Any],
    *,
    hard_front_block: bool,
) -> tuple[torch.Tensor, str]:
    """Apply a minimal local safety layer on top of the global A* command.

    Important design constraint for this smoke test:
      - the global A* path is responsible for planned obstacle avoidance;
      - local clearance safety should not fight A* during ordinary side passing;
      - local escape is reserved for a true hard frontal block.

    Earlier versions used the closest obstacle cell to trigger an ``escape`` even
    when the obstacle was beside the robot. That produced commands such as
    ``vy=-0.1, wz=-0.12`` while the A* follower wanted positive ``vy/wz`` around
    the obstacle. The result was a planner/safety tug-of-war near the box. This
    function fixes that by preserving the A* lateral/yaw direction unless the
    hard-front-block detector fires.
    """
    if raw_cmd.ndim != 2 or raw_cmd.shape[-1] != 3:
        raise ValueError(f"raw_cmd must have shape [N, 3], got {tuple(raw_cmd.shape)}")

    cmd = raw_cmd.detach().clone()
    if not cfg.get("enabled", True):
        return cmd, "disabled"

    edge_clearance = float(clearance_stats.get("edge_clearance", math.inf))
    if not math.isfinite(edge_clearance):
        return cmd, "no_obstacle"

    slow_distance = float(cfg.get("slow_distance", 1.05))
    crawl_distance = float(cfg.get("crawl_distance", 0.75))
    stop_distance = float(cfg.get("stop_distance", 0.58))
    max_vx_slow = float(cfg.get("max_vx_slow", 0.18))
    max_vx_crawl = float(cfg.get("max_vx_crawl", 0.08))
    escape_enabled = bool(cfg.get("escape_enabled", True))
    escape_vx = float(cfg.get("escape_vx", max_vx_crawl))
    escape_vy = abs(float(cfg.get("escape_vy", 0.10)))
    escape_wz = abs(float(cfg.get("escape_wz", 0.25)))

    # True emergency case: only a hard frontal obstacle may override the global
    # path follower. Even here, choose the escape side from the obstacle side.
    if hard_front_block:
        if not escape_enabled:
            cmd[:, :] = 0.0
            return cmd, "hard_front_stop"
        closest_xy_b = clearance_stats.get("closest_forward_xy_b") or clearance_stats.get("closest_xy_b")
        if closest_xy_b is not None:
            closest_y = float(closest_xy_b[1])
            side_sign = -1.0 if closest_y >= 0.0 else 1.0
        else:
            # Fall back to the current A* side preference if the diagnostic is missing.
            vy0 = float(cmd[0, 1].detach().cpu().item())
            side_sign = 1.0 if vy0 >= 0.0 else -1.0
        cmd[:, 0] = torch.clamp(cmd[:, 0], max=escape_vx)
        cmd[:, 1] = side_sign * escape_vy
        cmd[:, 2] = side_sign * escape_wz
        return cmd, "hard_front_escape"

    # Ordinary side/diagonal proximity: preserve A* vy/wz; only cap forward vx.
    # This keeps clearance conservative without destroying the planned turn.
    if edge_clearance < stop_distance:
        cmd[:, 0] = torch.clamp(cmd[:, 0], max=max_vx_crawl)
        return cmd, "side_close_crawl_vx"
    if edge_clearance < crawl_distance:
        cmd[:, 0] = torch.clamp(cmd[:, 0], max=max_vx_crawl)
        return cmd, "crawl_vx"
    if edge_clearance < slow_distance:
        cmd[:, 0] = torch.clamp(cmd[:, 0], max=max_vx_slow)
        return cmd, "slow_vx"
    return cmd, "none"

def effective_smoothing_cfg_for_clearance(cfg: dict[str, Any], clearance: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Return base or near-obstacle smoothing parameters."""
    if not cfg.get("enabled", True):
        return {**cfg, "enabled": False}, "disabled"
    d = float(clearance.get("min_forward_center_distance", math.inf))
    adaptive_distance = float(cfg.get("adaptive_distance", 0.90))
    if math.isfinite(d) and d < adaptive_distance:
        merged = dict(cfg)
        merged["alpha"] = float(cfg.get("near_alpha", cfg.get("alpha", 0.35)))
        merged["max_delta_vx"] = float(cfg.get("near_max_delta_vx", cfg.get("max_delta_vx", 0.03)))
        merged["max_delta_vy"] = float(cfg.get("near_max_delta_vy", cfg.get("max_delta_vy", 0.04)))
        merged["max_delta_wz"] = float(cfg.get("near_max_delta_wz", cfg.get("max_delta_wz", 0.08)))
        return merged, "near_obstacle"
    return dict(cfg), "nominal"


def smooth_command(
    raw_cmd: torch.Tensor,
    last_cmd: torch.Tensor | None,
    cfg: dict[str, Any],
) -> torch.Tensor:
    """Apply command/momentum continuity before proxy command limiting."""
    if last_cmd is None or not cfg.get("enabled", True):
        return raw_cmd
    alpha = float(cfg.get("alpha", 0.35))
    desired = alpha * raw_cmd + (1.0 - alpha) * last_cmd
    max_delta = torch.tensor(
        [[float(cfg.get("max_delta_vx", 0.03)), float(cfg.get("max_delta_vy", 0.04)), float(cfg.get("max_delta_wz", 0.08))]],
        dtype=raw_cmd.dtype,
        device=raw_cmd.device,
    )
    delta = torch.clamp(desired - last_cmd, min=-max_delta, max=max_delta)
    return last_cmd + delta


@configclass
class G1LidarAStarSceneCfg(InteractiveSceneCfg):
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

    lidar_mount = AssetBaseCfg(
        prim_path=CFG["lidar"]["prim_path"],
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(CFG["proxy"]["init_state"]["pos"])),
        spawn=sim_utils.CuboidCfg(
            size=tuple(CFG["lidar"].get("mount_size", [0.05, 0.05, 0.05])),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(CFG["lidar"].get("mount_color", [1.0, 0.5, 0.0]))),
        ),
    )

    if len(OBSTACLES) > 0:
        obstacle_0 = obstacle_asset_cfg(OBSTACLES[0])
    if len(OBSTACLES) > 1:
        obstacle_1 = obstacle_asset_cfg(OBSTACLES[1])
    if len(OBSTACLES) > 2:
        obstacle_2 = obstacle_asset_cfg(OBSTACLES[2])
    if len(OBSTACLES) > 3:
        obstacle_3 = obstacle_asset_cfg(OBSTACLES[3])
    if len(OBSTACLES) > 4:
        obstacle_4 = obstacle_asset_cfg(OBSTACLES[4])
    if len(OBSTACLES) > 5:
        obstacle_5 = obstacle_asset_cfg(OBSTACLES[5])

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


def make_proxy(device: torch.device, num_envs: int) -> G1CommandProxy:
    limits_cfg = CFG["proxy"]["command_limits"]
    limits = G1CommandProxyLimits(
        max_vx=float(limits_cfg["max_vx"]),
        max_vy=float(limits_cfg["max_vy"]),
        max_wz=float(limits_cfg["max_wz"]),
        max_ax=float(limits_cfg["max_ax"]),
        max_ay=float(limits_cfg["max_ay"]),
        max_awz=float(limits_cfg["max_awz"]),
        smoothing_alpha=float(limits_cfg["smoothing_alpha"]),
    )
    return G1CommandProxy(
        proxy_prim_paths=resolve_env_prim_paths(CFG["proxy"]["prim_path"], num_envs),
        num_envs=num_envs,
        device=device,
        init_pos=tuple(CFG["proxy"]["init_state"]["pos"]),
        init_yaw=float(CFG["proxy"]["init_state"]["yaw"]),
        limits=limits,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    sim_dt = sim.get_physics_dt()
    device = torch.device(args_cli.device)
    num_envs = int(CFG["scene"]["num_envs"])

    lidar_obs_cfg = LidarObservationCfg.from_mapping(CFG.get("lidar", {}).get("observations", {}))
    grid_cfg = OccupancyGridCfg.from_mapping(CFG.get("lidar", {}).get("occupancy_grid", {}))
    raw_astar_cfg = GridAStarCfg.from_mapping(CFG.get("grid_astar", {}))
    astar_cfg, body_radius, safety_margin, effective_inflation = compute_astar_cfg_with_body_radius(raw_astar_cfg)
    follower_cfg = PathFollowerCfg.from_mapping(CFG.get("grid_astar", {}).get("path_follower", {}))
    path_memory_cfg = CFG.get("grid_astar", {}).get("path_memory", {})
    command_smoothing_cfg = CFG.get("grid_astar", {}).get("command_smoothing", {})
    hard_block_cfg = CFG.get("grid_astar", {}).get("hard_block", {})
    clearance_safety_cfg = CFG.get("grid_astar", {}).get("clearance_safety", {})

    replan_period_steps = int(path_memory_cfg.get("replan_period_steps", 80))
    path_reached_distance = float(path_memory_cfg.get("path_reached_distance", 0.45))
    path_behind_distance = float(path_memory_cfg.get("path_behind_distance", 0.35))
    path_switch_weight = float(path_memory_cfg.get("path_switch_weight", 8.0))
    switch_min_improvement = float(path_memory_cfg.get("switch_min_improvement", 0.30))
    compare_samples = int(path_memory_cfg.get("compare_samples", 48))

    astar_debug_cfg = CFG.get("grid_astar", {}).get("debug", {})
    print_every = int(args_cli.print_every if args_cli.print_every is not None else astar_debug_cfg.get("print_every", CFG.get("lidar", {}).get("debug", {}).get("print_every", 10)))
    save_every = int(args_cli.save_every if args_cli.save_every is not None else astar_debug_cfg.get("save_every", CFG.get("lidar", {}).get("debug", {}).get("save_every", 20)))
    save_grid = bool(astar_debug_cfg.get("save_grid", True)) and not bool(args_cli.no_save_grid)
    grid_dir = PROJECT_ROOT / str(astar_debug_cfg.get("grid_dir", "outputs/lidar_astar_planner")) / args_cli.scenario

    proxy = make_proxy(device=device, num_envs=num_envs)
    proxy.reset(scene.env_origins)

    lidar_mount_paths = resolve_env_prim_paths(CFG["lidar"]["prim_path"], num_envs)
    mount_sync_cfg = LidarMountSyncCfg()
    sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)

    scene.reset()

    target_local_xy = torch.tensor(CFG["target"]["pos"][0:2], dtype=torch.float32, device=device)
    target_xy_w = scene.env_origins[:, 0:2].to(device=device) + target_local_xy
    success_radius = float(args_cli.target_success_radius if args_cli.target_success_radius is not None else CFG.get("target", {}).get("success_radius", 0.75))

    start_xy_w: PointXY = (
        float(proxy.pos_w[0, 0].detach().cpu().item()),
        float(proxy.pos_w[0, 1].detach().cpu().item()),
    )
    trajectory_w: list[PointXY] = [start_xy_w]

    print("[INFO]: Starting LiDAR A* planner smoke test...", flush=True)
    print(f"[INFO]: scenario: {args_cli.scenario} — {SCENARIO['description']}", flush=True)
    print(f"[INFO]: target_xy_w env0: {target_xy_w[0].detach().cpu().tolist()}", flush=True)
    print(f"[INFO]: success radius: {success_radius:.3f} m", flush=True)
    print(f"[INFO]: obstacles: {len(OBSTACLES)}", flush=True)
    print(f"[INFO]: planning grid source: {'lidar_only' if args_cli.lidar_only else 'lidar + known scenario footprints'}", flush=True)
    for i, obs in enumerate(OBSTACLES):
        print(f"  obstacle[{i}]: shape={obs.get('shape')} pos={obs.get('pos')} size/radius={obs.get('size', obs.get('radius'))} yaw={obs.get('yaw', 0.0)}", flush=True)
    print(f"[INFO]: lidar mesh paths: {resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), num_envs)}", flush=True)
    print(f"[INFO]: occupancy grid: H={grid_cfg.height}, W={grid_cfg.width}, resolution={grid_cfg.resolution}", flush=True)
    print(
        f"[INFO]: A*: inflation={astar_cfg.inflation_radius:.3f} "
        f"(body_radius={body_radius:.3f}, safety_margin={safety_margin:.3f}), "
        f"diagonal={astar_cfg.allow_diagonal}, heuristic_weight={astar_cfg.heuristic_weight}",
        flush=True,
    )
    print(
        f"[INFO]: follower: lookahead={follower_cfg.lookahead_distance}, min_forward_x={follower_cfg.min_forward_x}, "
        f"max=({follower_cfg.max_vx}, {follower_cfg.max_vy}, {follower_cfg.max_wz})",
        flush=True,
    )
    print(
        f"[INFO]: path memory: replan_period={replan_period_steps}, reached={path_reached_distance:.2f}, "
        f"path_switch_weight={path_switch_weight:.2f}, switch_min_improvement={switch_min_improvement:.2f}",
        flush=True,
    )
    print(
        f"[INFO]: command smoothing: alpha={float(command_smoothing_cfg.get('alpha', 0.35)):.2f}, "
        f"max_delta=({float(command_smoothing_cfg.get('max_delta_vx', 0.03)):.3f}, "
        f"{float(command_smoothing_cfg.get('max_delta_vy', 0.04)):.3f}, "
        f"{float(command_smoothing_cfg.get('max_delta_wz', 0.08)):.3f})",
        flush=True,
    )
    print(
        f"[INFO]: clearance safety: enabled={bool(clearance_safety_cfg.get('enabled', True))}, "
        f"slow/crawl/stop=({float(clearance_safety_cfg.get('slow_distance', 1.05)):.2f}, "
        f"{float(clearance_safety_cfg.get('crawl_distance', 0.75)):.2f}, "
        f"{float(clearance_safety_cfg.get('stop_distance', 0.58)):.2f}), "
        f"escape={bool(clearance_safety_cfg.get('escape_enabled', True))} "
        f"(hard-front only; side clearance preserves A* vy/wz)",
        flush=True,
    )
    print(f"[INFO]: shortcutting: enabled={not args_cli.disable_shortcut}", flush=True)
    print(f"[INFO]: run control: max_steps={int(args_cli.max_steps)}, print_every={print_every}, save_grid={save_grid}, save_every={save_every}", flush=True)
    print(
        "[INFO]: debug image markers: yellow=current robot origin, green=start, "
        "grey=covered trajectory, red=active path, blue=target",
        flush=True,
    )

    distance_history: list[float] = []
    clearance_history: list[float] = []
    edge_clearance_history: list[float] = []
    clearance_limit_count = 0
    avoidance_command_count = 0
    no_path_count = 0
    found_path_count = 0
    target_reached = False
    reached_step: int | None = None

    active_path_w: list[PointXY] = []
    active_path_cells: list[tuple[int, int]] = []
    last_plan_step = -10**9
    # Separate accepted-plan timestamp from attempted-replan timestamp.
    # If a periodic candidate is rejected, we still want to wait another
    # replan period before trying again; otherwise every subsequent frame
    # becomes a periodic replan attempt.
    last_replan_attempt_step = -10**9
    last_cmd: torch.Tensor | None = None
    last_astar_found = False
    last_astar_reason = "not_planned"
    last_astar_expansions = 0
    last_raw_path_len = 0
    last_short_path_len = 0
    last_path_action = "none"
    last_replan_reason = "none"
    last_path_delta = math.inf
    last_current_objective = math.inf
    last_candidate_objective = math.inf

    # Give RayCaster one frame to populate buffers after reset.
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    for step_idx in range(int(args_cli.max_steps)):
        if not simulation_app.is_running():
            break

        lidar_obs = build_lidar_proxy_observation(scene, proxy, lidar_name="lidar", cfg=lidar_obs_cfg)
        lidar_grid = rasterize_points_to_occupancy_grid(lidar_obs["lidar_points_b"], lidar_obs["lidar_valid"], grid_cfg)
        if args_cli.lidar_only:
            footprint_grid = torch.zeros_like(lidar_grid)
            grid = lidar_grid
        else:
            footprint_grid = rasterize_scenario_obstacle_footprints(
                OBSTACLES,
                proxy=proxy,
                scene=scene,
                grid_cfg=grid_cfg,
                device=device,
            )
            grid = torch.maximum(lidar_grid, footprint_grid)

        # Keep the active path in world frame, prune points already reached/behind,
        # and follow it between replans. This avoids re-solving a symmetric left/right
        # problem at every frame.
        if active_path_w:
            active_path_w = prune_active_path_w(
                active_path_w,
                proxy,
                reached_distance=path_reached_distance,
                behind_distance=path_behind_distance,
            )

        hard_blocked = is_hard_front_blocked(lidar_obs, hard_block_cfg)
        no_active_path = len(active_path_w) < 2
        periodic_replan = (step_idx - last_replan_attempt_step) >= replan_period_steps
        needs_replan = no_active_path or periodic_replan or hard_blocked

        target_vec = target_vector_body_frame(proxy.pos_w[:, 0:2], proxy.yaw, target_xy_w)
        goal_xy_b = target_vec[0, 0:2].detach().cpu().tolist()

        last_path_action = "follow_current"
        last_replan_reason = "keep_current_path"
        last_path_delta = math.inf
        last_current_objective = path_length_xy(active_path_w) if active_path_w else math.inf
        last_candidate_objective = math.inf
        astar_short_for_debug = None
        astar_result_for_debug = None
        astar_attempted = False

        if needs_replan:
            astar_attempted = True
            last_replan_attempt_step = step_idx
            if no_active_path:
                replan_reason = "no_active_path"
            elif hard_blocked:
                replan_reason = "hard_block"
            else:
                replan_reason = "periodic"

            astar_result = plan_grid_path_xy(
                occupancy_grid=grid[0],
                start_xy_b=(0.0, 0.0),
                goal_xy_b=goal_xy_b,
                grid_cfg=grid_cfg,
                planner_cfg=astar_cfg,
                allow_goal_relocation=True,
            )
            astar_short = astar_result if args_cli.disable_shortcut else shortcut_grid_path_result(astar_result, grid[0], grid_cfg, astar_cfg)
            astar_result_for_debug = astar_result
            astar_short_for_debug = astar_short
            last_astar_found = bool(astar_short.found)
            last_astar_reason = str(astar_short.reason) + ("+raw_path" if args_cli.disable_shortcut else "")
            last_astar_expansions = int(astar_short.expansions)
            last_raw_path_len = len(astar_result.path_cells)
            last_short_path_len = len(astar_short.path_cells)

            if astar_short.found and len(astar_short.path_xy_b) >= 2:
                candidate_path_w = path_body_to_world(astar_short.path_xy_b, proxy)
                candidate_len = path_length_xy(candidate_path_w)
                if active_path_w and len(active_path_w) >= 2 and not hard_blocked:
                    current_len = path_length_xy(active_path_w)
                    delta = path_delta_xy(active_path_w, candidate_path_w, compare_samples)
                    candidate_objective = candidate_len + path_switch_weight * delta
                    current_objective = current_len
                    accept = candidate_objective < (current_objective - switch_min_improvement)
                else:
                    current_len = math.inf
                    delta = 0.0
                    candidate_objective = candidate_len
                    current_objective = math.inf
                    accept = True

                last_path_delta = delta
                last_current_objective = current_objective
                last_candidate_objective = candidate_objective

                if accept:
                    active_path_w = candidate_path_w
                    active_path_cells = list(astar_short.path_cells)
                    last_plan_step = step_idx
                    last_path_action = "replanned_accept"
                    last_replan_reason = replan_reason
                else:
                    last_path_action = "replanned_reject_keep_current"
                    last_replan_reason = replan_reason
            else:
                no_path_count += 1
                last_path_action = "replan_failed_keep_current" if active_path_w else "replan_failed_stop"
                last_replan_reason = replan_reason

        if astar_attempted and last_astar_found:
            found_path_count += 1

        active_path_b = path_world_to_body(active_path_w, proxy) if active_path_w else []
        if active_path_b:
            active_path_cells = path_cells_from_body_path(active_path_b, grid_cfg)

        # Snapshot all image overlays in the same body frame as the grid built
        # above. The proxy is stepped after this point, so using these saved
        # values avoids mixing a pre-step grid with a post-step pose.
        debug_active_path_b = list(active_path_b)
        debug_covered_path_b = path_world_to_body(trajectory_w, proxy) if trajectory_w else []
        debug_start_xy_b = path_world_to_body([start_xy_w], proxy)[0]
        debug_target_xy_b = (float(target_vec[0, 0].detach().cpu().item()), float(target_vec[0, 1].detach().cpu().item()))
        debug_robot_xy_b: PointXY = (0.0, 0.0)

        follower_result = follow_body_frame_path(active_path_b, follower_cfg, device=device)
        raw_planner_cmd = follower_result.command
        if hard_blocked and not active_path_b:
            raw_planner_cmd = torch.zeros_like(raw_planner_cmd)

        clearance_stats = compute_obstacle_clearance_from_grid(
            grid[0],
            grid_cfg,
            body_radius=body_radius,
            cfg=clearance_safety_cfg,
        )
        clearance_history.append(float(clearance_stats.get("min_forward_center_distance", math.inf)))
        edge_clearance_history.append(float(clearance_stats.get("edge_clearance", math.inf)))

        safety_desired_cmd, safety_action_pre = apply_clearance_safety(
            raw_planner_cmd,
            clearance_stats,
            clearance_safety_cfg,
            hard_front_block=hard_blocked,
        )
        effective_smoothing_cfg, smoothing_mode = effective_smoothing_cfg_for_clearance(command_smoothing_cfg, clearance_stats)
        smoothed_cmd = smooth_command(safety_desired_cmd, last_cmd, effective_smoothing_cfg)
        planner_cmd, safety_action_post = apply_clearance_safety(
            smoothed_cmd,
            clearance_stats,
            clearance_safety_cfg,
            hard_front_block=hard_blocked,
        )
        clearance_action = safety_action_pre if safety_action_post in ("none", "disabled", "no_obstacle") else f"{safety_action_pre}|post:{safety_action_post}"
        if clearance_action not in ("none", "disabled", "no_obstacle"):
            clearance_limit_count += 1
        last_cmd = planner_cmd.detach().clone()

        target_distance = float(torch.linalg.norm(target_xy_w[0] - proxy.pos_w[0, 0:2]).detach().cpu().item())
        distance_history.append(target_distance)
        if target_distance <= success_radius:
            target_reached = True
            reached_step = step_idx
            planner_cmd = torch.zeros_like(planner_cmd)

        if abs(float(planner_cmd[0, 1].detach().cpu())) > 0.05 or abs(float(planner_cmd[0, 2].detach().cpu())) > 0.05:
            avoidance_command_count += 1

        proxy.set_command(planner_cmd)
        proxy.step(sim_dt)
        current_xy_w: PointXY = (
            float(proxy.pos_w[0, 0].detach().cpu().item()),
            float(proxy.pos_w[0, 1].detach().cpu().item()),
        )
        if not trajectory_w or math.hypot(current_xy_w[0] - trajectory_w[-1][0], current_xy_w[1] - trajectory_w[-1][1]) > 1.0e-4:
            trajectory_w.append(current_xy_w)
        sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if step_idx % print_every == 0 or target_reached:
            ranges = lidar_obs["lidar_ranges"]
            valid = lidar_obs["lidar_valid"]
            occupied = int((grid[0] > 0).detach().cpu().sum().item())
            lidar_occupied = int((lidar_grid[0] > 0).detach().cpu().sum().item())
            footprint_occupied = int((footprint_grid[0] > 0).detach().cpu().sum().item())
            valid_ranges = ranges[0][valid[0]]
            min_valid = float(valid_ranges.detach().cpu().min().item()) if valid_ranges.numel() else float("nan")
            rel_points = lidar_obs["lidar_points_b"][0][valid[0]]
            rel_distances = torch.linalg.norm(rel_points, dim=-1) if rel_points.numel() else torch.empty(0, device=device)
            min_rel_distance = float(rel_distances.detach().cpu().min().item()) if rel_distances.numel() else float("nan")

            print("-" * 80, flush=True)
            print(f"step               : {step_idx}", flush=True)
            print(f"lidar              : {summarize_lidar_observation(lidar_obs)}", flush=True)
            print(f"raw lidar min range: {min_valid:.3f}", flush=True)
            print(f"proxy-relative min : {min_rel_distance:.3f}  # lidar-hit based", flush=True)
            cxy = clearance_stats.get("closest_forward_xy_b")
            cxy_str = None if cxy is None else (round(float(cxy[0]), 3), round(float(cxy[1]), 3))
            print(
                f"obstacle clearance : center={float(clearance_stats.get('min_forward_center_distance', math.inf)):.3f}, "
                f"edge={float(clearance_stats.get('edge_clearance', math.inf)):.3f}, closest_xy_b={cxy_str}",
                flush=True,
            )
            print(f"hard front block   : {hard_blocked}", flush=True)
            print(f"occupied cells     : {occupied} (lidar={lidar_occupied}, footprint={footprint_occupied})", flush=True)
            print(f"target vector env0 : {[round(x, 3) for x in target_vec[0].detach().cpu().tolist()]}", flush=True)
            print(f"target distance    : {target_distance:.3f}", flush=True)
            print(f"target reached     : {target_reached}", flush=True)
            print(f"path action        : {last_path_action}", flush=True)
            print(f"replan reason      : {last_replan_reason}", flush=True)
            print(f"path objective cur/new: {last_current_objective:.3f} / {last_candidate_objective:.3f}", flush=True)
            print(f"path switch delta  : {last_path_delta:.3f}", flush=True)
            print(f"A* attempted       : {astar_attempted}", flush=True)
            print(f"A* found           : {last_astar_found}", flush=True)
            print(f"A* reason          : {last_astar_reason}", flush=True)
            print(f"A* expansions      : {last_astar_expansions}", flush=True)
            print(f"A* raw/active path : {last_raw_path_len} / {len(active_path_cells)}", flush=True)
            print(f"lookahead_xy_b     : {follower_result.lookahead_xy_b}", flush=True)
            print(f"raw planner cmd    : {[round(x, 3) for x in raw_planner_cmd[0].detach().cpu().tolist()]}", flush=True)
            print(f"safety desired cmd : {[round(x, 3) for x in safety_desired_cmd[0].detach().cpu().tolist()]} action={clearance_action}", flush=True)
            print(f"smoothing mode     : {smoothing_mode}", flush=True)
            print(f"planner command    : {[round(x, 3) for x in planner_cmd[0].detach().cpu().tolist()]}", flush=True)
            print(f"proxy cmd_current  : {[round(x, 3) for x in proxy.cmd_current[0].detach().cpu().tolist()]}", flush=True)
            print(f"proxy position     : {proxy.debug_state()['pos_w']}", flush=True)
            print(f"proxy yaw          : {proxy.debug_state()['yaw'][0]:.3f}", flush=True)
            print(f"image robot marker : body-frame origin (0.000, 0.000)", flush=True)
            print(f"image start_xy_b   : ({debug_start_xy_b[0]:.3f}, {debug_start_xy_b[1]:.3f})", flush=True)
            print(f"covered path points: {len(debug_covered_path_b)}", flush=True)

            if save_grid and save_every > 0 and step_idx % save_every == 0:
                save_astar_debug_image(
                    grid,
                    active_path_xy_b=debug_active_path_b,
                    covered_path_xy_b=debug_covered_path_b,
                    start_xy_b=debug_start_xy_b,
                    target_xy_b=debug_target_xy_b,
                    robot_xy_b=debug_robot_xy_b,
                    grid_cfg=grid_cfg,
                    out_dir=grid_dir,
                    step_idx=step_idx,
                )

        if target_reached:
            break

    if distance_history:
        print(
            f"[INFO]: target distance first/last/min: {distance_history[0]:.3f} / {distance_history[-1]:.3f} / {min(distance_history):.3f}",
            flush=True,
        )
    finite_clearances = [d for d in clearance_history if math.isfinite(d)]
    finite_edge_clearances = [d for d in edge_clearance_history if math.isfinite(d)]
    if finite_clearances:
        print(
            f"[INFO]: obstacle center clearance min/last: {min(finite_clearances):.3f} / {finite_clearances[-1]:.3f}",
            flush=True,
        )
    if finite_edge_clearances:
        print(
            f"[INFO]: obstacle edge clearance min/last: {min(finite_edge_clearances):.3f} / {finite_edge_clearances[-1]:.3f}",
            flush=True,
        )
    print(f"[INFO]: clearance limiter count: {clearance_limit_count}", flush=True)
    print(f"[INFO]: target reached: {target_reached} step={reached_step}", flush=True)
    print(f"[INFO]: A* found path count: {found_path_count}", flush=True)
    print(f"[INFO]: A* no-path count: {no_path_count}", flush=True)
    print(f"[INFO]: avoidance command count: {avoidance_command_count}", flush=True)

    if args_cli.assert_planner:
        if len(distance_history) < 2:
            raise RuntimeError("No target distance history was collected.")
        if not (min(distance_history) < distance_history[0] - 0.05):
            raise RuntimeError(
                "Expected the A* planner to reduce target distance by at least 0.05 m. "
                f"first={distance_history[0]:.3f}, min={min(distance_history):.3f}"
            )
        if bool(SCENARIO.get("expect_success", True)) and found_path_count <= 0:
            raise RuntimeError("Expected A* to find at least one path.")
        if bool(SCENARIO.get("expect_avoidance", False)) and avoidance_command_count <= 0:
            raise RuntimeError(f"Expected at least one lateral/yaw avoidance command for scenario {args_cli.scenario!r}.")
        min_edge_assert = CFG.get("grid_astar", {}).get("clearance_safety", {}).get("assert_min_edge_clearance", None)
        if min_edge_assert is not None and finite_edge_clearances and min(finite_edge_clearances) < float(min_edge_assert):
            raise RuntimeError(
                f"Expected obstacle edge clearance >= {float(min_edge_assert):.3f} m, "
                f"got {min(finite_edge_clearances):.3f} m."
            )
    if args_cli.assert_success and bool(SCENARIO.get("expect_success", True)) and not target_reached:
        raise RuntimeError(f"Expected to reach target within radius {success_radius:.3f} m.")

    print("[INFO]: LiDAR A* planner smoke test complete.", flush=True)

def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(eye=list(CFG["scene"]["viewer"]["eye"]), target=list(CFG["scene"]["viewer"]["target"]))

    ensure_proxy_roots_exist(num_envs=int(CFG["scene"]["num_envs"]))

    print("[INFO]: Creating LiDAR A* planner scene...", flush=True)
    print(f"[INFO]: scenario: {args_cli.scenario}", flush=True)
    print(f"[INFO]: RayCaster attachment prim path: {CFG['lidar']['prim_path']}", flush=True)
    print(f"[INFO]: RayCaster mesh prim paths: {resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), int(CFG['scene']['num_envs']))}", flush=True)
    scene_cfg = G1LidarAStarSceneCfg(num_envs=int(CFG["scene"]["num_envs"]), env_spacing=float(CFG["scene"]["env_spacing"]))
    scene = InteractiveScene(scene_cfg)
    print_stage_prim_diagnostics(num_envs=int(CFG["scene"]["num_envs"]))

    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Setup complete.", flush=True)
    print("[INFO]: RayCaster initialized; entering A* planner loop.", flush=True)

    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
