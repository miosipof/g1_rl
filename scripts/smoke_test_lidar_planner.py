#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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
# CLI + Isaac Sim launcher
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Smoke test for LiDAR occupancy-grid planning: static RayCaster hits "
        "are transformed into the moving proxy frame, then a DWA/DWB-style "
        "local velocity rollout planner commands the proxy toward a fixed target."
    )
)
parser.add_argument(
    "--config",
    type=str,
    default=str(PROJECT_ROOT / "configs" / "robot" / "g1_command_proxy.yaml"),
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=180)
parser.add_argument(
    "--assert_planner",
    action="store_true",
    help=(
        "Assert that the planner makes target progress and uses at least one "
        "lateral/yaw avoidance command when the debug cube is present."
    ),
)
parser.add_argument(
    "--disable_obstacle",
    action="store_true",
    help="Disable the debug cube/RayCaster obstacle target for a clear-path planner sanity check.",
)
AppLauncher.add_app_launcher_args(parser)
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
from g1_nav_rl.planning.local_velocity_rollout import (  # noqa: E402
    LocalVelocityRolloutCfg,
    plan_local_velocity_command,
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
from g1_nav_rl.utils.geometry import target_vector_body_frame  # noqa: E402


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml(args_cli.config)
if args_cli.num_envs is not None:
    CFG["scene"]["num_envs"] = int(args_cli.num_envs)
if args_cli.disable_obstacle:
    CFG.setdefault("debug_cube", {})["enabled"] = False


def resolve_env_prim_paths(template_path: str, num_envs: int) -> list[str]:
    paths = []
    for env_id in range(num_envs):
        env_ns = f"/World/envs/env_{env_id}"
        paths.append(template_path.replace("{ENV_REGEX_NS}", env_ns))
    return paths


def resolve_template_path(template_path: str, env_id: int = 0) -> str:
    env_ns = f"/World/envs/env_{env_id}"
    return template_path.replace("{ENV_REGEX_NS}", env_ns)


def resolve_mesh_prim_paths(mesh_paths: list[str], num_envs: int) -> list[str]:
    resolved: list[str] = []
    for path in mesh_paths:
        if "{ENV_REGEX_NS}" in path:
            if num_envs != 1:
                print(
                    "[WARN]: lidar.mesh_prim_paths contains {ENV_REGEX_NS}; "
                    "this smoke test is intended for num_envs=1.",
                    flush=True,
                )
            resolved.append(resolve_template_path(path, env_id=0))
        else:
            resolved.append(path)
    return resolved


def ensure_proxy_roots_exist(num_envs: int) -> None:
    stage = omni.usd.get_context().get_stage()
    proxy_paths = resolve_env_prim_paths(CFG["proxy"]["prim_path"], num_envs)

    for path in proxy_paths:
        proxy_xform = UsdGeom.Xform.Define(stage, path)
        ensure_canonical_xform_ops(proxy_xform.GetPrim())
        print(f"[INFO]: Ensured canonical proxy root exists: {path}", flush=True)


def print_stage_prim_diagnostics(num_envs: int) -> None:
    stage = omni.usd.get_context().get_stage()
    paths = [
        resolve_template_path(CFG["lidar"]["prim_path"], env_id=0),
        resolve_template_path(CFG["target"]["prim_path"], env_id=0),
    ]
    if CFG.get("debug_cube", {}).get("enabled", False):
        paths.append(resolve_template_path(CFG["debug_cube"]["prim_path"], env_id=0))
        paths.append(resolve_template_path(CFG["debug_cube"]["mesh_prim_path"], env_id=0))
    for path in CFG["lidar"].get("mesh_prim_paths", []):
        paths.append(resolve_template_path(path, env_id=0))

    print("[INFO]: Stage prim diagnostics before sim.reset():", flush=True)
    for path in dict.fromkeys(paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            print(f"  - {path}: INVALID", flush=True)
            continue
        is_mesh = bool(UsdGeom.Mesh(prim))
        is_xform = bool(UsdGeom.Xformable(prim))
        print(
            f"  - {path}: type={prim.GetTypeName()} is_mesh={is_mesh} is_xformable={is_xform}",
            flush=True,
        )


def save_occupancy_grid_image(grid: torch.Tensor, out_dir: Path, step_idx: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("[WARN]: Pillow is not installed. Skipping occupancy-grid save.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    image = occupancy_grid_to_uint8_image(grid, env_id=0).numpy()
    path = out_dir / f"planner_grid_{step_idx:06d}.png"
    Image.fromarray(image).save(path)
    print(f"[INFO]: Saved planner occupancy grid: {path}", flush=True)


@configclass
class G1LidarPlannerSceneCfg(InteractiveSceneCfg):
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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(CFG["proxy"]["body"]["color"]),
            ),
        ),
    )

    if CFG.get("target", {}).get("enabled", True):
        target = AssetBaseCfg(
            prim_path=CFG["target"]["prim_path"],
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(CFG["target"]["pos"])),
            spawn=sim_utils.CuboidCfg(
                size=tuple(CFG["target"]["size"]),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(CFG["target"]["color"]),
                ),
            ),
        )

    if CFG.get("debug_cube", {}).get("enabled", False):
        debug_cube = AssetBaseCfg(
            prim_path=CFG["debug_cube"]["prim_path"],
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(CFG["debug_cube"]["pos"])),
            spawn=sim_utils.MeshCuboidCfg(
                size=tuple(CFG["debug_cube"]["size"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(CFG["debug_cube"]["color"]),
                ),
            ),
        )

    lidar_mount = AssetBaseCfg(
        prim_path=CFG["lidar"]["prim_path"],
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(CFG["proxy"]["init_state"]["pos"])),
        spawn=sim_utils.CuboidCfg(
            size=tuple(CFG["lidar"].get("mount_size", [0.05, 0.05, 0.05])),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(CFG["lidar"].get("mount_color", [1.0, 0.5, 0.0])),
            ),
        ),
    )

    lidar = RayCasterCfg(
        prim_path=CFG["lidar"]["prim_path"],
        update_period=float(CFG["lidar"]["update_period"]),
        offset=RayCasterCfg.OffsetCfg(
            pos=tuple(CFG["lidar"]["offset"]["pos"]),
            rot=tuple(CFG["lidar"]["offset"]["rot_wxyz"]),
        ),
        mesh_prim_paths=resolve_mesh_prim_paths(
            list(CFG["lidar"]["mesh_prim_paths"]),
            int(CFG["scene"]["num_envs"]),
        ),
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
    planner_cfg = LocalVelocityRolloutCfg.from_mapping(CFG.get("local_planner", {}))

    planner_debug_cfg = CFG.get("local_planner", {}).get("debug", {})
    print_every = int(planner_debug_cfg.get("print_every", CFG.get("lidar", {}).get("debug", {}).get("print_every", 10)))
    save_every = int(planner_debug_cfg.get("save_every", CFG.get("lidar", {}).get("debug", {}).get("save_every", 20)))
    save_grid = bool(planner_debug_cfg.get("save_grid", True))
    grid_dir = PROJECT_ROOT / str(planner_debug_cfg.get("grid_dir", "outputs/lidar_planner"))

    proxy = make_proxy(device=device, num_envs=num_envs)
    proxy.reset(scene.env_origins)

    lidar_mount_paths = resolve_env_prim_paths(CFG["lidar"]["prim_path"], num_envs)
    mount_sync_cfg = LidarMountSyncCfg()
    sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)

    scene.reset()

    target_local_xy = torch.tensor(CFG["target"]["pos"][0:2], dtype=torch.float32, device=device)
    target_xy_w = scene.env_origins[:, 0:2].to(device=device) + target_local_xy

    print("[INFO]: Starting LiDAR planner smoke test...", flush=True)
    print(f"[INFO]: target_xy_w env0: {target_xy_w[0].detach().cpu().tolist()}", flush=True)
    print(f"[INFO]: lidar mesh paths: {resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), num_envs)}", flush=True)
    print(
        f"[INFO]: occupancy grid: H={grid_cfg.height}, W={grid_cfg.width}, resolution={grid_cfg.resolution}",
        flush=True,
    )
    print(
        f"[INFO]: planner samples: |vx|={len(planner_cfg.vx_samples)}, "
        f"|vy|={len(planner_cfg.vy_samples)}, |wz|={len(planner_cfg.wz_samples)}, "
        f"horizon={planner_cfg.horizon_s}s",
        flush=True,
    )

    distance_history: list[float] = []
    avoidance_command_count = 0
    blocked_count = 0

    # Give RayCaster one frame to populate buffers after reset.
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    for step_idx in range(int(args_cli.max_steps)):
        if not simulation_app.is_running():
            break

        lidar_obs = build_lidar_proxy_observation(scene, proxy, lidar_name="lidar", cfg=lidar_obs_cfg)
        grid = rasterize_points_to_occupancy_grid(
            lidar_obs["lidar_points_b"],
            lidar_obs["lidar_valid"],
            grid_cfg,
        )
        target_vec = target_vector_body_frame(proxy.pos_w[:, 0:2], proxy.yaw, target_xy_w)
        planner_cmd, planner_info = plan_local_velocity_command(
            occupancy_grid=grid,
            target_xy_b=target_vec,
            last_command=proxy.cmd_current,
            grid_cfg=grid_cfg,
            cfg=planner_cfg,
        )

        if abs(float(planner_cmd[0, 1].detach().cpu())) > 0.05 or abs(float(planner_cmd[0, 2].detach().cpu())) > 0.05:
            avoidance_command_count += 1
        if not bool(planner_info["any_collision_free"][0].detach().cpu()):
            blocked_count += 1

        proxy.set_command(planner_cmd)
        proxy.step(sim_dt)
        sync_lidar_mounts_to_proxy_usd(proxy.stage, lidar_mount_paths, proxy, mount_sync_cfg)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        target_distance = float(
            torch.linalg.norm(target_xy_w[0] - proxy.pos_w[0, 0:2]).detach().cpu().item()
        )
        distance_history.append(target_distance)

        if step_idx % print_every == 0:
            ranges = lidar_obs["lidar_ranges"]
            valid = lidar_obs["lidar_valid"]
            occupied = int((grid[0] > 0).detach().cpu().sum().item())
            valid_ranges = ranges[0][valid[0]]
            min_valid = float(valid_ranges.detach().cpu().min().item()) if valid_ranges.numel() else float("nan")
            rel_points = lidar_obs["lidar_points_b"][0][valid[0]]
            rel_distances = torch.linalg.norm(rel_points, dim=-1) if rel_points.numel() else torch.empty(0, device=device)
            min_rel_distance = (
                float(rel_distances.detach().cpu().min().item())
                if rel_distances.numel()
                else float("nan")
            )
            best_score = float(planner_info["best_score"][0].detach().cpu())
            best_clearance = float(planner_info["best_clearance"][0].detach().cpu())
            best_progress = float(planner_info["best_progress"][0].detach().cpu())
            effective_radius = float(planner_info["effective_collision_radius"][0].detach().cpu())
            any_free = bool(planner_info["any_collision_free"][0].detach().cpu())

            print("-" * 80, flush=True)
            print(f"step               : {step_idx}", flush=True)
            print(f"lidar              : {summarize_lidar_observation(lidar_obs)}", flush=True)
            print(f"raw lidar min range: {min_valid:.3f}", flush=True)
            print(f"proxy-relative min : {min_rel_distance:.3f}", flush=True)
            print(f"occupied cells     : {occupied}", flush=True)
            print(f"target vector env0 : {[round(x, 3) for x in target_vec[0].detach().cpu().tolist()]}", flush=True)
            print(f"target distance    : {target_distance:.3f}", flush=True)
            print(f"planner command    : {[round(x, 3) for x in planner_cmd[0].detach().cpu().tolist()]}", flush=True)
            print(f"proxy cmd_current  : {[round(x, 3) for x in proxy.cmd_current[0].detach().cpu().tolist()]}", flush=True)
            print(f"planner score      : {best_score:.3f}", flush=True)
            print(f"planner progress   : {best_progress:.3f}", flush=True)
            print(f"planner clearance  : {best_clearance:.3f}", flush=True)
            print(f"effective radius   : {effective_radius:.3f}", flush=True)
            print(f"any collision-free : {any_free}", flush=True)
            print(f"proxy position     : {proxy.debug_state()['pos_w']}", flush=True)
            print(f"proxy yaw          : {proxy.debug_state()['yaw'][0]:.3f}", flush=True)

            if save_grid and save_every > 0 and step_idx % save_every == 0:
                save_occupancy_grid_image(grid, grid_dir, step_idx)

    if distance_history:
        print(
            f"[INFO]: target distance first/last/min: "
            f"{distance_history[0]:.3f} / {distance_history[-1]:.3f} / {min(distance_history):.3f}",
            flush=True,
        )
    print(f"[INFO]: avoidance command count: {avoidance_command_count}", flush=True)
    print(f"[INFO]: blocked count: {blocked_count}", flush=True)

    if args_cli.assert_planner:
        if len(distance_history) < 2:
            raise RuntimeError("No target distance history was collected.")
        if not (distance_history[-1] < distance_history[0] - 0.05):
            raise RuntimeError(
                "Expected the planner to reduce target distance by at least 0.05 m. "
                f"first={distance_history[0]:.3f}, last={distance_history[-1]:.3f}"
            )
        if CFG.get("debug_cube", {}).get("enabled", False) and avoidance_command_count <= 0:
            raise RuntimeError(
                "Expected at least one lateral/yaw avoidance command with the debug cube enabled."
            )

    print("[INFO]: LiDAR planner smoke test complete.", flush=True)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=list(CFG["scene"]["viewer"]["eye"]),
        target=list(CFG["scene"]["viewer"]["target"]),
    )

    ensure_proxy_roots_exist(num_envs=int(CFG["scene"]["num_envs"]))

    print("[INFO]: Creating LiDAR planner scene...", flush=True)
    print(f"[INFO]: RayCaster attachment prim path: {CFG['lidar']['prim_path']}", flush=True)
    print(
        f"[INFO]: RayCaster mesh prim paths: "
        f"{resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), int(CFG['scene']['num_envs']))}",
        flush=True,
    )
    scene_cfg = G1LidarPlannerSceneCfg(
        num_envs=int(CFG["scene"]["num_envs"]),
        env_spacing=float(CFG["scene"]["env_spacing"]),
    )
    scene = InteractiveScene(scene_cfg)
    print_stage_prim_diagnostics(num_envs=int(CFG["scene"]["num_envs"]))

    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Setup complete.", flush=True)
    print("[INFO]: RayCaster initialized; entering planner loop.", flush=True)

    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
