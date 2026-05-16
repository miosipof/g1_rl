#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
parser = argparse.ArgumentParser(description="Smoke test for proxy-mounted lidar / RayCaster sensor.")
parser.add_argument(
    "--config",
    type=str,
    default=str(PROJECT_ROOT / "configs" / "robot" / "g1_command_proxy.yaml"),
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=120)
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


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml(args_cli.config)
if args_cli.num_envs is not None:
    CFG["scene"]["num_envs"] = int(args_cli.num_envs)


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
    """Resolve lidar mesh paths for the current smoke setup.

    RayCaster currently expects static mesh targets. The default smoke test uses
    one environment and a static debug cube, so a single resolved mesh path is
    the preferred configuration.
    """
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
    """Print the concrete USD prims that matter for RayCaster init."""
    stage = omni.usd.get_context().get_stage()

    paths = [
        resolve_template_path(CFG["lidar"]["prim_path"], env_id=0),
        resolve_template_path(CFG["debug_cube"]["prim_path"], env_id=0),
    ]
    for path in CFG["lidar"].get("mesh_prim_paths", []):
        paths.append(resolve_template_path(path, env_id=0))
    if CFG.get("debug_cube", {}).get("mesh_prim_path"):
        paths.append(resolve_template_path(CFG["debug_cube"]["mesh_prim_path"], env_id=0))

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
    path = out_dir / f"occupancy_grid_{step_idx:06d}.png"
    Image.fromarray(image).save(path)
    print(f"[INFO]: Saved occupancy grid: {path}", flush=True)


@configclass
class G1LidarSceneCfg(InteractiveSceneCfg):
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

    if CFG.get("debug_cube", {}).get("enabled", False):
        debug_cube = AssetBaseCfg(
            prim_path=CFG["debug_cube"]["prim_path"],
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(CFG["debug_cube"]["pos"]),
            ),
            # RayCaster parses USD Mesh prims into Warp meshes. A plain
            # CuboidCfg creates a USD shape prim, which is not a valid
            # RayCaster mesh target. MeshCuboidCfg creates:
            #   {debug_cube.prim_path}/geometry/mesh
            # and the lidar config targets that actual mesh prim.
            spawn=sim_utils.MeshCuboidCfg(
                size=tuple(CFG["debug_cube"]["size"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(CFG["debug_cube"]["color"]),
                ),
            ),
        )

    # RayCaster pose lookup is most reliable in headless mode when the
    # attachment path is a physics prim. The command proxy root is a manually
    # moved USD Xform, so for this first stationary lidar smoke test we attach
    # the sensor to a tiny kinematic rigid mount colocated with the proxy root.
    # The sensor offset below still represents the lidar frame relative to this
    # mount. Later, for moving lidar, make the proxy body / mount a proper
    # kinematic rigid object and update it via the sim/real adapter.
    lidar_mount = AssetBaseCfg(
        prim_path=CFG["lidar"]["prim_path"],
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=tuple(CFG["proxy"]["init_state"]["pos"]),
        ),
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

    debug_cfg = CFG.get("lidar", {}).get("debug", {})
    print_every = int(debug_cfg.get("print_every", 10))
    save_every = int(debug_cfg.get("save_every", 20))
    save_grid = bool(debug_cfg.get("save_grid", True))
    grid_dir = PROJECT_ROOT / str(debug_cfg.get("grid_dir", "outputs/lidar_grid"))

    proxy = make_proxy(device=device, num_envs=num_envs)
    proxy.reset(scene.env_origins)
    scene.reset()

    print("[INFO]: Starting lidar smoke test...", flush=True)
    print(f"[INFO]: lidar mesh paths: {resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), num_envs)}", flush=True)
    print(
        f"[INFO]: occupancy grid: H={grid_cfg.height}, W={grid_cfg.width}, "
        f"resolution={grid_cfg.resolution}",
        flush=True,
    )

    # Give sensors one frame to populate buffers after reset.
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    for step_idx in range(int(args_cli.max_steps)):
        if not simulation_app.is_running():
            break

        # Keep the proxy stationary for the first lidar smoke test. We only want
        # to validate the sensor, local 2D projection, and occupancy rasterization.
        zero_cmd = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
        proxy.set_command(zero_cmd)
        proxy.step(sim_dt)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        lidar_obs = build_lidar_proxy_observation(
            scene,
            proxy,
            lidar_name="lidar",
            cfg=lidar_obs_cfg,
        )
        grid = rasterize_points_to_occupancy_grid(
            lidar_obs["lidar_points_b"],
            lidar_obs["lidar_valid"],
            grid_cfg,
        )

        if step_idx % print_every == 0:
            ranges = lidar_obs["lidar_ranges"]
            valid = lidar_obs["lidar_valid"]
            occupied = int((grid[0] > 0).detach().cpu().sum().item())
            valid_ranges = ranges[0][valid[0]]
            min_valid = float(valid_ranges.detach().cpu().min().item()) if valid_ranges.numel() else float("nan")
            max_valid = float(valid_ranges.detach().cpu().max().item()) if valid_ranges.numel() else float("nan")

            print("-" * 80, flush=True)
            print(f"step               : {step_idx}", flush=True)
            print(f"lidar              : {summarize_lidar_observation(lidar_obs)}", flush=True)
            print(f"valid range min/max: {min_valid:.3f} / {max_valid:.3f}", flush=True)
            print(f"occupied cells     : {occupied}", flush=True)
            print(f"proxy position     : {proxy.debug_state()['pos_w']}", flush=True)
            print(f"proxy yaw          : {proxy.debug_state()['yaw'][0]:.3f}", flush=True)

            if save_grid and save_every > 0 and step_idx % save_every == 0:
                save_occupancy_grid_image(grid, grid_dir, step_idx)

    print("[INFO]: Lidar smoke test complete.", flush=True)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=list(CFG["scene"]["viewer"]["eye"]),
        target=list(CFG["scene"]["viewer"]["target"]),
    )

    ensure_proxy_roots_exist(num_envs=int(CFG["scene"]["num_envs"]))

    print("[INFO]: Creating lidar scene...", flush=True)
    print(f"[INFO]: RayCaster attachment prim path: {CFG['lidar']['prim_path']}", flush=True)
    print(
        f"[INFO]: RayCaster mesh prim paths: "
        f"{resolve_mesh_prim_paths(list(CFG['lidar']['mesh_prim_paths']), int(CFG['scene']['num_envs']))}",
        flush=True,
    )
    scene_cfg = G1LidarSceneCfg(
        num_envs=int(CFG["scene"]["num_envs"]),
        env_spacing=float(CFG["scene"]["env_spacing"]),
    )
    scene = InteractiveScene(scene_cfg)
    print_stage_prim_diagnostics(num_envs=int(CFG["scene"]["num_envs"]))

    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Setup complete.", flush=True)
    print("[INFO]: RayCaster initialized; entering simulation loop.", flush=True)

    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
