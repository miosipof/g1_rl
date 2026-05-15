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
parser = argparse.ArgumentParser(description="Deterministic A-to-B controller for G1 command proxy.")
parser.add_argument(
    "--config",
    type=str,
    default=str(PROJECT_ROOT / "configs" / "robot" / "g1_command_proxy.yaml"),
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=1200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

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
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from g1_nav_rl.robots.g1_command_proxy import (  # noqa: E402
    G1CommandProxy,
    G1CommandProxyLimits,
)
from g1_nav_rl.policies.deterministic_go_to_target import (  # noqa: E402
    GoToTargetControllerCfg,
    compute_go_to_target_command,
)
from g1_nav_rl.safety.depth_safety_filter import (  # noqa: E402
    DepthSafetyFilterCfg,
    apply_depth_safety_filter,
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


def ensure_proxy_roots_exist(num_envs: int) -> None:
    stage = omni.usd.get_context().get_stage()
    proxy_paths = resolve_env_prim_paths(CFG["proxy"]["prim_path"], num_envs)

    for path in proxy_paths:
        UsdGeom.Xform.Define(stage, path)
        print(f"[INFO]: Ensured proxy root exists: {path}", flush=True)



def save_camera_images(obs: dict[str, torch.Tensor], out_dir: Path, step_idx: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("[WARN]: Pillow is not installed. Skipping screenshot save.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = obs["rgb"][0].detach().cpu()
    if rgb.max().item() <= 1.0:
        rgb = (rgb * 255.0).clamp(0, 255)
    Image.fromarray(rgb.to(torch.uint8).numpy()).save(out_dir / f"rgb_{step_idx:06d}.png")

    depth = obs["depth"][0, :, :, 0].detach().cpu()
    dmin = float(CFG["observations"]["depth_min"])
    dmax = float(CFG["observations"]["depth_max"])
    depth_norm = ((depth - dmin) / max(dmax - dmin, 1e-6)).clamp(0.0, 1.0)

    # Near objects bright, far objects dark.
    depth_vis = ((1.0 - depth_norm) * 255.0).to(torch.uint8).numpy()
    Image.fromarray(depth_vis).save(out_dir / f"depth_{step_idx:06d}.png")

    print(f"[INFO]: Saved screenshots to {out_dir} at step {step_idx}", flush=True)


def build_camera_observation(scene: InteractiveScene, proxy: G1CommandProxy) -> dict[str, torch.Tensor]:
    camera = scene["camera"]

    rgb = camera.data.output["rgb"].detach().to(dtype=torch.float32).contiguous()
    if bool(CFG["observations"]["rgb_normalize"]) and float(rgb.max().item()) > 1.0:
        rgb = rgb / 255.0

    depth = camera.data.output["distance_to_image_plane"].detach().to(dtype=torch.float32).contiguous()
    depth = torch.nan_to_num(
        depth,
        nan=float(CFG["observations"]["depth_max"]),
        posinf=float(CFG["observations"]["depth_max"]),
        neginf=float(CFG["observations"]["depth_min"]),
    )
    depth = depth.clamp(
        min=float(CFG["observations"]["depth_min"]),
        max=float(CFG["observations"]["depth_max"]),
    )

    return {
        "rgb": rgb,
        "depth": depth,
        "proprio": proxy.observation(),
    }



@configclass
class G1GoToTargetSceneCfg(InteractiveSceneCfg):
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

    # Visual proxy body under the clean moving proxy root.
    proxy_body = AssetBaseCfg(
        prim_path=CFG["proxy"]["body_prim_path"],
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
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
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(CFG["target"]["pos"]),
            ),
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
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(CFG["debug_cube"]["pos"]),
            ),
            spawn=sim_utils.CuboidCfg(
                size=tuple(CFG["debug_cube"]["size"]),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(CFG["debug_cube"]["color"]),
                ),
            ),
        )

    camera = CameraCfg(
        prim_path=CFG["camera"]["prim_path"],
        update_period=float(CFG["camera"]["update_period"]),
        height=int(CFG["camera"]["height"]),
        width=int(CFG["camera"]["width"]),
        data_types=list(CFG["camera"]["data_types"]),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(CFG["camera"]["pinhole"]["focal_length"]),
            focus_distance=float(CFG["camera"]["pinhole"]["focus_distance"]),
            horizontal_aperture=float(CFG["camera"]["pinhole"]["horizontal_aperture"]),
            clipping_range=tuple(CFG["camera"]["pinhole"]["clipping_range"]),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(CFG["camera"]["offset"]["pos"]),
            rot=tuple(CFG["camera"]["offset"]["rot_wxyz"]),
            convention=str(CFG["camera"]["offset"]["convention"]),
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    sim_dt = sim.get_physics_dt()
    device = torch.device(args_cli.device)
    num_envs = int(CFG["scene"]["num_envs"])

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

    proxy_prim_paths = resolve_env_prim_paths(CFG["proxy"]["prim_path"], num_envs)

    proxy = G1CommandProxy(
        proxy_prim_paths=proxy_prim_paths,
        num_envs=num_envs,
        device=device,
        init_pos=tuple(CFG["proxy"]["init_state"]["pos"]),
        init_yaw=float(CFG["proxy"]["init_state"]["yaw"]),
        limits=limits,
    )

    proxy.reset(scene.env_origins)
    scene.reset()

    target_local_xy = torch.tensor(
        CFG["target"]["pos"][0:2],
        dtype=torch.float32,
        device=device,
    )
    target_xy_w = scene.env_origins[:, 0:2].to(device=device) + target_local_xy

    target_controller_cfg = GoToTargetControllerCfg.from_mapping(
        CFG["target"]["controller"],
        success_radius=float(CFG["target"]["success_radius"]),
    )
    depth_safety_cfg = DepthSafetyFilterCfg.from_mapping(CFG.get("obstacle_safety", {}))

    print_every = int(CFG["debug"]["print_every"])
    save_every = int(CFG["debug"]["save_every"])
    save_screenshots = bool(CFG["debug"]["save_screenshots"])
    screenshot_dir = PROJECT_ROOT / "outputs" / "go_to_target_camera"

    print("[INFO]: Starting go-to-target smoke test...", flush=True)
    print(f"[INFO]: Target XY world env0: {target_xy_w[0].detach().cpu().tolist()}", flush=True)

    reached_once = False

    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    last_obs = build_camera_observation(scene, proxy)
    last_safety_info = {
        "enabled": 0.0,
        "center_min": float("nan"),
        "left_min": float("nan"),
        "right_min": float("nan"),
        "active": 0.0,
    }

    for step_idx in range(int(args_cli.max_steps)):
        if not simulation_app.is_running():
            break

        raw_command, distance, yaw_error = compute_go_to_target_command(
            proxy,
            target_xy_w,
            target_controller_cfg,
        )

        if last_obs is not None:
            command, last_safety_info = apply_depth_safety_filter(last_obs["depth"], raw_command, depth_safety_cfg)
        else:
            command = raw_command

        proxy.set_command(command)
        proxy.step(sim_dt)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        obs = build_camera_observation(scene, proxy)
        last_obs = obs

        if step_idx % print_every == 0:
            obs = build_camera_observation(scene, proxy)
            state = proxy.debug_state()

            print("-" * 80, flush=True)
            print(f"step               : {step_idx}", flush=True)
            print(f"rgb                : {tuple(obs['rgb'].shape)} {obs['rgb'].dtype}", flush=True)
            print(f"depth              : {tuple(obs['depth'].shape)} {obs['depth'].dtype}", flush=True)
            print(f"proprio            : {tuple(obs['proprio'].shape)} {obs['proprio'].dtype}", flush=True)
            print(f"target xy env0     : {target_xy_w[0].detach().cpu().tolist()}", flush=True)
            print(f"distance env0      : {float(distance[0].detach().cpu()):.3f}", flush=True)
            print(f"yaw error env0     : {float(yaw_error[0].detach().cpu()):.3f}", flush=True)
            print(f"target command     : {state['cmd_target']}", flush=True)

            print(f"raw command        : {raw_command[0].detach().cpu().tolist()}", flush=True)
            print(
                "safety depth       : "
                f"center={last_safety_info['center_min']:.3f}, "
                f"left={last_safety_info['left_min']:.3f}, "
                f"right={last_safety_info['right_min']:.3f}, "
                f"active={bool(last_safety_info['active'])}",
                flush=True,
            )

            print(f"smoothed command   : {state['cmd_current']}", flush=True)
            print(f"proxy position     : {state['pos_w']}", flush=True)
            print(f"proxy yaw          : {state['yaw'][0]:.3f}", flush=True)
            print(f"depth min/max      : {obs['depth'].min().item():.4f} / {obs['depth'].max().item():.4f}", flush=True)

            if save_screenshots and save_every > 0 and step_idx % save_every == 0:
                save_camera_images(obs, screenshot_dir, step_idx)

        if float(distance[0].detach().cpu()) < float(CFG["target"]["success_radius"]):
            if not reached_once:
                print(
                    f"[INFO]: Target reached at step {step_idx}, "
                    f"distance={float(distance[0].detach().cpu()):.3f}",
                    flush=True,
                )
                reached_once = True

            # Let it settle for a few more steps, then stop the test.
            if step_idx > 50:
                break

    print("[INFO]: Go-to-target smoke test complete.", flush=True)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=list(CFG["scene"]["viewer"]["eye"]),
        target=list(CFG["scene"]["viewer"]["target"]),
    )

    ensure_proxy_roots_exist(num_envs=int(CFG["scene"]["num_envs"]))

    print("[INFO]: Creating go-to-target scene...", flush=True)
    scene_cfg = G1GoToTargetSceneCfg(
        num_envs=int(CFG["scene"]["num_envs"]),
        env_spacing=float(CFG["scene"]["env_spacing"]),
    )
    scene = InteractiveScene(scene_cfg)

    print("[INFO]: Resetting simulator...", flush=True)
    sim.reset()
    print("[INFO]: Setup complete.", flush=True)

    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()