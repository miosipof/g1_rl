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

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI + Isaac Sim launcher
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Smoke test: spawn Unitree G1 with RGB-D camera and a debug cube.")
parser.add_argument(
    "--config",
    type=str,
    default=str(PROJECT_ROOT / "configs" / "robot" / "unitree_g1.yaml"),
    help="Path to the robot/camera YAML config.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Optional override for number of environments in the YAML file.",
)
parser.add_argument(
    "--reset_interval",
    type=int,
    default=500,
    help="How many simulation steps between resets.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Isaac Lab imports after app launch
# -----------------------------------------------------------------------------
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_assets import G1_CFG  # noqa: E402

from g1_nav_rl.observations.rgbd_obs import build_rgbd_observation, summarize_observation  # noqa: E402


def load_yaml(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml(args_cli.config)
if args_cli.num_envs is not None:
    CFG["scene"]["num_envs"] = int(args_cli.num_envs)


def make_camera_prim_path(cfg: dict) -> str:
    robot_prim = cfg["robot"]["prim_path"].rstrip("/")
    mount_link = cfg["camera"]["mount_link"].strip("/")
    prim_name = cfg["camera"]["prim_name"].strip("/")
    return f"{robot_prim}/{mount_link}/{prim_name}"


def make_robot_cfg(cfg: dict) -> ArticulationCfg:
    if cfg["robot"]["asset_source"] != "builtin_isaaclab":
        raise ValueError("This smoke test only supports robot.asset_source = builtin_isaaclab")
    if cfg["robot"]["asset_name"] != "G1_CFG":
        raise ValueError("This smoke test only supports robot.asset_name = G1_CFG")

    robot_cfg = G1_CFG.replace(prim_path=cfg["robot"]["prim_path"])
    robot_cfg.init_state.pos = tuple(cfg["robot"]["init_state"]["pos"])
    robot_cfg.init_state.rot = tuple(cfg["robot"]["init_state"]["rot_wxyz"])
    return robot_cfg


def save_camera_images(obs: dict[str, torch.Tensor], out_dir: Path, step_idx: int) -> None:
    """Save RGB and depth snapshots if Pillow is available."""
    try:
        from PIL import Image
    except ImportError:
        print("[WARN]: Pillow is not installed. Skipping screenshot save.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = obs["rgb"][0].detach().cpu()
    if rgb.max().item() <= 1.0:
        rgb = (rgb * 255.0).clamp(0, 255)
    rgb_np = rgb.to(torch.uint8).numpy()
    Image.fromarray(rgb_np).save(out_dir / f"rgb_{step_idx:06d}.png")

    depth = obs["depth"][0, :, :, 0].detach().cpu()
    dmin = float(CFG["observations"]["depth_min"])
    dmax = float(CFG["observations"]["depth_max"])
    denom = max(dmax - dmin, 1e-6)
    depth_norm = ((depth - dmin) / denom).clamp(0.0, 1.0)
    depth_vis = ((1.0 - depth_norm) * 255.0).to(torch.uint8).numpy()
    Image.fromarray(depth_vis).save(out_dir / f"depth_{step_idx:06d}.png")

    print(f"[INFO]: Saved camera screenshots to {out_dir} at step {step_idx}")


ROBOT_CFG = make_robot_cfg(CFG)
CAMERA_PRIM_PATH = make_camera_prim_path(CFG)


@configclass
class G1SmokeSceneCfg(InteractiveSceneCfg):
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

    robot: ArticulationCfg = ROBOT_CFG

    if CFG.get("debug_cube", {}).get("enabled", False):
        debug_cube = AssetBaseCfg(
            prim_path=CFG["debug_cube"]["prim_path"],
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=tuple(CFG["debug_cube"]["pos"]),
            ),
            spawn=sim_utils.CuboidCfg(
                size=tuple(CFG["debug_cube"]["size"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(CFG["debug_cube"]["color"])
                ),
            ),
        )

    camera = CameraCfg(
        prim_path=CAMERA_PRIM_PATH,
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
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    local_count = 0
    global_step = 0

    print_every = int(CFG["debug"]["print_every"])
    save_every = int(CFG["debug"]["save_every"])
    save_screenshots = bool(CFG["debug"]["save_screenshots"])
    screenshot_dir = PROJECT_ROOT / CFG["debug"]["screenshot_dir"]

    while simulation_app.is_running():
        if local_count % args_cli.reset_interval == 0:
            local_count = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()
            print("[INFO]: Resetting robot state...")

        # Keep a simple pose target. This is not a balancing controller.
        joint_targets = robot.data.default_joint_pos.clone()
        robot.set_joint_position_target(joint_targets)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if global_step % print_every == 0:
            obs = build_rgbd_observation(
                scene=scene,
                robot_name="robot",
                camera_name="camera",
                depth_min=float(CFG["observations"]["depth_min"]),
                depth_max=float(CFG["observations"]["depth_max"]),
                normalize_rgb=bool(CFG["observations"]["rgb_normalize"]),
            )

            print("-" * 80)
            print(summarize_observation(obs))
            print(f"camera prim path : {CAMERA_PRIM_PATH}")
            print(f"base_lin_vel_b[0]: {obs['base_lin_vel_b'][0].tolist()}")
            print(f"base_ang_vel_b[0]: {obs['base_ang_vel_b'][0].tolist()}")
            print(f"depth min/max    : {obs['depth'].min().item():.4f} / {obs['depth'].max().item():.4f}")

            if save_screenshots and save_every > 0 and global_step % save_every == 0:
                save_camera_images(obs, screenshot_dir, global_step)

        local_count += 1
        global_step += 1


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=list(CFG["scene"]["viewer"]["eye"]),
        target=list(CFG["scene"]["viewer"]["target"]),
    )

    scene_cfg = G1SmokeSceneCfg(
        num_envs=int(CFG["scene"]["num_envs"]),
        env_spacing=float(CFG["scene"]["env_spacing"]),
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()