#!/usr/bin/env python3
"""Smoke test: Isaac RGB-D camera + person-like target + YOLO bbox -> Target2D.

This script is intentionally perception-only. It does not move the G1 proxy and
it does not invoke NavRuntime. The goal is to validate the Phase-5 bridge:

    RGB image + aligned depth + bbox + camera intrinsics + camera pose
        -> Target2D(x, y)

Import-order note: Isaac/Omniverse modules are imported only after AppLauncher
creates the SimulationApp. This avoids ``ModuleNotFoundError: No module named
'omni'`` when running outside the Isaac python bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402


# -----------------------------------------------------------------------------
# CLI + Isaac app launch
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Smoke test YOLO/person RGB-D target extraction in Isaac Sim.")
parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "yolo_person_target"))
parser.add_argument("--warmup_steps", type=int, default=20)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--focal_length", type=float, default=24.0)
parser.add_argument("--horizontal_aperture", type=float, default=20.955)
parser.add_argument("--focus_distance", type=float, default=400.0)
parser.add_argument("--clip_near", type=float, default=0.10)
parser.add_argument("--clip_far", type=float, default=100.0)
parser.add_argument("--camera_x", type=float, default=0.0, help="Camera/body x in world, meters.")
parser.add_argument("--camera_y", type=float, default=0.0, help="Camera/body y in world, meters.")
parser.add_argument("--camera_z", type=float, default=0.75, help="Camera height in world, meters.")
parser.add_argument("--person_x", type=float, default=3.0)
parser.add_argument("--person_y", type=float, default=0.0)
parser.add_argument("--person_yaw", type=float, default=3.141592653589793)
parser.add_argument("--person_asset_usd", type=str, default="", help="Optional realistic human USD asset. If omitted, use a simple proxy humanoid.")
parser.add_argument("--person_asset_scale", type=float, default=1.0)
parser.add_argument("--skip_person_asset_reference", action="store_true", help="Debug: do not add the provided person USD reference; useful to isolate asset-load crashes.")
parser.add_argument("--person_asset_reference_mode", type=str, default="reference", choices=["reference", "payload"], help="How to compose the person USD into the stage.")
parser.add_argument("--debug_checkpoints", action="store_true", default=True, help="Print detailed checkpoints around Isaac scene setup and data extraction.")
parser.add_argument("--skip_yolo", action="store_true", help="Do not run YOLO; use projected synthetic bbox instead.")
parser.add_argument("--yolo_model", type=str, default="yolov8n.pt")
parser.add_argument("--yolo_conf", type=float, default=0.25)
parser.add_argument("--yolo_iou", type=float, default=0.45)
parser.add_argument("--yolo_device", type=str, default="")
parser.add_argument("--target_class", type=str, default="person")
parser.add_argument("--min_confidence", type=float, default=0.25)
parser.add_argument("--selection_policy", type=str, default="nearest", choices=["nearest", "largest", "highest_confidence"])
parser.add_argument("--fallback_synthetic_bbox", action="store_true", help="Use projected known person bbox if YOLO is unavailable or no person is detected.")
parser.add_argument("--synthetic_bbox_w", type=float, default=120.0)
parser.add_argument("--synthetic_bbox_h", type=float, default=260.0)
parser.add_argument("--synthetic_point_z", type=float, default=1.0, help="World z of projected proxy bbox center.")
parser.add_argument("--depth_min", type=float, default=0.20)
parser.add_argument("--depth_max", type=float, default=8.0)
parser.add_argument("--crop_frac", type=float, default=0.50)
parser.add_argument("--depth_percentile", type=float, default=0.50)
parser.add_argument("--stand_off", type=float, default=0.8)
parser.add_argument("--assert_detection", action="store_true", help="Fail if YOLO/fallback detection is unavailable.")
parser.add_argument("--assert_target", action="store_true", help="Fail if bbox+depth does not produce Target2D.")
parser.add_argument("--assert_target_error", type=float, default=-1.0, help="Optional max XY error vs known person center. Disabled if <0.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Isaac/Omniverse imports after app launch
# -----------------------------------------------------------------------------
import torch  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from g1_nav_rl.navigation.simple_nav import Pose2D  # noqa: E402
from g1_nav_rl.perception.target_from_bbox import (  # noqa: E402
    CameraIntrinsics,
    Detection2D,
    TargetFromBBoxConfig,
    bbox_depth_to_target2d,
    make_forward_camera_T_body_camera,
    pose2d_to_T_world_body,
    select_detection_for_target,
)

SCRIPT_VERSION = "smoke_test_yolo_person_target_v3_asset_wrapper_reference"


def ckpt(msg: str) -> None:
    if bool(getattr(args_cli, "debug_checkpoints", False)):
        print(f"[CKPT]: {msg}", flush=True)


# -----------------------------------------------------------------------------
# Geometry / camera helpers
# -----------------------------------------------------------------------------
def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * float(yaw)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def compute_intrinsics() -> CameraIntrinsics:
    # Pinhole camera mm/mm -> pixel focal length. Vertical aperture is inferred
    # by Isaac from horizontal aperture and resolution for square pixels.
    fx = float(args_cli.focal_length) / float(args_cli.horizontal_aperture) * float(args_cli.width)
    fy = fx
    return CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=0.5 * (int(args_cli.width) - 1),
        cy=0.5 * (int(args_cli.height) - 1),
        width=int(args_cli.width),
        height=int(args_cli.height),
    )


def T_world_camera_assumed() -> np.ndarray:
    # Same convention as target_from_bbox.py: body/world x forward, y left, z up;
    # camera x right, y down, z optical forward.
    robot_pose = Pose2D(float(args_cli.camera_x), float(args_cli.camera_y), 0.0)
    T_world_body = pose2d_to_T_world_body(robot_pose, z=0.0)
    T_body_camera = make_forward_camera_T_body_camera(
        translation_body=(0.0, 0.0, float(args_cli.camera_z))
    )
    return T_world_body @ T_body_camera


def project_world_to_pixel(point_w: Sequence[float], intr: CameraIntrinsics, T_world_camera: np.ndarray) -> tuple[float, float, float] | None:
    p_w = np.array([float(point_w[0]), float(point_w[1]), float(point_w[2]), 1.0], dtype=np.float64)
    T_camera_world = np.linalg.inv(np.asarray(T_world_camera, dtype=np.float64))
    p_c = T_camera_world @ p_w
    x, y, z = float(p_c[0]), float(p_c[1]), float(p_c[2])
    if not math.isfinite(z) or z <= 1e-6:
        return None
    u = float(intr.fx) * x / z + float(intr.cx)
    v = float(intr.fy) * y / z + float(intr.cy)
    return u, v, z


def synthetic_person_bbox_from_projection(intr: CameraIntrinsics, T_world_camera: np.ndarray) -> Detection2D:
    projected = project_world_to_pixel(
        (float(args_cli.person_x), float(args_cli.person_y), float(args_cli.synthetic_point_z)),
        intr,
        T_world_camera,
    )
    if projected is None:
        raise RuntimeError("known person point projects behind the camera; check camera/person pose")
    u, v, _ = projected
    w = float(args_cli.synthetic_bbox_w)
    h = float(args_cli.synthetic_bbox_h)
    return Detection2D(
        class_name=str(args_cli.target_class),
        bbox_xyxy=(u - 0.5 * w, v - 0.5 * h, u + 0.5 * w, v + 0.5 * h),
        confidence=1.0,
    )


def tensor_rgb_to_numpy(rgb: torch.Tensor) -> np.ndarray:
    rgb0 = rgb[0].detach().cpu()
    if rgb0.dtype != torch.uint8:
        if float(rgb0.max().item()) <= 1.0:
            rgb0 = rgb0 * 255.0
        rgb0 = rgb0.clamp(0, 255).to(torch.uint8)
    return rgb0.numpy()


def tensor_depth_to_numpy(depth: torch.Tensor) -> np.ndarray:
    depth0 = depth[0].detach().cpu()
    if depth0.ndim == 3 and depth0.shape[-1] == 1:
        depth0 = depth0[..., 0]
    return depth0.to(torch.float32).numpy()


def save_outputs(out_dir: Path, rgb_np: np.ndarray, depth_np: np.ndarray, metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "rgb.npy", rgb_np)
    np.save(out_dir / "depth_m.npy", depth_np.astype(np.float32, copy=False))
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    try:
        from PIL import Image

        Image.fromarray(rgb_np).save(out_dir / "rgb.png")
        d = depth_np.astype(np.float32, copy=False)
        valid = np.isfinite(d)
        d_vis = np.zeros_like(d, dtype=np.uint8)
        if valid.any():
            d_clip = np.clip(d, float(args_cli.depth_min), float(args_cli.depth_max))
            d_norm = (d_clip - float(args_cli.depth_min)) / max(float(args_cli.depth_max) - float(args_cli.depth_min), 1e-6)
            d_vis = ((1.0 - d_norm) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(d_vis).save(out_dir / "depth_vis.png")
    except Exception as exc:
        print(f"[WARN]: could not save PNG previews: {exc}", flush=True)


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/PerceptionCamera"
USE_PROXY_PERSON = not bool(str(args_cli.person_asset_usd).strip())


@configclass
class YoloPersonTargetSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=700.0, color=(1.0, 1.0, 1.0)),
    )

    camera = CameraCfg(
        prim_path=CAMERA_PRIM_PATH,
        update_period=0.0,
        height=int(args_cli.height),
        width=int(args_cli.width),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args_cli.focal_length),
            focus_distance=float(args_cli.focus_distance),
            horizontal_aperture=float(args_cli.horizontal_aperture),
            clipping_range=(float(args_cli.clip_near), float(args_cli.clip_far)),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(float(args_cli.camera_x), float(args_cli.camera_y), float(args_cli.camera_z)),
            # ROS camera convention: optical axis forward. This matches the
            # T_world_camera_assumed() helper used for target extraction.
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    if USE_PROXY_PERSON:
        # A simple person-like proxy for camera/depth plumbing. It is not meant
        # to be a reliable YOLO-person visual target; pass --person_asset_usd for that.
        person_torso = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/ProxyPersonTorso",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(float(args_cli.person_x), float(args_cli.person_y), 1.05),
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
                pos=(float(args_cli.person_x), float(args_cli.person_y), 1.65),
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
                pos=(float(args_cli.person_x), float(args_cli.person_y) + 0.08, 0.45),
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
                pos=(float(args_cli.person_x), float(args_cli.person_y) - 0.08, 0.45),
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

    Important: do not put the payload/reference and transform ops on the same
    prim. Some character assets have authored xform/skeleton state on their
    root prim; composing them and then adding transform ops to the same prim can
    force expensive/fragile composition. Instead:

        /World/envs/env_0/PersonAsset          # wrapper Xform with transform
        /World/envs/env_0/PersonAsset/Model    # referenced/payload asset
    """
    if USE_PROXY_PERSON:
        ckpt("using proxy person; no external person USD reference")
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
    ckpt("getting USD stage before person reference")
    stage = omni.usd.get_context().get_stage()

    # Transform the lightweight wrapper first, before composing the heavy asset.
    ckpt(f"defining wrapper Xform at {wrapper_path}")
    wrapper_xform = UsdGeom.Xform.Define(stage, wrapper_path)
    wrapper_xform.ClearXformOpOrder()
    ckpt("setting wrapper translate op")
    wrapper_xform.AddTranslateOp().Set(Gf.Vec3d(float(args_cli.person_x), float(args_cli.person_y), 0.0))
    ckpt("setting wrapper rotateZ op")
    wrapper_xform.AddRotateZOp().Set(float(math.degrees(float(args_cli.person_yaw))))
    scale = float(args_cli.person_asset_scale)
    ckpt("setting wrapper scale op")
    wrapper_xform.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))

    ckpt(f"defining model Xform at {model_path}")
    model_xform = UsdGeom.Xform.Define(stage, model_path)
    model_prim = model_xform.GetPrim()
    mode = str(args_cli.person_asset_reference_mode)
    ckpt(f"adding person USD as {mode} on child model prim: {usd_path}")
    if mode == "payload":
        model_prim.GetPayloads().AddPayload(usd_path)
    else:
        model_prim.GetReferences().AddReference(usd_path)
    print(f"[INFO]: Added person asset ({mode}): {usd_path} -> {model_path}; wrapper={wrapper_path}", flush=True)

    # BBox computation can force full payload/reference loading. Keep it as a
    # best-effort diagnostic after all lightweight setup is complete.
    try:
        ckpt("computing person asset bbox before sim.reset")
        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
        bbox = bbox_cache.ComputeWorldBound(wrapper_xform.GetPrim()).ComputeAlignedBox()
        mn = bbox.GetMin()
        mx = bbox.GetMax()
        print(
            f"[INFO]: person asset bbox before reset: "
            f"min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
            f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})",
            flush=True,
        )
    except Exception as exc:
        print(f"[WARN]: could not compute person asset bbox before reset: {exc}", flush=True)

def run_yolo_if_requested(rgb_np: np.ndarray) -> list[Detection2D]:
    if bool(args_cli.skip_yolo):
        print("[INFO]: YOLO skipped by --skip_yolo", flush=True)
        return []
    try:
        from g1_nav_rl.perception.yolo_detector import YoloDetector, YoloDetectorConfig

        detector = YoloDetector(
            YoloDetectorConfig(
                model=str(args_cli.yolo_model),
                confidence=float(args_cli.yolo_conf),
                iou=float(args_cli.yolo_iou),
                device=str(args_cli.yolo_device) or None,
                verbose=False,
            )
        )
        detections = detector.detect(rgb_np)
        print(f"[INFO]: YOLO detections: {len(detections)}", flush=True)
        for det in detections:
            print(
                f"  det class={det.class_name!r} conf={det.confidence:.3f} "
                f"bbox={tuple(round(v, 1) for v in det.bbox_xyxy)} area={det.area:.1f}",
                flush=True,
            )
        return detections
    except Exception as exc:
        print(f"[WARN]: YOLO unavailable or failed: {exc}", flush=True)
        return []


def main() -> None:
    print(f"[INFO]: script version: {SCRIPT_VERSION}", flush=True)
    print(f"[INFO]: person source: {'proxy_humanoid' if USE_PROXY_PERSON else args_cli.person_asset_usd}", flush=True)

    ckpt("creating SimulationCfg")
    sim_cfg = sim_utils.SimulationCfg(dt=0.02, device=args_cli.device)
    ckpt("creating SimulationContext")
    sim = sim_utils.SimulationContext(sim_cfg)
    ckpt("SimulationContext created")
    sim.set_camera_view(eye=[-2.0, -3.0, 2.0], target=[float(args_cli.person_x), float(args_cli.person_y), 1.0])

    ckpt("creating InteractiveSceneCfg")
    scene_cfg = YoloPersonTargetSceneCfg(num_envs=1, env_spacing=1.0)
    ckpt("creating InteractiveScene")
    scene = InteractiveScene(scene_cfg)
    ckpt("InteractiveScene created")
    add_referenced_person_asset()

    ckpt("calling sim.reset()")
    sim.reset()
    ckpt("sim.reset() returned")
    ckpt("calling scene.reset()")
    scene.reset()
    ckpt("scene.reset() returned")

    ckpt(f"starting warmup steps: {max(1, int(args_cli.warmup_steps))}")
    for i in range(max(1, int(args_cli.warmup_steps))):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        if i in (0, max(1, int(args_cli.warmup_steps)) - 1):
            ckpt(f"warmup step {i + 1}/{max(1, int(args_cli.warmup_steps))} complete")

    ckpt("reading camera sensor output")
    camera = scene["camera"]
    rgb = camera.data.output.get("rgb")
    depth = camera.data.output.get("distance_to_image_plane")
    if rgb is None or depth is None:
        raise RuntimeError("camera did not produce rgb and distance_to_image_plane outputs")

    rgb_np = tensor_rgb_to_numpy(rgb)
    depth_np = tensor_depth_to_numpy(depth)
    depth_np = np.nan_to_num(depth_np, nan=float(args_cli.depth_max), posinf=float(args_cli.depth_max), neginf=float(args_cli.depth_min))
    depth_np = np.clip(depth_np, float(args_cli.depth_min), float(args_cli.depth_max)).astype(np.float32)

    intr = compute_intrinsics()
    T_wc = T_world_camera_assumed()
    ckpt("running YOLO or skipping/fallback")
    detections = run_yolo_if_requested(rgb_np)
    selected = select_detection_for_target(
        detections,
        allowed_classes={str(args_cli.target_class)},
        min_confidence=float(args_cli.min_confidence),
        policy=str(args_cli.selection_policy),
        depth_image=depth_np,
        depth_min_m=float(args_cli.depth_min),
        depth_max_m=float(args_cli.depth_max),
    )
    detection_source = "yolo"
    if selected is None and (bool(args_cli.fallback_synthetic_bbox) or bool(args_cli.skip_yolo)):
        selected = synthetic_person_bbox_from_projection(intr, T_wc)
        detection_source = "synthetic_projection"
        print(
            f"[INFO]: using fallback synthetic bbox: {tuple(round(v, 1) for v in selected.bbox_xyxy)}",
            flush=True,
        )

    if selected is None:
        msg = "no target detection selected"
        if bool(args_cli.assert_detection):
            raise AssertionError(msg)
        print(f"[WARN]: {msg}. Use --fallback_synthetic_bbox for geometry-only validation.", flush=True)

    target_result = None
    if selected is not None:
        target_result = bbox_depth_to_target2d(
            selected,
            depth_np,
            intr,
            T_wc,
            robot_pose=Pose2D(float(args_cli.camera_x), float(args_cli.camera_y), 0.0),
            config=TargetFromBBoxConfig(
                depth_min_m=float(args_cli.depth_min),
                depth_max_m=float(args_cli.depth_max),
                bbox_center_crop_frac=float(args_cli.crop_frac),
                depth_percentile=float(args_cli.depth_percentile),
                stand_off_distance_m=float(args_cli.stand_off),
            ),
        )
        if target_result is None:
            msg = "bbox+depth did not produce a valid Target2D"
            if bool(args_cli.assert_target):
                raise AssertionError(msg)
            print(f"[WARN]: {msg}", flush=True)

    known_xy = (float(args_cli.person_x), float(args_cli.person_y))
    metadata: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "person_source": "proxy_humanoid" if USE_PROXY_PERSON else str(args_cli.person_asset_usd),
        "known_person_xy": list(known_xy),
        "intrinsics": intr.to_dict(),
        "camera_xyz": [float(args_cli.camera_x), float(args_cli.camera_y), float(args_cli.camera_z)],
        "rgb_shape": list(rgb_np.shape),
        "depth_shape": list(depth_np.shape),
        "depth_minmax_m": [float(np.nanmin(depth_np)), float(np.nanmax(depth_np))],
        "detections": [
            {"class_name": d.class_name, "confidence": d.confidence, "bbox_xyxy": list(d.bbox_xyxy)}
            for d in detections
        ],
        "selected_detection": None,
        "detection_source": detection_source if selected is not None else "none",
        "target": None,
    }
    if selected is not None:
        metadata["selected_detection"] = {
            "class_name": selected.class_name,
            "confidence": selected.confidence,
            "bbox_xyxy": list(selected.bbox_xyxy),
        }
    if target_result is not None:
        object_xy = target_result.object_xy_w
        target_xy = (float(target_result.target.x), float(target_result.target.y))
        object_error = float(math.hypot(object_xy[0] - known_xy[0], object_xy[1] - known_xy[1]))
        target_error = float(math.hypot(target_xy[0] - known_xy[0], target_xy[1] - known_xy[1]))
        metadata["target"] = {
            "depth_m": float(target_result.depth_m),
            "point_camera": target_result.point_camera.tolist(),
            "point_world": target_result.point_world.tolist(),
            "object_xy_w": list(object_xy),
            "target_xy_w": list(target_xy),
            "object_error_xy_m": object_error,
            "target_error_xy_m": target_error,
            "used_stand_off": bool(target_result.used_stand_off),
        }
        print("[INFO]: bbox/depth target result", flush=True)
        print(f"  selected source/class/conf : {metadata['detection_source']} / {selected.class_name} / {selected.confidence:.3f}", flush=True)
        print(f"  bbox_xyxy                  : {tuple(round(v, 1) for v in selected.bbox_xyxy)}", flush=True)
        print(f"  depth_m                    : {target_result.depth_m:.3f}", flush=True)
        print(f"  object_xy_w                : ({object_xy[0]:.3f}, {object_xy[1]:.3f})", flush=True)
        print(f"  target_xy_w                : ({target_xy[0]:.3f}, {target_xy[1]:.3f})", flush=True)
        print(f"  known_person_xy            : ({known_xy[0]:.3f}, {known_xy[1]:.3f})", flush=True)
        print(f"  object_error_xy_m          : {object_error:.3f}", flush=True)
        print(f"  used_stand_off             : {target_result.used_stand_off}", flush=True)
        if float(args_cli.assert_target_error) >= 0.0 and object_error > float(args_cli.assert_target_error):
            raise AssertionError(
                f"object XY error {object_error:.3f} exceeds --assert_target_error {args_cli.assert_target_error:.3f}"
            )

    out_dir = Path(args_cli.output_dir)
    ckpt("saving outputs")
    save_outputs(out_dir, rgb_np, depth_np, metadata)
    print(f"[INFO]: saved outputs to {out_dir}", flush=True)
    print("[INFO]: yolo/person target smoke test complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
