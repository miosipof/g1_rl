from __future__ import annotations

from typing import Any

import torch


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """Detach and convert a tensor to float32."""
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
    return x.detach().to(dtype=torch.float32).contiguous()


def build_rgbd_observation(
    scene: Any,
    robot_name: str = "robot",
    camera_name: str = "camera",
    depth_min: float = 0.0,
    depth_max: float = 10.0,
    normalize_rgb: bool = True,
) -> dict[str, torch.Tensor]:
    """Build a minimal observation dictionary from an Isaac Lab scene.

    Returned keys:
        - rgb:            [N, H, W, 3]
        - depth:          [N, H, W, 1]
        - proprio:        [N, 9 + 2*num_joints]
        - base_lin_vel_b: [N, 3]
        - base_ang_vel_b: [N, 3]
        - gravity_b:      [N, 3]
        - joint_pos:      [N, num_joints]
        - joint_vel:      [N, num_joints]
    """
    robot = scene[robot_name]
    camera = scene[camera_name]

    rgb = _to_float_tensor(camera.data.output["rgb"])
    if normalize_rgb and float(rgb.max().item()) > 1.0:
        rgb = rgb / 255.0

    depth = _to_float_tensor(camera.data.output["distance_to_image_plane"])
    depth = torch.nan_to_num(depth, nan=depth_max, posinf=depth_max, neginf=depth_min)
    depth = depth.clamp(min=depth_min, max=depth_max)

    base_lin_vel_b = _to_float_tensor(robot.data.root_lin_vel_b)
    base_ang_vel_b = _to_float_tensor(robot.data.root_ang_vel_b)
    gravity_b = _to_float_tensor(robot.data.projected_gravity_b)
    joint_pos = _to_float_tensor(robot.data.joint_pos)
    joint_vel = _to_float_tensor(robot.data.joint_vel)

    proprio = torch.cat(
        [base_lin_vel_b, base_ang_vel_b, gravity_b, joint_pos, joint_vel],
        dim=-1,
    ).contiguous()

    return {
        "rgb": rgb,
        "depth": depth,
        "proprio": proprio,
        "base_lin_vel_b": base_lin_vel_b,
        "base_ang_vel_b": base_ang_vel_b,
        "gravity_b": gravity_b,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def summarize_observation(obs: dict[str, torch.Tensor]) -> str:
    """Pretty summary for quick smoke-test prints."""
    rgb = obs["rgb"]
    depth = obs["depth"]
    proprio = obs["proprio"]
    return (
        f"rgb={tuple(rgb.shape)} {rgb.dtype}, "
        f"depth={tuple(depth.shape)} {depth.dtype}, "
        f"proprio={tuple(proprio.shape)} {proprio.dtype}"
    )