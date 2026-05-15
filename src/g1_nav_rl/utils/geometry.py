# src/g1_nav_rl/utils/geometry.py

from __future__ import annotations

import math

import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles to quaternions in wxyz convention.

    Args:
        yaw: Tensor of shape [N] or scalar tensor.

    Returns:
        Tensor of shape [N, 4] with quaternion order [w, x, y, z].
    """
    half = 0.5 * yaw
    q = torch.zeros((*yaw.shape, 4), device=yaw.device, dtype=yaw.dtype)
    q[..., 0] = torch.cos(half)
    q[..., 3] = torch.sin(half)
    return q


def rotate_body_to_world_2d(
    vec_body: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Rotate 2D body-frame vectors into world frame.

    Args:
        vec_body: Tensor [..., 2].
        yaw: Tensor broadcastable to vec_body[..., 0].

    Returns:
        Tensor [..., 2].
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x_b = vec_body[..., 0]
    y_b = vec_body[..., 1]

    x_w = c * x_b - s * y_b
    y_w = s * x_b + c * y_b

    return torch.stack((x_w, y_w), dim=-1)


def rotate_world_to_body_2d(
    vec_world: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Rotate 2D world-frame vectors into body frame."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x_w = vec_world[..., 0]
    y_w = vec_world[..., 1]

    x_b = c * x_w + s * y_w
    y_b = -s * x_w + c * y_w

    return torch.stack((x_b, y_b), dim=-1)


def heading_to_target(
    pos_xy_w: torch.Tensor,
    target_xy_w: torch.Tensor,
) -> torch.Tensor:
    """Return world-frame heading angle from position to target."""
    delta = target_xy_w - pos_xy_w
    return torch.atan2(delta[..., 1], delta[..., 0])


def distance_xy(
    a_xy: torch.Tensor,
    b_xy: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Euclidean 2D distance."""
    return torch.linalg.norm(a_xy - b_xy, dim=-1).clamp_min(eps)