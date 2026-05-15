from __future__ import annotations

import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to the interval [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def rotate_world_to_body_2d(vec_world: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate 2D world-frame vectors into the robot body frame.

    Args:
        vec_world: Tensor with shape ``[..., 2]``.
        yaw: World-frame body yaw, broadcastable to ``vec_world[..., 0]``.

    Returns:
        Tensor with shape ``[..., 2]``.
    """
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x_w = vec_world[..., 0]
    y_w = vec_world[..., 1]

    x_b = c * x_w + s * y_w
    y_b = -s * x_w + c * y_w

    return torch.stack((x_b, y_b), dim=-1)


def rotate_body_to_world_2d(vec_body: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate 2D body-frame vectors into the world frame."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)

    x_b = vec_body[..., 0]
    y_b = vec_body[..., 1]

    x_w = c * x_b - s * y_b
    y_w = s * x_b + c * y_b

    return torch.stack((x_w, y_w), dim=-1)


def distance_xy(a_xy: torch.Tensor, b_xy: torch.Tensor) -> torch.Tensor:
    """Return Euclidean distance between batched XY positions."""
    return torch.linalg.norm(a_xy - b_xy, dim=-1)


def target_vector_body_frame(
    pos_xy_w: torch.Tensor,
    yaw: torch.Tensor,
    target_xy_w: torch.Tensor,
) -> torch.Tensor:
    """Build the navigation target vector used by the command-level environment.

    The returned tensor has shape ``[N, 4]``:

    ``[target_x_b, target_y_b, distance, yaw_error]``

    where ``target_x_b`` and ``target_y_b`` are the world target vector expressed
    in the proxy/robot body frame.
    """
    if pos_xy_w.ndim != 2 or pos_xy_w.shape[-1] != 2:
        raise ValueError(f"pos_xy_w must have shape [N, 2], got {tuple(pos_xy_w.shape)}")
    if target_xy_w.ndim != 2 or target_xy_w.shape[-1] != 2:
        raise ValueError(f"target_xy_w must have shape [N, 2], got {tuple(target_xy_w.shape)}")
    if yaw.ndim != 1:
        raise ValueError(f"yaw must have shape [N], got {tuple(yaw.shape)}")
    if pos_xy_w.shape[0] != target_xy_w.shape[0] or pos_xy_w.shape[0] != yaw.shape[0]:
        raise ValueError(
            "Batch sizes must match: "
            f"pos_xy_w={pos_xy_w.shape[0]}, target_xy_w={target_xy_w.shape[0]}, yaw={yaw.shape[0]}"
        )

    target_xy_w = target_xy_w.to(device=pos_xy_w.device, dtype=pos_xy_w.dtype)
    yaw = yaw.to(device=pos_xy_w.device, dtype=pos_xy_w.dtype)

    delta_w = target_xy_w - pos_xy_w
    delta_b = rotate_world_to_body_2d(delta_w, yaw)
    distance = torch.linalg.norm(delta_w, dim=-1)
    desired_yaw = torch.atan2(delta_w[:, 1], delta_w[:, 0])
    yaw_error = wrap_to_pi(desired_yaw - yaw)

    return torch.cat((delta_b, distance.unsqueeze(-1), yaw_error.unsqueeze(-1)), dim=-1)
