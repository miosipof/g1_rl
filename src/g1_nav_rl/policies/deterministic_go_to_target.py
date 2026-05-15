from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import torch


@runtime_checkable
class CommandProxyLike(Protocol):
    """Minimal state interface required by the deterministic go-to-target policy."""

    pos_w: torch.Tensor
    yaw: torch.Tensor


@dataclass(frozen=True)
class GoToTargetControllerCfg:
    """Configuration for the deterministic command-level go-to-target controller.

    The controller outputs a body-frame velocity command ``[vx, vy, wz]``:

    - ``vx``: forward/backward velocity command
    - ``vy``: lateral velocity command, positive to robot-left by convention
    - ``wz``: yaw-rate command

    The implementation intentionally mirrors the original logic from
    ``scripts/smoke_test_go_to_target.py``. It does not include command limits,
    acceleration limits, smoothing, or safety filtering; those remain downstream
    responsibilities of the command proxy and safety layer.
    """

    k_vx: float = 0.8
    k_vy: float = 0.8
    k_wz: float = 1.8
    slow_radius: float = 1.0
    success_radius: float = 0.20

    @classmethod
    def from_mapping(
        cls,
        controller_cfg: Mapping[str, object],
        *,
        success_radius: float,
    ) -> "GoToTargetControllerCfg":
        """Create a controller config from the existing YAML structure.

        Expected input is ``CFG["target"]["controller"]`` from
        ``configs/robot/g1_command_proxy.yaml`` plus ``CFG["target"]["success_radius"]``.
        """
        return cls(
            k_vx=float(controller_cfg["k_vx"]),
            k_vy=float(controller_cfg["k_vy"]),
            k_wz=float(controller_cfg["k_wz"]),
            slow_radius=float(controller_cfg["slow_radius"]),
            success_radius=float(success_radius),
        )


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _validate_state_tensors(pos_xy_w: torch.Tensor, yaw: torch.Tensor, target_xy_w: torch.Tensor) -> None:
    if not torch.is_tensor(pos_xy_w):
        raise TypeError(f"pos_xy_w must be a torch.Tensor, got {type(pos_xy_w)!r}")
    if not torch.is_tensor(yaw):
        raise TypeError(f"yaw must be a torch.Tensor, got {type(yaw)!r}")
    if not torch.is_tensor(target_xy_w):
        raise TypeError(f"target_xy_w must be a torch.Tensor, got {type(target_xy_w)!r}")

    if pos_xy_w.ndim != 2 or pos_xy_w.shape[-1] != 2:
        raise ValueError(f"pos_xy_w must have shape [N, 2], got {tuple(pos_xy_w.shape)}")
    if yaw.ndim != 1:
        raise ValueError(f"yaw must have shape [N], got {tuple(yaw.shape)}")
    if target_xy_w.ndim != 2 or target_xy_w.shape[-1] != 2:
        raise ValueError(f"target_xy_w must have shape [N, 2], got {tuple(target_xy_w.shape)}")
    if pos_xy_w.shape[0] != yaw.shape[0] or pos_xy_w.shape[0] != target_xy_w.shape[0]:
        raise ValueError(
            "Batch sizes must match: "
            f"pos_xy_w={pos_xy_w.shape[0]}, yaw={yaw.shape[0]}, target_xy_w={target_xy_w.shape[0]}"
        )


def compute_go_to_target_command_from_state(
    pos_xy_w: torch.Tensor,
    yaw: torch.Tensor,
    target_xy_w: torch.Tensor,
    cfg: GoToTargetControllerCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a body-frame ``[vx, vy, wz]`` command from pose and target tensors.

    Args:
        pos_xy_w: Current proxy/body position in world frame, shape ``[N, 2]``.
        yaw: Current proxy/body yaw in world frame, shape ``[N]``.
        target_xy_w: Target position in world frame, shape ``[N, 2]``.
        cfg: Deterministic controller gains and radii.

    Returns:
        ``(command, distance, yaw_error)`` where:

        - ``command`` has shape ``[N, 3]`` and stores ``[vx, vy, wz]`` in body frame.
        - ``distance`` has shape ``[N]`` and stores Euclidean distance to target.
        - ``yaw_error`` has shape ``[N]`` and stores desired-heading error in radians.
    """
    _validate_state_tensors(pos_xy_w, yaw, target_xy_w)

    target_xy_w = target_xy_w.to(device=pos_xy_w.device, dtype=pos_xy_w.dtype)
    yaw = yaw.to(device=pos_xy_w.device, dtype=pos_xy_w.dtype)

    delta_w = target_xy_w - pos_xy_w
    distance = torch.linalg.norm(delta_w, dim=-1)

    desired_yaw = torch.atan2(delta_w[:, 1], delta_w[:, 0])
    yaw_error = wrap_to_pi(desired_yaw - yaw)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # World vector expressed in body frame: R_world_body^T * delta_world.
    rel_x_b = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
    rel_y_b = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]

    slow_radius = max(float(cfg.slow_radius), 1e-6)
    speed_scale = (distance / slow_radius).clamp(0.0, 1.0)

    vx = float(cfg.k_vx) * rel_x_b * speed_scale
    vy = float(cfg.k_vy) * rel_y_b * speed_scale
    wz = float(cfg.k_wz) * yaw_error

    command = torch.stack([vx, vy, wz], dim=-1)

    reached = distance < float(cfg.success_radius)
    command[reached, :] = 0.0

    return command, distance, yaw_error


def compute_go_to_target_command(
    proxy: CommandProxyLike,
    target_xy_w: torch.Tensor,
    cfg: GoToTargetControllerCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a body-frame ``[vx, vy, wz]`` command for a ``G1CommandProxy``-like object.

    This is the direct reusable equivalent of the original smoke-test helper.
    The proxy must expose:

    - ``proxy.pos_w`` with shape ``[N, 3]``
    - ``proxy.yaw`` with shape ``[N]``
    """
    if not hasattr(proxy, "pos_w") or not hasattr(proxy, "yaw"):
        raise TypeError("proxy must expose pos_w and yaw tensors")

    pos_w = proxy.pos_w
    yaw = proxy.yaw

    if not torch.is_tensor(pos_w):
        raise TypeError(f"proxy.pos_w must be a torch.Tensor, got {type(pos_w)!r}")
    if pos_w.ndim != 2 or pos_w.shape[-1] < 2:
        raise ValueError(f"proxy.pos_w must have shape [N, >=2], got {tuple(pos_w.shape)}")

    return compute_go_to_target_command_from_state(
        pos_xy_w=pos_w[:, 0:2],
        yaw=yaw,
        target_xy_w=target_xy_w,
        cfg=cfg,
    )
