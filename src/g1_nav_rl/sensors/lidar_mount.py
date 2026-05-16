from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import torch
from pxr import Gf, UsdGeom

from g1_nav_rl.robots.g1_command_proxy import ensure_canonical_xform_ops


@dataclass(frozen=True)
class LidarMountSyncCfg:
    """Rigid/kinematic lidar mount synchronization config.

    The current command proxy is a manually moved USD Xform. Isaac Lab RayCaster
    is more stable in headless mode when attached to a physics prim, so the lidar
    smoke tests use a tiny kinematic rigid body mount. This helper keeps that
    mount colocated with the command proxy while preserving the RayCaster offset
    as the actual lidar-frame offset.
    """

    # Pose of the RayCaster attachment mount relative to the proxy root.
    # Usually zero: the RayCasterCfg.offset then represents the lidar frame.
    mount_offset_pos_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_yaw_offset: float = 0.0


def compute_lidar_mount_pose_w(
    proxy_pos_w: torch.Tensor,
    proxy_yaw: torch.Tensor,
    cfg: LidarMountSyncCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute world pose for a lidar mount synchronized to the proxy.

    Args:
        proxy_pos_w: Proxy root positions, shape ``[N, 3]``.
        proxy_yaw: Proxy yaw angles, shape ``[N]``.
        cfg: Mount synchronization config.

    Returns:
        ``(mount_pos_w, mount_yaw)``.
    """
    if cfg is None:
        cfg = LidarMountSyncCfg()

    if not torch.is_tensor(proxy_pos_w) or not torch.is_tensor(proxy_yaw):
        raise TypeError("proxy_pos_w and proxy_yaw must be torch tensors")
    if proxy_pos_w.ndim != 2 or proxy_pos_w.shape[-1] != 3:
        raise ValueError(f"proxy_pos_w must have shape [N, 3], got {tuple(proxy_pos_w.shape)}")
    if proxy_yaw.ndim != 1 or proxy_yaw.shape[0] != proxy_pos_w.shape[0]:
        raise ValueError(f"proxy_yaw must have shape [N], got {tuple(proxy_yaw.shape)}")

    offset_b = torch.tensor(
        cfg.mount_offset_pos_b,
        dtype=proxy_pos_w.dtype,
        device=proxy_pos_w.device,
    ).unsqueeze(0)

    c = torch.cos(proxy_yaw)
    s = torch.sin(proxy_yaw)

    offset_w = torch.zeros_like(proxy_pos_w)
    offset_w[:, 0] = c * offset_b[:, 0] - s * offset_b[:, 1]
    offset_w[:, 1] = s * offset_b[:, 0] + c * offset_b[:, 1]
    offset_w[:, 2] = offset_b[:, 2]

    mount_pos_w = proxy_pos_w + offset_w
    mount_yaw = torch.atan2(
        torch.sin(proxy_yaw + float(cfg.mount_yaw_offset)),
        torch.cos(proxy_yaw + float(cfg.mount_yaw_offset)),
    )
    return mount_pos_w.contiguous(), mount_yaw.contiguous()


def write_lidar_mounts_to_usd(
    stage,
    mount_prim_paths: Sequence[str],
    mount_pos_w: torch.Tensor,
    mount_yaw: torch.Tensor,
) -> None:
    """Write synchronized lidar mount poses to USD with canonical xform ops."""
    if len(mount_prim_paths) != int(mount_pos_w.shape[0]):
        raise ValueError(
            f"Expected {mount_pos_w.shape[0]} mount paths, got {len(mount_prim_paths)}"
        )

    pos_cpu = mount_pos_w.detach().cpu()
    yaw_cpu = mount_yaw.detach().cpu()

    for env_id, prim_path in enumerate(mount_prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            UsdGeom.Xform.Define(stage, prim_path)
            prim = stage.GetPrimAtPath(prim_path)

        translate_op, orient_op, scale_op = ensure_canonical_xform_ops(prim)
        x, y, z = pos_cpu[env_id].tolist()
        yaw_half = 0.5 * float(yaw_cpu[env_id])

        translate_op.Set(Gf.Vec3d(x, y, z))
        orient_op.Set(Gf.Quatd(math.cos(yaw_half), Gf.Vec3d(0.0, 0.0, math.sin(yaw_half))))
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))


def sync_lidar_mounts_to_proxy_usd(
    stage,
    mount_prim_paths: Sequence[str],
    proxy,
    cfg: LidarMountSyncCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synchronize kinematic lidar mount prims to a command proxy pose."""
    if not hasattr(proxy, "pos_w") or not hasattr(proxy, "yaw"):
        raise TypeError("proxy must expose pos_w and yaw tensors")

    mount_pos_w, mount_yaw = compute_lidar_mount_pose_w(proxy.pos_w, proxy.yaw, cfg)
    write_lidar_mounts_to_usd(stage, mount_prim_paths, mount_pos_w, mount_yaw)
    return mount_pos_w, mount_yaw
