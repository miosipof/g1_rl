from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import torch
import omni.usd
from pxr import Gf, UsdGeom


@dataclass
class G1CommandProxyLimits:
    max_vx: float
    max_vy: float
    max_wz: float
    max_ax: float
    max_ay: float
    max_awz: float
    smoothing_alpha: float



def ensure_canonical_xform_ops(prim) -> tuple[UsdGeom.XformOp, UsdGeom.XformOp, UsdGeom.XformOp]:
    """Ensure Isaac-Lab-compatible USD xform ops on a prim.

    Isaac Lab transform views validate that Xform prims use exactly this op order:

        xformOp:translate, xformOp:orient, xformOp:scale

    A raw ``UsdGeom.Xform.Define(...)`` prim has an empty op order, and
    ``XformCommonAPI.SetRotate(...)`` creates a rotateXYZ op. Both can break
    sensors such as RayCaster that initialize transform views during ``sim.reset``.
    """
    xformable = UsdGeom.Xformable(prim)
    ordered_ops = list(xformable.GetOrderedXformOps())
    ordered_names = [op.GetOpName() for op in ordered_ops]
    canonical_names = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

    if ordered_names != canonical_names:
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        orient_op.Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
        scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
        return translate_op, orient_op, scale_op

    op_by_name = {op.GetOpName(): op for op in ordered_ops}
    return (
        op_by_name["xformOp:translate"],
        op_by_name["xformOp:orient"],
        op_by_name["xformOp:scale"],
    )


class G1CommandProxy:
    """Command-level G1 proxy.

    This deliberately avoids PhysX rigid-body tensor writes.

    The proxy models the real robot command interface:

        move(vx, vy, wz)

    and updates a USD transform directly. Sensors attached under the proxy prim
    move with it.
    """

    def __init__(
        self,
        proxy_prim_paths: Sequence[str],
        num_envs: int,
        device: str | torch.device,
        init_pos: tuple[float, float, float],
        init_yaw: float,
        limits: G1CommandProxyLimits,
    ) -> None:
        self.proxy_prim_paths = list(proxy_prim_paths)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.init_pos_local = torch.tensor(init_pos, dtype=torch.float32, device=self.device)
        self.init_yaw = float(init_yaw)
        self.limits = limits

        if len(self.proxy_prim_paths) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} proxy prim paths, got {len(self.proxy_prim_paths)}"
            )

        self.stage = omni.usd.get_context().get_stage()

        self.pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.yaw = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self.cmd_target = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.cmd_current = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def reset(self, env_origins: torch.Tensor) -> None:
        env_origins = env_origins.to(device=self.device, dtype=torch.float32)

        self.pos_w[:] = env_origins + self.init_pos_local
        self.yaw[:] = self.init_yaw
        self.cmd_target.zero_()
        self.cmd_current.zero_()

        self.write_to_usd()

    def set_command(self, command: torch.Tensor) -> None:
        command = command.to(device=self.device, dtype=torch.float32)

        if command.ndim == 1:
            command = command.unsqueeze(0).repeat(self.num_envs, 1)

        if command.shape != (self.num_envs, 3):
            raise ValueError(f"Expected command shape {(self.num_envs, 3)}, got {tuple(command.shape)}")

        self.cmd_target[:, 0] = command[:, 0].clamp(-self.limits.max_vx, self.limits.max_vx)
        self.cmd_target[:, 1] = command[:, 1].clamp(-self.limits.max_vy, self.limits.max_vy)
        self.cmd_target[:, 2] = command[:, 2].clamp(-self.limits.max_wz, self.limits.max_wz)

    def step(self, dt: float) -> None:
        dt = float(dt)

        alpha = float(self.limits.smoothing_alpha)
        smoothed = (1.0 - alpha) * self.cmd_current + alpha * self.cmd_target

        delta = smoothed - self.cmd_current
        max_delta = torch.tensor(
            [
                self.limits.max_ax * dt,
                self.limits.max_ay * dt,
                self.limits.max_awz * dt,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        delta = torch.max(torch.min(delta, max_delta), -max_delta)
        self.cmd_current += delta

        vx_b = self.cmd_current[:, 0]
        vy_b = self.cmd_current[:, 1]
        wz = self.cmd_current[:, 2]

        cos_yaw = torch.cos(self.yaw)
        sin_yaw = torch.sin(self.yaw)

        vx_w = cos_yaw * vx_b - sin_yaw * vy_b
        vy_w = sin_yaw * vx_b + cos_yaw * vy_b

        self.pos_w[:, 0] += vx_w * dt
        self.pos_w[:, 1] += vy_w * dt
        self.yaw += wz * dt
        self.yaw = torch.atan2(torch.sin(self.yaw), torch.cos(self.yaw))

        self.write_to_usd()

    def write_to_usd(self) -> None:
        """Move the USD proxy prims directly."""
        pos_cpu = self.pos_w.detach().cpu()
        yaw_cpu = self.yaw.detach().cpu()

        for env_id, prim_path in enumerate(self.proxy_prim_paths):
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                UsdGeom.Xform.Define(self.stage, prim_path)
                prim = self.stage.GetPrimAtPath(prim_path)

            translate_op, orient_op, scale_op = ensure_canonical_xform_ops(prim)

            x, y, z = pos_cpu[env_id].tolist()
            yaw_half = 0.5 * float(yaw_cpu[env_id])

            translate_op.Set(Gf.Vec3d(x, y, z))
            orient_op.Set(
                Gf.Quatd(
                    math.cos(yaw_half),
                    Gf.Vec3d(0.0, 0.0, math.sin(yaw_half)),
                )
            )
            scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))

    def observation(self) -> torch.Tensor:
        """Return [x, y, yaw, vx, vy, wz]."""
        return torch.cat(
            [
                self.pos_w[:, 0:2],
                self.yaw.unsqueeze(-1),
                self.cmd_current,
            ],
            dim=-1,
        )

    def debug_state(self) -> dict[str, list[float]]:
        return {
            "pos_w": self.pos_w[0].detach().cpu().tolist(),
            "yaw": [float(self.yaw[0].detach().cpu())],
            "cmd_target": self.cmd_target[0].detach().cpu().tolist(),
            "cmd_current": self.cmd_current[0].detach().cpu().tolist(),
        }