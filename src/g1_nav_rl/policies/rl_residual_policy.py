"""Bounded residual policy on top of a deterministic local command."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from g1_nav_rl.features.path_features import clearance_sectors_from_grid, path_curvature_features
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg

from .local_command_policy import LocalPolicyInput, LocalPolicyOutput, TrackerState
from .residual_mlp import ResidualMLP, ResidualMLPCfg


@dataclass(frozen=True)
class RLResidualPolicyCfg:
    """Configuration for the first learned policy seam.

    The network emits unconstrained logits.  ``tanh`` then maps them into the
    residual limits below, so a loaded or randomly initialized model cannot
    command arbitrary velocities through this layer.
    """

    path_points: int = 8
    use_clearance_features: bool = True
    use_curvature_features: bool = True
    residual_limit_vx: float = 0.10
    residual_limit_vy: float = 0.10
    residual_limit_wz: float = 0.25
    hidden_sizes: tuple[int, ...] = (128, 128)
    activation: str = "elu"
    grid_cfg: OccupancyGridCfg = field(default_factory=OccupancyGridCfg)
    source: str = "rl_residual_policy"

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "RLResidualPolicyCfg":
        """Build a policy config from YAML/checkpoint mappings.

        Accepted keys use either the public training-config names
        (``residual_limits: {vx, vy, wz}``) or the dataclass field names.
        """
        if cfg is None:
            return cls()
        grid_raw = cfg.get("grid_cfg", cfg.get("grid")) if isinstance(cfg, Mapping) else None
        if isinstance(grid_raw, OccupancyGridCfg):
            grid_cfg = grid_raw
        elif isinstance(grid_raw, Mapping):
            grid_cfg = OccupancyGridCfg.from_mapping(grid_raw)
        else:
            grid_cfg = OccupancyGridCfg()
        limits = cfg.get("residual_limits", {}) if isinstance(cfg.get("residual_limits", {}), Mapping) else {}
        hidden = cfg.get("hidden_sizes", cls.hidden_sizes)
        return cls(
            path_points=int(cfg.get("path_points", cls.path_points)),
            use_clearance_features=bool(cfg.get("use_clearance_features", cls.use_clearance_features)),
            use_curvature_features=bool(cfg.get("use_curvature_features", cls.use_curvature_features)),
            residual_limit_vx=float(limits.get("vx", cfg.get("residual_limit_vx", cls.residual_limit_vx))),
            residual_limit_vy=float(limits.get("vy", cfg.get("residual_limit_vy", cls.residual_limit_vy))),
            residual_limit_wz=float(limits.get("wz", cfg.get("residual_limit_wz", cls.residual_limit_wz))),
            hidden_sizes=tuple(int(v) for v in hidden),
            activation=str(cfg.get("activation", cls.activation)),
            grid_cfg=grid_cfg,
            source=str(cfg.get("source", cls.source)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint/YAML-safe representation."""
        return {
            "path_points": int(self.path_points),
            "use_clearance_features": bool(self.use_clearance_features),
            "use_curvature_features": bool(self.use_curvature_features),
            "residual_limit_vx": float(self.residual_limit_vx),
            "residual_limit_vy": float(self.residual_limit_vy),
            "residual_limit_wz": float(self.residual_limit_wz),
            "hidden_sizes": [int(v) for v in self.hidden_sizes],
            "activation": str(self.activation),
            "grid_cfg": asdict(self.grid_cfg),
            "source": str(self.source),
        }

    def feature_dim(self) -> int:
        # target xy + distance + bearing
        dim = 4
        # fixed path points
        dim += 2 * int(self.path_points)
        # nominal command + previous command
        dim += 6
        # target visible, target age
        dim += 2
        # tracker state one-hot
        dim += 3
        if bool(self.use_clearance_features):
            dim += 3
        if bool(self.use_curvature_features):
            dim += 3
        return dim

    @property
    def residual_limits(self) -> torch.Tensor:
        return torch.tensor(
            [float(self.residual_limit_vx), float(self.residual_limit_vy), float(self.residual_limit_wz)],
            dtype=torch.float32,
        )

    def model_cfg(self) -> ResidualMLPCfg:
        return ResidualMLPCfg(
            input_dim=self.feature_dim(),
            hidden_sizes=tuple(int(v) for v in self.hidden_sizes),
            output_dim=3,
            activation=str(self.activation),
            zero_init_output=True,
        )


def build_compact_residual_features(policy_input: LocalPolicyInput, cfg: RLResidualPolicyCfg) -> torch.Tensor:
    """Build compact features for the residual smoother.

    This function is intentionally deterministic and independent from any RL
    library. It is used by BC training, PPO wrappers, offline replay, and the
    real dry-run runtime.
    """

    x = policy_input.to(policy_input.device)
    n = x.num_envs
    device = x.device
    target_xy = x.target_xy_b.to(dtype=torch.float32)
    target_distance = torch.linalg.norm(target_xy, dim=-1, keepdim=True)
    target_bearing = torch.atan2(target_xy[:, 1:2], target_xy[:, 0:1])

    path = x.path_xy_b.to(dtype=torch.float32)
    k = int(cfg.path_points)
    if path.shape[1] < k:
        pad = torch.zeros((n, k - path.shape[1], 2), dtype=torch.float32, device=device)
        path = torch.cat((path, pad), dim=1)
    elif path.shape[1] > k:
        path = path[:, :k, :]
    path_flat = path.reshape(n, 2 * k)

    visible = (
        torch.ones((n, 1), dtype=torch.float32, device=device)
        if x.target_visible is None
        else x.target_visible.to(device=device, dtype=torch.float32).reshape(n, 1)
    )
    age = (
        torch.zeros((n, 1), dtype=torch.float32, device=device)
        if x.target_age_s is None
        else x.target_age_s.to(device=device, dtype=torch.float32).reshape(n, 1)
    )
    tracker = (
        torch.full((n,), int(TrackerState.TRACKING), dtype=torch.long, device=device)
        if x.tracker_state is None
        else x.tracker_state.to(device=device, dtype=torch.long).clamp(0, 2)
    )
    tracker_one_hot = F.one_hot(tracker, num_classes=3).to(dtype=torch.float32)

    parts = [
        target_xy,
        target_distance,
        target_bearing,
        path_flat,
        x.nominal_command.to(dtype=torch.float32),
        x.last_command.to(dtype=torch.float32),
        visible,
        age,
        tracker_one_hot,
    ]

    if bool(cfg.use_clearance_features):
        if x.occupancy_grid is None:
            clearance = torch.full((n, 3), 6.0, dtype=torch.float32, device=device)
        else:
            grid_for_features = cfg.grid_cfg
            if (
                x.occupancy_grid.shape[1] != cfg.grid_cfg.height
                or x.occupancy_grid.shape[2] != cfg.grid_cfg.width
            ):
                # Checkpoints may be trained with a coarser synthetic grid than
                # the real runtime grid. Clearance features only need the same
                # body-frame extent, so infer resolution from the actual tensor
                # instead of failing on shape mismatch.
                inferred_res_x = (float(cfg.grid_cfg.x_max) - float(cfg.grid_cfg.x_min)) / float(x.occupancy_grid.shape[1])
                inferred_res_y = (float(cfg.grid_cfg.y_max) - float(cfg.grid_cfg.y_min)) / float(x.occupancy_grid.shape[2])
                inferred_res = 0.5 * (inferred_res_x + inferred_res_y)
                grid_for_features = OccupancyGridCfg(
                    x_min=cfg.grid_cfg.x_min,
                    x_max=cfg.grid_cfg.x_max,
                    y_min=cfg.grid_cfg.y_min,
                    y_max=cfg.grid_cfg.y_max,
                    resolution=inferred_res,
                    occupied_value=cfg.grid_cfg.occupied_value,
                )
            clearance = clearance_sectors_from_grid(x.occupancy_grid, grid_for_features).to(device=device)
            # Normalize to roughly [0,1] without losing monotonicity.
            clearance = clearance / 6.0
        parts.append(clearance)

    if bool(cfg.use_curvature_features):
        parts.append(path_curvature_features(path).to(device=device, dtype=torch.float32))

    if x.extra_features is not None:
        parts.append(x.extra_features.to(device=device, dtype=torch.float32))

    features = torch.cat(parts, dim=-1).contiguous()
    expected = cfg.feature_dim() + (0 if x.extra_features is None else x.extra_features.shape[-1])
    if features.shape[-1] != expected:
        raise RuntimeError(f"feature dimension mismatch: got {features.shape[-1]}, expected {expected}")
    return features


class RLResidualPolicy:
    """Apply a learned or zero-initialized residual to the nominal command."""

    def __init__(
        self,
        cfg: RLResidualPolicyCfg | None = None,
        *,
        model: ResidualMLP | None = None,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or RLResidualPolicyCfg()
        self.device = torch.device(device)
        self.model = model or ResidualMLP(self.cfg.model_cfg())
        self.model.to(self.device)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def reset(self, *, num_envs: int | None = None, device: torch.device | str | None = None) -> None:
        if device is not None and torch.device(device) != self.device:
            self.device = torch.device(device)
            self.model.to(self.device)

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        if isinstance(checkpoint, Mapping) and "policy_cfg" in checkpoint and isinstance(checkpoint["policy_cfg"], Mapping):
            loaded_cfg = RLResidualPolicyCfg.from_mapping(checkpoint["policy_cfg"])
            if loaded_cfg.feature_dim() != self.cfg.feature_dim() or tuple(loaded_cfg.hidden_sizes) != tuple(self.cfg.hidden_sizes):
                self.cfg = loaded_cfg
                self.model = ResidualMLP(self.cfg.model_cfg()).to(self.device)
            else:
                self.cfg = loaded_cfg
        if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, Mapping) and "policy_state_dict" in checkpoint:
            state_dict = checkpoint["policy_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def step(self, policy_input: LocalPolicyInput) -> LocalPolicyOutput:
        x = policy_input.to(self.device)
        features = build_compact_residual_features(x, self.cfg)
        with torch.no_grad():
            logits = self.model(features)
            limits = self.cfg.residual_limits.to(device=features.device)
            residual = torch.tanh(logits) * limits[None, :]
            command = x.nominal_command.to(device=features.device, dtype=torch.float32) + residual
        metadata: dict[str, Any] = {
            "policy": "rl_residual",
            "features": features.detach().cpu(),
            "residual_limits": self.cfg.residual_limits,
        }
        return LocalPolicyOutput(command=command.contiguous(), residual=residual.contiguous(), metadata=metadata)


__all__ = ["RLResidualPolicy", "RLResidualPolicyCfg", "build_compact_residual_features"]
