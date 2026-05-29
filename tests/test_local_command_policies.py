from __future__ import annotations

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.policies.ema_smoother_policy import EMASmootherPolicy, EMASmootherPolicyCfg
from g1_nav_rl.policies.local_command_policy import LocalPolicyInput, TrackerState
from g1_nav_rl.policies.local_rollout_policy import LocalRolloutPolicy, LocalRolloutPolicyCfg
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicy, RLResidualPolicyCfg, build_compact_residual_features


def _input(grid_cfg: OccupancyGridCfg | None = None) -> LocalPolicyInput:
    cfg = grid_cfg or OccupancyGridCfg()
    return LocalPolicyInput(
        occupancy_grid=torch.zeros((1, cfg.height, cfg.width), dtype=torch.float32),
        target_xy_b=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        path_xy_b=torch.tensor([[[0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.0]]], dtype=torch.float32),
        nominal_command=torch.tensor([[0.2, 0.0, 0.1]], dtype=torch.float32),
        last_command=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        target_visible=torch.tensor([1.0], dtype=torch.float32),
        target_age_s=torch.tensor([0.0], dtype=torch.float32),
        tracker_state=torch.tensor([int(TrackerState.TRACKING)], dtype=torch.long),
    )


def test_ema_smoother_limits_delta() -> None:
    pol = EMASmootherPolicy(EMASmootherPolicyCfg(alpha=1.0, max_delta_vx=0.05, max_delta_vy=0.05, max_delta_wz=0.05))
    out = pol.step(_input())
    assert torch.allclose(out.command, torch.tensor([[0.05, 0.0, 0.05]]))


def test_rl_residual_zero_initialization_reproduces_nominal() -> None:
    cfg = RLResidualPolicyCfg(path_points=4)
    inp = _input(cfg.grid_cfg)
    pol = RLResidualPolicy(cfg)
    out = pol.step(inp)
    assert torch.allclose(out.command, inp.nominal_command, atol=1e-6)
    assert out.residual is not None
    assert torch.allclose(out.residual, torch.zeros_like(out.residual), atol=1e-6)


def test_compact_features_dimension_matches_config() -> None:
    cfg = RLResidualPolicyCfg(path_points=4)
    features = build_compact_residual_features(_input(cfg.grid_cfg), cfg)
    assert features.shape == (1, cfg.feature_dim())


def test_local_rollout_policy_returns_command() -> None:
    grid_cfg = OccupancyGridCfg(x_min=-1.0, x_max=3.0, y_min=-2.0, y_max=2.0, resolution=0.2)
    pol = LocalRolloutPolicy(LocalRolloutPolicyCfg(grid_cfg=grid_cfg))
    out = pol.step(_input(grid_cfg))
    assert out.command.shape == (1, 3)
