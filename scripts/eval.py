#!/usr/bin/env python3
"""Smoke-evaluate the residual policy seam on a synthetic local input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.policies.local_command_policy import LocalPolicyInput, TrackerState
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicy, RLResidualPolicyCfg, build_compact_residual_features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    grid_cfg = OccupancyGridCfg()
    policy_cfg = RLResidualPolicyCfg(grid_cfg=grid_cfg)
    policy = RLResidualPolicy(policy_cfg, checkpoint_path=args.checkpoint, device=args.device)
    occupancy = torch.zeros((1, grid_cfg.height, grid_cfg.width), dtype=torch.float32, device=args.device)
    path = torch.tensor([[[0.3, 0.0], [0.6, 0.0], [0.9, 0.1], [1.2, 0.2], [1.5, 0.3], [1.8, 0.3], [2.1, 0.3], [2.4, 0.3]]], dtype=torch.float32, device=args.device)
    inp = LocalPolicyInput(
        occupancy_grid=occupancy,
        target_xy_b=torch.tensor([[2.5, 0.3]], dtype=torch.float32, device=args.device),
        path_xy_b=path,
        nominal_command=torch.tensor([[0.2, 0.0, 0.1]], dtype=torch.float32, device=args.device),
        last_command=torch.tensor([[0.15, 0.0, 0.05]], dtype=torch.float32, device=args.device),
        target_visible=torch.tensor([1.0], dtype=torch.float32, device=args.device),
        target_age_s=torch.tensor([0.0], dtype=torch.float32, device=args.device),
        tracker_state=torch.tensor([int(TrackerState.TRACKING)], dtype=torch.long, device=args.device),
    )
    features = build_compact_residual_features(inp, policy_cfg)
    out = policy.step(inp)
    print(json.dumps({
        "feature_dim": int(features.shape[-1]),
        "nominal_command": inp.nominal_command.detach().cpu().tolist()[0],
        "command": out.command.detach().cpu().tolist()[0],
        "residual": None if out.residual is None else out.residual.detach().cpu().tolist()[0],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
