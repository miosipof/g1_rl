#!/usr/bin/env python3
"""Train the compact residual local-command policy.

Current supported mode: behavior cloning from a teacher dataset produced by a
future data-collection script or external log conversion.

Expected ``.npz`` fields for BC:
  - ``features``: float array ``[N, F]`` from ``build_compact_residual_features``
  - ``residuals``: float array ``[N, 3]`` teacher residuals

The PPO config is present because this repo is moving toward Isaac Lab training,
but this script intentionally does not fake a full Isaac Lab PPO runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import yaml

from g1_nav_rl.policies.residual_mlp import ResidualMLP, ResidualMLPCfg
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicyCfg


def _policy_cfg_from_mapping(raw: dict[str, Any]) -> RLResidualPolicyCfg:
    policy = raw.get("policy", raw.get("residual_policy", {})) or {}
    limits = policy.get("residual_limits", {}) or {}
    return RLResidualPolicyCfg(
        path_points=int(policy.get("path_points", 8)),
        use_clearance_features=bool(policy.get("use_clearance_features", True)),
        use_curvature_features=bool(policy.get("use_curvature_features", True)),
        residual_limit_vx=float(limits.get("vx", policy.get("residual_limit_vx", 0.10))),
        residual_limit_vy=float(limits.get("vy", policy.get("residual_limit_vy", 0.10))),
        residual_limit_wz=float(limits.get("wz", policy.get("residual_limit_wz", 0.25))),
        hidden_sizes=tuple(int(v) for v in policy.get("hidden_sizes", [128, 128])),
        activation=str(policy.get("activation", "elu")),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping, got {type(data)!r}")
    return data


def train_bc(args: argparse.Namespace) -> Path:
    config = _load_yaml(args.config)
    policy_cfg = _policy_cfg_from_mapping(config)
    dataset = np.load(args.dataset)
    features = torch.as_tensor(dataset["features"], dtype=torch.float32, device=args.device)
    residuals = torch.as_tensor(dataset["residuals"], dtype=torch.float32, device=args.device)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N,F], got {tuple(features.shape)}")
    if residuals.ndim != 2 or residuals.shape[-1] != 3 or residuals.shape[0] != features.shape[0]:
        raise ValueError(f"residuals must have shape [N,3], got {tuple(residuals.shape)}")

    model_cfg = ResidualMLPCfg(
        input_dim=int(features.shape[-1]),
        hidden_sizes=policy_cfg.hidden_sizes,
        output_dim=3,
        activation=policy_cfg.activation,
        zero_init_output=False,
    )
    model = ResidualMLP(model_cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    limits = policy_cfg.residual_limits.to(device=args.device)
    target = torch.clamp(residuals, min=-limits[None, :], max=limits[None, :])

    batch_size = int(args.batch_size)
    num_samples = int(features.shape[0])
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(args.seed))
    for epoch in range(int(args.epochs)):
        perm = torch.randperm(num_samples, generator=generator, device=args.device)
        total = 0.0
        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            pred = torch.tanh(model(features[idx])) * limits[None, :]
            loss = torch.nn.functional.mse_loss(pred, target[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            total += float(loss.item()) * int(idx.numel())
        if epoch % max(1, int(args.log_every)) == 0 or epoch == int(args.epochs) - 1:
            print(json.dumps({"epoch": epoch, "mse": total / num_samples}))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "policy_bc.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "policy_cfg": policy_cfg.__dict__,
            "feature_dim": int(features.shape[-1]),
        },
        ckpt_path,
    )
    (out_dir / "policy_cfg.json").write_text(json.dumps(policy_cfg.__dict__, indent=2, default=str), encoding="utf-8")
    return ckpt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/ppo.yaml"))
    parser.add_argument("--mode", choices=["bc", "ppo", "inspect"], default="inspect")
    parser.add_argument("--dataset", type=Path, help="NPZ dataset with features/residuals for --mode bc")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/residual_bc"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    policy_cfg = _policy_cfg_from_mapping(config)
    if args.mode == "inspect":
        print(json.dumps({"policy_feature_dim": policy_cfg.feature_dim(), "policy_cfg": policy_cfg.__dict__}, indent=2, default=str))
        return 0
    if args.mode == "ppo":
        raise SystemExit(
            "PPO requires an Isaac Lab vectorized task wrapper. The compact residual policy and config are present; "
            "use --mode bc for teacher pretraining or wrap NavResidualCore in an Isaac Lab DirectRLEnv."
        )
    if args.dataset is None:
        raise SystemExit("--dataset is required for --mode bc")
    ckpt = train_bc(args)
    print(f"saved checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
