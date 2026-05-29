#!/usr/bin/env python3
"""Train the compact residual local-command policy.

Supported modes:

- ``inspect``: print the derived policy/env dimensions.
- ``bc``: behavior-clone residuals from an NPZ dataset with fields
  ``features`` [N,F] and ``residuals`` [N,3].
- ``ppo``: run a lightweight Isaac-independent PPO trainer on the synthetic
  residual navigation environment. This trains the same compact actor that the
  real dry-run runtime loads via ``RLResidualPolicy``.

The PPO mode is deliberately not a full Isaac Lab task. It is the fast training
implementation that validates the policy seam before Isaac vectorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import yaml

from g1_nav_rl.envs.synthetic_residual_env import SyntheticResidualEnvCfg, SyntheticResidualNavEnv
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.policies.residual_mlp import ResidualMLP, ResidualMLPCfg
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicyCfg
from g1_nav_rl.training.ppo import PPOTrainCfg, train_ppo



def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping, got {type(data)!r}")
    return data


def _merged_policy_mapping(config: Mapping[str, Any], env_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    policy = dict(config.get("policy", config.get("residual_policy", {})) or {})
    if env_config:
        policy.update(dict(env_config.get("residual_policy", {}) or {}))
        grid_raw = env_config.get("grid")
        if isinstance(grid_raw, Mapping):
            policy["grid_cfg"] = dict(grid_raw)
    return policy


def _policy_cfg_from_config(config: Mapping[str, Any], env_config: Mapping[str, Any] | None = None) -> RLResidualPolicyCfg:
    return RLResidualPolicyCfg.from_mapping(_merged_policy_mapping(config, env_config))


def train_bc(args: argparse.Namespace, config: Mapping[str, Any], env_config: Mapping[str, Any] | None) -> Path:
    policy_cfg = _policy_cfg_from_config(config, env_config)
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
            print(json.dumps({"epoch": epoch, "mse": total / max(1, num_samples)}))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "policy_bc.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "policy_cfg": policy_cfg.to_dict(),
            "feature_dim": int(features.shape[-1]),
            "dataset": str(args.dataset),
        },
        ckpt_path,
    )
    (out_dir / "policy_cfg.json").write_text(json.dumps(policy_cfg.to_dict(), indent=2), encoding="utf-8")
    return ckpt_path


def train_ppo_entry(args: argparse.Namespace, config: Mapping[str, Any], env_config: Mapping[str, Any] | None) -> Path:
    combined_env_cfg: dict[str, Any] = dict(env_config or {})
    # Also allow configs/train/ppo.yaml to be sufficient by itself. Merge nested
    # sections so synthetic-only fields such as obstacle_count are not lost when
    # the train config overrides num_envs or PPO-specific settings.
    for key in ("env", "planner", "reward", "grid", "residual_policy", "nav", "astar"):
        if key in config:
            existing = combined_env_cfg.get(key, {})
            if isinstance(existing, Mapping) and isinstance(config[key], Mapping):
                combined_env_cfg[key] = _deep_merge(existing, config[key])
            else:
                combined_env_cfg[key] = config[key]
    if args.num_envs is not None:
        combined_env_cfg.setdefault("env", {})
        combined_env_cfg["env"]["num_envs"] = int(args.num_envs)
    if args.total_steps is not None:
        config = dict(config)
        config.setdefault("training", {})
        config["training"]["total_steps"] = int(args.total_steps)

    policy_cfg = _policy_cfg_from_config(config, combined_env_cfg)
    # Keep env and policy grid configs aligned for clearance features.
    grid_raw = combined_env_cfg.get("grid")
    if isinstance(grid_raw, Mapping):
        policy_cfg = RLResidualPolicyCfg.from_mapping({**policy_cfg.to_dict(), "grid_cfg": OccupancyGridCfg.from_mapping(grid_raw)})
        combined_env_cfg["residual_policy"] = policy_cfg.to_dict()
    env_cfg = SyntheticResidualEnvCfg.from_mapping(combined_env_cfg)
    train_cfg = PPOTrainCfg.from_mapping(config.get("training", {}))
    env = SyntheticResidualNavEnv(env_cfg, device=args.device, seed=int(args.seed))
    return train_ppo(
        env,
        policy_cfg,
        train_cfg,
        output_dir=Path(args.output_dir),
        device=args.device,
        init_checkpoint=args.init_checkpoint,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/ppo.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env/synthetic_residual_env.yaml"))
    parser.add_argument("--mode", choices=["bc", "ppo", "inspect"], default="inspect")
    parser.add_argument("--dataset", type=Path, help="NPZ dataset with features/residuals for --mode bc")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/residual_train"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--num-envs", type=int, help="Override env.num_envs for PPO")
    parser.add_argument("--total-steps", type=int, help="Override training.total_steps for PPO")
    parser.add_argument("--init-checkpoint", type=Path, help="Optional BC checkpoint to initialize PPO actor")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    env_config = _load_yaml(args.env_config) if args.env_config and args.env_config.exists() else {}
    policy_cfg = _policy_cfg_from_config(config, env_config)
    if args.mode == "inspect":
        env_cfg = SyntheticResidualEnvCfg.from_mapping({**env_config, **{"env": config.get("env", {})}, "reward": config.get("reward", {})})
        print(
            json.dumps(
                {
                    "policy_feature_dim": policy_cfg.feature_dim(),
                    "policy_cfg": policy_cfg.to_dict(),
                    "synthetic_env": {
                        "num_envs": env_cfg.num_envs,
                        "episode_length": env_cfg.episode_length,
                        "dt": env_cfg.dt,
                        "grid_shape": [env_cfg.grid_cfg.height, env_cfg.grid_cfg.width],
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.mode == "bc":
        if args.dataset is None:
            raise SystemExit("--dataset is required for --mode bc")
        ckpt = train_bc(args, config, env_config)
        print(f"saved checkpoint: {ckpt}")
        return 0
    if args.mode == "ppo":
        ckpt = train_ppo_entry(args, config, env_config)
        print(f"saved checkpoint: {ckpt}")
        return 0
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
