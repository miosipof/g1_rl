#!/usr/bin/env python3
"""Collect behavior-cloning data for the residual command policy.

The collector runs the synthetic residual navigation environment, computes the
current compact feature vector, asks the deterministic local rollout planner for
a teacher command, and stores:

    features  [N, F]
    residuals [N, 3] = teacher_command - nominal_astar_command

The resulting NPZ can be passed to:

    python scripts/train.py --mode bc --dataset teacher_rollouts.npz
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
from g1_nav_rl.policies.local_rollout_policy import LocalRolloutPolicy, LocalRolloutPolicyCfg
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicyCfg, build_compact_residual_features



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


def _policy_cfg(config: Mapping[str, Any], env_config: Mapping[str, Any]) -> RLResidualPolicyCfg:
    policy = dict(config.get("policy", {}) or {})
    policy.update(dict(env_config.get("residual_policy", {}) or {}))
    if isinstance(env_config.get("grid"), Mapping):
        policy["grid_cfg"] = dict(env_config["grid"])
    return RLResidualPolicyCfg.from_mapping(policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/ppo.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env/synthetic_residual_env.yaml"))
    parser.add_argument("--output", type=Path, default=Path("runs/teacher/teacher_rollouts.npz"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", choices=["teacher", "nominal", "random"], default="teacher")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    env_config = _load_yaml(args.env_config) if args.env_config.exists() else {}
    env_config = dict(env_config)
    env_config = _deep_merge(env_config, {"reward": config.get("reward", {})})
    if "planner" in config:
        env_config = _deep_merge(env_config, {"planner": config["planner"]})
    env_config.setdefault("env", {})
    env_config["env"]["num_envs"] = int(args.num_envs)
    env_config["env"]["target_occlusion_prob"] = 0.0
    policy_cfg = _policy_cfg(config, env_config)
    if isinstance(env_config.get("grid"), Mapping):
        policy_cfg = RLResidualPolicyCfg.from_mapping({**policy_cfg.to_dict(), "grid_cfg": OccupancyGridCfg.from_mapping(env_config["grid"])})
        env_config["residual_policy"] = policy_cfg.to_dict()
    env = SyntheticResidualNavEnv(SyntheticResidualEnvCfg.from_mapping(env_config), device=args.device, seed=int(args.seed))
    teacher = LocalRolloutPolicy(LocalRolloutPolicyCfg(grid_cfg=env.cfg.grid_cfg))

    features_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    nominal_chunks: list[np.ndarray] = []
    teacher_chunks: list[np.ndarray] = []
    collected = 0
    iteration = 0
    while collected < int(args.samples):
        iteration += 1
        inp = env.current_policy_input()
        features = build_compact_residual_features(inp, policy_cfg)
        teacher_out = teacher.step(inp)
        nominal = inp.nominal_command
        residual = teacher_out.command - nominal
        residual = torch.clamp(residual, min=-policy_cfg.residual_limits.to(args.device)[None, :], max=policy_cfg.residual_limits.to(args.device)[None, :])
        features_chunks.append(features.detach().cpu().numpy())
        residual_chunks.append(residual.detach().cpu().numpy())
        nominal_chunks.append(nominal.detach().cpu().numpy())
        teacher_chunks.append(teacher_out.command.detach().cpu().numpy())
        collected += int(features.shape[0])

        if args.execute == "teacher":
            action = residual / policy_cfg.residual_limits.to(args.device)[None, :].clamp_min(1.0e-6)
        elif args.execute == "nominal":
            action = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=args.device)
        else:
            action = torch.empty((env.num_envs, 3), dtype=torch.float32, device=args.device).uniform_(-1.0, 1.0)
        _obs, _reward, _done, info = env.step(action)
        if iteration % max(1, int(args.log_every)) == 0:
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "collected": min(collected, int(args.samples)),
                        "mean_distance": float(info["distance"].mean().item()),
                        "success": int(info["success"].sum().item()),
                        "collision": int(info["collision"].sum().item()),
                    }
                )
            )

    features_np = np.concatenate(features_chunks, axis=0)[: int(args.samples)]
    residual_np = np.concatenate(residual_chunks, axis=0)[: int(args.samples)]
    nominal_np = np.concatenate(nominal_chunks, axis=0)[: int(args.samples)]
    teacher_np = np.concatenate(teacher_chunks, axis=0)[: int(args.samples)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features_np.astype(np.float32),
        residuals=residual_np.astype(np.float32),
        nominal_commands=nominal_np.astype(np.float32),
        teacher_commands=teacher_np.astype(np.float32),
        policy_cfg=json.dumps(policy_cfg.to_dict()),
    )
    print(f"saved dataset: {args.output} samples={features_np.shape[0]} feature_dim={features_np.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
