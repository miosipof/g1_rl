#!/usr/bin/env python3
"""Evaluate the residual policy seam on the synthetic navigation environment."""

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

import torch
import yaml

from g1_nav_rl.envs.synthetic_residual_env import SyntheticResidualEnvCfg, SyntheticResidualNavEnv
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicy, RLResidualPolicyCfg



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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/train/ppo.yaml"))
    parser.add_argument("--env-config", type=Path, default=Path("configs/env/synthetic_residual_env.yaml"))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--policy", choices=["zero", "checkpoint"], default="checkpoint")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    env_config = _load_yaml(args.env_config) if args.env_config.exists() else {}
    env_config = dict(env_config)
    env_config = _deep_merge(env_config, {"reward": config.get("reward", {})})
    if "planner" in config:
        env_config = _deep_merge(env_config, {"planner": config["planner"]})
    env_config.setdefault("env", {})
    env_config["env"]["num_envs"] = int(args.num_envs)
    policy_cfg = _policy_cfg(config, env_config)
    env_config["residual_policy"] = policy_cfg.to_dict()
    env = SyntheticResidualNavEnv(SyntheticResidualEnvCfg.from_mapping(env_config), device=args.device, seed=int(args.seed))
    policy = None
    if args.policy == "checkpoint" and args.checkpoint is not None:
        policy = RLResidualPolicy(policy_cfg, checkpoint_path=args.checkpoint, device=args.device)

    terminals = 0
    successes = 0
    collisions = 0
    timeouts = 0
    reward_sum = 0.0
    step_count = 0
    distance_sum = 0.0
    while terminals < int(args.episodes):
        if policy is None:
            action = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=args.device)
        else:
            out = policy.step(env.current_policy_input())
            action = out.residual / policy.cfg.residual_limits.to(args.device)[None, :].clamp_min(1.0e-6)
        _obs, reward, _done, info = env.step(action)
        reward_sum += float(reward.sum().item())
        step_count += env.num_envs
        distance_sum += float(info["distance"].sum().item())
        terminal = info["success"] | info["collision"] | info["timeout"]
        terminals += int(terminal.sum().item())
        successes += int(info["success"].sum().item())
        collisions += int(info["collision"].sum().item())
        timeouts += int(info["timeout"].sum().item())

    denom = max(1, terminals)
    summary = {
        "episodes": terminals,
        "successes": successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": successes / denom,
        "collision_rate": collisions / denom,
        "timeout_rate": timeouts / denom,
        "mean_reward_per_step": reward_sum / max(1, step_count),
        "mean_distance": distance_sum / max(1, step_count),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
