"""Minimal PPO implementation for the synthetic residual-command environment.

The goal is not to replace Isaac Lab training. This module provides a runnable,
repo-local training loop for the same compact policy that is deployed by
``RLResidualPolicy``. The saved actor network is compatible with the dry-run
runtime: runtime applies ``tanh(actor(features)) * residual_limits``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import json

import torch
from torch import nn
from torch.distributions import Normal

from g1_nav_rl.envs.synthetic_residual_env import SyntheticResidualNavEnv
from g1_nav_rl.policies.residual_mlp import ResidualMLP, ResidualMLPCfg
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicyCfg


@dataclass(frozen=True)
class PPOTrainCfg:
    total_steps: int = 200_000
    rollout_steps: int = 64
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.005
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 512
    target_kl: float = 0.05
    log_interval: int = 5
    checkpoint_interval: int = 25
    seed: int = 0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "PPOTrainCfg":
        if cfg is None:
            return cls()
        return cls(
            total_steps=int(cfg.get("total_steps", cls.total_steps)),
            rollout_steps=int(cfg.get("rollout_steps", cfg.get("num_steps", cls.rollout_steps))),
            learning_rate=float(cfg.get("learning_rate", cls.learning_rate)),
            gamma=float(cfg.get("gamma", cls.gamma)),
            gae_lambda=float(cfg.get("gae_lambda", cls.gae_lambda)),
            clip_range=float(cfg.get("clip_range", cls.clip_range)),
            entropy_coef=float(cfg.get("entropy_coef", cls.entropy_coef)),
            value_loss_coef=float(cfg.get("value_loss_coef", cls.value_loss_coef)),
            max_grad_norm=float(cfg.get("max_grad_norm", cls.max_grad_norm)),
            update_epochs=int(cfg.get("update_epochs", cls.update_epochs)),
            minibatch_size=int(cfg.get("minibatch_size", cls.minibatch_size)),
            target_kl=float(cfg.get("target_kl", cls.target_kl)),
            log_interval=int(cfg.get("log_interval", cls.log_interval)),
            checkpoint_interval=int(cfg.get("checkpoint_interval", cls.checkpoint_interval)),
            seed=int(cfg.get("seed", cls.seed)),
        )


class ResidualActorCritic(nn.Module):
    """Actor/critic for normalized residual actions."""

    def __init__(self, policy_cfg: RLResidualPolicyCfg, obs_dim: int) -> None:
        super().__init__()
        self.policy_cfg = policy_cfg
        self.actor = ResidualMLP(
            ResidualMLPCfg(
                input_dim=int(obs_dim),
                hidden_sizes=tuple(policy_cfg.hidden_sizes),
                output_dim=3,
                activation=str(policy_cfg.activation),
                zero_init_output=True,
            )
        )
        self.critic = ResidualMLP(
            ResidualMLPCfg(
                input_dim=int(obs_dim),
                hidden_sizes=tuple(policy_cfg.hidden_sizes),
                output_dim=1,
                activation=str(policy_cfg.activation),
                zero_init_output=False,
            )
        )
        self.log_std = nn.Parameter(torch.full((3,), -0.7, dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.actor(obs.to(dtype=torch.float32)))
        std = torch.exp(self.log_std).clamp(0.05, 2.0)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs.to(dtype=torch.float32)).squeeze(-1)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        raw_action = dist.rsample()
        action = raw_action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        # Stored actions are clipped normalized actions. Approximate their log
        # probability under the unclipped Gaussian. This is sufficient for the
        # debug/synthetic PPO layer; Isaac-side training can use tanh-corrected
        # distributions later.
        log_prob = dist.log_prob(actions.clamp(-1.0, 1.0)).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value


def _flatten_time_env(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("expected tensor with time/env dimensions")
    return x.reshape((-1,) + tuple(x.shape[2:]))


def save_policy_checkpoint(
    path: Path,
    model: ResidualActorCritic,
    policy_cfg: RLResidualPolicyCfg,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.actor.state_dict(),
            "critic_state_dict": model.critic.state_dict(),
            "log_std": model.log_std.detach().cpu(),
            "model_cfg": model.actor.cfg.__dict__,
            "policy_cfg": policy_cfg.to_dict(),
            "feature_dim": int(policy_cfg.feature_dim()),
            "extra": dict(extra or {}),
        },
        path,
    )


def train_ppo(
    env: SyntheticResidualNavEnv,
    policy_cfg: RLResidualPolicyCfg,
    train_cfg: PPOTrainCfg,
    *,
    output_dir: Path,
    device: torch.device | str = "cpu",
    init_checkpoint: Path | None = None,
) -> Path:
    device_t = torch.device(device)
    torch.manual_seed(int(train_cfg.seed))
    model = ResidualActorCritic(policy_cfg, obs_dim=env.observation_dim).to(device_t)
    if init_checkpoint is not None:
        checkpoint = torch.load(init_checkpoint, map_location=device_t)
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint else checkpoint
        model.actor.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.learning_rate))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ppo_train_cfg.json").write_text(json.dumps(asdict(train_cfg), indent=2), encoding="utf-8")
    (output_dir / "policy_cfg.json").write_text(json.dumps(policy_cfg.to_dict(), indent=2), encoding="utf-8")

    obs = env.reset().to(device_t)
    num_envs = env.num_envs
    steps_per_iter = int(train_cfg.rollout_steps) * num_envs
    num_iters = max(1, int(train_cfg.total_steps) // max(1, steps_per_iter))
    best_success = -1.0
    last_ckpt = output_dir / "policy_ppo_last.pt"

    for iteration in range(1, num_iters + 1):
        obs_buf = torch.zeros((train_cfg.rollout_steps, num_envs, env.observation_dim), dtype=torch.float32, device=device_t)
        action_buf = torch.zeros((train_cfg.rollout_steps, num_envs, 3), dtype=torch.float32, device=device_t)
        logprob_buf = torch.zeros((train_cfg.rollout_steps, num_envs), dtype=torch.float32, device=device_t)
        reward_buf = torch.zeros((train_cfg.rollout_steps, num_envs), dtype=torch.float32, device=device_t)
        done_buf = torch.zeros((train_cfg.rollout_steps, num_envs), dtype=torch.float32, device=device_t)
        value_buf = torch.zeros((train_cfg.rollout_steps, num_envs), dtype=torch.float32, device=device_t)
        success_count = 0.0
        collision_count = 0.0
        timeout_count = 0.0
        mean_distance = 0.0

        for t in range(int(train_cfg.rollout_steps)):
            obs_buf[t] = obs
            with torch.no_grad():
                action, log_prob, _entropy, value = model.act(obs)
            next_obs, reward, done, info = env.step(action.detach())
            action_buf[t] = action.detach()
            logprob_buf[t] = log_prob.detach()
            value_buf[t] = value.detach()
            reward_buf[t] = reward.to(device_t)
            done_buf[t] = done.to(device_t)
            success_count += float(info["success"].float().sum().item())
            collision_count += float(info["collision"].float().sum().item())
            timeout_count += float(info["timeout"].float().sum().item())
            mean_distance += float(info["distance"].mean().item())
            obs = next_obs.to(device_t)

        with torch.no_grad():
            next_value = model.value(obs)
        adv = torch.zeros_like(reward_buf)
        last_gae = torch.zeros((num_envs,), dtype=torch.float32, device=device_t)
        for t in reversed(range(int(train_cfg.rollout_steps))):
            if t == int(train_cfg.rollout_steps) - 1:
                next_nonterminal = 1.0 - done_buf[t]
                next_values = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_values = value_buf[t + 1]
            delta = reward_buf[t] + float(train_cfg.gamma) * next_values * next_nonterminal - value_buf[t]
            last_gae = delta + float(train_cfg.gamma) * float(train_cfg.gae_lambda) * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + value_buf

        b_obs = _flatten_time_env(obs_buf)
        b_actions = _flatten_time_env(action_buf)
        b_logprobs = _flatten_time_env(logprob_buf.unsqueeze(-1)).squeeze(-1)
        b_adv = _flatten_time_env(adv.unsqueeze(-1)).squeeze(-1)
        b_returns = _flatten_time_env(returns.unsqueeze(-1)).squeeze(-1)
        b_values = _flatten_time_env(value_buf.unsqueeze(-1)).squeeze(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1.0e-8)

        batch_size = b_obs.shape[0]
        generator = torch.Generator(device=device_t)
        generator.manual_seed(int(train_cfg.seed) + iteration)
        approx_kl_value = 0.0
        for _epoch in range(int(train_cfg.update_epochs)):
            perm = torch.randperm(batch_size, generator=generator, device=device_t)
            for start in range(0, batch_size, int(train_cfg.minibatch_size)):
                idx = perm[start : start + int(train_cfg.minibatch_size)]
                new_logprob, entropy, new_value = model.evaluate_actions(b_obs[idx], b_actions[idx])
                log_ratio = new_logprob - b_logprobs[idx]
                ratio = log_ratio.exp()
                pg_loss1 = -b_adv[idx] * ratio
                pg_loss2 = -b_adv[idx] * torch.clamp(ratio, 1.0 - float(train_cfg.clip_range), 1.0 + float(train_cfg.clip_range))
                policy_loss = torch.maximum(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * torch.nn.functional.mse_loss(new_value, b_returns[idx])
                entropy_loss = entropy.mean()
                loss = policy_loss + float(train_cfg.value_loss_coef) * value_loss - float(train_cfg.entropy_coef) * entropy_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.max_grad_norm))
                optimizer.step()
                with torch.no_grad():
                    approx_kl_value = float(((ratio - 1.0) - log_ratio).mean().abs().item())
            if approx_kl_value > float(train_cfg.target_kl):
                break

        success_rate = success_count / max(1.0, success_count + collision_count + timeout_count)
        summary = {
            "iteration": iteration,
            "steps": iteration * steps_per_iter,
            "mean_reward": float(reward_buf.mean().item()),
            "mean_return": float(returns.mean().item()),
            "success_rate_terminal": success_rate,
            "success_count": success_count,
            "collision_count": collision_count,
            "timeout_count": timeout_count,
            "mean_distance": mean_distance / max(1, int(train_cfg.rollout_steps)),
            "approx_kl": approx_kl_value,
            "log_std": model.log_std.detach().cpu().tolist(),
        }
        if iteration % max(1, int(train_cfg.log_interval)) == 0 or iteration == 1:
            print(json.dumps(summary))
        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        if success_rate > best_success:
            best_success = success_rate
            save_policy_checkpoint(output_dir / "policy_ppo_best.pt", model, policy_cfg, extra=summary)
        if iteration % max(1, int(train_cfg.checkpoint_interval)) == 0 or iteration == num_iters:
            save_policy_checkpoint(last_ckpt, model, policy_cfg, extra=summary)

    save_policy_checkpoint(last_ckpt, model, policy_cfg, extra={"final": True})
    return last_ckpt


__all__ = ["PPOTrainCfg", "ResidualActorCritic", "save_policy_checkpoint", "train_ppo"]
