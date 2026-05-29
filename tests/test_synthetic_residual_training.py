from __future__ import annotations

import torch

from g1_nav_rl.envs.synthetic_residual_env import SyntheticResidualEnvCfg, SyntheticResidualNavEnv
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.policies.local_command_policy import LocalPolicyInput, TrackerState
from g1_nav_rl.policies.residual_mlp import ResidualMLP
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicy, RLResidualPolicyCfg, build_compact_residual_features
from g1_nav_rl.training.ppo import PPOTrainCfg, ResidualActorCritic


def test_synthetic_residual_env_step_shapes() -> None:
    cfg = SyntheticResidualEnvCfg(num_envs=2, episode_length=20, grid_cfg=OccupancyGridCfg(resolution=0.20))
    # Keep policy grid aligned with the env grid for this small test.
    cfg = SyntheticResidualEnvCfg(
        num_envs=2,
        episode_length=20,
        grid_cfg=OccupancyGridCfg(resolution=0.20),
        residual_policy_cfg=RLResidualPolicyCfg(grid_cfg=OccupancyGridCfg(resolution=0.20)),
    )
    env = SyntheticResidualNavEnv(cfg, seed=1)
    obs = env.reset()
    assert obs.shape == (2, cfg.residual_policy_cfg.feature_dim())
    next_obs, reward, done, info = env.step(torch.zeros((2, 3)))
    assert next_obs.shape == obs.shape
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["command"].shape == (2, 3)


def test_residual_features_accept_runtime_grid_shape_mismatch() -> None:
    train_cfg = RLResidualPolicyCfg(grid_cfg=OccupancyGridCfg(resolution=0.10))
    runtime_grid_cfg = OccupancyGridCfg(resolution=0.05)
    policy_input = LocalPolicyInput(
        occupancy_grid=torch.zeros((1, runtime_grid_cfg.height, runtime_grid_cfg.width)),
        target_xy_b=torch.tensor([[2.0, 0.1]]),
        path_xy_b=torch.zeros((1, train_cfg.path_points, 2)),
        nominal_command=torch.zeros((1, 3)),
        last_command=torch.zeros((1, 3)),
        target_visible=torch.ones((1,)),
        target_age_s=torch.zeros((1,)),
        tracker_state=torch.tensor([int(TrackerState.TRACKING)]),
    )
    features = build_compact_residual_features(policy_input, train_cfg)
    assert features.shape == (1, train_cfg.feature_dim())


def test_actor_checkpoint_loads_into_runtime_policy(tmp_path) -> None:
    cfg = RLResidualPolicyCfg(grid_cfg=OccupancyGridCfg(resolution=0.10))
    actor_critic = ResidualActorCritic(cfg, obs_dim=cfg.feature_dim())
    ckpt = tmp_path / "policy.pt"
    torch.save({"model_state_dict": actor_critic.actor.state_dict(), "policy_cfg": cfg.to_dict()}, ckpt)
    policy = RLResidualPolicy(checkpoint_path=ckpt)
    assert isinstance(policy.model, ResidualMLP)
    assert policy.cfg.feature_dim() == cfg.feature_dim()


def test_ppo_cfg_from_mapping() -> None:
    cfg = PPOTrainCfg.from_mapping({"total_steps": 128, "rollout_steps": 16, "minibatch_size": 32})
    assert cfg.total_steps == 128
    assert cfg.rollout_steps == 16
    assert cfg.minibatch_size == 32
