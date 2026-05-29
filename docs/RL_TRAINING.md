# RL training implementation

This repo now has a runnable training path for the A*-conditioned residual command policy.

The learned policy is still intentionally narrow:

```text
RGB/YOLO/depth -> target estimate
LiDAR          -> occupancy grid
A*             -> nominal path/command
RL policy      -> bounded residual correction
safety         -> final command clamp/supervision
```

The training implementation does **not** train low-level humanoid locomotion and does **not** replace A*. It trains a compact local policy that smooths/refines `[vx, vy, wz]` commands.

## Components

```text
src/g1_nav_rl/envs/synthetic_residual_env.py
```

Fast Isaac-independent vectorized 2D environment. It samples rectangular obstacles and targets, builds a local occupancy grid, runs the A* nominal planner, exposes compact residual features, and applies residual actions.

```text
src/g1_nav_rl/training/ppo.py
```

Minimal PPO trainer for the synthetic environment. The saved actor checkpoint is compatible with `RLResidualPolicy` and can be loaded by dry-run/runtime scripts.

```text
scripts/collect_rollout_teacher_data.py
```

Collects behavior-cloning data from the deterministic local velocity rollout planner.

```text
scripts/train.py
```

Supports:

```text
--mode inspect
--mode bc
--mode ppo
```

```text
scripts/eval.py
```

Evaluates either the zero-residual baseline or a learned checkpoint in the synthetic environment.

## Inspect configuration

```bash
python scripts/train.py --mode inspect
```

## Collect teacher data

```bash
python scripts/collect_rollout_teacher_data.py \
  --samples 20000 \
  --num-envs 32 \
  --output runs/teacher/teacher_rollouts.npz
```

The output dataset contains:

```text
features:         [N, F]
residuals:        [N, 3]
nominal_commands: [N, 3]
teacher_commands: [N, 3]
```

## Behavior-cloning pretraining

```bash
python scripts/train.py \
  --mode bc \
  --dataset runs/teacher/teacher_rollouts.npz \
  --output-dir runs/residual_bc \
  --epochs 50 \
  --batch-size 256
```

This writes:

```text
runs/residual_bc/policy_bc.pt
```

## PPO training

From scratch:

```bash
python scripts/train.py \
  --mode ppo \
  --output-dir runs/residual_ppo \
  --num-envs 32 \
  --total-steps 200000
```

Fine-tune from behavior cloning:

```bash
python scripts/train.py \
  --mode ppo \
  --init-checkpoint runs/residual_bc/policy_bc.pt \
  --output-dir runs/residual_ppo_from_bc \
  --num-envs 32 \
  --total-steps 200000
```

The PPO trainer writes:

```text
policy_ppo_last.pt
policy_ppo_best.pt
train_log.jsonl
policy_cfg.json
ppo_train_cfg.json
```

## Evaluate

Zero-residual deterministic baseline:

```bash
python scripts/eval.py \
  --policy zero \
  --episodes 64 \
  --num-envs 16
```

Learned checkpoint:

```bash
python scripts/eval.py \
  --checkpoint runs/residual_ppo/policy_ppo_last.pt \
  --episodes 64 \
  --num-envs 16
```

## Dry-run with real adapter runtime

A checkpoint produced by BC or PPO can be loaded by the existing dry-run path:

```bash
python scripts/real_g1_follow_dryrun.py \
  --policy rl-residual \
  --rl-residual-checkpoint runs/residual_ppo/policy_ppo_last.pt \
  --steps 5 \
  --dt 0.1 \
  --json-lines
```

Dry-run must come before real walking. The safety supervisor remains authoritative.

## Current limitation

This is not yet the final Isaac Lab vectorized PPO task. It is the fast local training implementation for the same deployable residual policy. The next Isaac-specific step is to wrap the same feature/action/reward seam in an Isaac Lab `DirectRLEnv` so the policy can train with richer randomized scenes and sensor timing models.
