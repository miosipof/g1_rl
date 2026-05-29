from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping

import torch


@dataclass(frozen=True)
class NavigationRewardCfg:
    """Command-level reward terms for A*-conditioned residual navigation.

    The legacy reward ``progress - command_norm + success`` remains available;
    additional terms are optional and are ignored when their tensors are not
    provided to :func:`compute_navigation_reward_terms`.
    """

    progress_weight: float = 1.0
    command_penalty_weight: float = 0.01
    success_bonus: float = 1.0
    collision_penalty: float = 10.0
    near_obstacle_penalty_weight: float = 0.0
    near_obstacle_clearance_m: float = 0.45
    safety_intervention_penalty: float = 1.0
    residual_penalty_weight: float = 0.05
    command_delta_penalty_weight: float = 0.05
    command_jerk_penalty_weight: float = 0.0
    path_deviation_penalty_weight: float = 0.0
    time_penalty: float = 0.0
    target_lost_motion_penalty_weight: float = 0.5
    timeout_penalty: float = 0.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "NavigationRewardCfg":
        if cfg is None:
            return cls()
        names = {field.name for field in fields(cls)}
        kwargs = {name: float(cfg.get(name, getattr(cls, name))) for name in names if name in cfg}
        return cls(**kwargs)


def _zeros_like_progress(progress: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(progress, dtype=torch.float32)


def _validate_base(progress: torch.Tensor, command: torch.Tensor, success: torch.Tensor) -> None:
    if progress.ndim != 1:
        raise ValueError(f"progress must have shape [N], got {tuple(progress.shape)}")
    if command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError(f"command must have shape [N, 3], got {tuple(command.shape)}")
    if success.ndim != 1:
        raise ValueError(f"success must have shape [N], got {tuple(success.shape)}")
    if progress.shape[0] != command.shape[0] or progress.shape[0] != success.shape[0]:
        raise ValueError("progress, command, and success batch sizes must match")


def _optional_flag(value: torch.Tensor | None, progress: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(progress, dtype=torch.bool)
    if value.ndim != 1 or value.shape[0] != progress.shape[0]:
        raise ValueError(f"optional flag must have shape [N], got {tuple(value.shape)}")
    return value.to(device=progress.device, dtype=torch.bool)


def _optional_vector_norm(value: torch.Tensor | None, command: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros((command.shape[0],), dtype=torch.float32, device=command.device)
    if value.shape != command.shape:
        raise ValueError(f"optional command-like tensor must have shape {tuple(command.shape)}, got {tuple(value.shape)}")
    return torch.linalg.norm(value.to(device=command.device, dtype=torch.float32), dim=-1)


def compute_navigation_reward_terms(
    progress: torch.Tensor,
    command: torch.Tensor,
    success: torch.Tensor,
    cfg: NavigationRewardCfg,
    *,
    raw_residual: torch.Tensor | None = None,
    last_command: torch.Tensor | None = None,
    previous_command_delta: torch.Tensor | None = None,
    min_clearance: torch.Tensor | None = None,
    collision: torch.Tensor | None = None,
    safety_intervention: torch.Tensor | None = None,
    path_deviation: torch.Tensor | None = None,
    target_lost: torch.Tensor | None = None,
    timeout: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return decomposed reward terms, each shaped ``[N]``.

    Keeping terms explicit is important for this project because the policy is
    only supposed to smooth/refine A* commands.  If the learned policy wins by
    exploiting one scalar reward while causing safety interventions or stale
    target motion, the logs should expose that immediately.
    """

    _validate_base(progress, command, success)
    progress = progress.to(dtype=torch.float32)
    command = command.to(device=progress.device, dtype=torch.float32)
    success = success.to(device=progress.device, dtype=torch.bool)

    terms: dict[str, torch.Tensor] = {}
    terms["progress"] = float(cfg.progress_weight) * progress
    terms["command_norm"] = -float(cfg.command_penalty_weight) * torch.linalg.norm(command, dim=-1)
    terms["success"] = success.to(dtype=torch.float32) * float(cfg.success_bonus)

    collision_flag = _optional_flag(collision, progress)
    terms["collision"] = -collision_flag.to(dtype=torch.float32) * float(cfg.collision_penalty)

    safety_flag = _optional_flag(safety_intervention, progress)
    terms["safety_intervention"] = -safety_flag.to(dtype=torch.float32) * float(cfg.safety_intervention_penalty)

    residual_norm = _optional_vector_norm(raw_residual, command)
    terms["residual_norm"] = -float(cfg.residual_penalty_weight) * residual_norm

    if last_command is None:
        command_delta = torch.zeros_like(progress, dtype=torch.float32)
    else:
        command_delta = _optional_vector_norm(command - last_command.to(device=command.device), command)
    terms["command_delta"] = -float(cfg.command_delta_penalty_weight) * command_delta

    jerk_norm = _optional_vector_norm(previous_command_delta, command)
    terms["command_jerk"] = -float(cfg.command_jerk_penalty_weight) * jerk_norm

    if min_clearance is None:
        terms["near_obstacle"] = _zeros_like_progress(progress)
    else:
        if min_clearance.ndim != 1 or min_clearance.shape[0] != progress.shape[0]:
            raise ValueError(f"min_clearance must have shape [N], got {tuple(min_clearance.shape)}")
        clearance_deficit = torch.clamp(float(cfg.near_obstacle_clearance_m) - min_clearance.to(progress), min=0.0)
        terms["near_obstacle"] = -float(cfg.near_obstacle_penalty_weight) * clearance_deficit

    if path_deviation is None:
        terms["path_deviation"] = _zeros_like_progress(progress)
    else:
        if path_deviation.ndim != 1 or path_deviation.shape[0] != progress.shape[0]:
            raise ValueError(f"path_deviation must have shape [N], got {tuple(path_deviation.shape)}")
        terms["path_deviation"] = -float(cfg.path_deviation_penalty_weight) * path_deviation.to(progress)

    lost_flag = _optional_flag(target_lost, progress)
    forward_motion = torch.clamp(command[:, 0], min=0.0)
    terms["target_lost_motion"] = -lost_flag.to(dtype=torch.float32) * forward_motion * float(
        cfg.target_lost_motion_penalty_weight
    )

    timeout_flag = _optional_flag(timeout, progress)
    terms["timeout"] = -timeout_flag.to(dtype=torch.float32) * float(cfg.timeout_penalty)

    if float(cfg.time_penalty) != 0.0:
        terms["time"] = torch.full_like(progress, -float(cfg.time_penalty))
    else:
        terms["time"] = _zeros_like_progress(progress)
    return terms


def compute_navigation_reward(
    progress: torch.Tensor,
    command: torch.Tensor,
    success: torch.Tensor,
    cfg: NavigationRewardCfg,
    **kwargs: torch.Tensor | None,
) -> torch.Tensor:
    """Compute a batched scalar reward.

    The original call signature is preserved.  New callers may pass optional
    tensors such as ``raw_residual``, ``min_clearance``, ``collision`` or
    ``safety_intervention`` to activate additional reward terms.
    """

    terms = compute_navigation_reward_terms(progress, command, success, cfg, **kwargs)
    reward = torch.zeros_like(progress, dtype=torch.float32)
    for value in terms.values():
        reward = reward + value.to(device=reward.device, dtype=reward.dtype)
    return reward


__all__ = ["NavigationRewardCfg", "compute_navigation_reward", "compute_navigation_reward_terms"]
