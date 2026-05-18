from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg


@dataclass(frozen=True)
class LocalVelocityRolloutCfg:
    """DWA/DWB-style local command rollout planner config.

    The planner assumes the navigation body-frame convention used elsewhere in
    the repo: ``+x`` is forward and ``+y`` is left. Candidate commands are
    constant body-frame commands ``[vx, vy, wz]``. Each candidate is integrated
    for a short horizon, checked against a local occupancy grid, and scored.
    """

    vx_samples: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    vy_samples: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
    wz_samples: tuple[float, ...] = (-0.6, -0.3, 0.0, 0.3, 0.6)

    horizon_s: float = 2.0
    dt: float = 0.1
    robot_radius: float = 0.35
    collision_margin: float = 0.0
    goal_tolerance: float = 0.25

    # Scores are intentionally simple and inspectable.
    progress_weight: float = 5.0
    final_distance_weight: float = 0.5
    clearance_weight: float = 1.0
    command_smoothness_weight: float = 0.25
    yaw_rate_weight: float = 0.5
    lateral_speed_weight: float = 0.4
    forward_speed_weight: float = 0.05
    final_lateral_error_weight: float = 1.0
    final_yaw_weight: float = 0.5
    collision_penalty: float = 100.0
    out_of_bounds_penalty: float = 25.0
    clearance_cap: float = 2.0

    stop_when_blocked: bool = True

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "LocalVelocityRolloutCfg":
        """Create config from a YAML-style mapping."""
        if cfg is None:
            return cls()

        def tuple_float(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
            value = cfg.get(name, default)
            return tuple(float(x) for x in value)  # type: ignore[arg-type]

        return cls(
            vx_samples=tuple_float("vx_samples", cls.vx_samples),
            vy_samples=tuple_float("vy_samples", cls.vy_samples),
            wz_samples=tuple_float("wz_samples", cls.wz_samples),
            horizon_s=float(cfg.get("horizon_s", cls.horizon_s)),
            dt=float(cfg.get("dt", cls.dt)),
            robot_radius=float(cfg.get("robot_radius", cls.robot_radius)),
            collision_margin=float(cfg.get("collision_margin", cls.collision_margin)),
            goal_tolerance=float(cfg.get("goal_tolerance", cls.goal_tolerance)),
            progress_weight=float(cfg.get("progress_weight", cls.progress_weight)),
            final_distance_weight=float(cfg.get("final_distance_weight", cls.final_distance_weight)),
            clearance_weight=float(cfg.get("clearance_weight", cls.clearance_weight)),
            command_smoothness_weight=float(cfg.get("command_smoothness_weight", cls.command_smoothness_weight)),
            yaw_rate_weight=float(cfg.get("yaw_rate_weight", cls.yaw_rate_weight)),
            lateral_speed_weight=float(cfg.get("lateral_speed_weight", cls.lateral_speed_weight)),
            forward_speed_weight=float(cfg.get("forward_speed_weight", cls.forward_speed_weight)),
            final_lateral_error_weight=float(
                cfg.get("final_lateral_error_weight", cls.final_lateral_error_weight)
            ),
            final_yaw_weight=float(cfg.get("final_yaw_weight", cls.final_yaw_weight)),
            collision_penalty=float(cfg.get("collision_penalty", cls.collision_penalty)),
            out_of_bounds_penalty=float(cfg.get("out_of_bounds_penalty", cls.out_of_bounds_penalty)),
            clearance_cap=float(cfg.get("clearance_cap", cls.clearance_cap)),
            stop_when_blocked=bool(cfg.get("stop_when_blocked", cls.stop_when_blocked)),
        )


def make_candidate_commands(cfg: LocalVelocityRolloutCfg, *, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return candidate commands with shape ``[C, 3]``."""
    if cfg.dt <= 0.0:
        raise ValueError(f"dt must be positive, got {cfg.dt}")
    if cfg.horizon_s <= 0.0:
        raise ValueError(f"horizon_s must be positive, got {cfg.horizon_s}")
    if not cfg.vx_samples or not cfg.vy_samples or not cfg.wz_samples:
        raise ValueError("vx_samples, vy_samples, and wz_samples must be non-empty")

    commands = list(product(cfg.vx_samples, cfg.vy_samples, cfg.wz_samples))
    return torch.tensor(commands, dtype=torch.float32, device=device)


def rollout_constant_commands(
    candidate_commands: torch.Tensor,
    cfg: LocalVelocityRolloutCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate constant body-frame commands in the initial body frame.

    Args:
        candidate_commands: Tensor ``[C, 3]`` containing ``[vx, vy, wz]``.
        cfg: Rollout config.

    Returns:
        ``(trajectory_xy_b, trajectory_yaw)`` where shapes are ``[C, T, 2]``
        and ``[C, T]``. The first trajectory point is after one integration
        step, not the initial origin.
    """
    if candidate_commands.ndim != 2 or candidate_commands.shape[-1] != 3:
        raise ValueError(f"candidate_commands must have shape [C, 3], got {tuple(candidate_commands.shape)}")

    device = candidate_commands.device
    dtype = candidate_commands.dtype
    num_steps = max(1, int(round(cfg.horizon_s / cfg.dt)))
    num_candidates = candidate_commands.shape[0]

    pos = torch.zeros((num_candidates, 2), dtype=dtype, device=device)
    yaw = torch.zeros((num_candidates,), dtype=dtype, device=device)
    trajectory_xy = torch.zeros((num_candidates, num_steps, 2), dtype=dtype, device=device)
    trajectory_yaw = torch.zeros((num_candidates, num_steps), dtype=dtype, device=device)

    vx = candidate_commands[:, 0]
    vy = candidate_commands[:, 1]
    wz = candidate_commands[:, 2]

    for t in range(num_steps):
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        dx = c * vx - s * vy
        dy = s * vx + c * vy
        pos[:, 0] = pos[:, 0] + dx * cfg.dt
        pos[:, 1] = pos[:, 1] + dy * cfg.dt
        yaw = yaw + wz * cfg.dt
        trajectory_xy[:, t, :] = pos
        trajectory_yaw[:, t] = yaw

    return trajectory_xy.contiguous(), trajectory_yaw.contiguous()


def _occupied_cell_centers_xy_b(
    grid: torch.Tensor,
    grid_cfg: OccupancyGridCfg,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert one occupancy grid ``[H, W]`` to occupied cell centers ``[M, 2]``."""
    occupied = grid > float(threshold)
    idx = occupied.nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=grid.device)

    rows = idx[:, 0].to(dtype=torch.float32)
    cols = idx[:, 1].to(dtype=torch.float32)
    x = float(grid_cfg.x_max) - (rows + 0.5) * float(grid_cfg.resolution)
    y = float(grid_cfg.y_min) + (cols + 0.5) * float(grid_cfg.resolution)
    return torch.stack((x, y), dim=-1).contiguous()


def _trajectory_in_bounds(trajectory_xy_b: torch.Tensor, grid_cfg: OccupancyGridCfg) -> torch.Tensor:
    """Return mask ``[C]`` indicating whether all trajectory points are inside the grid."""
    x = trajectory_xy_b[..., 0]
    y = trajectory_xy_b[..., 1]
    in_bounds = (
        (x >= float(grid_cfg.x_min))
        & (x < float(grid_cfg.x_max))
        & (y >= float(grid_cfg.y_min))
        & (y < float(grid_cfg.y_max))
    )
    return in_bounds.all(dim=-1)


def _min_clearance_to_obstacles(
    trajectory_xy_b: torch.Tensor,
    occupied_xy_b: torch.Tensor,
    *,
    no_obstacle_clearance: float,
) -> torch.Tensor:
    """Return minimum Euclidean clearance per candidate trajectory."""
    num_candidates = trajectory_xy_b.shape[0]
    if occupied_xy_b.numel() == 0:
        return torch.full(
            (num_candidates,),
            float(no_obstacle_clearance),
            dtype=trajectory_xy_b.dtype,
            device=trajectory_xy_b.device,
        )

    # [C, T, M]
    distances = torch.linalg.norm(
        trajectory_xy_b[:, :, None, :] - occupied_xy_b[None, None, :, :].to(trajectory_xy_b),
        dim=-1,
    )
    return distances.amin(dim=(1, 2)).contiguous()


def _target_xy_from_input(target: torch.Tensor) -> torch.Tensor:
    """Accept ``[N, 2]`` target_xy_b or ``[N, 4]`` target_vector and return xy."""
    if target.ndim != 2 or target.shape[-1] not in (2, 4):
        raise ValueError(f"target must have shape [N, 2] or [N, 4], got {tuple(target.shape)}")
    return target[:, 0:2]


def plan_local_velocity_command(
    occupancy_grid: torch.Tensor,
    target_xy_b: torch.Tensor,
    last_command: torch.Tensor | None,
    grid_cfg: OccupancyGridCfg,
    cfg: LocalVelocityRolloutCfg | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Plan one local velocity command from a local occupancy grid and target.

    Args:
        occupancy_grid: Float tensor ``[N, H, W]``. Occupied cells should be > 0.5.
        target_xy_b: Tensor ``[N, 2]`` or target vector ``[N, 4]`` in body frame.
            Only the first two elements, ``[target_x_b, target_y_b]``, are used.
        last_command: Optional tensor ``[N, 3]`` for smoothness penalty.
        grid_cfg: Geometry of ``occupancy_grid``.
        cfg: Planner config.

    Returns:
        ``(command, info)`` where ``command`` has shape ``[N, 3]`` and ``info``
        contains score/debug tensors.
    """
    if cfg is None:
        cfg = LocalVelocityRolloutCfg()
    if occupancy_grid.ndim != 3:
        raise ValueError(f"occupancy_grid must have shape [N, H, W], got {tuple(occupancy_grid.shape)}")
    if occupancy_grid.shape[1] != grid_cfg.height or occupancy_grid.shape[2] != grid_cfg.width:
        raise ValueError(
            "occupancy_grid shape does not match grid_cfg: "
            f"grid={tuple(occupancy_grid.shape)}, expected H,W=({grid_cfg.height},{grid_cfg.width})"
        )
    if cfg.robot_radius < 0.0:
        raise ValueError(f"robot_radius must be non-negative, got {cfg.robot_radius}")
    if cfg.collision_margin < 0.0:
        raise ValueError(f"collision_margin must be non-negative, got {cfg.collision_margin}")

    occupancy_grid = occupancy_grid.detach().to(dtype=torch.float32).contiguous()
    device = occupancy_grid.device
    num_envs = occupancy_grid.shape[0]

    target_xy = _target_xy_from_input(target_xy_b).detach().to(device=device, dtype=torch.float32).contiguous()
    if target_xy.shape[0] != num_envs:
        raise ValueError(f"target batch mismatch: target N={target_xy.shape[0]}, grid N={num_envs}")

    if last_command is None:
        last_command_t = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    else:
        if last_command.ndim != 2 or last_command.shape != (num_envs, 3):
            raise ValueError(f"last_command must have shape [{num_envs}, 3], got {tuple(last_command.shape)}")
        last_command_t = last_command.detach().to(device=device, dtype=torch.float32).contiguous()

    candidate_commands = make_candidate_commands(cfg, device=device)
    trajectory_xy, trajectory_yaw = rollout_constant_commands(candidate_commands, cfg)
    num_candidates = candidate_commands.shape[0]

    commands_out = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    best_scores = torch.empty((num_envs,), dtype=torch.float32, device=device)
    best_indices = torch.empty((num_envs,), dtype=torch.long, device=device)
    best_clearance = torch.empty((num_envs,), dtype=torch.float32, device=device)
    best_progress = torch.empty((num_envs,), dtype=torch.float32, device=device)
    any_collision_free = torch.empty((num_envs,), dtype=torch.bool, device=device)
    selected_trajectories = torch.empty((num_envs, trajectory_xy.shape[1], 2), dtype=torch.float32, device=device)

    all_scores = torch.empty((num_envs, num_candidates), dtype=torch.float32, device=device)
    all_collision_free = torch.empty((num_envs, num_candidates), dtype=torch.bool, device=device)
    all_min_clearance = torch.empty((num_envs, num_candidates), dtype=torch.float32, device=device)
    all_progress = torch.empty((num_envs, num_candidates), dtype=torch.float32, device=device)

    in_bounds = _trajectory_in_bounds(trajectory_xy, grid_cfg)
    final_xy = trajectory_xy[:, -1, :]
    effective_collision_radius = float(cfg.robot_radius) + float(cfg.collision_margin)

    for env_id in range(num_envs):
        target = target_xy[env_id]
        initial_distance = torch.linalg.norm(target)

        if initial_distance <= float(cfg.goal_tolerance):
            # Already at target: command stop, keep info tensors coherent.
            scores = torch.full((num_candidates,), -float("inf"), dtype=torch.float32, device=device)
            zero_idx = torch.argmin(torch.linalg.norm(candidate_commands, dim=-1))
            scores[zero_idx] = 0.0
            all_scores[env_id] = scores
            all_collision_free[env_id] = True
            all_min_clearance[env_id] = float(cfg.clearance_cap)
            all_progress[env_id] = 0.0
            commands_out[env_id] = 0.0
            best_scores[env_id] = 0.0
            best_indices[env_id] = zero_idx
            best_clearance[env_id] = float(cfg.clearance_cap)
            best_progress[env_id] = 0.0
            any_collision_free[env_id] = True
            selected_trajectories[env_id] = trajectory_xy[zero_idx]
            continue

        occupied_xy = _occupied_cell_centers_xy_b(occupancy_grid[env_id], grid_cfg)
        min_clearance = _min_clearance_to_obstacles(
            trajectory_xy,
            occupied_xy,
            no_obstacle_clearance=float(cfg.clearance_cap),
        )
        collision_free = (min_clearance > effective_collision_radius) & in_bounds

        final_distance = torch.linalg.norm(final_xy - target[None, :], dim=-1)
        progress = initial_distance - final_distance
        command_delta = candidate_commands - last_command_t[env_id][None, :]
        smoothness_cost = torch.linalg.norm(command_delta, dim=-1)
        clearance_score = torch.clamp(min_clearance, 0.0, float(cfg.clearance_cap)) / max(float(cfg.clearance_cap), 1.0e-6)

        final_lateral_error = torch.abs(final_xy[:, 1] - target[1])
        final_yaw_error = torch.abs(trajectory_yaw[:, -1])

        scores = (
            float(cfg.progress_weight) * progress
            - float(cfg.final_distance_weight) * final_distance
            + float(cfg.clearance_weight) * clearance_score
            - float(cfg.command_smoothness_weight) * smoothness_cost
            - float(cfg.yaw_rate_weight) * torch.abs(candidate_commands[:, 2])
            - float(cfg.lateral_speed_weight) * torch.abs(candidate_commands[:, 1])
            + float(cfg.forward_speed_weight) * candidate_commands[:, 0]
            - float(cfg.final_lateral_error_weight) * final_lateral_error
            - float(cfg.final_yaw_weight) * final_yaw_error
        )
        scores = scores - torch.where(
            collision_free,
            torch.zeros_like(scores),
            torch.full_like(scores, float(cfg.collision_penalty)),
        )
        scores = scores - torch.where(
            in_bounds,
            torch.zeros_like(scores),
            torch.full_like(scores, float(cfg.out_of_bounds_penalty)),
        )

        has_free = bool(collision_free.any().item())
        if cfg.stop_when_blocked and not has_free:
            # If all candidates collide, prefer stopping rather than choosing a
            # high-scoring but unsafe candidate.
            stop_idx = torch.argmin(torch.linalg.norm(candidate_commands, dim=-1))
            best_idx = stop_idx
            selected_command = torch.zeros((3,), dtype=torch.float32, device=device)
        else:
            best_idx = torch.argmax(scores)
            selected_command = candidate_commands[best_idx]

        all_scores[env_id] = scores
        all_collision_free[env_id] = collision_free
        all_min_clearance[env_id] = min_clearance
        all_progress[env_id] = progress
        commands_out[env_id] = selected_command
        best_scores[env_id] = scores[best_idx]
        best_indices[env_id] = best_idx
        best_clearance[env_id] = min_clearance[best_idx]
        best_progress[env_id] = progress[best_idx]
        any_collision_free[env_id] = has_free
        selected_trajectories[env_id] = trajectory_xy[best_idx]

    info = {
        "candidate_commands": candidate_commands,
        "trajectory_xy_b": trajectory_xy,
        "trajectory_yaw": trajectory_yaw,
        "selected_trajectory_xy_b": selected_trajectories,
        "scores": all_scores,
        "collision_free": all_collision_free,
        "min_clearance": all_min_clearance,
        "progress": all_progress,
        "best_score": best_scores,
        "best_index": best_indices,
        "best_clearance": best_clearance,
        "best_progress": best_progress,
        "any_collision_free": any_collision_free,
        "effective_collision_radius": torch.full((num_envs,), effective_collision_radius, dtype=torch.float32, device=device),
    }
    return commands_out.contiguous(), info


__all__ = [
    "LocalVelocityRolloutCfg",
    "make_candidate_commands",
    "rollout_constant_commands",
    "plan_local_velocity_command",
]
