"""Synthetic vectorized training environment for A*-conditioned residual control.

This environment is intentionally Isaac-independent. It generates simple 2D
navigation episodes with rectangular obstacles, runs the same A*/feature seam as
real runtime code, and exposes a compact residual-control RL interface:

    observation = build_compact_residual_features(LocalPolicyInput)
    action      = normalized residual in [-1, 1]^3
    command     = nominal A* command + action * residual_limits

It is not a physics substitute for Isaac Sim. It is a fast training/debug layer
for BC/PPO before the policy is wrapped in an Isaac Lab task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch

from g1_nav_rl.features.path_features import fixed_body_path_points
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.simple_nav import (
    MotionCommand,
    Pose2D,
    SimpleNavConfig,
    SimpleNavigator,
    Target2D,
    angle_wrap,
    integrate_body_command,
    world_to_body_xy,
)
from g1_nav_rl.planning.grid_astar import GridAStarCfg
from g1_nav_rl.policies.local_command_policy import LocalPolicyInput, TrackerState
from g1_nav_rl.policies.rl_residual_policy import RLResidualPolicyCfg, build_compact_residual_features
from g1_nav_rl.rewards.navigation_reward import NavigationRewardCfg, compute_navigation_reward_terms


@dataclass(frozen=True)
class RectObstacle:
    """Axis-aligned rectangular obstacle in world frame."""

    cx: float
    cy: float
    sx: float
    sy: float

    def contains_xy(self, x: torch.Tensor, y: torch.Tensor, *, margin: float = 0.0) -> torch.Tensor:
        hx = 0.5 * float(self.sx) + float(margin)
        hy = 0.5 * float(self.sy) + float(margin)
        return (torch.abs(x - float(self.cx)) <= hx) & (torch.abs(y - float(self.cy)) <= hy)

    def as_dict(self) -> dict[str, float]:
        return {"cx": float(self.cx), "cy": float(self.cy), "sx": float(self.sx), "sy": float(self.sy)}


@dataclass(frozen=True)
class SyntheticResidualEnvCfg:
    """Configuration for the synthetic residual-control environment."""

    num_envs: int = 32
    episode_length: int = 300
    dt: float = 0.10
    success_radius: float = 0.55
    robot_radius: float = 0.30
    world_x_range: tuple[float, float] = (-0.5, 5.5)
    world_y_range: tuple[float, float] = (-2.5, 2.5)
    target_x_range: tuple[float, float] = (3.0, 5.0)
    target_y_range: tuple[float, float] = (-1.5, 1.5)
    obstacle_count: int = 2
    obstacle_size_x_range: tuple[float, float] = (0.35, 0.90)
    obstacle_size_y_range: tuple[float, float] = (0.35, 1.30)
    obstacle_x_range: tuple[float, float] = (1.0, 3.6)
    obstacle_y_range: tuple[float, float] = (-1.6, 1.6)
    obstacle_keepout_m: float = 0.60
    grid_cfg: OccupancyGridCfg = field(default_factory=lambda: OccupancyGridCfg(resolution=0.10))
    astar_cfg: GridAStarCfg = field(default_factory=lambda: GridAStarCfg(inflation_radius=0.35, max_expansions=20000))
    nav_cfg: SimpleNavConfig = field(
        default_factory=lambda: SimpleNavConfig(
            command_mode="heading_aligned",
            max_vx=0.30,
            max_vy=0.20,
            max_wz=0.60,
            path_lookahead_m=0.40,
            success_radius=0.55,
        )
    )
    residual_policy_cfg: RLResidualPolicyCfg = field(
        default_factory=lambda: RLResidualPolicyCfg(grid_cfg=OccupancyGridCfg(resolution=0.10))
    )
    reward: NavigationRewardCfg = field(
        default_factory=lambda: NavigationRewardCfg(
            progress_weight=1.0,
            command_penalty_weight=0.01,
            success_bonus=5.0,
            collision_penalty=10.0,
            near_obstacle_penalty_weight=0.35,
            near_obstacle_clearance_m=0.45,
            residual_penalty_weight=0.03,
            command_delta_penalty_weight=0.05,
            command_jerk_penalty_weight=0.02,
            path_deviation_penalty_weight=0.02,
            time_penalty=0.001,
            target_lost_motion_penalty_weight=0.5,
            timeout_penalty=1.0,
        )
    )
    path_points: int = 8
    path_spacing_m: float = 0.25
    command_max_vx: float = 0.35
    command_max_vy: float = 0.25
    command_max_wz: float = 0.70
    command_smoothing_alpha: float = 0.65
    command_delay_steps_max: int = 1
    replan_every: int = 10
    target_occlusion_prob: float = 0.10
    target_lost_after_s: float = 1.0
    noisy_target_std_m: float = 0.02
    lidar_dropout_prob: float = 0.00

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "SyntheticResidualEnvCfg":
        if cfg is None:
            return cls()
        grid_cfg = OccupancyGridCfg.from_mapping(cfg.get("grid")) if isinstance(cfg.get("grid"), Mapping) else OccupancyGridCfg(resolution=0.10)
        residual_raw = cfg.get("residual_policy") if isinstance(cfg.get("residual_policy"), Mapping) else {}
        residual_cfg = RLResidualPolicyCfg.from_mapping({**dict(residual_raw), "grid_cfg": grid_cfg})
        reward_cfg = NavigationRewardCfg.from_mapping(cfg.get("reward") if isinstance(cfg.get("reward"), Mapping) else None)
        nav_raw = cfg.get("nav") if isinstance(cfg.get("nav"), Mapping) else cfg.get("planner", {}).get("nav", {}) if isinstance(cfg.get("planner"), Mapping) else {}
        astar_raw = cfg.get("astar") if isinstance(cfg.get("astar"), Mapping) else cfg.get("planner", {}).get("astar", {}) if isinstance(cfg.get("planner"), Mapping) else {}
        nav_cfg = SimpleNavConfig(
            command_mode=str(nav_raw.get("command_mode", "heading_aligned")),
            max_vx=float(nav_raw.get("max_vx", 0.30)),
            max_vy=float(nav_raw.get("max_vy", 0.20)),
            max_wz=float(nav_raw.get("max_wz", 0.60)),
            path_lookahead_m=float(nav_raw.get("path_lookahead_m", 0.40)),
            success_radius=float(cfg.get("success_radius", 0.55)),
        )
        astar_cfg = GridAStarCfg.from_mapping(astar_raw)
        env_raw = cfg.get("env", cfg)
        if not isinstance(env_raw, Mapping):
            env_raw = {}
        return cls(
            num_envs=int(env_raw.get("num_envs", cfg.get("num_envs", cls.num_envs))),
            episode_length=int(env_raw.get("episode_length", env_raw.get("max_episode_steps", cls.episode_length))),
            dt=float(env_raw.get("control_dt", env_raw.get("dt", cls.dt))),
            success_radius=float(env_raw.get("success_radius", cfg.get("success_radius", cls.success_radius))),
            robot_radius=float(env_raw.get("robot_radius", cfg.get("robot_radius", cls.robot_radius))),
            obstacle_count=int(env_raw.get("obstacle_count", cfg.get("obstacle_count", cls.obstacle_count))),
            grid_cfg=grid_cfg,
            astar_cfg=astar_cfg,
            nav_cfg=nav_cfg,
            residual_policy_cfg=residual_cfg,
            reward=reward_cfg,
            path_points=int(residual_cfg.path_points),
            command_delay_steps_max=int(env_raw.get("command_delay_steps_max", cls.command_delay_steps_max)),
            replan_every=int(env_raw.get("replan_every", cfg.get("replan_every", cls.replan_every))),
            target_occlusion_prob=float(env_raw.get("target_occlusion_prob", cls.target_occlusion_prob)),
            noisy_target_std_m=float(env_raw.get("noisy_target_std_m", cls.noisy_target_std_m)),
            lidar_dropout_prob=float(env_raw.get("lidar_dropout_prob", cls.lidar_dropout_prob)),
        )


class SyntheticResidualNavEnv:
    """Vectorized, gym-like synthetic residual-control environment."""

    def __init__(
        self,
        cfg: SyntheticResidualEnvCfg | None = None,
        *,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or SyntheticResidualEnvCfg()
        self.device = torch.device(device)
        self.num_envs = int(self.cfg.num_envs)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.pose = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.target = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.command_current = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.command_target = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_command = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.previous_command_delta = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.episode_step = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.target_age_s = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.target_visible = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.obstacles: list[list[RectObstacle]] = [[] for _ in range(self.num_envs)]
        self.navigators = [SimpleNavigator(self.cfg.grid_cfg, self.cfg.astar_cfg, self.cfg.nav_cfg) for _ in range(self.num_envs)]
        self._last_policy_input: LocalPolicyInput | None = None
        self._last_nominal_command = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._last_min_clearance = torch.full((self.num_envs,), 6.0, dtype=torch.float32, device=self.device)
        self._last_path_deviation = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._last_obs = torch.empty((self.num_envs, self.cfg.residual_policy_cfg.feature_dim()), dtype=torch.float32, device=self.device)
        self.reset()

    @property
    def action_limits(self) -> torch.Tensor:
        return self.cfg.residual_policy_cfg.residual_limits.to(device=self.device)

    @property
    def observation_dim(self) -> int:
        return int(self.cfg.residual_policy_cfg.feature_dim())

    @property
    def action_dim(self) -> int:
        return 3

    def _rand_uniform(self, lo: float, hi: float, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=torch.float32, device=self.device).uniform_(float(lo), float(hi), generator=self.generator)

    def _sample_one_obstacles(self, env_id: int) -> list[RectObstacle]:
        obs: list[RectObstacle] = []
        for _ in range(int(self.cfg.obstacle_count)):
            # Rejection sampling to avoid placing an obstacle directly on start or target.
            chosen: RectObstacle | None = None
            for _attempt in range(40):
                sx = float(self._rand_uniform(*self.cfg.obstacle_size_x_range, (1,)).item())
                sy = float(self._rand_uniform(*self.cfg.obstacle_size_y_range, (1,)).item())
                cx = float(self._rand_uniform(*self.cfg.obstacle_x_range, (1,)).item())
                cy = float(self._rand_uniform(*self.cfg.obstacle_y_range, (1,)).item())
                cand = RectObstacle(cx, cy, sx, sy)
                tx, ty = float(self.target[env_id, 0].item()), float(self.target[env_id, 1].item())
                start_bad = cand.contains_xy(torch.tensor(0.0), torch.tensor(0.0), margin=self.cfg.obstacle_keepout_m).item()
                target_bad = cand.contains_xy(torch.tensor(tx), torch.tensor(ty), margin=self.cfg.obstacle_keepout_m).item()
                if not start_bad and not target_bad:
                    chosen = cand
                    break
            if chosen is None:
                chosen = RectObstacle(2.0, 1.3, 0.5, 0.5)
            obs.append(chosen)
        return obs

    def reset(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return self._last_obs

        self.pose[env_ids, :] = 0.0
        self.pose[env_ids, 2] = self._rand_uniform(-0.15, 0.15, (int(env_ids.numel()),))
        self.target[env_ids, 0] = self._rand_uniform(*self.cfg.target_x_range, (int(env_ids.numel()),))
        self.target[env_ids, 1] = self._rand_uniform(*self.cfg.target_y_range, (int(env_ids.numel()),))
        self.command_current[env_ids] = 0.0
        self.command_target[env_ids] = 0.0
        self.last_command[env_ids] = 0.0
        self.previous_command_delta[env_ids] = 0.0
        self.episode_step[env_ids] = 0
        self.target_age_s[env_ids] = 0.0
        self.target_visible[env_ids] = 1.0
        for idx in env_ids.detach().cpu().tolist():
            self.obstacles[int(idx)] = self._sample_one_obstacles(int(idx))
            self.navigators[int(idx)].reset()
        self._last_obs = self._build_observation()
        return self._last_obs

    def _body_grid_centers_world(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.grid_cfg
        rows = torch.arange(cfg.height, dtype=torch.float32, device=self.device)
        cols = torch.arange(cfg.width, dtype=torch.float32, device=self.device)
        x_b = float(cfg.x_max) - (rows + 0.5) * float(cfg.resolution)
        y_b = float(cfg.y_min) + (cols + 0.5) * float(cfg.resolution)
        xb = x_b[:, None].expand(cfg.height, cfg.width)
        yb = y_b[None, :].expand(cfg.height, cfg.width)
        yaw = self.pose[env_id, 2]
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        x_w = self.pose[env_id, 0] + c * xb - s * yb
        y_w = self.pose[env_id, 1] + s * xb + c * yb
        return x_w, y_w

    def _occupancy_grid_one(self, env_id: int) -> torch.Tensor:
        cfg = self.cfg.grid_cfg
        x_w, y_w = self._body_grid_centers_world(env_id)
        occ = torch.zeros((cfg.height, cfg.width), dtype=torch.float32, device=self.device)
        for obs in self.obstacles[env_id]:
            occ = torch.maximum(occ, obs.contains_xy(x_w, y_w).to(dtype=torch.float32))
        if float(self.cfg.lidar_dropout_prob) > 0.0:
            keep = torch.rand(occ.shape, dtype=occ.dtype, device=occ.device, generator=self.generator) > float(self.cfg.lidar_dropout_prob)
            occ = occ * keep.to(dtype=torch.float32)
        return occ

    def _occupancy_grid(self) -> torch.Tensor:
        return torch.stack([self._occupancy_grid_one(i) for i in range(self.num_envs)], dim=0).contiguous()

    def _pose2d(self, env_id: int) -> Pose2D:
        p = self.pose[env_id]
        return Pose2D(float(p[0].item()), float(p[1].item()), float(p[2].item()))

    def _target2d(self, env_id: int) -> Target2D:
        t = self.target[env_id]
        return Target2D(float(t[0].item()), float(t[1].item()))

    def _nominal_and_path(self, occupancy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nominal = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        path = torch.zeros((self.num_envs, int(self.cfg.path_points), 2), dtype=torch.float32, device=self.device)
        path_deviation = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            pose = self._pose2d(i)
            target = self._target2d(i)
            nav = self.navigators[i]
            should_replan = nav.plan_result is None or (int(self.episode_step[i].item()) % max(1, int(self.cfg.replan_every)) == 0)
            if should_replan:
                nav.plan(pose, target, occupancy[i], allow_goal_relocation=True)
            cmd = nav.command(pose, target).command
            nominal[i] = torch.tensor(cmd.tuple, dtype=torch.float32, device=self.device)
            remaining_body = nav.remaining_body_path(pose)
            if not remaining_body:
                remaining_body = [world_to_body_xy(target.xy, pose)]
            path_i = fixed_body_path_points(
                remaining_body,
                num_points=int(self.cfg.path_points),
                spacing_m=float(self.cfg.path_spacing_m),
                device=self.device,
            )
            path[i] = path_i
            if path_i.numel() > 0:
                # Distance from target vector to the local path endpoint is a coarse path-following diagnostic.
                tgt = torch.tensor(world_to_body_xy(target.xy, pose), dtype=torch.float32, device=self.device)
                path_deviation[i] = torch.linalg.norm(path_i[min(2, path_i.shape[0] - 1)] - tgt).clamp(max=5.0)
        return nominal, path, path_deviation

    def _min_clearance(self) -> torch.Tensor:
        clear = torch.full((self.num_envs,), 6.0, dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            x = self.pose[i, 0]
            y = self.pose[i, 1]
            vals = []
            for obs in self.obstacles[i]:
                dx = torch.clamp(torch.abs(x - obs.cx) - 0.5 * obs.sx, min=0.0)
                dy = torch.clamp(torch.abs(y - obs.cy) - 0.5 * obs.sy, min=0.0)
                vals.append(torch.sqrt(dx * dx + dy * dy))
            if vals:
                clear[i] = torch.stack(vals).amin()
        return clear

    def _collision(self) -> torch.Tensor:
        flags = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        for i in range(self.num_envs):
            x = self.pose[i, 0]
            y = self.pose[i, 1]
            for obs in self.obstacles[i]:
                if bool(obs.contains_xy(x, y, margin=float(self.cfg.robot_radius)).item()):
                    flags[i] = True
                    break
        return flags

    def _build_policy_input(self, occupancy: torch.Tensor, nominal: torch.Tensor, path: torch.Tensor) -> LocalPolicyInput:
        target_xy_b = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            target_xy_b[i] = torch.tensor(world_to_body_xy(self._target2d(i).xy, self._pose2d(i)), dtype=torch.float32, device=self.device)
        # Simulated detector noise and temporary visibility loss.
        if float(self.cfg.noisy_target_std_m) > 0.0:
            noise = torch.randn(target_xy_b.shape, dtype=torch.float32, device=self.device, generator=self.generator) * float(self.cfg.noisy_target_std_m)
            target_xy_b = target_xy_b + noise
        if float(self.cfg.target_occlusion_prob) > 0.0:
            visible_now = (torch.rand((self.num_envs,), dtype=torch.float32, device=self.device, generator=self.generator) >= float(self.cfg.target_occlusion_prob)).to(torch.float32)
            self.target_visible = visible_now
            self.target_age_s = torch.where(visible_now > 0.5, torch.zeros_like(self.target_age_s), self.target_age_s + float(self.cfg.dt))
        tracker = torch.where(
            self.target_visible > 0.5,
            torch.full((self.num_envs,), int(TrackerState.TRACKING), dtype=torch.long, device=self.device),
            torch.where(
                self.target_age_s < float(self.cfg.target_lost_after_s),
                torch.full((self.num_envs,), int(TrackerState.OCCLUDED_RECENT), dtype=torch.long, device=self.device),
                torch.full((self.num_envs,), int(TrackerState.LOST), dtype=torch.long, device=self.device),
            ),
        )
        return LocalPolicyInput(
            occupancy_grid=occupancy,
            target_xy_b=target_xy_b,
            path_xy_b=path,
            nominal_command=nominal,
            last_command=self.last_command,
            target_visible=self.target_visible,
            target_age_s=self.target_age_s,
            tracker_state=tracker,
        )

    def _build_observation(self) -> torch.Tensor:
        occupancy = self._occupancy_grid()
        nominal, path, path_deviation = self._nominal_and_path(occupancy)
        policy_input = self._build_policy_input(occupancy, nominal, path)
        self._last_policy_input = policy_input
        self._last_nominal_command = nominal
        self._last_min_clearance = self._min_clearance()
        self._last_path_deviation = path_deviation
        return build_compact_residual_features(policy_input, self.cfg.residual_policy_cfg)

    def step(self, action_normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if action_normalized.shape != (self.num_envs, 3):
            raise ValueError(f"action_normalized must have shape {(self.num_envs, 3)}, got {tuple(action_normalized.shape)}")
        prev_distance = torch.linalg.norm(self.target - self.pose[:, :2], dim=-1)
        prev_cmd = self.command_current.clone()
        residual = torch.clamp(action_normalized.to(self.device, dtype=torch.float32), -1.0, 1.0) * self.action_limits[None, :]
        nominal = self._last_nominal_command
        cmd_target = nominal + residual
        cmd_target[:, 0] = cmd_target[:, 0].clamp(0.0, float(self.cfg.command_max_vx))
        cmd_target[:, 1] = cmd_target[:, 1].clamp(-float(self.cfg.command_max_vy), float(self.cfg.command_max_vy))
        cmd_target[:, 2] = cmd_target[:, 2].clamp(-float(self.cfg.command_max_wz), float(self.cfg.command_max_wz))
        alpha = float(self.cfg.command_smoothing_alpha)
        self.command_target = cmd_target
        self.command_current = (1.0 - alpha) * self.command_current + alpha * self.command_target

        for i in range(self.num_envs):
            pose = self._pose2d(i)
            cmd = MotionCommand(float(self.command_current[i, 0].item()), float(self.command_current[i, 1].item()), float(self.command_current[i, 2].item()))
            nxt = integrate_body_command(pose, cmd, float(self.cfg.dt))
            self.pose[i] = torch.tensor([nxt.x, nxt.y, nxt.yaw], dtype=torch.float32, device=self.device)
        self.episode_step += 1

        new_distance = torch.linalg.norm(self.target - self.pose[:, :2], dim=-1)
        progress = prev_distance - new_distance
        success = new_distance <= float(self.cfg.success_radius)
        collision = self._collision()
        timeout = self.episode_step >= int(self.cfg.episode_length)
        done = success | collision | timeout
        lost = self.target_age_s >= float(self.cfg.target_lost_after_s)
        min_clearance = self._min_clearance()
        command_delta = self.command_current - prev_cmd
        jerk = command_delta - self.previous_command_delta
        self.previous_command_delta = command_delta.detach().clone()
        terms = compute_navigation_reward_terms(
            progress=progress,
            command=self.command_current,
            success=success,
            cfg=self.cfg.reward,
            raw_residual=residual,
            last_command=prev_cmd,
            previous_command_delta=jerk,
            min_clearance=min_clearance,
            collision=collision,
            safety_intervention=None,
            path_deviation=self._last_path_deviation,
            target_lost=lost,
            timeout=timeout,
        )
        reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for value in terms.values():
            reward = reward + value.to(device=self.device, dtype=torch.float32)

        self.last_command = self.command_current.detach().clone()
        info = {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "distance": new_distance,
            "progress": progress,
            "nominal_command": nominal.detach().clone(),
            "residual": residual.detach().clone(),
            "command": self.command_current.detach().clone(),
            "min_clearance": min_clearance.detach().clone(),
            "target_lost": lost.detach().clone(),
        }
        for name, term in terms.items():
            info[f"reward/{name}"] = term.detach().clone()

        if bool(done.any().item()):
            self.reset(torch.nonzero(done, as_tuple=False).flatten())
        else:
            self._last_obs = self._build_observation()
        return self._last_obs, reward, done.to(dtype=torch.float32), info

    def current_policy_input(self) -> LocalPolicyInput:
        if self._last_policy_input is None:
            self._build_observation()
        assert self._last_policy_input is not None
        return self._last_policy_input

    def scenario_metadata(self, env_id: int = 0) -> dict[str, Any]:
        return {
            "pose": self.pose[int(env_id)].detach().cpu().tolist(),
            "target": self.target[int(env_id)].detach().cpu().tolist(),
            "obstacles": [obs.as_dict() for obs in self.obstacles[int(env_id)]],
        }


__all__ = ["RectObstacle", "SyntheticResidualEnvCfg", "SyntheticResidualNavEnv"]
