from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import torch

from g1_nav_rl.rewards.navigation_reward import NavigationRewardCfg, compute_navigation_reward
from g1_nav_rl.safety.depth_safety_filter import (
    DepthSafetyFilterCfg,
    apply_depth_safety_filter,
    disabled_depth_safety_info,
)
from g1_nav_rl.utils.geometry import distance_xy, target_vector_body_frame


ObservationBuilder = Callable[[Any, Any], dict[str, torch.Tensor]]


@runtime_checkable
class CommandProxyLike(Protocol):
    """Minimal command-proxy interface required by ``G1NavEnv``."""

    num_envs: int
    device: torch.device
    pos_w: torch.Tensor
    yaw: torch.Tensor
    cmd_target: torch.Tensor
    cmd_current: torch.Tensor

    def reset(self, env_origins: torch.Tensor) -> None: ...

    def set_command(self, command: torch.Tensor) -> None: ...

    def step(self, dt: float) -> None: ...

    def observation(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class G1NavEnvCfg:
    """Configuration for the command-level navigation environment wrapper."""

    max_episode_steps: int = 500
    success_radius: float = 0.20
    use_depth_safety: bool = True
    reward: NavigationRewardCfg = field(default_factory=NavigationRewardCfg)
    depth_safety: DepthSafetyFilterCfg = field(default_factory=DepthSafetyFilterCfg)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "G1NavEnvCfg":
        """Build an env config from a plain mapping.

        Supported keys are deliberately small for now:

        - ``max_episode_steps``
        - ``success_radius``
        - ``use_depth_safety``
        - ``reward`` with ``progress_weight``, ``command_penalty_weight``, ``success_bonus``
        - ``depth_safety`` using ``DepthSafetyFilterCfg`` fields
        """
        if cfg is None:
            return cls()

        reward_raw = cfg.get("reward", {})
        if not isinstance(reward_raw, Mapping):
            raise ValueError("env.reward must be a mapping if provided")

        depth_safety_raw = cfg.get("depth_safety", None)
        if depth_safety_raw is not None and not isinstance(depth_safety_raw, Mapping):
            raise ValueError("env.depth_safety must be a mapping if provided")

        reward = NavigationRewardCfg(
            progress_weight=float(reward_raw.get("progress_weight", NavigationRewardCfg.progress_weight)),
            command_penalty_weight=float(
                reward_raw.get("command_penalty_weight", NavigationRewardCfg.command_penalty_weight)
            ),
            success_bonus=float(reward_raw.get("success_bonus", NavigationRewardCfg.success_bonus)),
        )

        depth_safety = (
            DepthSafetyFilterCfg.from_mapping(depth_safety_raw)
            if depth_safety_raw is not None
            else DepthSafetyFilterCfg()
        )

        return cls(
            max_episode_steps=int(cfg.get("max_episode_steps", cls.max_episode_steps)),
            success_radius=float(cfg.get("success_radius", cls.success_radius)),
            use_depth_safety=bool(cfg.get("use_depth_safety", cls.use_depth_safety)),
            reward=reward,
            depth_safety=depth_safety,
        )


class G1NavEnv:
    """Command-level navigation environment wrapper for the G1 proxy.

    This class does not create Isaac Sim scenes. It wraps objects that already
    exist in a smoke script or future task builder:

    - a simulation context with ``get_physics_dt()`` and ``step()``;
    - an Isaac Lab scene with ``env_origins``, ``reset()``, ``write_data_to_sim()``,
      ``update(dt)``, and ``__getitem__`` for sensors;
    - a ``G1CommandProxy``-compatible proxy;
    - an observation builder that returns at least ``rgb`` and ``depth``.

    The action is a body-frame command tensor ``[vx, vy, wz]``. The final command
    path is:

    ``action -> optional depth safety filter -> proxy.set_command() -> proxy.step()``.
    """

    def __init__(
        self,
        *,
        sim: Any,
        scene: Any,
        proxy: CommandProxyLike,
        target_xy_w: torch.Tensor,
        observation_builder: ObservationBuilder,
        cfg: G1NavEnvCfg | None = None,
    ) -> None:
        self.sim = sim
        self.scene = scene
        self.proxy = proxy
        self.observation_builder = observation_builder
        self.cfg = cfg if cfg is not None else G1NavEnvCfg()

        self.device = torch.device(getattr(proxy, "device", target_xy_w.device))
        self.num_envs = int(getattr(proxy, "num_envs", target_xy_w.shape[0]))
        self.dt = float(sim.get_physics_dt())

        self.target_xy_w = self._validate_target_xy(target_xy_w)

        self.episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.safety_intervention_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_distance = self._distance_to_target()

        self.last_raw_action = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_command = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_safety_info: dict[str, float] = disabled_depth_safety_info()
        self.last_obs: dict[str, torch.Tensor] | None = None

    def _validate_target_xy(self, target_xy_w: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(target_xy_w):
            raise TypeError(f"target_xy_w must be a torch.Tensor, got {type(target_xy_w)!r}")
        target_xy_w = target_xy_w.to(device=self.device, dtype=torch.float32)
        if target_xy_w.ndim == 1:
            if target_xy_w.shape[0] != 2:
                raise ValueError(f"1D target_xy_w must have shape [2], got {tuple(target_xy_w.shape)}")
            target_xy_w = target_xy_w.unsqueeze(0).repeat(self.num_envs, 1)
        if target_xy_w.shape != (self.num_envs, 2):
            raise ValueError(f"target_xy_w must have shape {(self.num_envs, 2)}, got {tuple(target_xy_w.shape)}")
        return target_xy_w

    def _validate_action(self, action: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(action):
            raise TypeError(f"action must be a torch.Tensor, got {type(action)!r}")
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            if action.shape[0] != 3:
                raise ValueError(f"1D action must have shape [3], got {tuple(action.shape)}")
            action = action.unsqueeze(0).repeat(self.num_envs, 1)
        if action.shape != (self.num_envs, 3):
            raise ValueError(f"action must have shape {(self.num_envs, 3)}, got {tuple(action.shape)}")
        return action

    def _distance_to_target(self) -> torch.Tensor:
        return distance_xy(self.proxy.pos_w[:, 0:2], self.target_xy_w)

    def _target_vector(self) -> torch.Tensor:
        return target_vector_body_frame(
            self.proxy.pos_w[:, 0:2],
            self.proxy.yaw,
            self.target_xy_w,
        )

    def _build_observation(self) -> dict[str, torch.Tensor]:
        sensor_obs = self.observation_builder(self.scene, self.proxy)
        if "rgb" not in sensor_obs:
            raise KeyError("observation_builder must return key 'rgb'")
        if "depth" not in sensor_obs:
            raise KeyError("observation_builder must return key 'depth'")

        obs = dict(sensor_obs)
        obs["target_vector"] = self._target_vector()
        obs["proxy_pose"] = torch.cat((self.proxy.pos_w[:, 0:2], self.proxy.yaw.unsqueeze(-1)), dim=-1)
        obs["last_command"] = self.last_command.clone()
        return obs

    def reset(self, *, target_xy_w: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Reset proxy state and return an observation dictionary."""
        if target_xy_w is not None:
            self.target_xy_w = self._validate_target_xy(target_xy_w)

        self.proxy.reset(self.scene.env_origins)
        self.scene.reset()

        self.episode_length.zero_()
        self.safety_intervention_count.zero_()
        self.last_raw_action.zero_()
        self.last_command.zero_()
        self.last_safety_info = disabled_depth_safety_info()

        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

        self.previous_distance = self._distance_to_target()
        self.last_obs = self._build_observation()
        return self.last_obs

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply one command-level action and advance the simulation.

        Args:
            action: Body-frame velocity command with shape ``[N, 3]`` or ``[3]``.

        Returns:
            ``(obs, reward, done, info)`` where ``done`` is ``terminated | truncated``.
            ``info`` also contains separate ``terminated`` and ``truncated`` flags.
        """
        action = self._validate_action(action)
        self.last_raw_action = action.clone()

        command = action
        safety_info = disabled_depth_safety_info()
        safety_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if self.cfg.use_depth_safety and self.last_obs is not None and "depth" in self.last_obs:
            command, safety_info = apply_depth_safety_filter(self.last_obs["depth"], action, self.cfg.depth_safety)
            safety_active = command.ne(action).any(dim=-1)
            self.safety_intervention_count += safety_active.to(dtype=torch.long)

        self.last_command = command.clone()

        self.proxy.set_command(command)
        self.proxy.step(self.dt)

        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

        self.episode_length += 1

        obs = self._build_observation()
        self.last_obs = obs

        distance = self._distance_to_target()
        progress = self.previous_distance - distance
        self.previous_distance = distance.clone()

        terminated = distance <= float(self.cfg.success_radius)
        truncated = self.episode_length >= int(self.cfg.max_episode_steps)
        done = terminated | truncated

        reward = compute_navigation_reward(
            progress=progress,
            command=command,
            success=terminated,
            cfg=self.cfg.reward,
        )

        command_norm = torch.linalg.norm(command, dim=-1)
        metrics = {
            "distance_to_target": distance,
            "collision_proxy": safety_active,
            "safety_intervention_count": self.safety_intervention_count.clone(),
            "command_norm": command_norm,
            "progress": progress,
            "episode_length": self.episode_length.clone(),
        }

        info: dict[str, Any] = {
            "metrics": metrics,
            "safety": safety_info,
            "terminated": terminated,
            "truncated": truncated,
            "raw_action": action,
            "final_command": command,
        }
        return obs, reward, done, info
