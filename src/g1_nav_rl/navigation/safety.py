"""Small LiDAR-based command safety filter.

This module is intentionally a post-planner filter. It does not plan, build maps,
modify A* paths, or talk to Isaac/G1. It takes a nominal body-frame command from
``NavRuntime``/``SimpleNavigator`` plus current LiDAR points in the robot body
frame and returns a locally safer ``MotionCommand``.

Frame convention matches the navigation smoke tests:

- ``+x`` is forward;
- ``+y`` is left;
- ``wz`` is positive yaw rate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Mapping

import torch

from g1_nav_rl.navigation.simple_nav import MotionCommand, Pose2D, Target2D


@dataclass(frozen=True)
class LidarSafetyConfig:
    """Configuration for :class:`LidarSafetyFilter`.

    The defaults are conservative but not intended to replace path planning.
    This filter should be enabled explicitly in scripts/tests.
    """

    enabled: bool = True

    # Sector definitions in degrees in the robot/body frame.
    front_sector_deg: float = 30.0
    side_inner_deg: float = 30.0
    side_outer_deg: float = 100.0

    # Use a low percentile instead of raw min to reduce single-ray noise.
    clearance_percentile: float = 0.10
    max_clearance: float = 8.0

    # Front-zone intervention distances.
    front_slow_distance: float = 1.20
    front_stop_distance: float = 0.70
    front_backup_distance: float = 0.45

    # Side-zone intervention distances. These are checked independently from
    # the front sector so that the robot does not keep moving along an obstacle
    # with insufficient lateral clearance. ``side_open_distance`` is retained
    # for front-blocked sidestep selection; the three side_* thresholds below
    # govern ordinary side-clearance behavior.
    side_open_distance: float = 0.90
    side_slow_distance: float = 0.90
    side_stop_distance: float = 0.65
    side_critical_distance: float = 0.45

    # Override commands.
    slow_min_scale: float = 0.25
    backup_speed: float = 0.12
    backup_lateral_speed: float = 0.06
    sidestep_speed: float = 0.12
    sidestep_wz: float = 0.0

    # Stuck detection. Set stuck_window_steps <= 1 to disable.
    stuck_window_steps: int = 50
    stuck_min_command_speed: float = 0.08
    stuck_min_pose_progress: float = 0.05
    stuck_min_target_progress: float = 0.05
    stuck_recovery_steps: int = 30

    # Command clipping after safety override.
    max_vx: float = 0.30
    max_vy: float = 0.25
    max_wz: float = 0.60


@dataclass(frozen=True)
class LidarSafetyInfo:
    """Debug information for one safety-filter application."""

    action: str
    reason: str
    override_active: bool
    scale: float
    front_clearance: float
    left_clearance: float
    right_clearance: float
    valid_points: int
    stuck_active: bool
    stuck_recovery_remaining: int


@dataclass(frozen=True)
class LidarSafetyResult:
    """Output of :class:`LidarSafetyFilter`."""

    command: MotionCommand
    info: LidarSafetyInfo


@dataclass(frozen=True)
class SectorClearances:
    front: float
    left: float
    right: float
    valid_points: int


def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(float(v), float(lo)), float(hi))


def _clip_command(cmd: MotionCommand, cfg: LidarSafetyConfig) -> MotionCommand:
    return MotionCommand(
        vx=_clip(cmd.vx, -abs(cfg.max_vx), abs(cfg.max_vx)),
        vy=_clip(cmd.vy, -abs(cfg.max_vy), abs(cfg.max_vy)),
        wz=_clip(cmd.wz, -abs(cfg.max_wz), abs(cfg.max_wz)),
    )


def _as_env0_points(points_b: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(points_b):
        raise TypeError(f"points_b must be a torch.Tensor, got {type(points_b)!r}")
    if points_b.ndim == 3:
        points_b = points_b[0]
    if points_b.ndim != 2 or points_b.shape[-1] != 2:
        raise ValueError(f"points_b must have shape [B,2] or [N,B,2], got {tuple(points_b.shape)}")
    return points_b.detach().to(dtype=torch.float32)


def _as_env0_valid(valid: torch.Tensor, *, expected_len: int) -> torch.Tensor:
    if not torch.is_tensor(valid):
        raise TypeError(f"valid must be a torch.Tensor, got {type(valid)!r}")
    if valid.ndim == 2:
        valid = valid[0]
    if valid.ndim != 1 or int(valid.shape[0]) != int(expected_len):
        raise ValueError(f"valid must have shape [B] or [N,B] with B={expected_len}, got {tuple(valid.shape)}")
    return valid.detach().to(dtype=torch.bool)


def _percentile_clearance(values: torch.Tensor, *, percentile: float, default: float) -> float:
    if values.numel() == 0:
        return float(default)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float(default)
    q = _clip(percentile, 0.0, 1.0)
    return float(torch.quantile(values.detach().to(dtype=torch.float32).cpu(), q).item())


def compute_lidar_sector_clearances(
    points_b: torch.Tensor,
    valid: torch.Tensor,
    *,
    cfg: LidarSafetyConfig | None = None,
) -> SectorClearances:
    """Compute front/left/right clearances from body-frame LiDAR hit points.

    Invalid/no-hit rays are treated as no obstacle. If a sector has no valid
    obstacle hit, its clearance is ``cfg.max_clearance``.
    """
    cfg = cfg or LidarSafetyConfig()
    points = _as_env0_points(points_b)
    mask = _as_env0_valid(valid, expected_len=int(points.shape[0]))

    finite = torch.isfinite(points).all(dim=-1)
    ranges = torch.linalg.norm(points, dim=-1)
    mask = mask & finite & torch.isfinite(ranges) & (ranges > 1.0e-6)
    if int(mask.sum().item()) <= 0:
        d = float(cfg.max_clearance)
        return SectorClearances(front=d, left=d, right=d, valid_points=0)

    angles = torch.atan2(points[:, 1], points[:, 0])
    abs_angles = torch.abs(angles)
    front = mask & (abs_angles <= math.radians(float(cfg.front_sector_deg))) & (points[:, 0] > 0.0)
    left = (
        mask
        & (angles >= math.radians(float(cfg.side_inner_deg)))
        & (angles <= math.radians(float(cfg.side_outer_deg)))
    )
    right = (
        mask
        & (angles <= -math.radians(float(cfg.side_inner_deg)))
        & (angles >= -math.radians(float(cfg.side_outer_deg)))
    )

    return SectorClearances(
        front=_percentile_clearance(ranges[front], percentile=float(cfg.clearance_percentile), default=float(cfg.max_clearance)),
        left=_percentile_clearance(ranges[left], percentile=float(cfg.clearance_percentile), default=float(cfg.max_clearance)),
        right=_percentile_clearance(ranges[right], percentile=float(cfg.clearance_percentile), default=float(cfg.max_clearance)),
        valid_points=int(mask.sum().detach().cpu().item()),
    )


class LidarSafetyFilter:
    """Deterministic LiDAR safety layer for nominal navigation commands."""

    def __init__(self, config: LidarSafetyConfig | None = None) -> None:
        self.config = config or LidarSafetyConfig()
        n = max(0, int(self.config.stuck_window_steps))
        self._pose_history: Deque[tuple[float, float]] = deque(maxlen=n if n > 0 else 1)
        self._target_distance_history: Deque[float] = deque(maxlen=n if n > 0 else 1)
        self._command_speed_history: Deque[float] = deque(maxlen=n if n > 0 else 1)
        self._recovery_remaining = 0
        self._recovery_side_sign = 1.0

    def reset(self) -> None:
        self._pose_history.clear()
        self._target_distance_history.clear()
        self._command_speed_history.clear()
        self._recovery_remaining = 0
        self._recovery_side_sign = 1.0

    @staticmethod
    def disabled_result(nominal_cmd: MotionCommand) -> LidarSafetyResult:
        return LidarSafetyResult(
            command=nominal_cmd,
            info=LidarSafetyInfo(
                action="disabled",
                reason="safety filter disabled",
                override_active=False,
                scale=1.0,
                front_clearance=math.inf,
                left_clearance=math.inf,
                right_clearance=math.inf,
                valid_points=0,
                stuck_active=False,
                stuck_recovery_remaining=0,
            ),
        )

    def _update_stuck_history(self, nominal_cmd: MotionCommand, pose: Pose2D | None, target: Target2D | None) -> None:
        if int(self.config.stuck_window_steps) <= 1 or pose is None:
            return
        speed = math.hypot(float(nominal_cmd.vx), float(nominal_cmd.vy))
        self._pose_history.append((float(pose.x), float(pose.y)))
        self._command_speed_history.append(speed)
        if target is not None:
            self._target_distance_history.append(math.hypot(float(target.x) - float(pose.x), float(target.y) - float(pose.y)))

    def _stuck_detected(self) -> bool:
        n = int(self.config.stuck_window_steps)
        if n <= 1 or len(self._pose_history) < n or len(self._command_speed_history) < n:
            return False
        avg_cmd = sum(self._command_speed_history) / max(1, len(self._command_speed_history))
        if avg_cmd < float(self.config.stuck_min_command_speed):
            return False
        x0, y0 = self._pose_history[0]
        x1, y1 = self._pose_history[-1]
        pose_progress = math.hypot(x1 - x0, y1 - y0)
        if pose_progress >= float(self.config.stuck_min_pose_progress):
            return False
        if len(self._target_distance_history) >= n:
            target_progress = float(self._target_distance_history[0]) - float(self._target_distance_history[-1])
            if target_progress >= float(self.config.stuck_min_target_progress):
                return False
        return True

    def _choose_side_sign(self, clearances: SectorClearances) -> float:
        # +1 means left (+vy), -1 means right (-vy).
        return 1.0 if float(clearances.left) >= float(clearances.right) else -1.0

    def _backup_command(self, clearances: SectorClearances, *, action: str, reason: str, stuck_active: bool) -> LidarSafetyResult:
        side = self._choose_side_sign(clearances)
        cmd = MotionCommand(
            vx=-abs(float(self.config.backup_speed)),
            vy=side * abs(float(self.config.backup_lateral_speed)),
            wz=0.0,
        )
        return self._result(cmd, clearances, action=action, reason=reason, scale=0.0, stuck_active=stuck_active)

    def _result(
        self,
        cmd: MotionCommand,
        clearances: SectorClearances,
        *,
        action: str,
        reason: str,
        scale: float,
        stuck_active: bool,
    ) -> LidarSafetyResult:
        final_cmd = _clip_command(cmd, self.config)
        return LidarSafetyResult(
            command=final_cmd,
            info=LidarSafetyInfo(
                action=action,
                reason=reason,
                override_active=action not in {"ok", "disabled"},
                scale=float(scale),
                front_clearance=float(clearances.front),
                left_clearance=float(clearances.left),
                right_clearance=float(clearances.right),
                valid_points=int(clearances.valid_points),
                stuck_active=bool(stuck_active),
                stuck_recovery_remaining=int(self._recovery_remaining),
            ),
        )

    def apply(
        self,
        nominal_cmd: MotionCommand,
        *,
        lidar_points_b: torch.Tensor,
        lidar_valid: torch.Tensor,
        pose: Pose2D | None = None,
        target: Target2D | None = None,
        step_index: int = 0,
    ) -> LidarSafetyResult:
        """Apply the safety filter to one nominal command."""
        if not bool(self.config.enabled):
            return self.disabled_result(nominal_cmd)

        clearances = compute_lidar_sector_clearances(lidar_points_b, lidar_valid, cfg=self.config)
        self._update_stuck_history(nominal_cmd, pose, target)

        # 1. Immediate critical front obstacle.
        if clearances.front < float(self.config.front_backup_distance):
            self._recovery_remaining = 0
            return self._backup_command(
                clearances,
                action="backup",
                reason=f"front_clearance={clearances.front:.3f} < backup={self.config.front_backup_distance:.3f}",
                stuck_active=False,
            )

        # 2. Continue active stuck recovery before ordinary slow/stop rules.
        if self._recovery_remaining > 0:
            self._recovery_remaining -= 1
            cmd = MotionCommand(
                vx=-abs(float(self.config.backup_speed)),
                vy=float(self._recovery_side_sign) * abs(float(self.config.backup_lateral_speed)),
                wz=0.0,
            )
            return self._result(
                cmd,
                clearances,
                action="stuck_recovery",
                reason="fixed-duration backup/sidestep recovery active",
                scale=0.0,
                stuck_active=True,
            )

        # 3. Start stuck recovery if commanded motion is not producing progress.
        if self._stuck_detected():
            self._recovery_side_sign = self._choose_side_sign(clearances)
            self._recovery_remaining = max(0, int(self.config.stuck_recovery_steps) - 1)
            cmd = MotionCommand(
                vx=-abs(float(self.config.backup_speed)),
                vy=float(self._recovery_side_sign) * abs(float(self.config.backup_lateral_speed)),
                wz=0.0,
            )
            return self._result(
                cmd,
                clearances,
                action="stuck_recovery",
                reason="nominal command effort with insufficient pose/target progress",
                scale=0.0,
                stuck_active=True,
            )

        # 4. Stop or sidestep when front is too close.
        if clearances.front < float(self.config.front_stop_distance):
            left_open = clearances.left >= float(self.config.side_open_distance)
            right_open = clearances.right >= float(self.config.side_open_distance)
            if left_open or right_open:
                side = self._choose_side_sign(clearances)
                # Respect which side is actually open if only one side is open.
                if left_open and not right_open:
                    side = 1.0
                elif right_open and not left_open:
                    side = -1.0
                cmd = MotionCommand(vx=0.0, vy=side * abs(float(self.config.sidestep_speed)), wz=float(self.config.sidestep_wz))
                return self._result(
                    cmd,
                    clearances,
                    action="sidestep_left" if side > 0 else "sidestep_right",
                    reason=f"front_clearance={clearances.front:.3f} < stop={self.config.front_stop_distance:.3f}; side open",
                    scale=0.0,
                    stuck_active=False,
                )
            cmd = MotionCommand(vx=0.0, vy=0.0, wz=0.0)
            return self._result(
                cmd,
                clearances,
                action="stop",
                reason=f"front_clearance={clearances.front:.3f} < stop={self.config.front_stop_distance:.3f}; no side open",
                scale=0.0,
                stuck_active=False,
            )

        # 5. Side-critical: move away from an obstacle that is too close on
        # either side even when the front sector is clear. This catches
        # sustained side-following cases where the nominal path runs too close
        # to a long obstacle.
        side_critical = float(self.config.side_critical_distance)
        if clearances.left < side_critical or clearances.right < side_critical:
            if clearances.left < clearances.right:
                # Obstacle is closer on the left: step right (-vy).
                side = -1.0
                action = "sidestep_right"
                side_name = "left"
                side_clearance = float(clearances.left)
            else:
                # Obstacle is closer on the right: step left (+vy).
                side = 1.0
                action = "sidestep_left"
                side_name = "right"
                side_clearance = float(clearances.right)
            cmd = MotionCommand(vx=0.0, vy=side * abs(float(self.config.sidestep_speed)), wz=float(self.config.sidestep_wz))
            return self._result(
                cmd,
                clearances,
                action=action,
                reason=f"{side_name}_clearance={side_clearance:.3f} < side_critical={side_critical:.3f}",
                scale=0.0,
                stuck_active=False,
            )

        # 6. Side-slow/side-stop: reduce forward motion when lateral clearance
        # is tight and never allow lateral velocity into the close side.
        side_min = min(float(clearances.left), float(clearances.right))
        side_slow = float(self.config.side_slow_distance)
        side_stop = float(self.config.side_stop_distance)
        if side_min < side_slow:
            denom = max(1.0e-6, side_slow - side_stop)
            side_scale = (side_min - side_stop) / denom
            side_scale = _clip(side_scale, float(self.config.slow_min_scale), 1.0)
            vx = float(nominal_cmd.vx)
            vy = float(nominal_cmd.vy)
            wz = float(nominal_cmd.wz)
            action = "side_slow"
            reasons: list[str] = []

            if vx > 0.0:
                vx *= side_scale

            if float(clearances.left) < side_stop and vy > 0.0:
                vy = 0.0
                action = "side_stop"
                reasons.append(f"suppress_left_vy left_clearance={clearances.left:.3f} < side_stop={side_stop:.3f}")
            elif float(clearances.left) < side_slow:
                reasons.append(f"left_clearance={clearances.left:.3f} < side_slow={side_slow:.3f}")

            if float(clearances.right) < side_stop and vy < 0.0:
                vy = 0.0
                action = "side_stop"
                reasons.append(f"suppress_right_vy right_clearance={clearances.right:.3f} < side_stop={side_stop:.3f}")
            elif float(clearances.right) < side_slow:
                reasons.append(f"right_clearance={clearances.right:.3f} < side_slow={side_slow:.3f}")

            cmd = MotionCommand(vx=vx, vy=vy, wz=wz)
            return self._result(
                cmd,
                clearances,
                action=action,
                reason="; ".join(reasons) if reasons else f"side_clearance={side_min:.3f} < side_slow={side_slow:.3f}",
                scale=side_scale,
                stuck_active=False,
            )

        # 7. Slow down forward command when a front obstacle is nearby. The
        # front-slow rule is checked after side rules because side-critical
        # clearance is the stronger local safety condition in side-following.
        if nominal_cmd.vx > 0.0 and clearances.front < float(self.config.front_slow_distance):
            denom = max(1.0e-6, float(self.config.front_slow_distance) - float(self.config.front_stop_distance))
            scale = (float(clearances.front) - float(self.config.front_stop_distance)) / denom
            scale = _clip(scale, float(self.config.slow_min_scale), 1.0)
            cmd = MotionCommand(vx=float(nominal_cmd.vx) * scale, vy=float(nominal_cmd.vy), wz=float(nominal_cmd.wz))
            return self._result(
                cmd,
                clearances,
                action="slow",
                reason=f"front_clearance={clearances.front:.3f} < slow={self.config.front_slow_distance:.3f}",
                scale=scale,
                stuck_active=False,
            )

        return self._result(nominal_cmd, clearances, action="ok", reason="clear", scale=1.0, stuck_active=False)

    def apply_from_lidar_obs(
        self,
        nominal_cmd: MotionCommand,
        lidar_obs: Mapping[str, torch.Tensor],
        *,
        pose: Pose2D | None = None,
        target: Target2D | None = None,
        step_index: int = 0,
    ) -> LidarSafetyResult:
        """Apply using the dictionary returned by ``build_lidar_proxy_observation``."""
        return self.apply(
            nominal_cmd,
            lidar_points_b=lidar_obs["lidar_points_b"],
            lidar_valid=lidar_obs["lidar_valid"],
            pose=pose,
            target=target,
            step_index=step_index,
        )
