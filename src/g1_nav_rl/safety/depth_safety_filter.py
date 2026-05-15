from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class DepthSafetyFilterCfg:
    """Configuration for the RGB-D near-field safety reflex.

    The filter modifies a body-frame command ``[vx, vy, wz]`` using a simple
    front-depth rule copied from the original go-to-target smoke test:

    - inspect a vertical band of the depth image,
    - treat the center crop as the forward obstacle region,
    - reduce/stop positive forward velocity when the center region is close,
    - sidestep and yaw toward the side whose crop has greater free depth.

    This is a near-field command filter, not a planner and not a collision
    model. It should be applied downstream of policy output and upstream of the
    command proxy or real robot command adapter.
    """

    enabled: bool = True
    vertical_band: tuple[float, float] = (0.25, 0.90)
    center_width_frac: float = 0.60
    safety_distance: float = 2.0
    hard_stop_distance: float = 1.1
    forward_scale_at_safety: float = 0.0
    sidestep_speed: float = 0.30
    yaw_away_rate: float = 0.50

    @classmethod
    def from_mapping(cls, safety_cfg: Mapping[str, object] | None) -> "DepthSafetyFilterCfg":
        """Create a filter config from the existing YAML ``obstacle_safety`` block."""
        if safety_cfg is None:
            return cls(enabled=False)

        vertical_band_raw = safety_cfg.get("vertical_band", cls.vertical_band)
        if not isinstance(vertical_band_raw, (list, tuple)) or len(vertical_band_raw) != 2:
            raise ValueError(
                "obstacle_safety.vertical_band must contain two fractions, "
                f"got {vertical_band_raw!r}"
            )

        return cls(
            enabled=bool(safety_cfg.get("enabled", cls.enabled)),
            vertical_band=(float(vertical_band_raw[0]), float(vertical_band_raw[1])),
            center_width_frac=float(safety_cfg.get("center_width_frac", cls.center_width_frac)),
            safety_distance=float(safety_cfg.get("safety_distance", cls.safety_distance)),
            hard_stop_distance=float(safety_cfg.get("hard_stop_distance", cls.hard_stop_distance)),
            forward_scale_at_safety=float(
                safety_cfg.get("forward_scale_at_safety", cls.forward_scale_at_safety)
            ),
            sidestep_speed=float(safety_cfg.get("sidestep_speed", cls.sidestep_speed)),
            yaw_away_rate=float(safety_cfg.get("yaw_away_rate", cls.yaw_away_rate)),
        )


def disabled_depth_safety_info() -> dict[str, float]:
    """Return the scalar info structure used by the smoke tests when disabled."""
    return {
        "enabled": 0.0,
        "center_min": float("nan"),
        "left_min": float("nan"),
        "right_min": float("nan"),
        "active": 0.0,
    }


def _depth_to_batched_2d(depth: torch.Tensor) -> torch.Tensor:
    """Normalize supported depth tensor layouts to ``[N, H, W]``.

    Primary expected input is Isaac Lab camera depth with shape ``[N, H, W, 1]``.
    ``[N, H, W]``, ``[H, W, 1]``, and ``[H, W]`` are accepted for convenience.
    """
    if not torch.is_tensor(depth):
        raise TypeError(f"depth must be a torch.Tensor, got {type(depth)!r}")

    if depth.ndim == 4:
        if depth.shape[-1] != 1:
            raise ValueError(f"4D depth must have shape [N, H, W, 1], got {tuple(depth.shape)}")
        return depth[..., 0]

    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            return depth[..., 0].unsqueeze(0)
        return depth

    if depth.ndim == 2:
        return depth.unsqueeze(0)

    raise ValueError(
        "depth must have shape [N, H, W, 1], [N, H, W], [H, W, 1], or [H, W], "
        f"got {tuple(depth.shape)}"
    )


def _validate_command(command: torch.Tensor, num_envs: int) -> None:
    if not torch.is_tensor(command):
        raise TypeError(f"command must be a torch.Tensor, got {type(command)!r}")
    if command.ndim != 2 or command.shape[-1] != 3:
        raise ValueError(f"command must have shape [N, 3], got {tuple(command.shape)}")
    if command.shape[0] != num_envs:
        raise ValueError(f"depth batch size {num_envs} does not match command batch size {command.shape[0]}")


def _safe_crop_indices(height: int, width: int, cfg: DepthSafetyFilterCfg) -> tuple[int, int, int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"depth image must have positive height and width, got {(height, width)}")

    y0_frac, y1_frac = cfg.vertical_band
    y0 = int(height * y0_frac)
    y1 = int(height * y1_frac)
    y0 = max(0, min(height - 1, y0))
    y1 = max(y0 + 1, min(height, y1))

    center_width = int(width * float(cfg.center_width_frac))
    center_width = max(1, min(width, center_width))
    cx0 = max(0, (width - center_width) // 2)
    cx1 = min(width, cx0 + center_width)

    return y0, y1, cx0, cx1


def compute_depth_safety_metrics(
    depth: torch.Tensor,
    cfg: DepthSafetyFilterCfg,
) -> dict[str, torch.Tensor]:
    """Compute batched depth metrics used by the safety filter.

    Returns tensors with shape ``[N]``:

    - ``center_min``: closest depth in the center crop,
    - ``left_min``: closest depth in the left crop,
    - ``right_min``: closest depth in the right crop,
    - ``center_blocked``: ``center_min < safety_distance``,
    - ``hard_blocked``: ``center_min < hard_stop_distance``,
    - ``go_left``: left crop has greater minimum depth than right crop.
    """
    depth_2d = _depth_to_batched_2d(depth)
    _, height, width = depth_2d.shape
    y0, y1, cx0, cx1 = _safe_crop_indices(height, width, cfg)

    band = depth_2d[:, y0:y1, :]

    center_min = band[:, :, cx0:cx1].amin(dim=(1, 2))

    if cx0 > 0:
        left_min = band[:, :, :cx0].amin(dim=(1, 2))
    else:
        left_min = center_min

    if cx1 < width:
        right_min = band[:, :, cx1:].amin(dim=(1, 2))
    else:
        right_min = center_min

    center_blocked = center_min < float(cfg.safety_distance)
    hard_blocked = center_min < float(cfg.hard_stop_distance)
    go_left = left_min > right_min

    return {
        "center_min": center_min,
        "left_min": left_min,
        "right_min": right_min,
        "center_blocked": center_blocked,
        "hard_blocked": hard_blocked,
        "go_left": go_left,
    }


def depth_safety_info_for_env0(metrics: Mapping[str, torch.Tensor], *, enabled: bool) -> dict[str, float]:
    """Convert batched metrics into the scalar info dictionary used by smoke logs."""
    if not enabled:
        return disabled_depth_safety_info()

    center_min = metrics["center_min"]
    left_min = metrics["left_min"]
    right_min = metrics["right_min"]
    center_blocked = metrics["center_blocked"]

    return {
        "enabled": 1.0,
        "center_min": float(center_min[0].detach().cpu()),
        "left_min": float(left_min[0].detach().cpu()),
        "right_min": float(right_min[0].detach().cpu()),
        "active": float(center_blocked[0].detach().cpu()),
    }


def apply_depth_safety_filter(
    depth: torch.Tensor,
    command: torch.Tensor,
    cfg: DepthSafetyFilterCfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Modify ``[vx, vy, wz]`` commands using the front-depth safety rule.

    Args:
        depth: Depth tensor, preferably shape ``[N, H, W, 1]``.
        command: Body-frame command tensor with shape ``[N, 3]``.
        cfg: Safety filter configuration.

    Returns:
        ``(filtered_command, info)``. ``filtered_command`` has shape ``[N, 3]``.
        ``info`` preserves the scalar smoke-test logging format for environment 0.
    """
    depth_2d = _depth_to_batched_2d(depth)
    _validate_command(command, depth_2d.shape[0])

    if not cfg.enabled:
        return command, disabled_depth_safety_info()

    metrics = compute_depth_safety_metrics(depth_2d, cfg)
    filtered = command.clone()

    center_blocked = metrics["center_blocked"]
    hard_blocked = metrics["hard_blocked"]
    go_left = metrics["go_left"]
    moving_forward = filtered[:, 0] > 0.0

    filtered[:, 0] = torch.where(
        center_blocked & moving_forward,
        filtered[:, 0] * float(cfg.forward_scale_at_safety),
        filtered[:, 0],
    )

    filtered[:, 0] = torch.where(
        hard_blocked & moving_forward,
        torch.zeros_like(filtered[:, 0]),
        filtered[:, 0],
    )

    sidestep = torch.where(
        go_left,
        torch.full_like(filtered[:, 1], float(cfg.sidestep_speed)),
        torch.full_like(filtered[:, 1], -float(cfg.sidestep_speed)),
    )
    yaw_bump = torch.where(
        go_left,
        torch.full_like(filtered[:, 2], float(cfg.yaw_away_rate)),
        torch.full_like(filtered[:, 2], -float(cfg.yaw_away_rate)),
    )

    filtered[:, 1] = torch.where(center_blocked, sidestep, filtered[:, 1])
    filtered[:, 2] = torch.where(center_blocked, filtered[:, 2] + yaw_bump, filtered[:, 2])

    return filtered, depth_safety_info_for_env0(metrics, enabled=True)
