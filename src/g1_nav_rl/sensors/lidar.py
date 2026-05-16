from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from g1_nav_rl.utils.geometry import rotate_world_to_body_2d


@dataclass(frozen=True)
class LidarObservationCfg:
    """Post-processing config for proxy-mounted ray-cast lidar observations."""

    min_range: float = 0.05
    max_range: float = 8.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "LidarObservationCfg":
        if cfg is None:
            return cls()
        return cls(
            min_range=float(cfg.get("min_range", cls.min_range)),
            max_range=float(cfg.get("max_range", cls.max_range)),
        )


def _to_float_tensor(x: torch.Tensor) -> torch.Tensor:
    """Detach and convert a tensor to contiguous float32."""
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
    return x.detach().to(dtype=torch.float32).contiguous()


def ray_hits_to_ranges(
    ray_hits_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    *,
    min_range: float = 0.05,
    max_range: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert RayCaster world-frame hit points to sanitized ranges.

    Args:
        ray_hits_w: World-frame ray hit points with shape ``[N, B, 3]``.
        sensor_pos_w: World-frame sensor origins with shape ``[N, 3]``.
        min_range: Minimum accepted range in meters.
        max_range: Maximum range returned for invalid/no-hit rays.

    Returns:
        ``(ranges, valid)``:

        - ``ranges`` has shape ``[N, B]`` and contains finite values. Invalid rays
          are filled with ``max_range``.
        - ``valid`` has shape ``[N, B]`` and marks finite hits inside the accepted
          range interval.
    """
    ray_hits_w = _to_float_tensor(ray_hits_w)
    sensor_pos_w = _to_float_tensor(sensor_pos_w)

    if ray_hits_w.ndim != 3 or ray_hits_w.shape[-1] != 3:
        raise ValueError(f"ray_hits_w must have shape [N, B, 3], got {tuple(ray_hits_w.shape)}")
    if sensor_pos_w.ndim != 2 or sensor_pos_w.shape[-1] != 3:
        raise ValueError(f"sensor_pos_w must have shape [N, 3], got {tuple(sensor_pos_w.shape)}")
    if ray_hits_w.shape[0] != sensor_pos_w.shape[0]:
        raise ValueError(
            f"Batch mismatch: ray_hits_w has N={ray_hits_w.shape[0]}, "
            f"sensor_pos_w has N={sensor_pos_w.shape[0]}"
        )

    raw_ranges = torch.linalg.norm(ray_hits_w - sensor_pos_w.unsqueeze(1), dim=-1)
    finite_hits = torch.isfinite(ray_hits_w).all(dim=-1) & torch.isfinite(raw_ranges)
    valid = finite_hits & (raw_ranges >= float(min_range)) & (raw_ranges <= float(max_range))
    ranges = torch.where(valid, raw_ranges, torch.full_like(raw_ranges, float(max_range)))
    return ranges.contiguous(), valid.contiguous()


def ray_hits_to_proxy_points_2d(
    ray_hits_w: torch.Tensor,
    proxy_pos_w: torch.Tensor,
    proxy_yaw: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project world-frame hit points into proxy/body-frame 2D coordinates.

    The returned frame is the navigation body frame: ``+x`` forward, ``+y`` left.
    Invalid points, when a mask is supplied, are set to zero.
    """
    ray_hits_w = _to_float_tensor(ray_hits_w)
    proxy_pos_w = _to_float_tensor(proxy_pos_w)
    proxy_yaw = _to_float_tensor(proxy_yaw)

    if ray_hits_w.ndim != 3 or ray_hits_w.shape[-1] != 3:
        raise ValueError(f"ray_hits_w must have shape [N, B, 3], got {tuple(ray_hits_w.shape)}")
    if proxy_pos_w.ndim != 2 or proxy_pos_w.shape[-1] != 3:
        raise ValueError(f"proxy_pos_w must have shape [N, 3], got {tuple(proxy_pos_w.shape)}")
    if proxy_yaw.ndim != 1:
        raise ValueError(f"proxy_yaw must have shape [N], got {tuple(proxy_yaw.shape)}")

    delta_xy_w = ray_hits_w[..., 0:2] - proxy_pos_w[:, None, 0:2]
    yaw = proxy_yaw[:, None].expand(delta_xy_w.shape[:-1])
    points_xy_b = rotate_world_to_body_2d(delta_xy_w, yaw)

    if valid is not None:
        if valid.shape != points_xy_b.shape[:-1]:
            raise ValueError(f"valid mask must have shape {tuple(points_xy_b.shape[:-1])}, got {tuple(valid.shape)}")
        points_xy_b = torch.where(valid.unsqueeze(-1), points_xy_b, torch.zeros_like(points_xy_b))

    return points_xy_b.contiguous()


def build_lidar_proxy_observation(
    scene: Any,
    proxy: Any,
    *,
    lidar_name: str = "lidar",
    cfg: LidarObservationCfg | None = None,
    min_range: float | None = None,
    max_range: float | None = None,
) -> dict[str, torch.Tensor]:
    """Build proxy-mounted lidar observations from an Isaac Lab RayCaster.

    This function intentionally avoids importing Isaac/Omniverse modules. It only
    assumes that ``scene[lidar_name].data`` exposes ``ray_hits_w`` and ``pos_w``.

    Returned keys:
        - ``lidar_ranges``: ``[N, B]`` finite range values; no-hit rays are max range.
        - ``lidar_points_b``: ``[N, B, 2]`` hit points in proxy/body frame.
        - ``lidar_valid``: ``[N, B]`` boolean valid-hit mask.
        - ``lidar_sensor_pos_w``: ``[N, 3]`` world-frame lidar origin.
        - ``lidar_ray_hits_w``: ``[N, B, 3]`` raw world-frame hit points.
    """
    if cfg is None:
        cfg = LidarObservationCfg(
            min_range=0.05 if min_range is None else float(min_range),
            max_range=8.0 if max_range is None else float(max_range),
        )
    else:
        if min_range is not None or max_range is not None:
            raise ValueError("Pass either cfg or min_range/max_range, not both")

    lidar = scene[lidar_name]
    ray_hits_w = _to_float_tensor(lidar.data.ray_hits_w)
    sensor_pos_w = _to_float_tensor(lidar.data.pos_w)

    if not hasattr(proxy, "pos_w") or not hasattr(proxy, "yaw"):
        raise TypeError("proxy must expose pos_w and yaw tensors")

    ranges, valid = ray_hits_to_ranges(
        ray_hits_w,
        sensor_pos_w,
        min_range=cfg.min_range,
        max_range=cfg.max_range,
    )
    points_xy_b = ray_hits_to_proxy_points_2d(ray_hits_w, proxy.pos_w, proxy.yaw, valid=valid)

    return {
        "lidar_ranges": ranges,
        "lidar_points_b": points_xy_b,
        "lidar_valid": valid,
        "lidar_sensor_pos_w": sensor_pos_w,
        "lidar_ray_hits_w": ray_hits_w,
    }


def summarize_lidar_observation(obs: dict[str, torch.Tensor]) -> str:
    """Return a compact summary for smoke-test logging."""
    ranges = obs["lidar_ranges"]
    points = obs["lidar_points_b"]
    valid = obs["lidar_valid"]
    valid_count = int(valid[0].detach().cpu().sum().item()) if valid.numel() else 0
    min_range = float(ranges[0].detach().cpu().min().item()) if ranges.numel() else float("nan")
    return (
        f"ranges={tuple(ranges.shape)} {ranges.dtype}, "
        f"points_b={tuple(points.shape)} {points.dtype}, "
        f"valid_env0={valid_count}, min_range_env0={min_range:.3f}"
    )
