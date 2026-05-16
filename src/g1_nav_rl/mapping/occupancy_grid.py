from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class OccupancyGridCfg:
    """Local body-frame occupancy grid config.

    The grid frame uses the navigation body convention: ``+x`` forward and
    ``+y`` left. The tensor layout is ``[N, H, W]`` where rows run from
    ``x_max`` down to ``x_min`` so that forward space appears at the top in a
    saved image.
    """

    x_min: float = -2.0
    x_max: float = 6.0
    y_min: float = -4.0
    y_max: float = 4.0
    resolution: float = 0.05
    occupied_value: float = 1.0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "OccupancyGridCfg":
        if cfg is None:
            return cls()
        return cls(
            x_min=float(cfg.get("x_min", cls.x_min)),
            x_max=float(cfg.get("x_max", cls.x_max)),
            y_min=float(cfg.get("y_min", cls.y_min)),
            y_max=float(cfg.get("y_max", cls.y_max)),
            resolution=float(cfg.get("resolution", cls.resolution)),
            occupied_value=float(cfg.get("occupied_value", cls.occupied_value)),
        )

    @property
    def height(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def width(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


def rasterize_points_to_occupancy_grid(
    points_xy_b: torch.Tensor,
    valid: torch.Tensor | None,
    cfg: OccupancyGridCfg,
) -> torch.Tensor:
    """Rasterize body-frame 2D points into a local occupancy grid.

    Args:
        points_xy_b: Tensor with shape ``[N, B, 2]`` in body frame.
        valid: Optional boolean mask with shape ``[N, B]``.
        cfg: Grid geometry and resolution.

    Returns:
        Float tensor with shape ``[N, H, W]``. Occupied cells contain
        ``cfg.occupied_value`` and free/unknown cells are zero.
    """
    if not torch.is_tensor(points_xy_b):
        raise TypeError(f"points_xy_b must be a torch.Tensor, got {type(points_xy_b)!r}")
    if points_xy_b.ndim != 3 or points_xy_b.shape[-1] != 2:
        raise ValueError(f"points_xy_b must have shape [N, B, 2], got {tuple(points_xy_b.shape)}")
    if cfg.resolution <= 0.0:
        raise ValueError(f"resolution must be positive, got {cfg.resolution}")
    if cfg.x_max <= cfg.x_min:
        raise ValueError("x_max must be greater than x_min")
    if cfg.y_max <= cfg.y_min:
        raise ValueError("y_max must be greater than y_min")

    points_xy_b = points_xy_b.detach().to(dtype=torch.float32).contiguous()
    device = points_xy_b.device
    num_envs = points_xy_b.shape[0]
    height = cfg.height
    width = cfg.width

    if valid is None:
        valid_mask = torch.isfinite(points_xy_b).all(dim=-1)
    else:
        if valid.shape != points_xy_b.shape[:-1]:
            raise ValueError(f"valid must have shape {tuple(points_xy_b.shape[:-1])}, got {tuple(valid.shape)}")
        valid_mask = valid.to(device=device, dtype=torch.bool) & torch.isfinite(points_xy_b).all(dim=-1)

    x = points_xy_b[..., 0]
    y = points_xy_b[..., 1]

    in_bounds = (
        (x >= cfg.x_min)
        & (x < cfg.x_max)
        & (y >= cfg.y_min)
        & (y < cfg.y_max)
    )
    mask = valid_mask & in_bounds

    # Row 0 corresponds to x_max, so forward space is at the top of images.
    rows = torch.floor((cfg.x_max - x) / cfg.resolution).to(dtype=torch.long)
    cols = torch.floor((y - cfg.y_min) / cfg.resolution).to(dtype=torch.long)
    rows = rows.clamp(0, height - 1)
    cols = cols.clamp(0, width - 1)

    grid = torch.zeros((num_envs, height, width), dtype=torch.float32, device=device)
    env_ids = torch.arange(num_envs, device=device).unsqueeze(1).expand_as(rows)
    grid[env_ids[mask], rows[mask], cols[mask]] = float(cfg.occupied_value)
    return grid.contiguous()


def occupancy_grid_to_uint8_image(grid: torch.Tensor, *, env_id: int = 0) -> torch.Tensor:
    """Convert one occupancy grid slice to a uint8 grayscale image tensor."""
    if grid.ndim != 3:
        raise ValueError(f"grid must have shape [N, H, W], got {tuple(grid.shape)}")
    if not 0 <= env_id < grid.shape[0]:
        raise ValueError(f"env_id {env_id} out of bounds for grid batch {grid.shape[0]}")

    image = grid[env_id].detach().cpu().clamp(0.0, 1.0)
    return (image * 255.0).to(dtype=torch.uint8)
