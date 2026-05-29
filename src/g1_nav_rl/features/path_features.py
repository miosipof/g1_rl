"""Fixed-size features extracted from A* path candidates and occupancy grids."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.simple_nav import PointXY, Pose2D, world_to_body_xy


def _as_tensor_path(path_xy_b: Sequence[PointXY], *, device: torch.device | str = "cpu") -> torch.Tensor:
    if not path_xy_b:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    return torch.tensor([(float(x), float(y)) for x, y in path_xy_b], dtype=torch.float32, device=device)


def sample_polyline_by_distance(
    path_xy: torch.Tensor,
    *,
    num_points: int,
    start_distance: float = 0.0,
    spacing: float | None = None,
) -> torch.Tensor:
    """Sample a 2D polyline into exactly ``num_points`` points.

    If the polyline is shorter than the requested distance range, the final
    point is repeated. Empty paths return zeros. This deterministic padding is
    preferred over variable-length path tensors for policy inputs.
    """

    if path_xy.ndim != 2 or path_xy.shape[-1] != 2:
        raise ValueError(f"path_xy must have shape [M, 2], got {tuple(path_xy.shape)}")
    if num_points < 0:
        raise ValueError("num_points must be non-negative")
    if num_points == 0:
        return torch.empty((0, 2), dtype=path_xy.dtype, device=path_xy.device)
    if path_xy.shape[0] == 0:
        return torch.zeros((num_points, 2), dtype=path_xy.dtype, device=path_xy.device)
    if path_xy.shape[0] == 1:
        return path_xy[0].repeat(num_points, 1)

    deltas = path_xy[1:] - path_xy[:-1]
    seg_lengths = torch.linalg.norm(deltas, dim=-1)
    total_length = float(seg_lengths.sum().item())
    if total_length <= 1.0e-9:
        return path_xy[-1].repeat(num_points, 1)

    if spacing is None:
        spacing_value = total_length / max(1, num_points - 1)
    else:
        spacing_value = max(1.0e-6, float(spacing))
    distances = torch.arange(num_points, dtype=path_xy.dtype, device=path_xy.device) * float(spacing_value)
    distances = distances + float(start_distance)
    distances = torch.clamp(distances, max=total_length)

    cum = torch.cat((torch.zeros((1,), dtype=path_xy.dtype, device=path_xy.device), torch.cumsum(seg_lengths, dim=0)))
    out = torch.empty((num_points, 2), dtype=path_xy.dtype, device=path_xy.device)
    for i, d in enumerate(distances):
        idx = int(torch.searchsorted(cum, d, right=True).item()) - 1
        idx = max(0, min(idx, path_xy.shape[0] - 2))
        seg_len = max(float(seg_lengths[idx].item()), 1.0e-9)
        t = float((d - cum[idx]).item()) / seg_len
        out[i] = path_xy[idx] + float(t) * (path_xy[idx + 1] - path_xy[idx])
    return out.contiguous()


def fixed_body_path_points(
    path_xy_b: Sequence[PointXY] | torch.Tensor,
    *,
    num_points: int = 8,
    spacing_m: float = 0.25,
    start_distance_m: float = 0.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert a body-frame path into ``[num_points, 2]`` fixed samples."""

    if torch.is_tensor(path_xy_b):
        path = path_xy_b.to(device=device, dtype=torch.float32)
    else:
        path = _as_tensor_path(path_xy_b, device=device)
    return sample_polyline_by_distance(
        path,
        num_points=int(num_points),
        start_distance=float(start_distance_m),
        spacing=float(spacing_m),
    )


def path_points_from_world_path(
    path_xy_w: Sequence[PointXY],
    pose: Pose2D,
    *,
    num_points: int = 8,
    spacing_m: float = 0.25,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Transform a world-frame path to body frame and sample fixed points."""

    path_b = [world_to_body_xy(xy, pose) for xy in path_xy_w]
    return fixed_body_path_points(path_b, num_points=num_points, spacing_m=spacing_m, device=device)


def path_length_tensor(path_xy: torch.Tensor) -> torch.Tensor:
    """Return path length for ``[M,2]`` or batched ``[N,M,2]`` tensors."""

    if path_xy.ndim == 2:
        if path_xy.shape[0] <= 1:
            return torch.zeros((), dtype=path_xy.dtype, device=path_xy.device)
        return torch.linalg.norm(path_xy[1:] - path_xy[:-1], dim=-1).sum()
    if path_xy.ndim == 3:
        if path_xy.shape[1] <= 1:
            return torch.zeros((path_xy.shape[0],), dtype=path_xy.dtype, device=path_xy.device)
        return torch.linalg.norm(path_xy[:, 1:] - path_xy[:, :-1], dim=-1).sum(dim=-1)
    raise ValueError(f"path_xy must have shape [M,2] or [N,M,2], got {tuple(path_xy.shape)}")


def path_curvature_features(path_xy_b: torch.Tensor) -> torch.Tensor:
    """Return simple curvature features from fixed body-frame path samples.

    Output shape is ``[N, 3]``:
    ``mean_abs_turn``, ``max_abs_turn``, ``first_turn`` in radians.
    """

    if path_xy_b.ndim != 3 or path_xy_b.shape[-1] != 2:
        raise ValueError(f"path_xy_b must have shape [N, K, 2], got {tuple(path_xy_b.shape)}")
    n = path_xy_b.shape[0]
    if path_xy_b.shape[1] < 3:
        return torch.zeros((n, 3), dtype=path_xy_b.dtype, device=path_xy_b.device)

    seg = path_xy_b[:, 1:] - path_xy_b[:, :-1]
    heading = torch.atan2(seg[..., 1], seg[..., 0])
    turns = (heading[:, 1:] - heading[:, :-1] + math.pi) % (2.0 * math.pi) - math.pi
    abs_turns = torch.abs(turns)
    return torch.stack(
        (
            abs_turns.mean(dim=-1),
            abs_turns.amax(dim=-1),
            turns[:, 0],
        ),
        dim=-1,
    ).contiguous()


def clearance_sectors_from_grid(
    occupancy_grid: torch.Tensor,
    grid_cfg: OccupancyGridCfg,
    *,
    occupancy_threshold: float = 0.5,
    front_half_width_m: float = 0.45,
    side_min_x_m: float = -0.25,
    side_max_x_m: float = 2.0,
    max_clearance_m: float = 6.0,
) -> torch.Tensor:
    """Return front/left/right minimum obstacle distances from a local grid.

    The output shape is ``[N, 3]``. Values are capped at ``max_clearance_m``.
    Empty sectors receive ``max_clearance_m``. This is deliberately coarse: it is
    meant as a compact neural feature, not a hard safety rule.
    """

    if occupancy_grid.ndim != 3:
        raise ValueError(f"occupancy_grid must have shape [N,H,W], got {tuple(occupancy_grid.shape)}")
    if occupancy_grid.shape[1] != grid_cfg.height or occupancy_grid.shape[2] != grid_cfg.width:
        raise ValueError("occupancy_grid shape does not match grid_cfg")

    device = occupancy_grid.device
    dtype = torch.float32
    rows = torch.arange(grid_cfg.height, device=device, dtype=dtype)
    cols = torch.arange(grid_cfg.width, device=device, dtype=dtype)
    x = float(grid_cfg.x_max) - (rows + 0.5) * float(grid_cfg.resolution)
    y = float(grid_cfg.y_min) + (cols + 0.5) * float(grid_cfg.resolution)
    xx = x[:, None].expand(grid_cfg.height, grid_cfg.width)
    yy = y[None, :].expand(grid_cfg.height, grid_cfg.width)
    dist = torch.linalg.norm(torch.stack((xx, yy), dim=-1), dim=-1)

    occ = occupancy_grid > float(occupancy_threshold)
    front_mask = (xx >= 0.0) & (torch.abs(yy) <= float(front_half_width_m))
    left_mask = (xx >= float(side_min_x_m)) & (xx <= float(side_max_x_m)) & (yy > float(front_half_width_m))
    right_mask = (xx >= float(side_min_x_m)) & (xx <= float(side_max_x_m)) & (yy < -float(front_half_width_m))

    out = []
    for mask in (front_mask, left_mask, right_mask):
        d = torch.where(occ & mask[None, :, :], dist[None, :, :], torch.full_like(occupancy_grid, float(max_clearance_m)))
        out.append(d.flatten(start_dim=1).amin(dim=1))
    return torch.stack(out, dim=-1).to(dtype=torch.float32).contiguous()


__all__ = [
    "clearance_sectors_from_grid",
    "fixed_body_path_points",
    "path_curvature_features",
    "path_length_tensor",
    "path_points_from_world_path",
    "sample_polyline_by_distance",
]
