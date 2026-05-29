from __future__ import annotations

import torch

from g1_nav_rl.features.path_features import (
    clearance_sectors_from_grid,
    fixed_body_path_points,
    path_curvature_features,
    sample_polyline_by_distance,
)
from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg


def test_sample_polyline_by_distance_pads_short_path() -> None:
    path = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    sampled = sample_polyline_by_distance(path, num_points=5, spacing=0.5)
    assert sampled.shape == (5, 2)
    assert torch.allclose(sampled[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(sampled[2], torch.tensor([1.0, 0.0]))
    assert torch.allclose(sampled[-1], torch.tensor([1.0, 0.0]))


def test_fixed_body_path_points_empty_returns_zeros() -> None:
    pts = fixed_body_path_points([], num_points=4)
    assert pts.shape == (4, 2)
    assert torch.count_nonzero(pts) == 0


def test_path_curvature_features_shape() -> None:
    path = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]], dtype=torch.float32)
    curv = path_curvature_features(path)
    assert curv.shape == (1, 3)
    assert curv[0, 1] > 1.0


def test_clearance_sectors_from_grid_detects_front_obstacle() -> None:
    cfg = OccupancyGridCfg(x_min=-1.0, x_max=2.0, y_min=-1.0, y_max=1.0, resolution=0.5)
    grid = torch.zeros((1, cfg.height, cfg.width), dtype=torch.float32)
    # x about 1.25, y about 0.25 -> front sector.
    row = int((cfg.x_max - 1.25) / cfg.resolution)
    col = int((0.25 - cfg.y_min) / cfg.resolution)
    grid[0, row, col] = 1.0
    sectors = clearance_sectors_from_grid(grid, cfg, max_clearance_m=5.0)
    assert sectors.shape == (1, 3)
    assert sectors[0, 0] < 2.0
    assert sectors[0, 1] == 5.0
