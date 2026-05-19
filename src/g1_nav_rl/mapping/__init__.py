"""Mapping utilities.

Keep package imports lightweight. Heavy adapters such as ``slam_cloud`` should be
imported from their concrete module to avoid circular imports with navigation
modules.
"""

from .occupancy_grid import (
    OccupancyGridCfg,
    occupancy_grid_to_uint8_image,
    rasterize_points_to_occupancy_grid,
)

__all__ = [
    "OccupancyGridCfg",
    "occupancy_grid_to_uint8_image",
    "rasterize_points_to_occupancy_grid",
]
