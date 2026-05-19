"""Adapter from Unitree/SLAM point-cloud snapshots to planner occupancy grids.

The real-robot snapshot format we observed is a JSON metadata object containing
paths like::

    {
      "files": {
        "slam_map_npy_xyz_m": "snapshots/slamtest_slam_map.npy",
        "slam_map_ply_xyz_m": "snapshots/slamtest_slam_map.ply",
        "slam_pose_4x4_npy": "snapshots/slamtest_slam_pose.npy"
      },
      "env": {
        "LIDAR_SELF_FILTER_RADIUS": "0.3",
        "LIDAR_SELF_FILTER_Z": "0.24"
      }
    }

This module converts that map evidence into the same local body-frame occupancy
``torch.Tensor`` expected by ``SimpleNavigator`` / ``NavRuntime``:

    shape [1, H, W], dtype float32, occupied cells = 1.0, unknown/free = 0.0

No SLAM, no planning, and no command transport is implemented here. This is only
an adapter from saved/received SLAM data to the planner grid format.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg
from g1_nav_rl.navigation.simple_nav import Pose2D

PointXY = tuple[float, float]


@dataclass(frozen=True)
class SlamSnapshotFiles:
    """Resolved file paths for one SLAM snapshot."""

    slam_map_npy_xyz_m: Path
    slam_pose_4x4_npy: Path
    slam_map_ply_xyz_m: Path | None = None


@dataclass(frozen=True)
class SlamCloudGridConfig:
    """Configuration for converting a 3D SLAM cloud into a 2D local grid.

    The resulting grid is local/body-frame: ``+x`` is robot-forward and ``+y``
    is robot-left, matching :class:`OccupancyGridCfg` and the A* planner.

    Args:
        grid_cfg: Local body-frame grid geometry.
        z_min/z_max: Collision-relevant height band. By default this filter is
            applied after transforming the cloud into body frame.
        height_filter_frame: ``"body"`` or ``"map"``. Use ``"body"`` when the
            pose transform is reliable. Use ``"map"`` if the SLAM frame is known
            to be gravity-aligned and easier to tune.
        min_points_per_cell: Require this many points before a cell is occupied.
        self_filter_radius: Remove points close to the robot XY origin in body
            frame. If ``None``, try to read ``LIDAR_SELF_FILTER_RADIUS`` from the
            JSON ``env`` field.
        self_filter_z: Optional vertical threshold used with self filtering. If
            set, points inside ``self_filter_radius`` and with ``z <= self_filter_z``
            are removed. If ``None``, any point inside the radius is removed.
        pose_convention: Convention for the 4x4 pose matrix:
            - ``"map_from_body"``: matrix maps body-frame points into map frame;
              this is the default and common robotics convention.
            - ``"body_from_map"``: matrix maps map-frame points into body frame.
    """

    grid_cfg: OccupancyGridCfg = OccupancyGridCfg()
    z_min: float = -0.80
    z_max: float = 1.20
    height_filter_frame: str = "body"
    min_points_per_cell: int = 1
    self_filter_radius: float | None = None
    self_filter_z: float | None = None
    pose_convention: str = "map_from_body"
    occupied_value: float = 1.0


@dataclass(frozen=True)
class SlamOccupancyResult:
    """Output from converting a SLAM snapshot into planner-ready occupancy."""

    occupancy_grid: torch.Tensor
    grid_cfg: OccupancyGridCfg
    pose: Pose2D
    points_map: np.ndarray
    points_body: np.ndarray
    selected_points_body: np.ndarray
    selected_points_map: np.ndarray
    counts: np.ndarray
    files: SlamSnapshotFiles | None
    stats: dict[str, Any]

    @property
    def planner_grid(self) -> torch.Tensor:
        """2D ``[H, W]`` grid accepted directly by ``SimpleNavigator``/``NavRuntime``."""
        return self.occupancy_grid[0]

    @property
    def occupied_cells(self) -> int:
        return int((self.planner_grid > 0.0).sum().item())


class SlamSnapshotFormatError(ValueError):
    """Raised when the snapshot JSON/dict does not contain required fields."""


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_slam_info_json(path: str | Path) -> dict[str, Any]:
    """Load the JSON metadata object shown in the robot snapshot example."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SlamSnapshotFormatError(f"SLAM info JSON must contain an object, got {type(data)!r}")
    return data


def resolve_slam_snapshot_files(info: Mapping[str, Any], *, root: str | Path | None = None) -> SlamSnapshotFiles:
    """Resolve snapshot file paths from the JSON/dict ``files`` section."""
    files = info.get("files")
    if not isinstance(files, Mapping):
        raise SlamSnapshotFormatError("SLAM info must contain a 'files' object")

    map_rel = files.get("slam_map_npy_xyz_m")
    pose_rel = files.get("slam_pose_4x4_npy")
    ply_rel = files.get("slam_map_ply_xyz_m")
    if not map_rel:
        raise SlamSnapshotFormatError("SLAM info files missing 'slam_map_npy_xyz_m'")
    if not pose_rel:
        raise SlamSnapshotFormatError("SLAM info files missing 'slam_pose_4x4_npy'")

    base = Path(root) if root is not None else Path.cwd()

    def resolve(value: Any) -> Path:
        p = Path(str(value))
        return p if p.is_absolute() else base / p

    return SlamSnapshotFiles(
        slam_map_npy_xyz_m=resolve(map_rel),
        slam_pose_4x4_npy=resolve(pose_rel),
        slam_map_ply_xyz_m=resolve(ply_rel) if ply_rel else None,
    )


def resolve_slam_snapshot_files_from_json(path: str | Path) -> tuple[dict[str, Any], SlamSnapshotFiles]:
    """Load metadata JSON and resolve files relative to the JSON directory."""
    p = Path(path)
    info = load_slam_info_json(p)
    return info, resolve_slam_snapshot_files(info, root=p.parent)


def load_slam_arrays(files: SlamSnapshotFiles) -> tuple[np.ndarray, np.ndarray]:
    """Load ``slam_map.npy`` and ``slam_pose.npy`` arrays."""
    if not files.slam_map_npy_xyz_m.exists():
        raise FileNotFoundError(f"SLAM map not found: {files.slam_map_npy_xyz_m}")
    if not files.slam_pose_4x4_npy.exists():
        raise FileNotFoundError(f"SLAM pose not found: {files.slam_pose_4x4_npy}")
    points = np.load(files.slam_map_npy_xyz_m)
    pose = np.load(files.slam_pose_4x4_npy)
    return validate_points_xyz(points), validate_pose_4x4(pose)


def validate_points_xyz(points: Any) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape [N, 3], got {arr.shape}")
    return arr


def validate_pose_4x4(pose: Any) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError(f"pose_4x4 must have shape [4, 4], got {arr.shape}")
    return arr


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid-ish 4x4 transform."""
    T = validate_pose_4x4(T)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def transform_points(points_xyz: np.ndarray, T_to_from: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to XYZ points."""
    points_xyz = validate_points_xyz(points_xyz)
    T_to_from = validate_pose_4x4(T_to_from)
    if points_xyz.shape[0] == 0:
        return points_xyz.copy()
    R = T_to_from[:3, :3]
    t = T_to_from[:3, 3]
    return (points_xyz @ R.T + t[None, :]).astype(np.float32, copy=False)


def map_points_to_body(points_map: np.ndarray, pose_4x4: np.ndarray, *, pose_convention: str = "map_from_body") -> np.ndarray:
    """Transform map-frame cloud points into the robot/body frame."""
    if pose_convention == "map_from_body":
        T_body_map = invert_transform(pose_4x4)
    elif pose_convention == "body_from_map":
        T_body_map = validate_pose_4x4(pose_4x4)
    else:
        raise ValueError("pose_convention must be 'map_from_body' or 'body_from_map'")
    return transform_points(points_map, T_body_map)


def pose4x4_to_pose2d(pose_4x4: np.ndarray, *, pose_convention: str = "map_from_body") -> Pose2D:
    """Convert a 4x4 pose into world/map-frame planar pose for NavRuntime."""
    T = validate_pose_4x4(pose_4x4)
    if pose_convention == "body_from_map":
        T = invert_transform(T)
    elif pose_convention != "map_from_body":
        raise ValueError("pose_convention must be 'map_from_body' or 'body_from_map'")
    yaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    return Pose2D(float(T[0, 3]), float(T[1, 3]), float(yaw))


def _env_self_filter(info: Mapping[str, Any]) -> tuple[float | None, float | None]:
    env = info.get("env")
    if not isinstance(env, Mapping):
        return None, None
    return _as_float_or_none(env.get("LIDAR_SELF_FILTER_RADIUS")), _as_float_or_none(env.get("LIDAR_SELF_FILTER_Z"))


def _effective_config(config: SlamCloudGridConfig, info: Mapping[str, Any] | None) -> SlamCloudGridConfig:
    if info is None:
        return config
    env_radius, env_z = _env_self_filter(info)
    radius = config.self_filter_radius if config.self_filter_radius is not None else env_radius
    z = config.self_filter_z if config.self_filter_z is not None else env_z
    return SlamCloudGridConfig(
        grid_cfg=config.grid_cfg,
        z_min=config.z_min,
        z_max=config.z_max,
        height_filter_frame=config.height_filter_frame,
        min_points_per_cell=config.min_points_per_cell,
        self_filter_radius=radius,
        self_filter_z=z,
        pose_convention=config.pose_convention,
        occupied_value=config.occupied_value,
    )


def filter_slam_points(
    points_map: np.ndarray,
    points_body: np.ndarray,
    *,
    cfg: SlamCloudGridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply finite, height, self, and local-grid crop filters.

    Returns ``(mask, selected_map, selected_body)``.
    """
    points_map = validate_points_xyz(points_map)
    points_body = validate_points_xyz(points_body)
    if points_map.shape != points_body.shape:
        raise ValueError("points_map and points_body must have the same shape")

    finite = np.isfinite(points_map).all(axis=1) & np.isfinite(points_body).all(axis=1)

    if cfg.height_filter_frame == "body":
        z_values = points_body[:, 2]
    elif cfg.height_filter_frame == "map":
        z_values = points_map[:, 2]
    else:
        raise ValueError("height_filter_frame must be 'body' or 'map'")

    mask = finite & (z_values >= float(cfg.z_min)) & (z_values <= float(cfg.z_max))

    if cfg.self_filter_radius is not None and float(cfg.self_filter_radius) > 0.0:
        r_xy = np.linalg.norm(points_body[:, :2], axis=1)
        inside = r_xy <= float(cfg.self_filter_radius)
        if cfg.self_filter_z is not None:
            inside = inside & (points_body[:, 2] <= float(cfg.self_filter_z))
        mask = mask & ~inside

    g = cfg.grid_cfg
    x = points_body[:, 0]
    y = points_body[:, 1]
    in_grid = (x >= g.x_min) & (x < g.x_max) & (y >= g.y_min) & (y < g.y_max)
    mask = mask & in_grid
    return mask, points_map[mask].copy(), points_body[mask].copy()


def body_points_to_occupancy_grid(
    points_body_xyz: np.ndarray,
    *,
    grid_cfg: OccupancyGridCfg,
    min_points_per_cell: int = 1,
    occupied_value: float = 1.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Rasterize body-frame XYZ points into the planner occupancy tensor."""
    points = validate_points_xyz(points_body_xyz)
    H, W = grid_cfg.height, grid_cfg.width
    counts = np.zeros((H, W), dtype=np.int32)
    if points.shape[0] > 0:
        x = points[:, 0]
        y = points[:, 1]
        rows = np.floor((grid_cfg.x_max - x) / grid_cfg.resolution).astype(np.int64)
        cols = np.floor((y - grid_cfg.y_min) / grid_cfg.resolution).astype(np.int64)
        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        np.add.at(counts, (rows[valid], cols[valid]), 1)
    occupied = counts >= max(1, int(min_points_per_cell))
    grid = np.zeros((1, H, W), dtype=np.float32)
    grid[0, occupied] = float(occupied_value)
    return torch.as_tensor(grid, dtype=torch.float32, device=device or "cpu"), counts


def slam_cloud_to_occupancy_grid(
    points_xyz_m: np.ndarray,
    pose_4x4: np.ndarray,
    *,
    config: SlamCloudGridConfig | None = None,
    info: Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
    files: SlamSnapshotFiles | None = None,
) -> SlamOccupancyResult:
    """Convert loaded SLAM arrays into a local body-frame occupancy grid."""
    raw_cfg = config or SlamCloudGridConfig()
    cfg = _effective_config(raw_cfg, info)
    points_map = validate_points_xyz(points_xyz_m)
    pose_4x4 = validate_pose_4x4(pose_4x4)
    points_body = map_points_to_body(points_map, pose_4x4, pose_convention=cfg.pose_convention)
    mask, selected_map, selected_body = filter_slam_points(points_map, points_body, cfg=cfg)
    grid, counts = body_points_to_occupancy_grid(
        selected_body,
        grid_cfg=cfg.grid_cfg,
        min_points_per_cell=cfg.min_points_per_cell,
        occupied_value=cfg.occupied_value,
        device=device,
    )
    finite = np.isfinite(points_map).all(axis=1)
    stats = {
        "points_total": int(points_map.shape[0]),
        "points_finite": int(finite.sum()),
        "points_selected": int(selected_body.shape[0]),
        "occupied_cells": int((counts >= max(1, int(cfg.min_points_per_cell))).sum()),
        "z_min": float(cfg.z_min),
        "z_max": float(cfg.z_max),
        "height_filter_frame": str(cfg.height_filter_frame),
        "self_filter_radius": None if cfg.self_filter_radius is None else float(cfg.self_filter_radius),
        "self_filter_z": None if cfg.self_filter_z is None else float(cfg.self_filter_z),
        "pose_convention": str(cfg.pose_convention),
        "grid": {
            "x_min": float(cfg.grid_cfg.x_min),
            "x_max": float(cfg.grid_cfg.x_max),
            "y_min": float(cfg.grid_cfg.y_min),
            "y_max": float(cfg.grid_cfg.y_max),
            "resolution": float(cfg.grid_cfg.resolution),
            "height": int(cfg.grid_cfg.height),
            "width": int(cfg.grid_cfg.width),
        },
    }
    if selected_body.shape[0] > 0:
        stats["selected_body_min_xyz_m"] = selected_body.min(axis=0).astype(float).tolist()
        stats["selected_body_max_xyz_m"] = selected_body.max(axis=0).astype(float).tolist()
    else:
        stats["selected_body_min_xyz_m"] = None
        stats["selected_body_max_xyz_m"] = None

    return SlamOccupancyResult(
        occupancy_grid=grid,
        grid_cfg=cfg.grid_cfg,
        pose=pose4x4_to_pose2d(pose_4x4, pose_convention=cfg.pose_convention),
        points_map=points_map,
        points_body=points_body,
        selected_points_body=selected_body,
        selected_points_map=selected_map,
        counts=counts,
        files=files,
        stats=stats,
    )


def slam_info_to_occupancy_grid(
    info: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    config: SlamCloudGridConfig | None = None,
    device: torch.device | str | None = None,
) -> SlamOccupancyResult:
    """Load snapshot arrays from an info dict and return planner occupancy."""
    files = resolve_slam_snapshot_files(info, root=root)
    points, pose = load_slam_arrays(files)
    return slam_cloud_to_occupancy_grid(points, pose, config=config, info=info, device=device, files=files)


def slam_info_json_to_occupancy_grid(
    path: str | Path,
    *,
    config: SlamCloudGridConfig | None = None,
    device: torch.device | str | None = None,
) -> SlamOccupancyResult:
    """Load the snapshot JSON format directly and return planner occupancy."""
    info, files = resolve_slam_snapshot_files_from_json(path)
    points, pose = load_slam_arrays(files)
    return slam_cloud_to_occupancy_grid(points, pose, config=config, info=info, device=device, files=files)


def save_occupancy_debug_png(
    occupancy_grid: torch.Tensor,
    path: str | Path,
    *,
    pose: Pose2D | None = None,
    target_xy_b: PointXY | None = None,
) -> None:
    """Save a small grayscale debug PNG for an occupancy grid.

    This helper is intentionally dependency-light. Occupied cells are dark,
    unknown/free cells are white. The robot origin is marked if possible.
    """
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - depends on optional pillow
        raise RuntimeError("Pillow is required to save occupancy debug PNGs") from exc

    if occupancy_grid.ndim == 3:
        grid = occupancy_grid[0]
    elif occupancy_grid.ndim == 2:
        grid = occupancy_grid
    else:
        raise ValueError(f"occupancy_grid must have shape [1,H,W] or [H,W], got {tuple(occupancy_grid.shape)}")

    occ = (grid.detach().cpu().numpy() > 0.0)
    img = np.full(occ.shape, 255, dtype=np.uint8)
    img[occ] = 30
    rgb = np.stack([img, img, img], axis=-1)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    # Without grid_cfg we cannot map coordinates here; this function is only a
    # quick image dump. Coordinate overlays are handled by scripts.
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
