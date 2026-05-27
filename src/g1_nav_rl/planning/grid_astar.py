"""Grid-based A* path planning for local occupancy maps.

This module is adapted for the ``g1_nav_rl`` occupancy-grid convention from
PythonRobotics' A* grid-planning example:

    https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py

PythonRobotics is MIT licensed. The implementation below keeps the same core
idea—8-connected grid A* with a Euclidean heuristic—but is rewritten around our
local body-frame occupancy-grid representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Mapping, Sequence

import torch

from g1_nav_rl.mapping.occupancy_grid import OccupancyGridCfg


Cell = tuple[int, int]
Point2D = tuple[float, float]


@dataclass(frozen=True)
class GridAStarCfg:
    """Configuration for grid A* planning.

    Args:
        occupancy_threshold: Grid values greater than or equal to this value are
            treated as occupied.
        inflation_radius: Additional obstacle inflation radius in meters. This
            should include the proxy/body radius plus safety margin if the input
            grid is only point occupancy.
        allow_diagonal: If true, use 8-connected motion. If false, use 4-connected
            motion.
        heuristic_weight: Weight on Euclidean heuristic. ``1.0`` is standard A*.
            Values above 1 can be faster but are no longer guaranteed optimal.
        max_expansions: Optional cap on node expansions; ``0`` means unlimited.
    """

    occupancy_threshold: float = 0.5
    inflation_radius: float = 0.35
    allow_diagonal: bool = True
    heuristic_weight: float = 1.0
    max_expansions: int = 0

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "GridAStarCfg":
        if cfg is None:
            return cls()
        return cls(
            occupancy_threshold=float(cfg.get("occupancy_threshold", cls.occupancy_threshold)),
            inflation_radius=float(cfg.get("inflation_radius", cls.inflation_radius)),
            allow_diagonal=bool(cfg.get("allow_diagonal", cls.allow_diagonal)),
            heuristic_weight=float(cfg.get("heuristic_weight", cls.heuristic_weight)),
            max_expansions=int(cfg.get("max_expansions", cls.max_expansions)),
        )


@dataclass(frozen=True)
class GridAStarResult:
    """Result from a grid A* query."""

    found: bool
    start_cell: Cell
    goal_cell: Cell
    path_cells: list[Cell]
    path_xy_b: list[Point2D]
    cost: float
    expansions: int
    reason: str = ""


@dataclass(frozen=True)
class PathFollowerCfg:
    """Simple waypoint follower for body-frame A* paths.

    The lookahead point is selected along the path arc length, not merely by
    Euclidean distance from the origin. ``min_forward_x`` prevents the follower
    from selecting points that are almost sideways next to the robot, which can
    make the proxy stall while trying to pass an obstacle.
    """

    lookahead_distance: float = 1.00
    min_forward_x: float = 0.20
    max_vx: float = 0.30
    max_vy: float = 0.25
    max_wz: float = 0.60
    k_vx: float = 0.60
    k_vy: float = 0.80
    k_wz: float = 0.80
    stop_distance: float = 0.20

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, object] | None) -> "PathFollowerCfg":
        if cfg is None:
            return cls()
        return cls(
            lookahead_distance=float(cfg.get("lookahead_distance", cls.lookahead_distance)),
            min_forward_x=float(cfg.get("min_forward_x", cls.min_forward_x)),
            max_vx=float(cfg.get("max_vx", cls.max_vx)),
            max_vy=float(cfg.get("max_vy", cls.max_vy)),
            max_wz=float(cfg.get("max_wz", cls.max_wz)),
            k_vx=float(cfg.get("k_vx", cls.k_vx)),
            k_vy=float(cfg.get("k_vy", cls.k_vy)),
            k_wz=float(cfg.get("k_wz", cls.k_wz)),
            stop_distance=float(cfg.get("stop_distance", cls.stop_distance)),
        )


@dataclass(frozen=True)
class PathFollowerResult:
    """Output from following a body-frame waypoint path."""

    command: torch.Tensor
    lookahead_xy_b: Point2D | None
    target_reached: bool
    path_length: int


def xy_to_grid_cell(xy_b: Sequence[float], cfg: OccupancyGridCfg) -> Cell:
    """Convert body-frame ``(x, y)`` to ``(row, col)`` grid indices.

    The mapping matches :func:`rasterize_points_to_occupancy_grid`: row 0 is
    ``x_max``/forward, and columns increase from ``y_min`` to ``y_max``.
    """
    if len(xy_b) != 2:
        raise ValueError(f"xy_b must have length 2, got {xy_b!r}")
    x = float(xy_b[0])
    y = float(xy_b[1])
    row = math.floor((cfg.x_max - x) / cfg.resolution)
    col = math.floor((y - cfg.y_min) / cfg.resolution)
    return int(row), int(col)


def grid_cell_to_xy(cell: Cell, cfg: OccupancyGridCfg) -> Point2D:
    """Convert ``(row, col)`` grid index to body-frame cell-center ``(x, y)``."""
    row, col = cell
    x = cfg.x_max - (float(row) + 0.5) * cfg.resolution
    y = cfg.y_min + (float(col) + 0.5) * cfg.resolution
    return float(x), float(y)


def is_cell_in_bounds(cell: Cell, *, height: int, width: int) -> bool:
    row, col = cell
    return 0 <= row < height and 0 <= col < width


def inflate_occupancy_grid(
    occupancy_grid: torch.Tensor,
    *,
    cfg: OccupancyGridCfg,
    inflation_radius: float,
    occupancy_threshold: float = 0.5,
) -> torch.Tensor:
    """Inflate occupied cells by a square radius.

    This is a conservative max-pool dilation. It is sufficient for local planning
    and avoids adding SciPy/OpenCV dependencies.

    Args:
        occupancy_grid: Tensor with shape ``[H, W]`` or ``[N, H, W]``.
        cfg: Occupancy-grid geometry.
        inflation_radius: Inflation radius in meters. ``<=0`` returns a binary
            occupancy mask without dilation.
        occupancy_threshold: Values >= threshold are occupied.

    Returns:
        Boolean tensor with the same leading shape as ``occupancy_grid``.
    """
    if not torch.is_tensor(occupancy_grid):
        raise TypeError(f"occupancy_grid must be a torch.Tensor, got {type(occupancy_grid)!r}")
    if occupancy_grid.ndim not in (2, 3):
        raise ValueError(f"occupancy_grid must have shape [H, W] or [N, H, W], got {tuple(occupancy_grid.shape)}")

    occupied = occupancy_grid >= float(occupancy_threshold)
    radius_cells = int(math.ceil(max(0.0, float(inflation_radius)) / cfg.resolution))
    if radius_cells <= 0:
        return occupied.contiguous()

    import torch.nn.functional as F

    squeeze_batch = occupancy_grid.ndim == 2
    if squeeze_batch:
        data = occupied[None, None].to(dtype=torch.float32)
    else:
        data = occupied[:, None].to(dtype=torch.float32)

    kernel = 2 * radius_cells + 1
    inflated = F.max_pool2d(data, kernel_size=kernel, stride=1, padding=radius_cells)
    inflated = inflated[:, 0] > 0.0
    if squeeze_batch:
        return inflated[0].contiguous()
    return inflated.contiguous()


def _motion_model(allow_diagonal: bool) -> tuple[tuple[int, int, float], ...]:
    motions: list[tuple[int, int, float]] = [
        (-1, 0, 1.0),  # forward in body x, smaller row
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
    ]
    if allow_diagonal:
        root2 = math.sqrt(2.0)
        motions.extend([
            (-1, -1, root2),
            (-1, 1, root2),
            (1, -1, root2),
            (1, 1, root2),
        ])
    return tuple(motions)


def _heuristic(cell: Cell, goal: Cell) -> float:
    return math.hypot(float(cell[0] - goal[0]), float(cell[1] - goal[1]))


def _reconstruct_path(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _nearest_free_cell(start_cell: tuple[int, int], occupied: torch.Tensor) -> tuple[int, int] | None:
    """Return the nearest free grid cell to ``start_cell`` using BFS.

    ``occupied`` is expected to be a boolean/integer occupancy tensor with shape
    ``[H, W]`` where True/non-zero means blocked.  This function is used when
    the requested A* goal lies in inflated occupancy and goal relocation is
    enabled.

    Important implementation detail: do not iterate over a deque while appending
    to it.  Python raises ``RuntimeError: deque mutated during iteration``.
    Use the standard ``while queue: popleft()`` BFS pattern instead.
    """
    if occupied.ndim != 2:
        raise ValueError(f"occupied must have shape [H, W], got {tuple(occupied.shape)}")

    height, width = int(occupied.shape[0]), int(occupied.shape[1])
    start_row, start_col = int(start_cell[0]), int(start_cell[1])

    def in_bounds(row: int, col: int) -> bool:
        return 0 <= row < height and 0 <= col < width

    def is_free(row: int, col: int) -> bool:
        # .item() is explicit and works for bool/integer scalar tensors.
        return not bool(occupied[row, col].item())

    if not in_bounds(start_row, start_col):
        return None
    if is_free(start_row, start_col):
        return (start_row, start_col)

    from collections import deque

    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    visited: set[tuple[int, int]] = {(start_row, start_col)}

    # 8-connected search keeps behavior consistent with diagonal A* grids and
    # usually finds a nearby free target with fewer ring artifacts.
    neighbors = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    while queue:
        row, col = queue.popleft()
        for dr, dc in neighbors:
            nr, nc = row + dr, col + dc
            cell = (nr, nc)
            if cell in visited or not in_bounds(nr, nc):
                continue
            if is_free(nr, nc):
                return cell
            visited.add(cell)
            queue.append(cell)

    return None

def astar_grid_search(
    occupied: torch.Tensor,
    start_cell: Cell,
    goal_cell: Cell,
    cfg: GridAStarCfg | None = None,
) -> tuple[list[Cell], dict[str, object]]:
    """Run A* directly on a boolean occupancy grid.

    Args:
        occupied: Boolean-like tensor with shape ``[H, W]``. ``True`` means
            occupied/untraversable.
        start_cell: ``(row, col)`` start index.
        goal_cell: ``(row, col)`` goal index.
        cfg: A* options.

    Returns:
        ``(path_cells, info)``. ``path_cells`` is empty when no path was found.
    """
    planner_cfg = cfg or GridAStarCfg(inflation_radius=0.0)
    if occupied.ndim != 2:
        raise ValueError(f"occupied must have shape [H, W], got {tuple(occupied.shape)}")
    occupied = occupied.to(dtype=torch.bool, device="cpu").contiguous()
    height, width = int(occupied.shape[0]), int(occupied.shape[1])

    if not is_cell_in_bounds(start_cell, height=height, width=width):
        return [], {"found": False, "reason": "start_out_of_bounds", "expansions": 0}
    if not is_cell_in_bounds(goal_cell, height=height, width=width):
        return [], {"found": False, "reason": "goal_out_of_bounds", "expansions": 0}
    if bool(occupied[start_cell[0], start_cell[1]]):
        return [], {"found": False, "reason": "start_occupied", "expansions": 0}
    if bool(occupied[goal_cell[0], goal_cell[1]]):
        return [], {"found": False, "reason": "goal_occupied", "expansions": 0}

    motions = _motion_model(planner_cfg.allow_diagonal)
    open_heap: list[tuple[float, int, Cell]] = []
    heapq.heappush(open_heap, (0.0, 0, start_cell))

    came_from: dict[Cell, Cell] = {}
    g_cost: dict[Cell, float] = {start_cell: 0.0}
    closed: set[Cell] = set()
    push_count = 1
    expansions = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_cell:
            path = _reconstruct_path(came_from, current)
            return path, {
                "found": True,
                "reason": "goal_reached",
                "expansions": expansions,
                "cost": g_cost[current],
            }

        closed.add(current)
        expansions += 1
        if planner_cfg.max_expansions > 0 and expansions >= planner_cfg.max_expansions:
            return [], {"found": False, "reason": "max_expansions", "expansions": expansions}

        row, col = current
        for dr, dc, move_cost in motions:
            nxt = (row + dr, col + dc)
            if not is_cell_in_bounds(nxt, height=height, width=width):
                continue
            if bool(occupied[nxt[0], nxt[1]]):
                continue
            if nxt in closed:
                continue

            tentative_g = g_cost[current] + move_cost
            if tentative_g >= g_cost.get(nxt, math.inf):
                continue

            came_from[nxt] = current
            g_cost[nxt] = tentative_g
            f_cost = tentative_g + planner_cfg.heuristic_weight * _heuristic(nxt, goal_cell)
            heapq.heappush(open_heap, (f_cost, push_count, nxt))
            push_count += 1

    return [], {"found": False, "reason": "open_set_empty", "expansions": expansions}


def plan_grid_path_xy(
    occupancy_grid: torch.Tensor,
    start_xy_b: Sequence[float],
    goal_xy_b: Sequence[float],
    grid_cfg: OccupancyGridCfg,
    planner_cfg: GridAStarCfg | None = None,
    *,
    allow_goal_relocation: bool = True,
    allow_start_relocation: bool = True,
) -> GridAStarResult:
    """Plan a body-frame path on a local occupancy grid.

    Args:
        occupancy_grid: Occupancy tensor ``[H, W]`` or batched ``[1, H, W]``.
        start_xy_b: Start point in body frame, usually ``(0, 0)``.
        goal_xy_b: Goal point in body frame.
        grid_cfg: Occupancy-grid geometry.
        planner_cfg: A* config.
        allow_goal_relocation: If true and the goal lies in an occupied cell,
            search from the nearest free cell instead. This is useful when the
            target itself is represented by/inside an obstacle blob.
        allow_start_relocation: If true and the start lies in inflated occupancy,
            search from the nearest free cell.  Stepwise real/sim controllers
            should usually disable this so being inside the safety margin is a
            hard planning failure instead of planning from a fictitious start.
    """
    cfg = planner_cfg or GridAStarCfg()
    grid = occupancy_grid
    if grid.ndim == 3:
        if grid.shape[0] != 1:
            raise ValueError("plan_grid_path_xy currently expects a single grid or batch size 1")
        grid = grid[0]
    if grid.ndim != 2:
        raise ValueError(f"occupancy_grid must have shape [H, W] or [1, H, W], got {tuple(occupancy_grid.shape)}")

    inflated = inflate_occupancy_grid(
        grid,
        cfg=grid_cfg,
        inflation_radius=cfg.inflation_radius,
        occupancy_threshold=cfg.occupancy_threshold,
    )
    start_cell = xy_to_grid_cell(start_xy_b, grid_cfg)
    goal_cell = xy_to_grid_cell(goal_xy_b, grid_cfg)
    height, width = int(inflated.shape[0]), int(inflated.shape[1])

    if not is_cell_in_bounds(start_cell, height=height, width=width):
        return GridAStarResult(False, start_cell, goal_cell, [], [], math.inf, 0, "start_out_of_bounds")
    if not is_cell_in_bounds(goal_cell, height=height, width=width):
        return GridAStarResult(False, start_cell, goal_cell, [], [], math.inf, 0, "goal_out_of_bounds")

    if bool(inflated[start_cell[0], start_cell[1]]):
        if not allow_start_relocation:
            return GridAStarResult(False, start_cell, goal_cell, [], [], math.inf, 0, "start_occupied")
        adjusted_start = _nearest_free_cell(start_cell, inflated)
        if adjusted_start is None:
            return GridAStarResult(False, start_cell, goal_cell, [], [], math.inf, 0, "no_free_start_cell")
    else:
        adjusted_start = start_cell

    adjusted_goal = goal_cell
    if bool(inflated[goal_cell[0], goal_cell[1]]):
        if not allow_goal_relocation:
            return GridAStarResult(False, adjusted_start, goal_cell, [], [], math.inf, 0, "goal_occupied")
        nearest_goal = _nearest_free_cell(goal_cell, inflated)
        if nearest_goal is None:
            return GridAStarResult(False, adjusted_start, goal_cell, [], [], math.inf, 0, "no_free_goal_cell")
        adjusted_goal = nearest_goal

    path_cells, info = astar_grid_search(inflated, adjusted_start, adjusted_goal, cfg)
    found = bool(info.get("found", False))
    path_xy = [grid_cell_to_xy(cell, grid_cfg) for cell in path_cells]
    return GridAStarResult(
        found=found,
        start_cell=adjusted_start,
        goal_cell=adjusted_goal,
        path_cells=path_cells,
        path_xy_b=path_xy,
        cost=float(info.get("cost", math.inf)),
        expansions=int(info.get("expansions", 0)),
        reason=str(info.get("reason", "")),
    )


def bresenham_cells(a: Cell, b: Cell) -> list[Cell]:
    """Return grid cells on a line segment using Bresenham's algorithm."""
    r0, c0 = a
    r1, c1 = b
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    cells: list[Cell] = []

    if dc > dr:
        err = dc / 2.0
        r = r0
        for c in range(c0, c1 + sc, sc):
            cells.append((r, c))
            err -= dr
            if err < 0:
                r += sr
                err += dc
    else:
        err = dr / 2.0
        c = c0
        for r in range(r0, r1 + sr, sr):
            cells.append((r, c))
            err -= dc
            if err < 0:
                c += sc
                err += dr
    return cells


def has_line_of_sight(occupied: torch.Tensor, a: Cell, b: Cell) -> bool:
    """Return true if the segment from ``a`` to ``b`` crosses no occupied cells."""
    if occupied.ndim != 2:
        raise ValueError(f"occupied must have shape [H, W], got {tuple(occupied.shape)}")
    occupied = occupied.to(dtype=torch.bool, device="cpu")
    height, width = int(occupied.shape[0]), int(occupied.shape[1])
    for cell in bresenham_cells(a, b):
        if not is_cell_in_bounds(cell, height=height, width=width):
            return False
        if bool(occupied[cell[0], cell[1]]):
            return False
    return True


def shortcut_path_cells(path_cells: Sequence[Cell], occupied: torch.Tensor) -> list[Cell]:
    """Shortcut a grid path by greedily skipping line-of-sight intermediate nodes."""
    if len(path_cells) <= 2:
        return list(path_cells)

    shortened: list[Cell] = [path_cells[0]]
    i = 0
    while i < len(path_cells) - 1:
        j = len(path_cells) - 1
        while j > i + 1:
            if has_line_of_sight(occupied, path_cells[i], path_cells[j]):
                break
            j -= 1
        shortened.append(path_cells[j])
        i = j
    return shortened


def shortcut_grid_path_result(
    result: GridAStarResult,
    occupancy_grid: torch.Tensor,
    grid_cfg: OccupancyGridCfg,
    planner_cfg: GridAStarCfg | None = None,
) -> GridAStarResult:
    """Shortcut a :class:`GridAStarResult` using inflated occupancy."""
    if not result.found or len(result.path_cells) <= 2:
        return result
    cfg = planner_cfg or GridAStarCfg()
    grid = occupancy_grid[0] if occupancy_grid.ndim == 3 else occupancy_grid
    inflated = inflate_occupancy_grid(
        grid,
        cfg=grid_cfg,
        inflation_radius=cfg.inflation_radius,
        occupancy_threshold=cfg.occupancy_threshold,
    )
    cells = shortcut_path_cells(result.path_cells, inflated)
    xy = [grid_cell_to_xy(cell, grid_cfg) for cell in cells]
    return GridAStarResult(
        found=True,
        start_cell=result.start_cell,
        goal_cell=result.goal_cell,
        path_cells=cells,
        path_xy_b=xy,
        cost=result.cost,
        expansions=result.expansions,
        reason=result.reason + "+shortcut",
    )


def select_lookahead_point(
    path_xy_b: Sequence[Point2D],
    lookahead_distance: float,
    *,
    min_forward_x: float = 0.0,
) -> Point2D | None:
    """Select a forward lookahead point along path arc length.

    The previous implementation picked the first waypoint whose Euclidean
    distance from the robot exceeded ``lookahead_distance``. Near obstacle
    corners this could select a point that was mostly sideways (e.g.
    ``x ~= 0, y ~= 1``), which makes the command-level proxy stop making
    forward progress.

    This version walks along the polyline and interpolates a point at the
    requested arc length. If possible, it also enforces ``x >= min_forward_x``.
    """
    if not path_xy_b:
        return None

    # Prefer only waypoints that are in front of the robot. If none exist, fall
    # back to the original path so the caller still gets a deterministic command.
    forward_path = [(float(x), float(y)) for x, y in path_xy_b if float(x) >= min_forward_x]
    path = forward_path if forward_path else [(float(x), float(y)) for x, y in path_xy_b]
    if len(path) == 1:
        return path[0]

    # Start from the robot origin, then walk along the path. This makes the
    # lookahead distance mean "distance to travel along the path", not just a
    # radial threshold.
    prev_x, prev_y = 0.0, 0.0
    remaining = max(0.0, float(lookahead_distance))
    for x, y in path:
        seg_len = math.hypot(x - prev_x, y - prev_y)
        if seg_len >= remaining and seg_len > 1.0e-8:
            t = remaining / seg_len
            lx = prev_x + t * (x - prev_x)
            ly = prev_y + t * (y - prev_y)
            return float(lx), float(ly)
        remaining -= seg_len
        prev_x, prev_y = x, y

    return path[-1]


def follow_body_frame_path(
    path_xy_b: Sequence[Point2D],
    cfg: PathFollowerCfg | None = None,
    *,
    device: torch.device | str | None = None,
) -> PathFollowerResult:
    """Convert a body-frame waypoint path into one command ``[vx, vy, wz]``.

    This is intentionally simple: it selects a lookahead waypoint and commands
    forward/lateral velocity toward it. Yaw is used to face the waypoint.
    """
    follower_cfg = cfg or PathFollowerCfg()
    dev = device if device is not None else "cpu"
    if not path_xy_b:
        command = torch.zeros((1, 3), dtype=torch.float32, device=dev)
        return PathFollowerResult(command, None, False, 0)

    final_x, final_y = path_xy_b[-1]
    final_dist = math.hypot(final_x, final_y)
    if final_dist <= follower_cfg.stop_distance:
        command = torch.zeros((1, 3), dtype=torch.float32, device=dev)
        return PathFollowerResult(command, (final_x, final_y), True, len(path_xy_b))

    lookahead = select_lookahead_point(
        path_xy_b,
        follower_cfg.lookahead_distance,
        min_forward_x=follower_cfg.min_forward_x,
    )
    if lookahead is None:
        command = torch.zeros((1, 3), dtype=torch.float32, device=dev)
        return PathFollowerResult(command, None, False, len(path_xy_b))

    lx, ly = lookahead
    heading_error = math.atan2(ly, max(lx, 1.0e-6))
    vx = max(0.0, min(follower_cfg.max_vx, follower_cfg.k_vx * lx))
    vy = max(-follower_cfg.max_vy, min(follower_cfg.max_vy, follower_cfg.k_vy * ly))
    wz = max(-follower_cfg.max_wz, min(follower_cfg.max_wz, follower_cfg.k_wz * heading_error))

    command = torch.tensor([[vx, vy, wz]], dtype=torch.float32, device=dev)
    return PathFollowerResult(command, lookahead, False, len(path_xy_b))
