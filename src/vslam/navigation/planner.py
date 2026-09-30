"""A* on an occupancy grid, 8-connected, Euclidean heuristic."""

from __future__ import annotations

import heapq

import numpy as np

from vslam.navigation.occupancy import OccupancyGrid


def astar(grid: OccupancyGrid, start_xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
    start = _xy_to_rowcol(grid, start_xy)
    goal = _xy_to_rowcol(grid, goal_xy)
    start = _nearest_free(grid, start)
    goal = _nearest_free(grid, goal)
    if start is None or goal is None:
        return np.zeros((0, 2))

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
    openh: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    gscore = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    def heuristic(a: tuple[int, int]) -> float:
        return float(np.hypot(a[0] - goal[0], a[1] - goal[1]))

    visited: set[tuple[int, int]] = set()
    while openh:
        _, cur = heapq.heappop(openh)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            break
        for (di, dj), c in zip(nbrs, costs):
            nxt = (cur[0] + di, cur[1] + dj)
            if not grid.free(nxt[0], nxt[1]):
                continue
            ng = gscore[cur] + c
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                parent[nxt] = cur
                heapq.heappush(openh, (ng + heuristic(nxt), nxt))
    if goal not in parent:
        return np.zeros((0, 2))
    path_ij = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path_ij.append(cur)
        cur = parent[cur]
    path_ij.reverse()
    cells = np.array([[col, row] for row, col in path_ij], dtype=np.float64)
    return grid.cell_to_world(cells)


def _xy_to_rowcol(grid: OccupancyGrid, xy: np.ndarray) -> tuple[int, int]:
    col, row = grid.world_to_cell(xy)[0]
    return int(row), int(col)


def _nearest_free(grid: OccupancyGrid, cell: tuple[int, int], radius: int = 25) -> tuple[int, int] | None:
    if grid.free(*cell):
        return cell
    i0, j0 = cell
    for r in range(1, radius + 1):
        for i in range(i0 - r, i0 + r + 1):
            for j in (j0 - r, j0 + r):
                if grid.free(i, j):
                    return (i, j)
        for j in range(j0 - r, j0 + r + 1):
            for i in (i0 - r, i0 + r):
                if grid.free(i, j):
                    return (i, j)
    return None
