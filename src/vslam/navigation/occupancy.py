"""Project sparse SLAM points to a 2D occupancy grid for planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OccupancyGrid:
    grid: np.ndarray  # uint8: 0 free, 100 occupied
    origin_xy: np.ndarray
    resolution: float

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return ((xy - self.origin_xy) / self.resolution).astype(int)

    def cell_to_world(self, ij: np.ndarray) -> np.ndarray:
        ij = np.asarray(ij, dtype=np.float64).reshape(-1, 2)
        return (ij + 0.5) * self.resolution + self.origin_xy

    def free(self, row: int, col: int) -> bool:
        h, w = self.grid.shape
        if row < 0 or col < 0 or row >= h or col >= w:
            return False
        return self.grid[row, col] < 50


def points_to_occupancy(
    Xw: np.ndarray,
    resolution: float = 0.05,
    inflate_radius: float = 0.25,
    height_min: float = 0.05,
    height_max: float = 1.7,
    occ_count: int = 3,
    pad: float = 0.8,
) -> OccupancyGrid:
    if len(Xw) == 0:
        g = np.zeros((20, 20), dtype=np.uint8)
        return OccupancyGrid(g, np.array([-0.5, -0.5]), resolution)
    z = Xw[:, 2]
    pts = Xw[(z >= height_min) & (z <= height_max)][:, :2]
    if len(pts) == 0:
        pts = Xw[:, :2]
    mn = pts.min(axis=0) - pad
    mx = pts.max(axis=0) + pad
    wh = np.ceil((mx - mn) / resolution).astype(int) + 1
    counts = np.zeros((int(wh[1]), int(wh[0])), dtype=np.int32)
    ix = ((pts[:, 0] - mn[0]) / resolution).astype(int)
    iy = ((pts[:, 1] - mn[1]) / resolution).astype(int)
    valid = (ix >= 0) & (iy >= 0) & (ix < wh[0]) & (iy < wh[1])
    np.add.at(counts, (iy[valid], ix[valid]), 1)
    occ = np.zeros(counts.shape, dtype=np.uint8)
    occ[counts >= occ_count] = 100
    inflate = max(1, int(round(inflate_radius / resolution)))
    occ = _inflate(occ, inflate)
    return OccupancyGrid(occ, mn, resolution)


def _inflate(occ: np.ndarray, radius: int) -> np.ndarray:
    ys, xs = np.where(occ == 100)
    out = occ.copy()
    h, w = occ.shape
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        out[y0:y1, x0:x1] = 100
    return out
