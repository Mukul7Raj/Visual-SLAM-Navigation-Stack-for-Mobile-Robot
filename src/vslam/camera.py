"""Pinhole camera model used throughout tracking, mapping, and simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(self, Xc: np.ndarray) -> np.ndarray:
        """Project camera-frame 3D points (N,3) to pixels (N,2)."""
        z = np.clip(Xc[:, 2], 1e-8, None)
        u = self.fx * Xc[:, 0] / z + self.cx
        v = self.fy * Xc[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)

    def in_image(self, uv: np.ndarray, border: int = 8) -> np.ndarray:
        u, v = uv[:, 0], uv[:, 1]
        return (
            (u >= border)
            & (v >= border)
            & (u < self.width - border)
            & (v < self.height - border)
        )

    def normalize(self, uv: np.ndarray) -> np.ndarray:
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        return np.stack([x, y], axis=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            width=int(d["width"]),
            height=int(d["height"]),
        )
