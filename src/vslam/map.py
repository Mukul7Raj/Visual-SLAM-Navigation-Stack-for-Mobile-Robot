"""Keyframe / map-point graph used by tracking and local BA."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    frame_id: int
    image: np.ndarray
    kps: list
    descriptors: np.ndarray
    T_cw: np.ndarray | None = None
    timestamp: float = 0.0
    is_keyframe: bool = False
    # index in this frame -> map point id
    mp_ids: dict[int, int] = field(default_factory=dict)


@dataclass
class MapPoint:
    mp_id: int
    Xw: np.ndarray
    descriptor: np.ndarray
    observations: dict[int, int] = field(default_factory=dict)  # kf_id -> kp index
    outlier: bool = False


class Map:
    def __init__(self) -> None:
        self.keyframes: dict[int, Frame] = {}
        self.points: dict[int, MapPoint] = {}
        self._next_mp = 0
        self.pose_graph: list[tuple[int, int]] = []

    def add_keyframe(self, frame: Frame) -> None:
        frame.is_keyframe = True
        self.keyframes[frame.frame_id] = frame
        ids = sorted(self.keyframes)
        if len(ids) >= 2:
            self.pose_graph.append((ids[-2], ids[-1]))

    def add_point(self, Xw: np.ndarray, descriptor: np.ndarray) -> MapPoint:
        mp = MapPoint(mp_id=self._next_mp, Xw=Xw.astype(np.float64).copy(), descriptor=descriptor.copy())
        self.points[mp.mp_id] = mp
        self._next_mp += 1
        return mp

    def observe(self, mp_id: int, frame: Frame, kp_idx: int) -> None:
        mp = self.points[mp_id]
        mp.observations[frame.frame_id] = kp_idx
        frame.mp_ids[kp_idx] = mp_id

    def active_points(self) -> list[MapPoint]:
        return [p for p in self.points.values() if not p.outlier and len(p.observations) >= 2]

    def trajectory_cw(self) -> list[tuple[int, np.ndarray]]:
        items = sorted(self.keyframes.items())
        return [(fid, f.T_cw.copy()) for fid, f in items if f.T_cw is not None]
