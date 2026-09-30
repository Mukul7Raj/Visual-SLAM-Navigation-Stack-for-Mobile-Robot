"""Unicycle (differential-drive) tracking of an A* path."""

from __future__ import annotations

import numpy as np


def path_to_unicycle(
    path_xy: np.ndarray,
    dt: float = 0.1,
    v_max: float = 0.35,
    w_max: float = 1.2,
    k_v: float = 1.2,
    k_w: float = 2.5,
) -> list[dict]:
    if len(path_xy) < 2:
        return []
    pose = np.array([path_xy[0, 0], path_xy[0, 1], _heading(path_xy[0], path_xy[1])], dtype=np.float64)
    cmds = []
    target_i = 1
    steps = 0
    max_steps = 4000
    while target_i < len(path_xy) and steps < max_steps:
        goal = path_xy[target_i]
        dx, dy = goal[0] - pose[0], goal[1] - pose[1]
        dist = float(np.hypot(dx, dy))
        if dist < 0.08:
            target_i += 1
            continue
        desired = np.arctan2(dy, dx)
        heading_err = _wrap(desired - pose[2])
        v = float(np.clip(k_v * dist * np.cos(heading_err), 0.0, v_max))
        w = float(np.clip(k_w * heading_err, -w_max, w_max))
        pose[0] += v * np.cos(pose[2]) * dt
        pose[1] += v * np.sin(pose[2]) * dt
        pose[2] = _wrap(pose[2] + w * dt)
        cmds.append({"v": v, "omega": w, "x": pose[0], "y": pose[1], "yaw": pose[2]})
        steps += 1
    return cmds


def _heading(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arctan2(b[1] - a[1], b[0] - a[0]))


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))
