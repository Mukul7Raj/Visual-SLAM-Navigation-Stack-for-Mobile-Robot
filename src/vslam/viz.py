"""Matplotlib figures used in the README and interview walkthrough."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vslam.geometry import invert_pose
from vslam.navigation.occupancy import OccupancyGrid


def save_tracking_preview(image: np.ndarray, kps, path: Path, n: int = 80) -> None:
    vis = image.copy()
    for kp in kps[:n]:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0, 220, 80), -1)
    cv2.imwrite(str(path), vis)


def plot_trajectory(ate: dict, out: Path, title: str = "Trajectory (Sim3-aligned)") -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    if "gt" not in ate or "aligned" not in ate:
        ax.text(0.5, 0.5, "Not enough poses to plot ATE", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return
    gt = ate["gt"]
    al = ate["aligned"]
    ax.plot(gt[:, 0], gt[:, 1], color="#1d4ed8", lw=2.2, label="Ground truth")
    ax.plot(al[:, 0], al[:, 1], color="#dc2626", lw=1.8, ls="--", label="SLAM (aligned)")
    ax.scatter(gt[0, 0], gt[0, 1], c="#16a34a", s=40, zorder=5, label="Start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{title}\nATE RMSE = {ate['ate_rmse']:.3f} m")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_occupancy(grid: OccupancyGrid, path_xy: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    h, w = grid.grid.shape
    extent = [
        grid.origin_xy[0],
        grid.origin_xy[0] + w * grid.resolution,
        grid.origin_xy[1],
        grid.origin_xy[1] + h * grid.resolution,
    ]
    show = np.full(grid.grid.shape, 0.85)
    show[grid.grid == 0] = 1.0
    show[grid.grid == 100] = 0.12
    ax.imshow(show, origin="lower", cmap="gray", extent=extent, vmin=0, vmax=1)
    if len(path_xy):
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="#f59e0b", lw=2.0, label="A* path")
        ax.scatter(path_xy[0, 0], path_xy[0, 1], c="#16a34a", s=36, label="Start")
        ax.scatter(path_xy[-1, 0], path_xy[-1, 1], c="#dc2626", s=36, label="Goal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Occupancy grid from sparse map + A*")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_map_3d(Xw: np.ndarray, poses_cw: list[np.ndarray], out: Path) -> None:
    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    if len(Xw):
        step = max(1, len(Xw) // 2500)
        P = Xw[::step]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c="#64748b", alpha=0.5)
    C = np.vstack([invert_pose(T)[:3, 3] for T in poses_cw]) if poses_cw else np.zeros((0, 3))
    if len(C):
        ax.plot(C[:, 0], C[:, 1], C[:, 2], color="#dc2626", lw=1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Sparse map and keyframe trajectory")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
