"""End-to-end interview demo: simulate indoor run, SLAM, evaluate, plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vslam.config import build_system, load_yaml
from vslam.evaluation import ate_rmse, rpe_rmse
from vslam.geometry import invert_pose
from vslam.navigation import astar, path_to_unicycle, points_to_occupancy
from vslam.sim import generate_indoor_sequence
from vslam.viz import plot_map_3d, plot_occupancy, plot_trajectory, save_tracking_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual SLAM + navigation demo")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="outputs/demo")
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_frames = args.frames or int(cfg.get("simulation", {}).get("n_frames", 160))
    seed = int(cfg.get("simulation", {}).get("seed", 7))
    dt = float(cfg.get("simulation", {}).get("dt", 0.12))

    slam = build_system(cfg)
    seq = generate_indoor_sequence(slam.camera, n_frames=n_frames, seed=seed, dt=dt)
    seq.save(out / "sequence")

    for i, (img, ts) in enumerate(zip(seq.images, seq.timestamps)):
        frame = slam.process(img, timestamp=ts, frame_id=i)
        if i == max(2, n_frames // 5):
            save_tracking_preview(img, frame.kps, out / "orb_preview.png")

    est = [T for _, T in slam.stats.tracked_poses]
    # Align frame indices: tracked poses may skip nothing; GT is every frame
    gt = [seq.T_cw[fid] for fid, _ in slam.stats.tracked_poses]
    ate = ate_rmse(est, gt)
    rpe = rpe_rmse(est, gt, delta=5)

    Xw = np.array([p.Xw for p in slam.map.active_points()]) if slam.map.active_points() else np.zeros((0, 3))
    nav_cfg = cfg.get("navigation", {})
    grid = points_to_occupancy(
        Xw,
        resolution=float(nav_cfg.get("resolution", 0.05)),
        inflate_radius=float(nav_cfg.get("inflate_radius", 0.25)),
        height_min=float(nav_cfg.get("height_min", 0.05)),
        height_max=float(nav_cfg.get("height_max", 1.7)),
        occ_count=int(nav_cfg.get("occ_count", 3)),
    )
    if est:
        start = invert_pose(est[0])[:3, 3][:2]
        goal = start + np.array([2.5, 1.6])
    else:
        start = np.array([2.0, 2.0])
        goal = np.array([5.0, 4.0])
    path = astar(grid, start, goal)
    cmds = path_to_unicycle(path)

    kf_poses = [f.T_cw for f in slam.map.keyframes.values() if f.T_cw is not None]
    if "gt" in ate:
        plot_trajectory(ate, out / "trajectory_ate.png")
    else:
        plot_trajectory({"ate_rmse": float("nan")}, out / "trajectory_ate.png")
    plot_occupancy(grid, path, out / "occupancy_plan.png")
    plot_map_3d(Xw, kf_poses, out / "sparse_map.png")

    metrics = {
        "initialized": slam.stats.initialized,
        "frames": slam.stats.n_frames,
        "keyframes": slam.stats.n_keyframes,
        "map_points": slam.stats.n_points,
        "lost_frames": slam.stats.n_lost,
        "loop_closures": slam.stats.n_loops,
        "ate_rmse_m": ate.get("ate_rmse"),
        "ate_mean_m": ate.get("ate_mean"),
        "rpe_trans_rmse": rpe.get("rpe_trans_rmse"),
        "rpe_rot_rmse_deg": rpe.get("rpe_rot_rmse_deg"),
        "path_cells": int(len(path)),
        "control_steps": len(cmds),
        "last_ba": slam.stats.ba_reports[-1] if slam.stats.ba_reports else None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
