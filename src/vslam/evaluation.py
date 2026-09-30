"""TUM-style trajectory metrics: ATE RMSE after Sim(3) alignment, and RPE."""

from __future__ import annotations

import numpy as np

from vslam.geometry import invert_pose, se3_log, umeyama


def poses_to_centers(Ts_cw: list[np.ndarray]) -> np.ndarray:
    return np.vstack([invert_pose(T)[:3, 3] for T in Ts_cw])


def ate_rmse(est_cw: list[np.ndarray], gt_cw: list[np.ndarray]) -> dict:
    n = min(len(est_cw), len(gt_cw))
    if n < 3:
        return {"ate_rmse": float("nan"), "n": n}
    est = poses_to_centers(est_cw[:n])
    gt = poses_to_centers(gt_cw[:n])
    S = umeyama(est, gt, with_scale=True)
    est_h = np.c_[est, np.ones(n)]
    aligned = (S @ est_h.T).T[:, :3]
    err = np.linalg.norm(aligned - gt, axis=1)
    return {
        "ate_rmse": float(np.sqrt(np.mean(err**2))),
        "ate_mean": float(np.mean(err)),
        "ate_median": float(np.median(err)),
        "ate_max": float(np.max(err)),
        "n": n,
        "scale": float(np.linalg.det(S[:3, :3]) ** (1.0 / 3.0)),
        "aligned": aligned,
        "gt": gt,
        "S": S,
    }


def rpe_rmse(est_cw: list[np.ndarray], gt_cw: list[np.ndarray], delta: int = 1) -> dict:
    n = min(len(est_cw), len(gt_cw))
    trans, rot = [], []
    for i in range(n - delta):
        e_rel = est_cw[i + delta] @ invert_pose(est_cw[i])
        g_rel = gt_cw[i + delta] @ invert_pose(gt_cw[i])
        err = invert_pose(g_rel) @ e_rel
        xi = se3_log(err)
        trans.append(np.linalg.norm(xi[3:]))
        rot.append(np.linalg.norm(xi[:3]))
    if not trans:
        return {"rpe_trans_rmse": float("nan"), "rpe_rot_rmse_deg": float("nan")}
    trans = np.asarray(trans)
    rot = np.degrees(np.asarray(rot))
    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(trans**2))),
        "rpe_rot_rmse_deg": float(np.sqrt(np.mean(rot**2))),
        "n": int(len(trans)),
    }
