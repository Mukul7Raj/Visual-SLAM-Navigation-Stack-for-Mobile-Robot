"""Local bundle adjustment over a sliding window of keyframes.

Minimizes Huber-weighted reprojection error. The oldest window pose is
held fixed to freeze gauge (the first two keyframes already pin scale).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vslam.camera import Camera
from vslam.geometry import pose_matrix, transform_points
from vslam.map import Map


def local_bundle_adjust(world: Map, camera: Camera, window: int = 6, max_nfev: int = 40) -> dict:
    kf_ids = sorted(world.keyframes)
    if len(kf_ids) < 3:
        return {"success": False, "reason": "too_few_keyframes"}
    window_ids = kf_ids[-window:]
    window_set = set(window_ids)

    points = []
    for mp in world.points.values():
        if mp.outlier:
            continue
        obs = [(fid, idx) for fid, idx in mp.observations.items() if fid in window_set]
        if len(obs) < 2:
            continue
        points.append((mp, obs))
    if len(points) < 8:
        return {"success": False, "reason": "too_few_points"}

    # Parameterization: 6*(N-1) pose params + 3*M point params. First KF fixed.
    n_pose = len(window_ids) - 1
    n_pts = len(points)

    x0 = np.zeros(n_pose * 6 + n_pts * 3, dtype=np.float64)
    for i, fid in enumerate(window_ids[1:]):
        T = world.keyframes[fid].T_cw
        x0[i * 6 : i * 6 + 3] = Rotation.from_matrix(T[:3, :3]).as_rotvec()
        x0[i * 6 + 3 : i * 6 + 6] = T[:3, 3]
    base = n_pose * 6
    for j, (mp, _) in enumerate(points):
        x0[base + j * 3 : base + j * 3 + 3] = mp.Xw

    T_fixed = world.keyframes[window_ids[0]].T_cw.copy()

    def unpack(x: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
        poses = {window_ids[0]: T_fixed}
        for i, fid in enumerate(window_ids[1:]):
            r = x[i * 6 : i * 6 + 3]
            t = x[i * 6 + 3 : i * 6 + 6]
            poses[fid] = pose_matrix(Rotation.from_rotvec(r).as_matrix(), t)
        Xs = x[base:].reshape(-1, 3)
        return poses, Xs

    def residual(x: np.ndarray) -> np.ndarray:
        poses, Xs = unpack(x)
        errs = []
        for j, (mp, obs) in enumerate(points):
            X = Xs[j]
            for fid, kp_idx in obs:
                frame = world.keyframes[fid]
                uv = np.array(frame.kps[kp_idx].pt, dtype=np.float64)
                Xc = transform_points(poses[fid], X.reshape(1, 3))[0]
                if Xc[2] < 1e-4:
                    errs.extend([10.0, 10.0])
                    continue
                pred = camera.project(Xc.reshape(1, 3))[0]
                errs.extend((pred - uv).tolist())
        return np.asarray(errs, dtype=np.float64)

    sol = least_squares(
        residual,
        x0,
        method="trf",
        loss="huber",
        f_scale=2.0,
        max_nfev=max_nfev,
        xtol=1e-5,
        ftol=1e-5,
    )
    poses, Xs = unpack(sol.x)
    for fid, T in poses.items():
        world.keyframes[fid].T_cw = T
    for j, (mp, _) in enumerate(points):
        mp.Xw = Xs[j]
    return {
        "success": bool(sol.cost < np.inf),
        "cost": float(sol.cost),
        "n_keyframes": len(window_ids),
        "n_points": n_pts,
        "nfev": int(sol.nfev),
    }
