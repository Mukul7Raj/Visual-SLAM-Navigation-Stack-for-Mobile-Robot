"""Place-recognition style loop detection on ORB descriptors.

This is a lightweight stand-in for DBoW2: candidate keyframes are scored
by inlier match count, then a PnP/Sim(3) correction is applied if a loop
is accepted. Enough to discuss loop closure in interviews without a
pretrained visual vocabulary.
"""

from __future__ import annotations

import numpy as np

from vslam.camera import Camera
from vslam.features import ORBExtractor
from vslam.geometry import pnp_ransac, umeyama, invert_pose, transform_points
from vslam.map import Frame, Map


def detect_loop(
    query: Frame,
    world: Map,
    extractor: ORBExtractor,
    camera: Camera,
    min_gap: int = 18,
    min_inliers: int = 35,
) -> dict | None:
    if query.T_cw is None:
        return None
    best = None
    for fid, kf in world.keyframes.items():
        if query.frame_id - fid < min_gap:
            continue
        matches = extractor.match(query.descriptors, kf.descriptors)
        if len(matches) < min_inliers:
            continue
        Xw, uv = [], []
        for m in matches:
            mp_id = kf.mp_ids.get(m.trainIdx)
            if mp_id is None or mp_id not in world.points:
                continue
            mp = world.points[mp_id]
            if mp.outlier:
                continue
            Xw.append(mp.Xw)
            uv.append(query.kps[m.queryIdx].pt)
        if len(Xw) < min_inliers:
            continue
        solved = pnp_ransac(np.asarray(Xw), np.asarray(uv, dtype=np.float64), camera)
        if solved is None:
            continue
        T_loop, inliers = solved
        if len(inliers) < min_inliers:
            continue
        score = len(inliers)
        if best is None or score > best["score"]:
            best = {
                "candidate_id": fid,
                "T_cw_loop": T_loop,
                "score": score,
                "Xw": np.asarray(Xw)[inliers],
            }
    return best


def apply_loop_correction(query: Frame, detection: dict, world: Map) -> np.ndarray:
    """Align the current trajectory to the loop-closure pose via Sim(3).

    Uses camera centers of recent keyframes vs. the implied centers from
    the relative transform, then applies Umeyama to all keyframes and points.
    """
    T_cw_loop = detection["T_cw_loop"]
    T_wc_curr = invert_pose(query.T_cw)
    T_wc_loop = invert_pose(T_cw_loop)
    src = T_wc_curr[:3, 3].reshape(1, 3)
    dst = T_wc_loop[:3, 3].reshape(1, 3)
    # Include a few recent centers warped by the same relative delta.
    delta = T_wc_loop @ invert_pose(T_wc_curr)
    src_pts = []
    dst_pts = []
    for fid, kf in world.keyframes.items():
        C = invert_pose(kf.T_cw)[:3, 3]
        src_pts.append(C)
        C2 = (delta @ np.append(C, 1.0))[:3]
        dst_pts.append(C2)
    src_pts.append(src[0])
    dst_pts.append(dst[0])
    S = umeyama(np.asarray(src_pts), np.asarray(dst_pts), with_scale=True)
    _apply_sim3(world, S)
    query.T_cw = T_cw_loop
    world.pose_graph.append((detection["candidate_id"], query.frame_id))
    return S


def _apply_sim3(world: Map, S: np.ndarray) -> None:
    for kf in world.keyframes.values():
        T_wc = invert_pose(kf.T_cw)
        C = transform_points(S, T_wc[:3, 3].reshape(1, 3))[0]
        R = S[:3, :3] / (np.linalg.det(S[:3, :3]) ** (1.0 / 3.0) + 1e-12)
        # Keep rotation from Sim(3) linear part (orthonormalize)
        u, _, vt = np.linalg.svd(R)
        R = u @ vt
        T_wc_new = np.eye(4)
        T_wc_new[:3, :3] = R @ T_wc[:3, :3]
        T_wc_new[:3, 3] = C
        from vslam.geometry import invert_pose as inv

        kf.T_cw = inv(T_wc_new)
    for mp in world.points.values():
        mp.Xw = transform_points(S, mp.Xw.reshape(1, 3))[0]
