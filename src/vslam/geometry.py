"""SE(3) helpers, two-view geometry, triangulation, and PnP."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from vslam.camera import Camera


def pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ X.T + T[:3, 3:4]).T


def se3_log(T: np.ndarray) -> np.ndarray:
    r = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.hstack([r, T[:3, 3]])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    R = Rotation.from_rotvec(xi[:3]).as_matrix()
    return pose_matrix(R, xi[3:])


def triangulation_angles_deg(C1: np.ndarray, C2: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Parallax angle at each triangulated point."""
    v1 = X - C1.reshape(1, 3)
    v2 = X - C2.reshape(1, 3)
    n1 = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    n1 = np.clip(n1, 1e-12, None)
    cosang = np.sum(v1 * v2, axis=1) / n1
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def recover_two_view(
    uv1: np.ndarray,
    uv2: np.ndarray,
    camera: Camera,
    ransac_thresh_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Estimate T_21 (frame2 from frame1) via the essential matrix.

    Returns (T_21, mask, inlier indices) or None.
    """
    if len(uv1) < 8:
        return None
    pts1 = np.ascontiguousarray(uv1, dtype=np.float64)
    pts2 = np.ascontiguousarray(uv2, dtype=np.float64)
    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        camera.K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_thresh_px,
    )
    if E is None or mask is None:
        return None
    # findEssentialMat can stack several 3x3 hypotheses; recoverPose needs 3x3.
    best_n, best = 0, None
    for i in range(0, E.shape[0], 3):
        Ei = np.ascontiguousarray(E[i : i + 3], dtype=np.float64)
        n, R, t, pose_mask = cv2.recoverPose(Ei, pts1, pts2, camera.K, mask=mask.copy())
        if n > best_n:
            best_n, best = n, (R, t, pose_mask)
    if best is None or best_n < 20:
        return None
    R, t, pose_mask = best
    inliers = np.where(pose_mask.ravel() > 0)[0]
    T_21 = pose_matrix(R, t.reshape(3))
    return T_21, pose_mask, inliers


def triangulate(
    T_cw1: np.ndarray,
    T_cw2: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    # Normalized rays + [R|t] (not K[R|t]) is stable across OpenCV versions.
    n1 = np.ascontiguousarray(camera.normalize(np.asarray(uv1, dtype=np.float64)).T)
    n2 = np.ascontiguousarray(camera.normalize(np.asarray(uv2, dtype=np.float64)).T)
    P1 = np.ascontiguousarray(T_cw1[:3, :], dtype=np.float64)
    P2 = np.ascontiguousarray(T_cw2[:3, :], dtype=np.float64)
    Xh = cv2.triangulatePoints(P1, P2, n1, n2)
    w = Xh[3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return (Xh[:3] / w).T


def pnp_ransac(
    Xw: np.ndarray,
    uv: np.ndarray,
    camera: Camera,
    iterations: int = 100,
    reproj_thresh: float = 4.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(Xw) < 6:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        Xw.astype(np.float32),
        uv.astype(np.float32),
        camera.K,
        None,
        iterationsCount=iterations,
        reprojectionError=reproj_thresh,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 10:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = pose_matrix(R, tvec.reshape(3))
    return T, inliers.ravel()


def reprojection_errors(T_cw: np.ndarray, Xw: np.ndarray, uv: np.ndarray, camera: Camera) -> np.ndarray:
    Xc = transform_points(T_cw, Xw)
    pred = camera.project(Xc)
    return np.linalg.norm(pred - uv, axis=1)


def camera_center(T_cw: np.ndarray) -> np.ndarray:
    return invert_pose(T_cw)[:3, 3]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """Sim(3)/SE(3) alignment mapping src -> dst. Used for ATE and loop correction."""
    assert src.shape == dst.shape
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    if with_scale:
        var_s = np.sum(src_c**2) / n
        scale = float(np.sum(S) / max(var_s, 1e-12))
    else:
        scale = 1.0
    t = mu_d - scale * R @ mu_s
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T
