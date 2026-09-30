import numpy as np

from vslam.camera import Camera
from vslam.geometry import pose_matrix, recover_two_view, triangulate, umeyama


def _cam() -> Camera:
    return Camera(525, 525, 320, 240, 640, 480)


def test_umeyama_recovers_sim3():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    t = np.array([0.4, -0.2, 0.1])
    dst = 1.7 * (src @ R.T) + t
    S = umeyama(src, dst, with_scale=True)
    pred = (S @ np.c_[src, np.ones(len(src))].T).T[:, :3]
    assert np.mean(np.linalg.norm(pred - dst, axis=1)) < 1e-6


def test_two_view_and_triangulation():
    cam = _cam()
    rng = np.random.default_rng(1)
    X = rng.uniform([-1, -1, 3], [1, 1, 6], size=(80, 3))
    T1 = np.eye(4)
    T2 = pose_matrix(np.eye(3), np.array([0.25, 0.0, 0.0]))
    uv1 = cam.project((T1[:3, :3] @ X.T + T1[:3, 3:4]).T)
    uv2 = cam.project((T2[:3, :3] @ X.T + T2[:3, 3:4]).T)
    rec = recover_two_view(uv1, uv2, cam)
    assert rec is not None
    T21, _, inl = rec
    Xh = triangulate(T1, T21, uv1[inl], uv2[inl], cam)
    t = T21[:3, 3] / np.linalg.norm(T21[:3, 3])
    assert abs(t[0]) > 0.9
    Xc = (T21[:3, :3] @ Xh.T + T21[:3, 3:4]).T
    assert np.median(Xh[:, 2]) > 0
    assert np.median(Xc[:, 2]) > 0
