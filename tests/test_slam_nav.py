import numpy as np

from vslam.camera import Camera
from vslam.features import ORBExtractor
from vslam.navigation import astar, path_to_unicycle, points_to_occupancy
from vslam.sim import generate_indoor_sequence
from vslam.slam import SLAMConfig, VisualSLAM
from vslam.evaluation import ate_rmse


def test_occupancy_and_astar():
    X = []
    for x in np.linspace(0, 4, 40):
        X.append([x, 0.0, 0.8])
        X.append([x, 3.0, 0.8])
    for y in np.linspace(0, 3, 30):
        X.append([0.0, y, 0.8])
        X.append([4.0, y, 0.8])
    grid = points_to_occupancy(np.asarray(X), resolution=0.1, inflate_radius=0.15, occ_count=1)
    path = astar(grid, np.array([1.0, 1.5]), np.array([3.0, 1.5]))
    assert len(path) > 5
    cmds = path_to_unicycle(path, dt=0.1)
    assert len(cmds) > 5
    assert all("v" in c and "omega" in c for c in cmds[:3])


def test_slam_initializes_on_synthetic_room():
    cam = Camera(525, 525, 320, 240, 640, 480)
    seq = generate_indoor_sequence(cam, n_frames=36, seed=3, dt=0.15)
    slam = VisualSLAM(
        cam,
        ORBExtractor(n_features=900, ratio_test=0.75),
        SLAMConfig(ba_window=4, ba_max_nfev=15, min_init_matches=40, min_track_inliers=15),
    )
    for i, img in enumerate(seq.images):
        slam.process(img, frame_id=i)
    assert slam.stats.initialized
    assert slam.stats.n_keyframes >= 2
    assert slam.stats.n_points >= 20
    est = [T for _, T in slam.stats.tracked_poses]
    gt = [seq.T_cw[fid] for fid, _ in slam.stats.tracked_poses]
    ate = ate_rmse(est, gt)
    assert ate["n"] >= 10
    assert ate["ate_rmse"] < 1.5
