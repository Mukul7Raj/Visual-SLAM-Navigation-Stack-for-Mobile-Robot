# Visual SLAM & Navigation Stack for Mobile Robot

Monocular **visual SLAM** (ORB features, two-view init, PnP tracking, sliding-window **bundle adjustment**, loop detection) plus a **navigation** layer that turns the sparse map into an inflated occupancy grid, runs **A\***, and tracks the path with a **differential-drive** controller.

Python / OpenCV implementation you can run on Windows or Linux. Gazebo + ROS 2 launch files live under `sim/gazebo` for a TurtleBot-style indoor world.

Repository layout matches a talking-track for intern / new-grad robotics interviews: every module is small enough to open live and explain.

## Architecture

```mermaid
flowchart LR
  cam[Monocular frames] --> orb[ORB extract + ratio match]
  orb --> init[Essential matrix init]
  init --> track[PnP-RANSAC tracking]
  track --> kf[Keyframe + triangulate]
  kf --> ba[Local Huber BA]
  ba --> loop[Loop detect vs old KFs]
  loop --> map[Sparse map + poses]
  map --> grid[Height-filtered occupancy]
  grid --> astar[A* global plan]
  astar --> dd[Unicycle v, omega]
```

| Stage | What it does | File |
| --- | --- | --- |
| Camera | Pinhole `K`, project / normalize | `src/vslam/camera.py` |
| Features | ORB + Lowe ratio + grid NMS | `src/vslam/features.py` |
| Geometry | Essential, triangulate, PnP, Sim(3) Umeyama | `src/vslam/geometry.py` |
| Frontend | Init, track, keyframes | `src/vslam/slam.py` |
| Local mapping | Sliding-window reprojection BA | `src/vslam/bundle_adjustment.py` |
| Loop closure | Match to old keyframes, snap pose in map frame | `src/vslam/loop_closure.py` |
| Metrics | ATE RMSE (Sim3) and RPE | `src/vslam/evaluation.py` |
| Nav | Sparse points → grid → A* → unicycle | `src/vslam/navigation/` |
| Sim | Textured indoor room + GT trajectory | `src/vslam/sim/indoor.py` |

This is **not** a drop-in of ORB-SLAM3 (no DBoW2 vocabulary, no covisibility graph thread). It follows the same *ideas* so you can contrast it with Mur-Artal / Campos papers in an interview.

## Quick start

```bash
python -m pip install -e ".[dev]"
pytest
python -m vslam.cli --config configs/default.yaml --out outputs/demo --frames 80
```

Artifacts in `outputs/demo/`:

- `trajectory_ate.png` — ground truth vs Sim(3)-aligned SLAM
- `sparse_map.png` — 3D points and keyframes
- `occupancy_plan.png` — grid + A* path
- `metrics.json` — ATE / RPE / keyframe counts
- `sequence/` — RGB frames + TUM-format `groundtruth.txt`

## Evaluation

Monocular SLAM has a **free scale**. Absolute Trajectory Error is computed after Umeyama Sim(3) alignment of camera centers (same protocol as TUM RGB-D / ORB-SLAM papers). Relative Pose Error uses a fixed frame delta and does **not** absorb scale the same way — useful for talking about drift.

On the synthetic indoor loop (textured 8×6 m room, wheeled rectangle path), you should see:

- successful two-view init within the first ~10 frames
- tens of keyframes and hundreds of map points
- ATE typically well below a meter after alignment (run-dependent)

If tracking is lost, `lost_frames` in `metrics.json` goes up; the frontend holds the last pose rather than fabricating motion.

## Navigation stack

1. Keep map points with height in `[height_min, height_max]` (reject floor/ceiling clutter).
2. Histogram into a 2D grid; cells with enough hits are **occupied**, the rest of the bounding box is **free** (optimistic, typical for sparse vision maps).
3. Inflate obstacles by robot radius.
4. A* (8-connected) from the first camera center toward a goal offset.
5. Pure-pursuit-style unicycle commands `(v, ω)` for a differential-drive base.

## Gazebo / hardware

See [`sim/gazebo/README.md`](sim/gazebo/README.md). Calibrate the camera so `configs/default.yaml` matches the real `K`. Replay a bag or subscribe to `/camera/image_raw` and call `VisualSLAM.process`.

## Interview notes

Be ready to derive or sketch:

1. **Why ORB** — FAST corners + BRIEF binary descriptors, Hamming match, good for CPU robots.
2. **Essential matrix** — `E = [t]_× R`, five-point / RANSAC, `recoverPose` cheirality.
3. **Gauge freedom** — monocular scale; first pose fixed in BA; scale from first baseline (unit `t` from `recoverPose`).
4. **PnP vs two-view** — after the map exists, pose is 3D–2D (`solvePnPRansac`), not 2D–2D.
5. **Bundle adjustment** — state = poses + points, residual = `π(T X) − uv`, Huber loss, oldest window pose locked.
6. **Loop closure** — place recognition + geometric verify; full ORB-SLAM uses DBoW2 + pose-graph Sim(3). Here: descriptor match to old keyframes + PnP.
7. **ATE vs RPE** — global consistency vs local drift.
8. **Sparse map → occupancy** — why inflation, height gates, and why this is weaker than a lidar occupancy grid.

Honest limitations: single thread, no IMU, no relocalization bag-of-words, BA is dense SciPy (`least_squares`) not g2o/gtsam, synthetic textures are richer than a blank wall.

## Project layout

```
configs/default.yaml
src/vslam/           # library
scripts/run_pipeline.py
sim/gazebo/          # world + ROS 2 launch
tests/
```

## Resume bullets (backed by this repo)

- Implemented a monocular visual SLAM pipeline using ORB features, essential-matrix initialization, PnP tracking, and sliding-window bundle adjustment for indoor localization and mapping.
- Built a navigation layer that converts the sparse map to an inflated occupancy grid, plans with A*, and tracks with a differential-drive controller.
- Evaluated trajectory error (ATE / RPE after Sim(3) alignment) on a ground-truth indoor sequence; provided Gazebo/ROS 2 bringup for a wheeled platform.
