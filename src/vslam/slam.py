"""Monocular visual SLAM frontend + local mapping.

Pipeline (ORB-SLAM style, single-threaded for interview clarity):
  1. ORB extract
  2. Two-view initialization (essential matrix + triangulation)
  3. Track map points with descriptor matching + PnP-RANSAC
  4. Keyframe insertion and new-point triangulation
  5. Sliding-window bundle adjustment
  6. Loop detection against old keyframes
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vslam.bundle_adjustment import local_bundle_adjust
from vslam.camera import Camera
from vslam.features import ORBExtractor, keypoints_xy
from vslam.geometry import (
    camera_center,
    invert_pose,
    pnp_ransac,
    recover_two_view,
    reprojection_errors,
    se3_log,
    transform_points,
    triangulate,
    triangulation_angles_deg,
)
from vslam.loop_closure import detect_loop
from vslam.map import Frame, Map


@dataclass
class SLAMConfig:
    min_init_matches: int = 80
    min_init_parallax_deg: float = 1.2
    min_track_inliers: int = 25
    keyframe_min_translation: float = 0.08
    keyframe_min_rotation_deg: float = 8.0
    keyframe_min_tracked_ratio: float = 0.65
    ba_window: int = 6
    ba_max_nfev: int = 40
    loop_min_gap: int = 18
    loop_min_inliers: int = 35


@dataclass
class SLAMStats:
    initialized: bool = False
    n_frames: int = 0
    n_keyframes: int = 0
    n_points: int = 0
    n_lost: int = 0
    n_loops: int = 0
    last_inliers: int = 0
    ba_reports: list[dict] = field(default_factory=list)
    tracked_poses: list[tuple[int, np.ndarray]] = field(default_factory=list)


class VisualSLAM:
    def __init__(self, camera: Camera, extractor: ORBExtractor, cfg: SLAMConfig | None = None) -> None:
        self.camera = camera
        self.extractor = extractor
        self.cfg = cfg or SLAMConfig()
        self.map = Map()
        self.stats = SLAMStats()
        self._prev: Frame | None = None
        self._ref: Frame | None = None
        self._pending_init: Frame | None = None

    def process(self, image: np.ndarray, timestamp: float = 0.0, frame_id: int | None = None) -> Frame:
        fid = self.stats.n_frames if frame_id is None else frame_id
        kps, desc = self.extractor.detect(image)
        frame = Frame(frame_id=fid, image=image, kps=kps, descriptors=desc, timestamp=timestamp)
        self.stats.n_frames += 1

        if not self.stats.initialized:
            self._try_initialize(frame)
        else:
            ok = self._track(frame)
            if not ok:
                self.stats.n_lost += 1
                if self._prev is not None and self._prev.T_cw is not None:
                    frame.T_cw = self._prev.T_cw.copy()
            else:
                if self._should_insert_keyframe(frame):
                    self._insert_keyframe(frame)
                    report = local_bundle_adjust(
                        self.map, self.camera, window=self.cfg.ba_window, max_nfev=self.cfg.ba_max_nfev
                    )
                    self.stats.ba_reports.append(report)
                    self._maybe_loop(frame)

        if frame.T_cw is not None:
            self.stats.tracked_poses.append((frame.frame_id, frame.T_cw.copy()))
        self._prev = frame
        self.stats.n_keyframes = len(self.map.keyframes)
        self.stats.n_points = len(self.map.active_points())
        return frame

    def _try_initialize(self, frame: Frame) -> None:
        if self._pending_init is None:
            if len(frame.kps) >= self.cfg.min_init_matches:
                self._pending_init = frame
            return
        ref = self._pending_init
        matches = self.extractor.match(ref.descriptors, frame.descriptors, ref.kps, frame.kps)
        if len(matches) < self.cfg.min_init_matches:
            # Keep the original reference so baseline can grow.
            return
        uv1 = keypoints_xy(ref.kps)[[m.queryIdx for m in matches]]
        uv2 = keypoints_xy(frame.kps)[[m.trainIdx for m in matches]]
        recovered = recover_two_view(uv1, uv2, self.camera)
        if recovered is None:
            return
        T_21, _, inliers = recovered
        uv1, uv2 = uv1[inliers], uv2[inliers]
        match_in = [matches[i] for i in inliers]
        T_cw1 = np.eye(4)
        T_cw2 = T_21
        X = triangulate(T_cw1, T_cw2, uv1, uv2, self.camera)
        C1, C2 = camera_center(T_cw1), camera_center(T_cw2)
        angles = triangulation_angles_deg(C1, C2, X)
        Xc1 = transform_points(T_cw1, X)
        Xc2 = transform_points(T_cw2, X)
        err1 = reprojection_errors(T_cw1, X, uv1, self.camera)
        err2 = reprojection_errors(T_cw2, X, uv2, self.camera)
        good = (
            (Xc1[:, 2] > 0.05)
            & (Xc2[:, 2] > 0.05)
            & (err1 < 3.0)
            & (err2 < 3.0)
            & (angles > self.cfg.min_init_parallax_deg)
            & (angles < 40.0)
        )
        if int(np.count_nonzero(good)) < 50:
            return
        ref.T_cw = T_cw1
        frame.T_cw = T_cw2
        self.map.add_keyframe(ref)
        self.map.add_keyframe(frame)
        for i, ok in enumerate(good):
            if not ok:
                continue
            m = match_in[i]
            mp = self.map.add_point(X[i], ref.descriptors[m.queryIdx])
            self.map.observe(mp.mp_id, ref, m.queryIdx)
            self.map.observe(mp.mp_id, frame, m.trainIdx)
        self._ref = frame
        self.stats.initialized = True
        self.stats.last_inliers = int(np.count_nonzero(good))

    def _track(self, frame: Frame) -> bool:
        ref = self._ref if self._ref is not None else self._prev
        if ref is None or ref.T_cw is None:
            return False
        matches = self.extractor.match(ref.descriptors, frame.descriptors, ref.kps, frame.kps)
        Xw, uv, pairs = [], [], []
        for m in matches:
            mp_id = ref.mp_ids.get(m.queryIdx)
            if mp_id is None:
                continue
            mp = self.map.points.get(mp_id)
            if mp is None or mp.outlier:
                continue
            Xw.append(mp.Xw)
            uv.append(frame.kps[m.trainIdx].pt)
            pairs.append((mp_id, m.trainIdx))
        if len(Xw) < self.cfg.min_track_inliers:
            # Fallback: match against last keyframe descriptors of all map points
            return self._track_against_map(frame)
        solved = pnp_ransac(np.asarray(Xw), np.asarray(uv, dtype=np.float64), self.camera)
        if solved is None:
            return self._track_against_map(frame)
        T, inliers = solved
        frame.T_cw = T
        for i in inliers:
            mp_id, kp_idx = pairs[i]
            frame.mp_ids[kp_idx] = mp_id
        self.stats.last_inliers = len(inliers)
        return len(inliers) >= self.cfg.min_track_inliers

    def _track_against_map(self, frame: Frame) -> bool:
        kfs = list(self.map.keyframes.values())[-4:]
        Xw, uv, pairs = [], [], []
        seen = set()
        for kf in kfs:
            matches = self.extractor.match(kf.descriptors, frame.descriptors)
            for m in matches:
                mp_id = kf.mp_ids.get(m.queryIdx)
                if mp_id is None or mp_id in seen:
                    continue
                mp = self.map.points.get(mp_id)
                if mp is None or mp.outlier:
                    continue
                seen.add(mp_id)
                Xw.append(mp.Xw)
                uv.append(frame.kps[m.trainIdx].pt)
                pairs.append((mp_id, m.trainIdx))
        if len(Xw) < self.cfg.min_track_inliers:
            return False
        solved = pnp_ransac(np.asarray(Xw), np.asarray(uv, dtype=np.float64), self.camera)
        if solved is None:
            return False
        T, inliers = solved
        frame.T_cw = T
        for i in inliers:
            mp_id, kp_idx = pairs[i]
            frame.mp_ids[kp_idx] = mp_id
        self.stats.last_inliers = len(inliers)
        return True

    def _should_insert_keyframe(self, frame: Frame) -> bool:
        if frame.T_cw is None or self._ref is None or self._ref.T_cw is None:
            return False
        rel = frame.T_cw @ invert_pose(self._ref.T_cw)
        xi = se3_log(rel)
        trans = float(np.linalg.norm(xi[3:]))
        rot_deg = float(np.degrees(np.linalg.norm(xi[:3])))
        tracked_ratio = len(frame.mp_ids) / max(len(frame.kps), 1)
        if trans > self.cfg.keyframe_min_translation or rot_deg > self.cfg.keyframe_min_rotation_deg:
            return True
        if tracked_ratio < self.cfg.keyframe_min_tracked_ratio and trans > 0.03:
            return True
        return False

    def _insert_keyframe(self, frame: Frame) -> None:
        assert frame.T_cw is not None and self._ref is not None and self._ref.T_cw is not None
        ref = self._ref
        matches = self.extractor.match(ref.descriptors, frame.descriptors, ref.kps, frame.kps)
        uv1, uv2, idx_pairs = [], [], []
        for m in matches:
            if m.trainIdx in frame.mp_ids:
                continue
            if m.queryIdx in ref.mp_ids:
                # already a map point — just add observation
                mp_id = ref.mp_ids[m.queryIdx]
                self.map.observe(mp_id, frame, m.trainIdx)
                continue
            uv1.append(ref.kps[m.queryIdx].pt)
            uv2.append(frame.kps[m.trainIdx].pt)
            idx_pairs.append((m.queryIdx, m.trainIdx))
        self.map.add_keyframe(frame)
        if len(uv1) >= 12:
            X = triangulate(
                ref.T_cw,
                frame.T_cw,
                np.asarray(uv1, dtype=np.float64),
                np.asarray(uv2, dtype=np.float64),
                self.camera,
            )
            C1, C2 = camera_center(ref.T_cw), camera_center(frame.T_cw)
            angles = triangulation_angles_deg(C1, C2, X)
            Xc1 = transform_points(ref.T_cw, X)
            Xc2 = transform_points(frame.T_cw, X)
            err1 = reprojection_errors(ref.T_cw, X, np.asarray(uv1), self.camera)
            err2 = reprojection_errors(frame.T_cw, X, np.asarray(uv2), self.camera)
            good = (
                (Xc1[:, 2] > 0.05)
                & (Xc2[:, 2] > 0.05)
                & (err1 < 3.0)
                & (err2 < 3.0)
                & (angles > 0.8)
                & (angles < 45.0)
                & np.isfinite(X).all(axis=1)
            )
            for i, ok in enumerate(good):
                if not ok:
                    continue
                q, t = idx_pairs[i]
                mp = self.map.add_point(X[i], ref.descriptors[q])
                self.map.observe(mp.mp_id, ref, q)
                self.map.observe(mp.mp_id, frame, t)
        self._ref = frame

    def _maybe_loop(self, frame: Frame) -> None:
        det = detect_loop(
            frame,
            self.map,
            self.extractor,
            self.camera,
            min_gap=self.cfg.loop_min_gap,
            min_inliers=self.cfg.loop_min_inliers,
        )
        if det is None:
            return
        # PnP pose is already in the map frame of the old keyframe — snap tracking to it.
        frame.T_cw = det["T_cw_loop"]
        self.map.pose_graph.append((det["candidate_id"], frame.frame_id))
        self.stats.n_loops += 1
        local_bundle_adjust(self.map, self.camera, window=min(12, len(self.map.keyframes)), max_nfev=self.cfg.ba_max_nfev)
