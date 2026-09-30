from __future__ import annotations

from pathlib import Path

import yaml

from vslam.camera import Camera
from vslam.features import ORBExtractor
from vslam.slam import SLAMConfig, VisualSLAM


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system(cfg: dict) -> VisualSLAM:
    camera = Camera.from_dict(cfg["camera"])
    fcfg = cfg.get("features", {})
    extractor = ORBExtractor(
        n_features=int(fcfg.get("n_features", 1200)),
        scale_factor=float(fcfg.get("scale_factor", 1.2)),
        n_levels=int(fcfg.get("n_levels", 8)),
        ratio_test=float(fcfg.get("ratio_test", 0.72)),
        max_hamming=int(fcfg.get("max_hamming", 64)),
    )
    scfg = cfg.get("slam", {})
    slam_cfg = SLAMConfig(
        min_init_matches=int(scfg.get("min_init_matches", 80)),
        min_init_parallax_deg=float(scfg.get("min_init_parallax_deg", 1.2)),
        min_track_inliers=int(scfg.get("min_track_inliers", 25)),
        keyframe_min_translation=float(scfg.get("keyframe_min_translation", 0.08)),
        keyframe_min_rotation_deg=float(scfg.get("keyframe_min_rotation_deg", 8.0)),
        keyframe_min_tracked_ratio=float(scfg.get("keyframe_min_tracked_ratio", 0.65)),
        ba_window=int(scfg.get("ba_window", 6)),
        ba_max_nfev=int(scfg.get("ba_max_nfev", 40)),
        loop_min_gap=int(scfg.get("loop_min_gap", 18)),
        loop_min_inliers=int(scfg.get("loop_min_inliers", 35)),
    )
    return VisualSLAM(camera, extractor, slam_cfg)
