"""ORB extraction and Lowe-ratio matching."""

from __future__ import annotations

import cv2
import numpy as np


class ORBExtractor:
    def __init__(
        self,
        n_features: int = 1200,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        ratio_test: float = 0.72,
        max_hamming: int = 64,
    ) -> None:
        self.ratio_test = ratio_test
        self.max_hamming = max_hamming
        self.orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=12,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect(self, image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            return [], np.zeros((0, 32), dtype=np.uint8)
        return list(kps), desc

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        kps1: list[cv2.KeyPoint] | None = None,
        kps2: list[cv2.KeyPoint] | None = None,
    ) -> list[cv2.DMatch]:
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        raw = self.matcher.knnMatch(desc1, desc2, k=2)
        good: list[cv2.DMatch] = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance and m.distance <= self.max_hamming:
                good.append(m)
        if kps1 is not None and kps2 is not None:
            good = _grid_nms(good, kps1, kps2)
        return good


def _grid_nms(matches: list[cv2.DMatch], kps1: list[cv2.KeyPoint], kps2: list[cv2.KeyPoint]) -> list[cv2.DMatch]:
    """Keep the strongest match per 32px cell in the first image to reduce clustering."""
    best: dict[tuple[int, int], cv2.DMatch] = {}
    for m in matches:
        x, y = kps1[m.queryIdx].pt
        key = (int(x) // 32, int(y) // 32)
        prev = best.get(key)
        if prev is None or m.distance < prev.distance:
            best[key] = m
    return list(best.values())


def keypoints_xy(kps: list[cv2.KeyPoint]) -> np.ndarray:
    if not kps:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array([kp.pt for kp in kps], dtype=np.float64)
