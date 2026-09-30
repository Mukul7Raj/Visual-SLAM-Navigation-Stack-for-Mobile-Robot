"""Indoor textured-room renderer + ground-truth camera trajectory.

Produces a monocular sequence a wheeled robot would see while looping
an office-sized rectangle. No Gazebo required for CI / interviews on
Windows; the same intrinsics are used in the ROS 2 launch files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vslam.camera import Camera
from vslam.geometry import invert_pose, pose_matrix, transform_points


@dataclass
class Sequence:
    images: list[np.ndarray]
    T_cw: list[np.ndarray]
    timestamps: list[float]
    camera: Camera

    def save(self, root: Path) -> None:
        root = Path(root)
        img_dir = root / "rgb"
        img_dir.mkdir(parents=True, exist_ok=True)
        with open(root / "groundtruth.txt", "w", encoding="utf-8") as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for i, (img, T, ts) in enumerate(zip(self.images, self.T_cw, self.timestamps)):
                cv2.imwrite(str(img_dir / f"{i:05d}.png"), img)
                Twc = invert_pose(T)
                t = Twc[:3, 3]
                q = _rot_to_quat(Twc[:3, :3])
                f.write(f"{ts:.5f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()  # x,y,z,w
    return q


def _wall_texture(rng: np.random.Generator, size: int = 640) -> np.ndarray:
    tex = rng.integers(50, 200, size=(size, size), dtype=np.uint8)
    checker = np.indices((size, size)).sum(axis=0) // 24 % 2
    tex = np.where(checker, np.clip(tex.astype(np.int16) + 40, 0, 255), tex).astype(np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    for _ in range(80):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        w, h = int(rng.integers(12, 70)), int(rng.integers(12, 70))
        color = int(rng.integers(10, 255))
        cv2.rectangle(tex, (x, y), (min(size - 1, x + w), min(size - 1, y + h)), color, -1)
    for _ in range(40):
        x, y = int(rng.integers(0, size)), int(rng.integers(0, size))
        r = int(rng.integers(6, 28))
        cv2.circle(tex, (x, y), r, int(rng.integers(0, 255)), -1)
    for i in range(12):
        cv2.putText(
            tex,
            f"W{i}{int(rng.integers(10, 99))}",
            (int(rng.integers(10, size - 80)), int(rng.integers(30, size - 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            int(rng.integers(0, 255)),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)


def _quads(room: tuple[float, float, float]) -> list[tuple[np.ndarray, np.ndarray]]:
    L, W, H = room
    # (4,3) world corners in order matching texture: tl, tr, br, bl in image space later
    walls = [
        np.array([[0, 0, H], [L, 0, H], [L, 0, 0], [0, 0, 0]], dtype=np.float64),
        np.array([[L, 0, H], [L, W, H], [L, W, 0], [L, 0, 0]], dtype=np.float64),
        np.array([[L, W, H], [0, W, H], [0, W, 0], [L, W, 0]], dtype=np.float64),
        np.array([[0, W, H], [0, 0, H], [0, 0, 0], [0, W, 0]], dtype=np.float64),
    ]
    floor = np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0]], dtype=np.float64)
    ceiling = np.array([[0, 0, H], [0, W, H], [L, W, H], [L, 0, H]], dtype=np.float64)
    return walls + [floor, ceiling]


def _render_pose(T_cw: np.ndarray, textures: list[np.ndarray], quads: list[np.ndarray], camera: Camera) -> np.ndarray:
    img = np.full((camera.height, camera.width, 3), 18, dtype=np.uint8)
    # painter: far walls first by mean camera-z
    order = []
    for i, Q in enumerate(quads):
        Xc = transform_points(T_cw, Q)
        order.append((float(np.mean(Xc[:, 2])), i))
    for _, i in sorted(order, reverse=True):
        Q = quads[i]
        uv = camera.project(transform_points(T_cw, Q))
        Xc = transform_points(T_cw, Q)
        if np.any(Xc[:, 2] < 0.15):
            continue
        if not np.any(camera.in_image(uv, border=-20)):
            continue
        h, w = textures[i].shape[:2]
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst = uv.astype(np.float32)
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            continue
        warped = cv2.warpPerspective(textures[i], H, (camera.width, camera.height))
        mask = cv2.warpPerspective(
            np.full(textures[i].shape[:2], 255, dtype=np.uint8), H, (camera.width, camera.height)
        )
        img[mask > 0] = warped[mask > 0]
    return img


def _rectangle_path(n: int, z: float = 1.15) -> list[np.ndarray]:
    """Loop a 4.2m x 2.8m rectangle twice, camera looking along the tangent."""
    waypoints = np.array(
        [
            [1.6, 1.4, z],
            [6.4, 1.4, z],
            [6.4, 4.6, z],
            [1.6, 4.6, z],
            [1.6, 1.4, z],
        ],
        dtype=np.float64,
    )
    segs = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        segs.append((a, b, np.linalg.norm(b - a)))
    total = sum(s[2] for s in segs) * 2  # two laps
    dist = np.linspace(0, total - 1e-6, n)
    poses = []
    lap = sum(s[2] for s in segs)
    for d in dist:
        dmod = d % lap
        acc = 0.0
        origin, dest = segs[0][0], segs[0][1]
        alpha = 0.0
        for a, b, length in segs:
            if dmod <= acc + length:
                origin, dest = a, b
                alpha = (dmod - acc) / max(length, 1e-9)
                break
            acc += length
        p = origin + alpha * (dest - origin)
        tangent = dest - origin
        yaw = np.arctan2(tangent[1], tangent[0])
        # camera: x right, y down, z forward  —  world: z up
        # R_wc: world from camera
        cy, sy = np.cos(yaw), np.sin(yaw)
        # robot looks along yaw in xy, camera pitched slightly down
        pitch = np.deg2rad(8.0)
        cp, sp = np.cos(pitch), np.sin(pitch)
        # camera z-axis in world = forward (cos pitch in xy, down a bit)
        z_cam = np.array([cy * cp, sy * cp, -sp])
        z_cam /= np.linalg.norm(z_cam)
        x_cam = np.array([-sy, cy, 0.0])
        y_cam = np.cross(z_cam, x_cam)
        y_cam /= np.linalg.norm(y_cam)
        x_cam = np.cross(y_cam, z_cam)
        R_wc = np.stack([x_cam, y_cam, z_cam], axis=1)
        T_wc = pose_matrix(R_wc, p)
        poses.append(invert_pose(T_wc))
    return poses


def generate_indoor_sequence(
    camera: Camera,
    n_frames: int = 160,
    seed: int = 7,
    dt: float = 0.12,
    room: tuple[float, float, float] = (8.0, 6.0, 2.8),
) -> Sequence:
    rng = np.random.default_rng(seed)
    quads = [q for q in _quads(room)]
    textures = [_wall_texture(rng) for _ in quads]
    Ts = _rectangle_path(n_frames)
    images = [_render_pose(T, textures, quads, camera) for T in Ts]
    stamps = [i * dt for i in range(n_frames)]
    return Sequence(images=images, T_cw=Ts, timestamps=stamps, camera=camera)
