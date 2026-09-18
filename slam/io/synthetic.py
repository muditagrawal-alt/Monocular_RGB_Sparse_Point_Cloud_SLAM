"""Synthetic sequence generator with exact ground truth.

Real datasets are large, slow to fetch, and their ground truth needs alignment
before it means anything. A synthetic scene gives exact known poses and
structure, so correctness tests are fast, deterministic, and unambiguous --
which makes this the primary regression signal for the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera import Camera, camera_from_hfov
from ..types import Pose


@dataclass
class SyntheticSequence:
    frames: list[np.ndarray]
    poses: list[Pose]
    """Ground-truth camera-to-world poses."""
    points: np.ndarray
    camera: Camera
    fps: float = 30.0

    @property
    def trajectory(self) -> np.ndarray:
        return np.array([p.center for p in self.poses])

    def write_video(self, path: str | Path, fps: float | None = None) -> str:
        """Encode to an mp4 so the full upload path can be exercised."""
        path = str(path)
        fps = fps or self.fps
        h, w = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {path}")
        try:
            for f in self.frames:
                writer.write(f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()
        return path


def _render(camera: Camera, pose: Pose, points: np.ndarray, colors: np.ndarray,
            inner: np.ndarray, width: int, height: int) -> np.ndarray:
    """Render 3D points as landmarks with distinctive local appearance.

    Each point is drawn as a disc with a per-point inner marker (its own
    colour, relative size and offset direction). This matters more than it
    looks: an earlier version drew uniform flat discs, and because every
    landmark then looked alike, only 2.7% of ORB descriptor matches between
    revisited views were geometrically correct and loop closure could never
    fire. Real scenes carry distinctive local texture, so the fixture must too,
    otherwise it tests the renderer rather than the SLAM system.
    """
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    cam_pts = pose.world_to_camera(points)
    in_front = cam_pts[:, 2] > 0.2
    if not in_front.any():
        return img

    idx = np.flatnonzero(in_front)
    proj = camera.project(cam_pts[in_front])
    depths = cam_pts[in_front, 2]

    order = np.argsort(-depths)   # paint far points first
    for i in order:
        x, y = proj[i]
        if not (-20 <= x < width + 20 and -20 <= y < height + 20):
            continue
        src = idx[i]
        radius = max(2, int(round(80.0 / depths[i])))
        shade = float(np.clip(1.4 - depths[i] / 22.0, 0.25, 1.0))
        base = tuple(int(c * shade) for c in colors[src])
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(img, (cx, cy), radius, base, -1, lineType=cv2.LINE_AA)

        # per-point inner marker -> a locally unique intensity pattern
        r_in = max(1, int(radius * (0.30 + 0.35 * inner[src, 3])))
        off = radius * 0.35
        ox = int(round(cx + off * np.cos(inner[src, 4] * 2 * np.pi)))
        oy = int(round(cy + off * np.sin(inner[src, 4] * 2 * np.pi)))
        inner_col = tuple(int(c * shade) for c in inner[src, :3] * 255)
        cv2.circle(img, (ox, oy), r_in, inner_col, -1, lineType=cv2.LINE_AA)

    return cv2.GaussianBlur(img, (3, 3), 0)


def make_sequence(n_frames: int = 90, width: int = 640, height: int = 360,
                  n_points: int = 1400, motion: str = "orbit",
                  loop: bool = True, seed: int = 0) -> SyntheticSequence:
    """Generate a sequence with known poses.

    `motion="orbit"` sweeps a circular arc around a point cloud; with
    `loop=True` it closes the full circle so loop-closure behaviour can be
    exercised. `motion="forward"` translates through the scene, and
    `motion="strafe"` moves sideways -- the best-conditioned case for
    monocular initialisation.
    """
    rng = np.random.default_rng(seed)
    camera = camera_from_hfov(width, height, 65.0)

    # A central cluster plus a surrounding shell. The cluster keeps structure
    # in view across a full orbit (so tracks survive and a loop can actually
    # be detected), while the shell spreads depth, which is what makes
    # triangulation well conditioned.
    pts: list[np.ndarray] = []
    n_cluster = n_points // 2
    for _ in range(n_cluster):
        pts.append(rng.normal(0.0, 1.5, 3))
    for _ in range(n_points - n_cluster):
        theta = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(7.0, 13.0)
        z = rng.uniform(-3.5, 3.5)
        pts.append([r * np.cos(theta), z, r * np.sin(theta)])
    points = np.array(pts, dtype=np.float64)
    colors = rng.integers(70, 255, (len(points), 3)).astype(np.uint8)
    # per-point inner-marker appearance: rgb (0-1), relative size, angle
    inner = rng.random((len(points), 5))

    poses: list[Pose] = []
    for i in range(n_frames):
        u = i / max(1, n_frames - 1)
        if motion == "orbit":
            # Orbit an object looking inward -- the classic object-scan loop.
            # An outward-looking orbit rotates the view so fast that KLT tracks
            # die before enough parallax accumulates to initialise at all.
            span = 2 * np.pi if loop else np.pi * 0.6
            a = span * u
            radius = 6.0
            centre = np.array([radius * np.sin(a), 0.0, -radius * np.cos(a)])
            fwd = -centre / np.linalg.norm(centre)   # look at the origin
        elif motion == "forward":
            centre = np.array([0.0, 0.0, -6.0 + 8.0 * u])
            fwd = np.array([0.0, 0.0, 1.0])
        elif motion == "strafe":
            centre = np.array([-2.5 + 5.0 * u, 0.0, 0.0])
            fwd = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"unknown motion '{motion}'")

        fwd = fwd / np.linalg.norm(fwd)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(world_up, fwd)
        n_right = np.linalg.norm(right)
        if n_right < 1e-8:      # looking straight up/down
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / n_right
        up = np.cross(fwd, right)
        # camera-to-world: columns are the camera axes in world coordinates
        R = np.column_stack([right, up, fwd])
        poses.append(Pose(R, centre))

    frames = [_render(camera, p, points, colors, inner, width, height) for p in poses]
    return SyntheticSequence(frames=frames, poses=poses, points=points,
                             camera=camera, fps=30.0)
