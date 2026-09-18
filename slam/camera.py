"""Pinhole camera model and intrinsics resolution.

Monocular SLAM on an arbitrary upload has no calibration available, so
`resolve_intrinsics` implements the fallback chain from IMPLEMENTATION_PLAN.md
section 2.2 and records which path was taken so the UI can be honest about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

IntrinsicsSource = Literal["user", "metadata", "fov_heuristic"]

DEFAULT_HFOV_DEG = 60.0
"""Typical horizontal field of view for phone / webcam footage."""


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics for a single lens. No distortion model: consumer video
    is usually already rectified by the encoder, and estimating distortion from
    an uncalibrated clip is less stable than ignoring it."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: IntrinsicsSource = "fov_heuristic"

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.width / self.fx))

    def scaled(self, scale: float) -> Camera:
        """Intrinsics for a frame resized by `scale`."""
        return Camera(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=int(round(self.width * scale)),
            height=int(round(self.height * scale)),
            source=self.source,
        )

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """Project Nx3 camera-frame points to Nx2 pixels. Caller filters depth."""
        pts = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
        z = np.where(np.abs(pts[:, 2]) < 1e-12, 1e-12, pts[:, 2])
        return np.stack([self.fx * pts[:, 0] / z + self.cx,
                         self.fy * pts[:, 1] / z + self.cy], axis=1)

    def unproject(self, pixels: np.ndarray, depth: np.ndarray | float = 1.0) -> np.ndarray:
        """Back-project Nx2 pixels to Nx3 camera-frame rays at `depth`."""
        px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
        d = (np.full(len(px), float(depth)) if isinstance(depth, (int | float))
             else np.asarray(depth, dtype=np.float64).reshape(-1))
        return np.stack([(px[:, 0] - self.cx) / self.fx * d,
                         (px[:, 1] - self.cy) / self.fy * d,
                         d], axis=1)


def camera_from_hfov(width: int, height: int, hfov_deg: float = DEFAULT_HFOV_DEG,
                     source: IntrinsicsSource = "fov_heuristic") -> Camera:
    """Build intrinsics from an assumed horizontal field of view."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    if not 5.0 < hfov_deg < 175.0:
        raise ValueError(f"implausible horizontal FOV: {hfov_deg}")
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Camera(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                  width=width, height=height, source=source)


def resolve_intrinsics(width: int, height: int, *,
                       focal_px: float | None = None,
                       hfov_deg: float | None = None,
                       metadata_focal_px: float | None = None) -> Camera:
    """Resolve intrinsics using the documented fallback chain.

    Priority: explicit user focal length -> user FOV -> container metadata ->
    the 60 deg HFOV heuristic. `Camera.source` records which applied.
    """
    if focal_px is not None:
        if focal_px <= 0:
            raise ValueError("focal length must be positive")
        return Camera(fx=focal_px, fy=focal_px, cx=width / 2.0, cy=height / 2.0,
                      width=width, height=height, source="user")
    if hfov_deg is not None:
        return camera_from_hfov(width, height, hfov_deg, source="user")
    if metadata_focal_px is not None and metadata_focal_px > 0:
        return Camera(fx=metadata_focal_px, fy=metadata_focal_px,
                      cx=width / 2.0, cy=height / 2.0,
                      width=width, height=height, source="metadata")
    return camera_from_hfov(width, height, DEFAULT_HFOV_DEG)
