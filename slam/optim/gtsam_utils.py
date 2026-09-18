"""Conversions between this codebase's pose convention and GTSAM's.

GTSAM's `Pose3` is a body-to-world transform and its projection factors expect
the camera pose in world coordinates, which matches the camera-to-world `Pose`
used throughout this project. The conversion is therefore direct -- but it is
isolated here and covered by a round-trip test, because a silent convention
mismatch shows up only as a subtly wrong map.
"""

from __future__ import annotations

import gtsam
import numpy as np

from ..camera import Camera
from ..types import Pose

POSE_CHAR = "x"
LANDMARK_CHAR = "l"


def pose_key(index: int) -> int:
    return int(gtsam.symbol(POSE_CHAR, index))


def landmark_key(index: int) -> int:
    return int(gtsam.symbol(LANDMARK_CHAR, index))


def to_gtsam_pose(pose: Pose) -> gtsam.Pose3:
    """Camera-to-world `Pose` -> `gtsam.Pose3`."""
    return gtsam.Pose3(gtsam.Rot3(pose.R), gtsam.Point3(*pose.t))


def from_gtsam_pose(pose3: gtsam.Pose3) -> Pose:
    return Pose(pose3.rotation().matrix(), np.asarray(pose3.translation()).reshape(3))


def to_gtsam_calibration(camera: Camera) -> gtsam.Cal3_S2:
    return gtsam.Cal3_S2(camera.fx, camera.fy, 0.0, camera.cx, camera.cy)


def robust_pixel_noise(sigma: float, huber_k: float) -> gtsam.noiseModel.Base:
    """Isotropic pixel noise wrapped in a Huber M-estimator.

    Robust kernels are drift-control layer L1 inside the optimiser: without
    them a handful of surviving outlier observations can drag an entire
    bundle-adjustment solution off the true minimum.
    """
    base = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k), base)


def pose_noise(sigma_trans: float, sigma_rot: float) -> gtsam.noiseModel.Diagonal:
    """6-DoF pose noise. GTSAM orders Pose3 tangent as (rotation, translation)."""
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([sigma_rot] * 3 + [sigma_trans] * 3, dtype=np.float64))


def robust_pose_noise(sigma_trans: float, sigma_rot: float,
                      huber_k: float) -> gtsam.noiseModel.Base:
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(huber_k),
        pose_noise(sigma_trans, sigma_rot))
