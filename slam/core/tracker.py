"""Frame-to-map pose estimation.

This is drift-control layer L2 from IMPLEMENTATION_PLAN.md section 4.6. Chaining
frame-to-frame relative poses integrates every small error forever, so the
trajectory drifts without bound. Estimating each pose against the *map* instead
anchors it to accumulated structure, so error stops compounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..camera import Camera
from ..config import TrackingConfig
from ..types import Pose, SlamMap


@dataclass
class TrackingResult:
    """Outcome of estimating one frame's pose."""

    success: bool
    pose: Pose | None = None
    n_inliers: int = 0
    n_correspondences: int = 0
    inlier_mask: np.ndarray | None = None
    mean_reproj_error: float = 0.0
    reason: str = ""


class MotionModel:
    """Constant-velocity pose predictor.

    Seeding solvePnP with a prediction rather than the previous pose measurably
    improves both convergence and RANSAC inlier counts under fast motion.
    """

    def __init__(self) -> None:
        self._prev: Pose | None = None
        self._velocity: Pose | None = None

    def predict(self, current: Pose | None) -> Pose | None:
        if current is None:
            return None
        if self._velocity is None:
            return current.copy()
        return current.compose(self._velocity)

    def update(self, pose: Pose) -> None:
        if self._prev is not None:
            self._velocity = self._prev.inverse().compose(pose)
        self._prev = pose.copy()

    def reset(self) -> None:
        self._prev = None
        self._velocity = None


class Tracker:
    """Estimates camera pose from 2D observations of known 3D landmarks."""

    def __init__(self, camera: Camera, config: TrackingConfig) -> None:
        self.camera = camera
        self.cfg = config
        self.motion = MotionModel()

    def track(self, slam_map: SlamMap, points: np.ndarray, track_ids: np.ndarray,
              track_to_landmark: dict[int, int],
              last_pose: Pose | None = None) -> TrackingResult:
        """Estimate the current pose via PnP against the existing map.

        `track_to_landmark` maps a persistent track id to a landmark id, which
        is how a 2D observation in this frame is associated with 3D structure
        triangulated earlier.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        track_ids = np.asarray(track_ids).reshape(-1)

        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        source_idx: list[int] = []
        for i, tid in enumerate(track_ids):
            lm_id = track_to_landmark.get(int(tid))
            if lm_id is None:
                continue
            lm = slam_map.landmarks.get(lm_id)
            if lm is None or lm.is_outlier:
                continue
            obj_pts.append(lm.position)
            img_pts.append(points[i])
            source_idx.append(i)

        n_corr = len(obj_pts)
        if n_corr < self.cfg.min_inliers:
            return TrackingResult(False, n_correspondences=n_corr,
                                  reason=f"only {n_corr} 2D-3D correspondences")

        obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
        img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)

        use_guess = False
        rvec = tvec = None
        if self.cfg.use_motion_prior:
            predicted = self.motion.predict(last_pose)
            if predicted is not None:
                r, t = predicted.rvec_tvec
                rvec = r.reshape(3, 1).copy()
                tvec = t.reshape(3, 1).copy()
                use_guess = True

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.camera.K, None,
            rvec=rvec, tvec=tvec, useExtrinsicGuess=use_guess,
            iterationsCount=self.cfg.pnp_iterations,
            reprojectionError=self.cfg.pnp_reproj_threshold_px,
            confidence=self.cfg.pnp_confidence,
            flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok or inliers is None or len(inliers) < self.cfg.min_inliers:
            n_in = 0 if inliers is None else len(inliers)
            return TrackingResult(False, n_correspondences=n_corr, n_inliers=n_in,
                                  reason=f"PnP failed or too few inliers ({n_in})")

        inlier_idx = inliers.reshape(-1)

        # Refine on the inlier set only; the RANSAC solution is coarse.
        if len(inlier_idx) >= 6:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj[inlier_idx], img[inlier_idx], self.camera.K, None, rvec, tvec)

        pose = Pose.from_rvec_tvec(rvec, tvec)

        proj, _ = cv2.projectPoints(obj[inlier_idx], rvec, tvec, self.camera.K, None)
        errs = np.linalg.norm(proj.reshape(-1, 2) - img[inlier_idx].reshape(-1, 2), axis=1)

        mask = np.zeros(len(track_ids), dtype=bool)
        mask[[source_idx[i] for i in inlier_idx]] = True

        self.motion.update(pose)
        return TrackingResult(True, pose=pose, n_inliers=len(inlier_idx),
                              n_correspondences=n_corr, inlier_mask=mask,
                              mean_reproj_error=float(errs.mean()), reason="ok")
