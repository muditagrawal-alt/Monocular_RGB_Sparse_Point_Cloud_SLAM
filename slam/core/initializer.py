"""Two-view initialisation with automatic planar/general model selection.

Monocular initialisation fails in two well-known ways (see
IMPLEMENTATION_PLAN.md section 2.3):

* **Pure rotation** gives no baseline, so depth is unrecoverable. Triangulated
  points shoot off to infinity and the map is garbage.
* **Planar scenes** make the essential matrix rank-deficient and its
  decomposition unstable; a homography is the correct model there.

Following ORB-SLAM, this scores a homography and an essential matrix on the
same correspondences and refuses to initialise until the motion is
well-conditioned, rather than accepting a bad map and drifting forever after.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..camera import Camera
from ..config import InitConfig
from ..geometry import filter_triangulated, parallax_angles_deg, triangulate
from ..types import Pose


@dataclass
class InitResult:
    """Outcome of an initialisation attempt."""

    success: bool
    reason: str = ""
    pose: Pose | None = None
    """Second camera pose; the first is fixed at the origin."""
    points_world: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None
    model: str = ""
    median_parallax_deg: float = 0.0
    n_triangulated: int = 0
    scale_normalisation: float = 1.0


def _symmetric_transfer_score(H: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray,
                              threshold: float) -> tuple[float, np.ndarray]:
    """Score a homography by symmetric transfer error (ORB-SLAM style).

    Each inlier contributes `threshold^2 - error`, so a model is rewarded both
    for having many inliers and for fitting them tightly.
    """
    n = len(pts_a)
    if n == 0:
        return 0.0, np.zeros(0, dtype=bool)
    ones = np.ones((n, 1))
    a_h = np.hstack([pts_a, ones])
    b_h = np.hstack([pts_b, ones])

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0, np.zeros(n, dtype=bool)

    fwd = (H @ a_h.T).T
    bwd = (H_inv @ b_h.T).T
    fw = np.where(np.abs(fwd[:, 2:3]) < 1e-12, 1e-12, fwd[:, 2:3])
    bw = np.where(np.abs(bwd[:, 2:3]) < 1e-12, 1e-12, bwd[:, 2:3])
    err_f = ((fwd[:, :2] / fw - pts_b) ** 2).sum(axis=1)
    err_b = ((bwd[:, :2] / bw - pts_a) ** 2).sum(axis=1)

    t2 = threshold ** 2
    chi_h = 5.99 * t2   # 95% chi-square, 2 DoF
    inliers = (err_f < chi_h) & (err_b < chi_h)
    score = float(np.sum(np.clip(chi_h - err_f[inliers], 0, None))
                  + np.sum(np.clip(chi_h - err_b[inliers], 0, None)))
    return score, inliers


def _epipolar_score(F: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray,
                    threshold: float) -> tuple[float, np.ndarray]:
    """Score a fundamental matrix by symmetric epipolar distance."""
    n = len(pts_a)
    if n == 0:
        return 0.0, np.zeros(0, dtype=bool)
    ones = np.ones((n, 1))
    a_h = np.hstack([pts_a, ones])
    b_h = np.hstack([pts_b, ones])

    l_b = (F @ a_h.T).T          # epipolar lines in image b
    l_a = (F.T @ b_h.T).T        # and in image a
    num_b = np.einsum("ij,ij->i", b_h, l_b) ** 2
    num_a = np.einsum("ij,ij->i", a_h, l_a) ** 2
    den_b = np.maximum((l_b[:, :2] ** 2).sum(axis=1), 1e-12)
    den_a = np.maximum((l_a[:, :2] ** 2).sum(axis=1), 1e-12)
    err_b, err_a = num_b / den_b, num_a / den_a

    t2 = threshold ** 2
    chi_f = 3.84 * t2   # 95% chi-square, 1 DoF (point-to-line)
    chi_h = 5.99 * t2   # scored against the H threshold for comparability
    inliers = (err_a < chi_f) & (err_b < chi_f)
    score = float(np.sum(np.clip(chi_h - err_a[inliers], 0, None))
                  + np.sum(np.clip(chi_h - err_b[inliers], 0, None)))
    return score, inliers


class Initializer:
    """Builds the first map from two sufficiently separated views."""

    def __init__(self, camera: Camera, config: InitConfig) -> None:
        self.camera = camera
        self.cfg = config

    def try_initialize(self, pts_ref: np.ndarray, pts_cur: np.ndarray) -> InitResult:
        """Attempt initialisation from corresponding points in two frames.

        `pts_ref` and `pts_cur` must be the same length and index the same
        tracks (which the KLT front end guarantees).
        """
        pts_ref = np.asarray(pts_ref, dtype=np.float64).reshape(-1, 2)
        pts_cur = np.asarray(pts_cur, dtype=np.float64).reshape(-1, 2)
        if len(pts_ref) != len(pts_cur):
            raise ValueError("correspondence count mismatch")
        if len(pts_ref) < self.cfg.min_correspondences:
            return InitResult(False, f"too few correspondences ({len(pts_ref)})")

        thr = self.cfg.ransac_threshold_px

        # --- score both models on the same data ---
        H, _ = cv2.findHomography(pts_ref, pts_cur, cv2.RANSAC, thr,
                                  confidence=self.cfg.ransac_confidence)
        F, _ = cv2.findFundamentalMat(pts_ref, pts_cur, cv2.FM_RANSAC, thr,
                                      self.cfg.ransac_confidence)
        score_h, _ = (_symmetric_transfer_score(H, pts_ref, pts_cur, thr)
                      if H is not None and H.shape == (3, 3) else (0.0, None))
        if F is not None and F.shape != (3, 3):
            F = F[:3, :3] if F.size >= 9 else None  # type: ignore[assignment]
        score_f, _ = (_epipolar_score(F, pts_ref, pts_cur, thr)
                      if F is not None else (0.0, None))

        total = score_h + score_f
        if total <= 0.0:
            return InitResult(False, "no geometric model fits the correspondences")
        h_ratio = score_h / total

        # A dominant homography means a planar or rotation-only view pair. The
        # essential matrix cannot be trusted there, so wait for better motion
        # rather than initialise on an ill-conditioned decomposition.
        if h_ratio > self.cfg.homography_score_ratio:
            return InitResult(False,
                              f"degenerate view pair (planar/rotation, H ratio {h_ratio:.2f})",
                              model="homography")

        # --- recover pose from the essential matrix ---
        E, e_mask = cv2.findEssentialMat(pts_ref, pts_cur, self.camera.K, cv2.RANSAC,
                                         self.cfg.ransac_confidence, thr)
        if E is None or E.shape != (3, 3):
            return InitResult(False, "essential matrix estimation failed")

        n_good, R_cw, t_cw, pose_mask = cv2.recoverPose(
            E, pts_ref, pts_cur, self.camera.K, mask=e_mask)
        if n_good < self.cfg.min_triangulated:
            return InitResult(False, f"recoverPose kept only {n_good} points")

        mask = pose_mask.reshape(-1).astype(bool)
        if mask.sum() < self.cfg.min_triangulated:
            return InitResult(False, f"too few pose inliers ({int(mask.sum())})")

        # recoverPose returns the transform taking frame ref -> cur, i.e. a
        # world-to-camera pose for cur given ref at the origin.
        pose_ref = Pose()
        pose_cur = Pose.from_world_to_camera(R_cw, t_cw.reshape(3))

        pa, pb = pts_ref[mask], pts_cur[mask]
        pts3d = triangulate(self.camera, pose_ref, pose_cur, pa, pb)

        parallax = parallax_angles_deg(pose_ref, pose_cur, pts3d)
        median_parallax = float(np.median(parallax)) if len(parallax) else 0.0
        if median_parallax < self.cfg.min_parallax_deg:
            return InitResult(False,
                              f"insufficient parallax ({median_parallax:.2f} deg)",
                              model="essential", median_parallax_deg=median_parallax)

        good = filter_triangulated(
            self.camera, pose_ref, pose_cur, pa, pb, pts3d,
            min_parallax_deg=self.cfg.min_parallax_deg,
            max_reproj_error_px=thr * 4.0, min_depth=1e-4)
        if good.sum() < self.cfg.min_triangulated:
            return InitResult(False, f"only {int(good.sum())} points survived filtering",
                              model="essential", median_parallax_deg=median_parallax)

        pts3d_good = pts3d[good]

        # Monocular scale is arbitrary (plan section 2.1): normalise median
        # scene depth to 1.0 so the map starts in well-conditioned units and
        # every later pose is expressed in the same consistent "SLAM units".
        depths = pose_ref.world_to_camera(pts3d_good)[:, 2]
        depths = depths[depths > 0]
        med_depth = float(np.median(depths)) if len(depths) else 1.0
        if not np.isfinite(med_depth) or med_depth <= 1e-9:
            return InitResult(False, "degenerate scene depth", model="essential")

        inv = 1.0 / med_depth
        pts3d_scaled = pts3d_good * inv
        pose_cur_scaled = Pose(pose_cur.R, pose_cur.t * inv)

        full_mask = np.zeros(len(pts_ref), dtype=bool)
        full_mask[np.flatnonzero(mask)[good]] = True

        return InitResult(
            success=True, reason="ok", pose=pose_cur_scaled, points_world=pts3d_scaled,
            inlier_mask=full_mask, model="essential",
            median_parallax_deg=median_parallax, n_triangulated=int(good.sum()),
            scale_normalisation=inv)
