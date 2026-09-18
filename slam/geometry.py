"""Multi-view geometry helpers: triangulation, parallax, and Sim(3) alignment."""

from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera
from .types import Pose


def projection_matrix(camera: Camera, pose: Pose) -> np.ndarray:
    """3x4 projection matrix P = K [R|t]_world->camera."""
    R_cw = pose.R.T
    t_cw = -R_cw @ pose.t
    return camera.K @ np.hstack([R_cw, t_cw.reshape(3, 1)])


def triangulate(camera: Camera, pose_a: Pose, pose_b: Pose,
                pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Linear triangulation of matched pixels into Nx3 world points."""
    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    if len(pts_a) == 0:
        return np.zeros((0, 3))
    Pa = projection_matrix(camera, pose_a)
    Pb = projection_matrix(camera, pose_b)
    hom = cv2.triangulatePoints(Pa, Pb, pts_a.T, pts_b.T)
    w = hom[3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return (hom[:3] / w).T


def parallax_angles_deg(pose_a: Pose, pose_b: Pose, points_world: np.ndarray) -> np.ndarray:
    """Angle at each 3D point subtended by the two camera centres.

    Small parallax means depth is poorly constrained, so this gates which
    triangulated points are allowed into the map.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    if len(pts) == 0:
        return np.zeros(0)
    ray_a = pts - pose_a.center
    ray_b = pts - pose_b.center
    na = np.linalg.norm(ray_a, axis=1)
    nb = np.linalg.norm(ray_b, axis=1)
    denom = np.where((na * nb) < 1e-12, 1e-12, na * nb)
    cos = np.clip(np.einsum("ij,ij->i", ray_a, ray_b) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def reprojection_errors(camera: Camera, pose: Pose, points_world: np.ndarray,
                        pixels: np.ndarray) -> np.ndarray:
    """Per-point reprojection error in pixels. Points behind the camera get inf."""
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    px = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    if len(pts) == 0:
        return np.zeros(0)
    cam_pts = pose.world_to_camera(pts)
    proj = camera.project(cam_pts)
    err = np.linalg.norm(proj - px, axis=1)
    return np.where(cam_pts[:, 2] <= 0, np.inf, err)


def filter_triangulated(camera: Camera, pose_a: Pose, pose_b: Pose,
                        pts_a: np.ndarray, pts_b: np.ndarray, points_world: np.ndarray,
                        *, min_parallax_deg: float, max_reproj_error_px: float,
                        min_depth: float, max_depth: float | None = None) -> np.ndarray:
    """Boolean mask of triangulated points that are geometrically trustworthy.

    Applies the four standard gates: positive depth in both views, sufficient
    parallax, bounded reprojection error, and a sanity bound on range.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)

    depth_a = pose_a.world_to_camera(pts)[:, 2]
    depth_b = pose_b.world_to_camera(pts)[:, 2]
    ok = (depth_a > min_depth) & (depth_b > min_depth) & np.isfinite(pts).all(axis=1)

    if max_depth is not None:
        ok &= (depth_a < max_depth) & (depth_b < max_depth)

    ok &= parallax_angles_deg(pose_a, pose_b, pts) >= min_parallax_deg

    err_a = reprojection_errors(camera, pose_a, pts, pts_a)
    err_b = reprojection_errors(camera, pose_b, pts, pts_b)
    ok &= (err_a <= max_reproj_error_px) & (err_b <= max_reproj_error_px)
    return ok


def relative_pose(pose_a: Pose, pose_b: Pose) -> Pose:
    """Transform from frame b into frame a (a^-1 * b)."""
    return pose_a.inverse().compose(pose_b)


def align_sim3(estimated: np.ndarray, reference: np.ndarray,
               with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama alignment of two Nx3 trajectories -> (scale, R, t).

    Monocular SLAM recovers geometry only up to a similarity transform, so any
    comparison against ground truth must solve for this 7-DoF alignment first
    (IMPLEMENTATION_PLAN.md section 2.1).
    """
    est = np.asarray(estimated, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if len(est) != len(ref):
        raise ValueError(f"trajectory length mismatch: {len(est)} vs {len(ref)}")
    if len(est) < 3:
        raise ValueError("need at least 3 poses to align")

    mu_e, mu_r = est.mean(axis=0), ref.mean(axis=0)
    ec, rc = est - mu_e, ref - mu_r

    C = (rc.T @ ec) / len(est)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_e = (ec ** 2).sum() / len(est)
        scale = float(np.trace(np.diag(D) @ S) / var_e) if var_e > 1e-18 else 1.0
    else:
        scale = 1.0

    t = mu_r - scale * R @ mu_e
    return scale, R, t


def ate_rmse(estimated: np.ndarray, reference: np.ndarray,
             align: bool = True, with_scale: bool = True) -> float:
    """Absolute trajectory error (RMSE) after optional Sim(3) alignment."""
    est = np.asarray(estimated, dtype=np.float64).reshape(-1, 3)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if align:
        s, R, t = align_sim3(est, ref, with_scale)
        est = (s * (R @ est.T).T) + t
    return float(np.sqrt(((est - ref) ** 2).sum(axis=1).mean()))
