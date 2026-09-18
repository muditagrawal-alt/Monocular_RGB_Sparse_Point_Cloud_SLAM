import numpy as np
import pytest

from slam.geometry import (
    align_sim3,
    ate_rmse,
    filter_triangulated,
    parallax_angles_deg,
    reprojection_errors,
    triangulate,
)
from slam.types import Pose


@pytest.fixture
def scene(rng):
    pts = np.column_stack([rng.uniform(-2, 2, 150), rng.uniform(-1.5, 1.5, 150),
                           rng.uniform(4, 9, 150)])
    return pts, Pose(), Pose(np.eye(3), np.array([0.6, 0.0, 0.0]))


def test_triangulation_recovers_known_points(camera, scene):
    pts, a, b = scene
    out = triangulate(camera, a, b, camera.project(a.world_to_camera(pts)),
                      camera.project(b.world_to_camera(pts)))
    assert np.abs(out - pts).max() < 1e-8


def test_triangulation_of_empty_input(camera, scene):
    _, a, b = scene
    assert triangulate(camera, a, b, np.zeros((0, 2)), np.zeros((0, 2))).shape == (0, 3)


def test_parallax_is_larger_for_wider_baseline(scene):
    pts, a, _ = scene
    near = parallax_angles_deg(a, Pose(np.eye(3), np.array([0.1, 0, 0])), pts)
    far = parallax_angles_deg(a, Pose(np.eye(3), np.array([2.0, 0, 0])), pts)
    assert np.median(far) > np.median(near)


def test_points_behind_camera_get_infinite_error(camera):
    err = reprojection_errors(camera, Pose(), np.array([[0.0, 0.0, -5.0]]),
                              np.array([[320.0, 180.0]]))
    assert not np.isfinite(err[0])


def test_filter_rejects_zero_parallax(camera, scene):
    """Identical viewpoints give no baseline, so nothing may be accepted."""
    pts, a, _ = scene
    px = camera.project(a.world_to_camera(pts))
    out = triangulate(camera, a, a, px, px)
    good = filter_triangulated(camera, a, a, px, px, out, min_parallax_deg=1.0,
                               max_reproj_error_px=4.0, min_depth=1e-4)
    assert good.sum() == 0


def test_filter_accepts_well_conditioned_points(camera, scene):
    pts, a, b = scene
    pa, pb = camera.project(a.world_to_camera(pts)), camera.project(b.world_to_camera(pts))
    out = triangulate(camera, a, b, pa, pb)
    good = filter_triangulated(camera, a, b, pa, pb, out, min_parallax_deg=0.5,
                               max_reproj_error_px=4.0, min_depth=1e-4)
    assert good.mean() > 0.9


def test_sim3_recovers_similarity_transform(rng):
    traj = np.column_stack([np.linspace(0, 5, 60), np.sin(np.linspace(0, 6, 60)),
                            np.zeros(60)])
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    warped = 2.7 * (R @ traj.T).T + np.array([3.0, -1.0, 2.0])
    scale, _, _ = align_sim3(warped, traj)
    assert scale == pytest.approx(1 / 2.7, rel=1e-6)
    assert ate_rmse(warped, traj) < 1e-9


def test_scale_correction_matters_for_monocular(rng):
    """Monocular results are only meaningful after 7-DoF alignment."""
    traj = np.column_stack([np.linspace(0, 5, 40), np.zeros(40), np.zeros(40)])
    scaled = traj * 3.0
    assert ate_rmse(scaled, traj, with_scale=True) < 1e-9
    assert ate_rmse(scaled, traj, with_scale=False) > 1.0


def test_align_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        align_sim3(np.zeros((5, 3)), np.zeros((6, 3)))
