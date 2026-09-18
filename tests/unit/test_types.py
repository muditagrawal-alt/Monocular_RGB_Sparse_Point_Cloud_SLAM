import cv2
import numpy as np

from slam.types import Pose, SlamMap


def random_rotation(rng):
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    return R


def test_inverse_is_exact(rng):
    p = Pose(random_rotation(rng), rng.normal(size=3))
    assert np.allclose(p.compose(p.inverse()).matrix, np.eye(4), atol=1e-12)


def test_world_camera_round_trip(rng):
    p = Pose(random_rotation(rng), rng.normal(size=3))
    pts = rng.normal(size=(40, 3))
    assert np.allclose(p.camera_to_world(p.world_to_camera(pts)), pts, atol=1e-12)


def test_opencv_convention_bridge(rng):
    """Pose must agree with OpenCV's world-to-camera rvec/tvec convention.

    A mismatch here produces a mirrored trajectory that still looks plausible,
    so it is worth pinning explicitly.
    """
    p = Pose(random_rotation(rng), rng.normal(size=3))
    rvec, tvec = p.rvec_tvec
    back = Pose.from_rvec_tvec(rvec, tvec)
    assert np.allclose(back.R, p.R, atol=1e-12)
    assert np.allclose(back.t, p.t, atol=1e-12)


def test_matches_cv2_project_points(rng, camera):
    p = Pose(random_rotation(rng), rng.normal(size=3) * 0.3)
    pts = np.column_stack([rng.uniform(-2, 2, 30), rng.uniform(-2, 2, 30),
                           rng.uniform(4, 10, 30)])
    rvec, tvec = p.rvec_tvec
    expected, _ = cv2.projectPoints(pts, rvec, tvec, camera.K, None)
    assert np.allclose(camera.project(p.world_to_camera(pts)),
                       expected.reshape(-1, 2), atol=1e-6)


def test_map_landmark_lifecycle():
    m = SlamMap()
    lm = m.add_landmark(np.array([1.0, 2.0, 3.0]))
    assert m.stats()["landmarks"] == 1
    lm.is_outlier = True
    assert m.stats()["active_landmarks"] == 0
    m.remove_landmark(lm.id)
    assert m.stats()["landmarks"] == 0


def test_point_cloud_empty_map_is_well_formed():
    pts, cols = SlamMap().point_cloud()
    assert pts.shape == (0, 3) and cols.shape == (0, 3)
