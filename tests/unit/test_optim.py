"""Tests for the drift-control backend (requirement 4)."""

import numpy as np
import pytest

from slam.config import LocalBAConfig, PoseGraphConfig
from slam.optim.gtsam_utils import from_gtsam_pose, to_gtsam_calibration, to_gtsam_pose
from slam.optim.local_ba import LocalBundleAdjuster
from slam.optim.loop_closure import BoWVocabulary
from slam.optim.pose_graph import PoseGraphOptimizer
from slam.types import Keyframe, Pose, SlamMap


def test_gtsam_pose_round_trip(rng):
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1
    p = Pose(R, rng.normal(size=3))
    back = from_gtsam_pose(to_gtsam_pose(p))
    assert np.allclose(back.R, p.R, atol=1e-12)
    assert np.allclose(back.t, p.t, atol=1e-12)


def test_gtsam_projection_matches_our_camera(camera, rng):
    """A convention mismatch here would surface only as a subtly wrong map.

    The pose is perturbed rather than fully random: projection is undefined for
    a point behind the camera, and an unconstrained random rotation puts it
    there often enough to make this test flaky depending on how much of the
    shared rng earlier tests consumed.
    """
    import gtsam

    point = np.array([0.8, -0.4, 6.0])
    for _ in range(8):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-0.3, 0.3)          # small rotation, keeps the point ahead
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        pose = Pose(R, rng.normal(size=3) * 0.2)

        if pose.world_to_camera(point.reshape(1, 3))[0, 2] <= 0.1:
            continue                            # behind the camera, try another

        gtsam_cam = gtsam.PinholeCameraCal3_S2(to_gtsam_pose(pose),
                                               to_gtsam_calibration(camera))
        expected = np.asarray(gtsam_cam.project(gtsam.Point3(*point))).ravel()
        assert np.allclose(camera.project(pose.world_to_camera(point.reshape(1, 3)))[0],
                           expected, atol=1e-9)
        return
    pytest.fail("could not generate a pose with the test point in front of the camera")


@pytest.fixture
def perturbed_map(camera, rng):
    """A small map with known truth, perturbed so BA has something to fix."""
    n_kf, n_lm = 8, 150
    truth_pts = np.column_stack([rng.uniform(-3, 3, n_lm), rng.uniform(-2, 2, n_lm),
                                 rng.uniform(5, 11, n_lm)])
    truth_poses = [Pose(np.eye(3), np.array([0.3 * i, 0.01 * i, 0.0])) for i in range(n_kf)]

    m = SlamMap()
    ids = np.arange(n_lm)
    for i, pose in enumerate(truth_poses):
        obs = camera.project(pose.world_to_camera(truth_pts)) + rng.normal(0, 0.5, (n_lm, 2))
        noisy = pose if i < 2 else Pose(pose.R, pose.t + rng.normal(0, 0.08, 3))
        m.add_keyframe(Keyframe(id=m.new_keyframe_id(), frame_index=i * 5,
                                timestamp=i * 0.16, pose=noisy, points=obs, track_ids=ids))
    for j in range(n_lm):
        lm = m.add_landmark(truth_pts[j] + rng.normal(0, 0.15, 3))
        for i in range(n_kf):
            m.observe(lm.id, i, j)
    return m, truth_poses, truth_pts


def test_local_ba_reduces_pose_and_structure_error(camera, perturbed_map):
    m, truth_poses, truth_pts = perturbed_map
    before_pose = np.mean([np.linalg.norm(m.keyframes[i].pose.t - truth_poses[i].t)
                           for i in range(len(truth_poses))])
    before_lm = np.mean([np.linalg.norm(m.landmarks[j].position - truth_pts[j])
                         for j in range(len(truth_pts))])

    result = LocalBundleAdjuster(camera, LocalBAConfig(window_size=8)).optimize(m)
    assert result.success and result.n_factors > 100

    after_pose = np.mean([np.linalg.norm(m.keyframes[i].pose.t - truth_poses[i].t)
                          for i in range(len(truth_poses))])
    after_lm = np.mean([np.linalg.norm(m.landmarks[j].position - truth_pts[j])
                        for j in range(len(truth_pts))])
    assert after_pose < before_pose * 0.5
    assert after_lm < before_lm * 0.5
    assert result.final_error < result.initial_error


def test_local_ba_declines_when_underconstrained(camera):
    m = SlamMap()
    m.add_keyframe(Keyframe(id=m.new_keyframe_id(), frame_index=0, timestamp=0.0,
                            pose=Pose(), points=np.zeros((0, 2)),
                            track_ids=np.zeros(0, dtype=np.int64)))
    assert not LocalBundleAdjuster(camera, LocalBAConfig()).optimize(m).success


def _circle_map(noise_sigma=0.03, n=60, seed=0):
    def truth(i):
        a = 2 * np.pi * i / n
        c, s = np.cos(a), np.sin(a)
        return Pose(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]),
                    np.array([5 * c, 5 * s, 0.0]))

    rng = np.random.default_rng(seed)
    m = SlamMap()
    drift = truth(0)
    for i in range(n):
        if i > 0:
            rel = truth(i - 1).inverse().compose(truth(i))
            drift = drift.compose(Pose(rel.R, rel.t + rng.normal(0, noise_sigma, 3)))
        m.add_keyframe(Keyframe(id=m.new_keyframe_id(), frame_index=i, timestamp=i * 0.1,
                                pose=drift.copy() if i > 0 else truth(0),
                                points=np.zeros((0, 2)),
                                track_ids=np.zeros(0, dtype=np.int64)))
    return m, truth, n


def test_pose_graph_removes_accumulated_drift():
    """The core of requirement 4: a loop constraint must redistribute drift."""
    m, truth, n = _circle_map()

    def ate():
        return float(np.sqrt(np.mean(
            [np.linalg.norm(m.keyframes[i].pose.t - truth(i).t) ** 2 for i in range(n)])))

    before = ate()
    loop = truth(n - 1).inverse().compose(truth(0))
    result = PoseGraphOptimizer(PoseGraphConfig()).optimize(m, [(n - 1, 0, loop)])

    assert result.success and result.n_loop_factors == 1
    assert ate() < before * 0.6
    assert result.max_correction > 0


def test_pose_graph_preserves_odometry_for_comparison():
    """Pre-optimisation poses must survive so the UI can show before/after."""
    m, truth, n = _circle_map()
    PoseGraphOptimizer(PoseGraphConfig()).optimize(
        m, [(n - 1, 0, truth(n - 1).inverse().compose(truth(0)))])
    assert all(m.keyframes[i].odometry_pose is not None for i in range(n))


def test_pose_graph_without_loops_is_a_noop():
    m, _, _ = _circle_map()
    poses = [m.keyframes[i].pose.t.copy() for i in m.keyframe_ids]
    result = PoseGraphOptimizer(PoseGraphConfig()).optimize(m, [])
    assert not result.success and "no loop" in result.reason
    assert all(np.allclose(m.keyframes[i].pose.t, poses[i]) for i in m.keyframe_ids)


def test_vocabulary_similarity_is_self_consistent(rng):
    desc = rng.integers(0, 256, (600, 32), dtype=np.uint8)
    vocab = BoWVocabulary(size=32)
    assert vocab.train(desc)
    a = vocab.describe(desc[:100])
    assert float(a @ vocab.describe(desc[:100])) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_vocabulary_declines_insufficient_data(rng):
    assert not BoWVocabulary(size=256).train(rng.integers(0, 256, (10, 32), dtype=np.uint8))
