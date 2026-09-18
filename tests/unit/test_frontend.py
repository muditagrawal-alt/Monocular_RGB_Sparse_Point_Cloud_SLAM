import cv2
import numpy as np

from slam.config import FrontendConfig, InitConfig
from slam.core.features import FeatureTracker, compute_orb_at_points
from slam.core.initializer import Initializer
from slam.types import Pose


def gray(seq, i):
    return cv2.cvtColor(seq.frames[i], cv2.COLOR_BGR2GRAY)


def test_tracker_seeds_and_maintains_tracks(strafe_sequence):
    tracker = FeatureTracker(FrontendConfig())
    first = tracker.track(gray(strafe_sequence, 0))
    assert first.is_first and first.n_added > 200

    for i in range(1, 30):
        result = tracker.track(gray(strafe_sequence, i))
    assert result.n_tracked > 100


def test_track_ids_are_stable(strafe_sequence):
    """A track id must follow the same physical point across frames.

    Tracks legitimately die as the view changes (on this sequence roughly a
    third survive five frames), so this asserts that a substantial cohort
    persists rather than a specific count.
    """
    tracker = FeatureTracker(FrontendConfig())
    first = tracker.track(gray(strafe_sequence, 0))
    original = set(first.track_ids.tolist())
    for i in range(1, 6):
        result = tracker.track(gray(strafe_sequence, i))
    survivors = original & set(result.track_ids.tolist())
    assert len(survivors) > 100
    assert survivors < original          # and some genuinely expire


def test_tracker_replenishes_lost_tracks(strafe_sequence):
    """Reseeding must keep the live track count healthy as old tracks die."""
    tracker = FeatureTracker(FrontendConfig())
    for i in range(20):
        result = tracker.track(gray(strafe_sequence, i))
    assert len(result.points) > 300


def test_features_cover_the_image(strafe_sequence):
    """Grid seeding must spread corners, not clump them in one region."""
    tracker = FeatureTracker(FrontendConfig())
    result = tracker.track(gray(strafe_sequence, 0))
    h, w = strafe_sequence.frames[0].shape[:2]
    cells = {(int(y / (h / 6)), int(x / (w / 8))) for x, y in result.points}
    assert len(cells) >= 24


def test_descriptors_align_with_tracked_points(strafe_sequence):
    tracker = FeatureTracker(FrontendConfig())
    result = tracker.track(gray(strafe_sequence, 0))
    desc, kept = compute_orb_at_points(gray(strafe_sequence, 0), result.points)
    assert len(desc) == len(kept)
    assert kept.max() < len(result.points)
    assert len(set(kept.tolist())) == len(kept)


def test_drop_prunes_tracks(strafe_sequence):
    tracker = FeatureTracker(FrontendConfig())
    tracker.track(gray(strafe_sequence, 0))
    n = tracker.n_tracks
    mask = np.zeros(n, dtype=bool)
    mask[: n // 2] = True
    tracker.drop(mask)
    assert tracker.n_tracks == n // 2


def _views(camera, rng, pts, pose_b, noise=0.3):
    a = Pose()
    return (camera.project(a.world_to_camera(pts)) + rng.normal(0, noise, (len(pts), 2)),
            camera.project(pose_b.world_to_camera(pts)) + rng.normal(0, noise, (len(pts), 2)))


def test_initialises_on_well_conditioned_motion(camera, rng):
    pts = np.column_stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300),
                           rng.uniform(4, 12, 300)])
    b = Pose(np.eye(3), np.array([0.8, 0.05, 0.1]))
    result = Initializer(camera, InitConfig()).try_initialize(*_views(camera, rng, pts, b))
    assert result.success
    direction = result.pose.t / np.linalg.norm(result.pose.t)
    assert float(direction @ (b.t / np.linalg.norm(b.t))) > 0.99
    depths = Pose().world_to_camera(result.points_world)[:, 2]
    assert 0.9 < np.median(depths) < 1.1   # normalised to unit scale


def test_rejects_pure_rotation(camera, rng):
    """No baseline means no recoverable depth; initialising anyway poisons the map."""
    pts = np.column_stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300),
                           rng.uniform(4, 12, 300)])
    th = 0.06
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    assert not Initializer(camera, InitConfig()).try_initialize(
        *_views(camera, rng, pts, Pose(R, np.zeros(3)))).success


def test_rejects_planar_scene(camera, rng):
    pts = np.column_stack([rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300),
                           np.full(300, 8.0)])
    result = Initializer(camera, InitConfig()).try_initialize(
        *_views(camera, rng, pts, Pose(np.eye(3), np.array([0.8, 0, 0]))))
    assert not result.success


def test_rejects_too_few_correspondences(camera, rng):
    pts = np.column_stack([rng.uniform(-3, 3, 20), rng.uniform(-2, 2, 20),
                           rng.uniform(4, 12, 20)])
    result = Initializer(camera, InitConfig()).try_initialize(
        *_views(camera, rng, pts, Pose(np.eye(3), np.array([0.8, 0, 0]))))
    assert not result.success and "correspond" in result.reason
