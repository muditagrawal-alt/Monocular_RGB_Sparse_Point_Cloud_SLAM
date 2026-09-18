"""Requirement 6: a 10-second clip must process in 10 seconds or less.

This is a hard deliverable, so it is asserted in the test suite rather than
checked by hand. The threshold is the requirement itself; the margin measured
during development is roughly 0.7-0.9x realtime on a laptop CPU.
"""

import pytest

from slam.config import SlamConfig
from slam.io.synthetic import make_sequence
from slam.pipeline import SlamPipeline


@pytest.fixture(scope="module")
def ten_second_video(tmp_path_factory):
    """300 frames at 30 fps -- exactly the assignment's stated case."""
    seq = make_sequence(n_frames=300, motion="orbit", loop=True, seed=0)
    path = tmp_path_factory.mktemp("perf") / "ten_seconds.mp4"
    seq.write_video(path)
    return str(path)


@pytest.mark.slow
def test_ten_second_clip_within_budget(ten_second_video):
    result = SlamPipeline(SlamConfig()).run(ten_second_video)
    assert result.success, result.reason
    assert result.video_duration_s == pytest.approx(10.0, abs=0.2)
    assert result.realtime_factor <= 1.0, (
        f"processed {result.video_duration_s:.1f}s of video in "
        f"{result.processing_time_s:.1f}s (realtime factor "
        f"{result.realtime_factor:.2f}); budget is 1.0")
    assert result.summary()["within_budget"]


@pytest.mark.slow
def test_no_single_stage_dominates(ten_second_video):
    """Guards against a regression that quietly makes one stage the bottleneck.

    Bundle adjustment and loop closure are the two expensive stages; each has
    previously regressed to several times its intended cost (a 223 ms Hamming
    call and a per-point reprojection loop), so this pins them.
    """
    result = SlamPipeline(SlamConfig()).run(ten_second_video)
    timings = result.timings
    for stage in ("local_ba", "loop_closure", "frontend", "tracking"):
        share = getattr(timings, stage) / timings.total
        assert share < 0.6, f"{stage} takes {share:.0%} of total runtime"


def test_adaptive_quality_engages_when_budget_is_tight(tmp_path):
    """The guard must reduce work up front rather than miss the deadline.

    On a fast machine the 30 fps decimation cap already keeps projected cost
    well inside budget, so this drives the guard the way a slower machine
    would: by tightening the allowed time per second of video.
    """
    seq = make_sequence(n_frames=120, motion="strafe", seed=0)
    path = tmp_path / "clip.mp4"
    seq.write_video(path, fps=30.0)

    from slam.io.video import probe_video
    info = probe_video(str(path))

    relaxed = SlamConfig()
    _, _, reduced_relaxed = SlamPipeline(relaxed)._plan_quality(info)
    assert not reduced_relaxed, "should not reduce quality when comfortably in budget"

    tight = SlamConfig()
    tight.budget.realtime_factor_target = 0.1     # emulate a much slower CPU
    width, features, reduced = SlamPipeline(tight)._plan_quality(info)
    assert reduced
    assert width < relaxed.frontend.target_width
    assert features <= relaxed.frontend.max_features
    assert width >= tight.budget.adaptive_min_width
    assert features >= tight.budget.adaptive_min_features


def test_adaptive_quality_can_be_disabled(tmp_path):
    seq = make_sequence(n_frames=60, motion="strafe", seed=0)
    path = tmp_path / "clip2.mp4"
    seq.write_video(path, fps=30.0)
    from slam.io.video import probe_video

    config = SlamConfig()
    config.budget.realtime_factor_target = 0.1
    config.budget.adaptive_quality = False
    width, features, reduced = SlamPipeline(config)._plan_quality(probe_video(str(path)))
    assert not reduced
    assert width == config.frontend.target_width
    assert features == config.frontend.max_features


def test_frontend_cost_per_frame(strafe_sequence):
    """The front end is the per-frame cost that scales with video length.

    Measured at roughly 6 ms/frame at 640x360; 20 ms would blow the budget on
    a 30 fps clip.
    """
    import time

    import cv2

    from slam.config import FrontendConfig
    from slam.core.features import FeatureTracker

    tracker = FeatureTracker(FrontendConfig())
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in strafe_sequence.frames[:60]]
    for f in frames[:5]:
        tracker.track(f)                       # warm up
    start = time.perf_counter()
    for f in frames[5:]:
        tracker.track(f)
    per_frame_ms = (time.perf_counter() - start) / len(frames[5:]) * 1000
    assert per_frame_ms < 20.0, f"front end at {per_frame_ms:.1f} ms/frame"
