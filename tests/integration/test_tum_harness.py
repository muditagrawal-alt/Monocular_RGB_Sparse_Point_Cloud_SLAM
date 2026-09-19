"""Tests for the image-sequence path and the TUM evaluation harness.

The real dataset is hundreds of megabytes and its download host is not
reachable from every network, so these exercise the harness against a synthetic
sequence written in exact TUM layout. That proves the machinery -- timestamp
association, Sim(3) alignment, ATE -- independently of the data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from benchmarks.make_tum_fixture import write_fixture  # noqa: E402
from benchmarks.tum import associate, evaluate, read_tum_trajectory  # noqa: E402
from slam.config import SlamConfig
from slam.io.video import ImageSequenceDecoder
from slam.pipeline import SlamPipeline


@pytest.fixture(scope="module")
def tum_fixture(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("tum"), n_frames=120, motion="strafe")


def test_fixture_has_tum_layout(tum_fixture):
    assert (tum_fixture / "rgb").is_dir()
    assert (tum_fixture / "rgb.txt").exists()
    assert (tum_fixture / "groundtruth.txt").exists()


def test_image_decoder_reads_timestamps(tum_fixture):
    decoder = ImageSequenceDecoder(tum_fixture / "rgb",
                                   timestamps_file=tum_fixture / "rgb.txt")
    frames = list(decoder)
    assert len(frames) == 120
    assert frames[0].timestamp == pytest.approx(0.0, abs=1e-6)
    # 120 frames at 30 fps spans just under four seconds
    assert frames[-1].timestamp == pytest.approx(119 / 30.0, abs=1e-3)
    assert all(f.gray.ndim == 2 for f in frames)


def test_image_decoder_orders_by_name_without_timestamps(tum_fixture):
    decoder = ImageSequenceDecoder(tum_fixture / "rgb")
    frames = list(decoder)
    assert len(frames) == 120
    assert [f.index for f in frames] == list(range(120))


def test_image_decoder_downscales(tum_fixture):
    decoder = ImageSequenceDecoder(tum_fixture / "rgb", target_width=320)
    assert decoder.out_width == 320
    assert next(iter(decoder)).gray.shape[1] == 320


def test_decoder_rejects_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        ImageSequenceDecoder(tmp_path / "nope")


def test_groundtruth_parsing(tum_fixture):
    times, positions = read_tum_trajectory(tum_fixture / "groundtruth.txt")
    assert len(times) == len(positions) > 120
    assert positions.shape[1] == 3
    assert np.all(np.diff(times) > 0)          # strictly increasing
    assert times[0] > 1e9                       # epoch-style, as the real data uses


def test_association_pairs_nearest_timestamps():
    est = np.array([0.0, 1.0, 2.0])
    gt = np.array([-0.005, 0.99, 2.3])
    pairs = associate(est, gt, max_difference=0.02)
    # 0.0 and 1.0 match; 2.0 is 0.3 away from anything and must be dropped
    assert pairs == [(0, 0), (1, 1)]


def test_association_never_reuses_a_groundtruth_pose():
    est = np.array([0.0, 0.001, 0.002])
    gt = np.array([0.0])
    assert len(associate(est, gt, max_difference=0.02)) == 1


@pytest.mark.slow
def test_end_to_end_evaluation_scores_known_ground_truth(tum_fixture):
    """The harness must recover a small error on a sequence it has truth for."""
    result = SlamPipeline(SlamConfig()).run(
        tum_fixture / "rgb", timestamps_file=tum_fixture / "rgb.txt")
    assert result.success, result.reason

    first = float(next(
        line.split()[0] for line in (tum_fixture / "rgb.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")))
    evaluation = evaluate(result, tum_fixture / "groundtruth.txt", "fixture", first)

    assert evaluation.n_matched > 100
    assert evaluation.tracked_fraction > 0.8
    assert evaluation.ate_rmse < 0.5
    # Monocular scale is arbitrary, so alignment must solve for it rather than
    # assuming unity.
    assert evaluation.scale > 0
    assert np.isfinite(evaluation.ate_median)
