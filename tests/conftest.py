"""Shared fixtures. Synthetic sequences are cached per session because
generating and encoding them costs more than the tests themselves."""

from __future__ import annotations

import numpy as np
import pytest

from slam.camera import camera_from_hfov
from slam.io.synthetic import make_sequence


@pytest.fixture(scope="session")
def camera():
    return camera_from_hfov(640, 360)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def strafe_sequence():
    return make_sequence(n_frames=90, motion="strafe", seed=1)


@pytest.fixture(scope="session")
def loop_sequence():
    return make_sequence(n_frames=300, motion="orbit", loop=True, seed=2)


@pytest.fixture(scope="session")
def strafe_video(strafe_sequence, tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "strafe.mp4"
    strafe_sequence.write_video(path)
    return str(path)


@pytest.fixture(scope="session")
def loop_video(loop_sequence, tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "loop.mp4"
    loop_sequence.write_video(path)
    return str(path)
