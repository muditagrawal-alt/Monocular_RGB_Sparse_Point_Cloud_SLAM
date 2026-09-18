import numpy as np
import pytest

from slam.camera import DEFAULT_HFOV_DEG, camera_from_hfov, resolve_intrinsics


def test_projection_round_trip(camera):
    pixels = np.array([[0.0, 0.0], [320.0, 180.0], [639.0, 359.0], [12.5, 300.25]])
    depths = np.array([1.0, 5.0, 12.0, 0.7])
    back = camera.project(camera.unproject(pixels, depths))
    assert np.allclose(back, pixels, atol=1e-9)


def test_fov_is_preserved_under_scaling(camera):
    assert camera_from_hfov(1920, 1080, 60.0).scaled(1 / 3).hfov_deg == pytest.approx(60.0)


def test_scaling_keeps_principal_point_centred():
    scaled = camera_from_hfov(1920, 1080).scaled(0.25)
    assert scaled.cx == pytest.approx(scaled.width / 2)
    assert scaled.cy == pytest.approx(scaled.height / 2)


def test_intrinsics_resolution_priority():
    """Explicit user input must win over metadata, which wins over the heuristic."""
    assert resolve_intrinsics(640, 360, focal_px=500.0).source == "user"
    assert resolve_intrinsics(640, 360, hfov_deg=70.0).source == "user"
    assert resolve_intrinsics(640, 360, metadata_focal_px=500.0).source == "metadata"
    assert resolve_intrinsics(640, 360).source == "fov_heuristic"


def test_heuristic_matches_documented_default():
    cam = resolve_intrinsics(640, 360)
    assert cam.hfov_deg == pytest.approx(DEFAULT_HFOV_DEG)


def test_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        camera_from_hfov(0, 360)
    with pytest.raises(ValueError):
        camera_from_hfov(640, 360, hfov_deg=200.0)
    with pytest.raises(ValueError):
        resolve_intrinsics(640, 360, focal_px=-1.0)


def test_project_handles_zero_depth_without_crashing(camera):
    out = camera.project(np.array([[1.0, 1.0, 0.0]]))
    assert np.isfinite(out).all()
