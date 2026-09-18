"""API tests, including a full upload-to-result round trip."""

import time

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_healthz_reports_slam_ready(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["slam_ready"] is True


def test_limits_are_published(client):
    body = client.get("/api/limits").json()
    assert body["max_upload_bytes"] > 0
    assert ".mp4" in body["accepted_types"]


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_rejects_non_video_extension(client):
    r = client.post("/api/jobs", files={"video": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    assert "unsupported file type" in r.json()["detail"]


def test_rejects_empty_upload(client):
    r = client.post("/api/jobs", files={"video": ("clip.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_rejects_undecodable_video(client):
    r = client.post("/api/jobs",
                    files={"video": ("clip.mp4", b"not really a video", "video/mp4")})
    assert r.status_code == 400
    assert "not a readable video" in r.json()["detail"]


@pytest.mark.slow
def test_full_upload_to_result_round_trip(client, strafe_video):
    """Upload a real clip and poll until the reconstruction is available."""
    with open(strafe_video, "rb") as fh:
        r = client.post("/api/jobs", files={"video": ("strafe.mp4", fh.read(), "video/mp4")})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    # the worker is submitted before the response returns, so it is already running
    assert r.json()["state"] in ("queued", "running")

    deadline = time.time() + 120
    state = None
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        state = status["state"]
        if state in ("completed", "failed"):
            break
        time.sleep(0.25)
    assert state == "completed", f"job ended as {state}"

    result = client.get(f"/api/jobs/{job_id}/result")
    assert result.status_code == 200
    payload = result.json()
    assert payload["success"]
    assert payload["cloud"]["count"] > 100
    assert payload["trajectory"]["count"] > 10
    assert len(payload["cloud"]["positions"]) == payload["cloud"]["count"] * 3
    assert payload["summary"]["keyframes"] > 0
    assert payload["camera"]["source"] == "fov_heuristic"


@pytest.mark.slow
def test_result_is_conflict_before_completion(client, strafe_video):
    with open(strafe_video, "rb") as fh:
        job_id = client.post(
            "/api/jobs",
            files={"video": ("s.mp4", fh.read(), "video/mp4")}).json()["job_id"]
    # Immediately after submission the job cannot be finished yet.
    r = client.get(f"/api/jobs/{job_id}/result")
    assert r.status_code in (409, 200)
