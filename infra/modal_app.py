"""Modal deployment for the SLAM service.

Modal is a good fit for this workload for one reason: it bills for CPU by the
second and scales to zero between requests, so a reconstruction service that is
idle most of the day costs almost nothing while still getting real cores when
someone actually uploads a video. The free Starter credit renews monthly, and a
10 second clip costs roughly 30 CPU-seconds.

Deploy with:

    modal deploy infra/modal_app.py

The frontend must be built first (`npm --prefix web run build`), since the
image copies `web/dist` rather than running node itself.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

# Python 3.11 specifically: gtsam publishes no wheels for 3.13+, and the pinned
# dependency set is verified against 3.11. libGL and libglib are needed by
# OpenCV even in the headless build; ffmpeg provides the demuxers it uses to
# read uploaded video.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install_from_requirements(ROOT / "requirements.txt")
    .env({
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        # Processing width and the feature budget are the two levers that move
        # runtime, and the right value depends on how fast the host is. These
        # are the values measured on this hardware; see README.
        "SLAM_TARGET_WIDTH": "448",
        "SLAM_MAX_FEATURES": "560",
        "SLAM_LOOP_VERIFICATIONS": "100",
        "SLAM_VERIFY_COST_S": "0.030",
        "SLAM_RTF_TARGET": "0.80",
    })
    .add_local_dir(ROOT / "slam", "/app/slam", copy=True)
    .add_local_dir(ROOT / "api", "/app/api", copy=True)
    .add_local_dir(ROOT / "web" / "dist", "/app/web/dist", copy=True)
    .workdir("/app")
)

app = modal.App("monocular-slam", image=image)


@app.function(
    # Physical cores. The pipeline runs at roughly 2x parallelism (the threaded
    # decoder overlaps the SLAM thread), so 4 leaves headroom for uvicorn to
    # stay responsive to progress polling while the worker saturates its cores.
    cpu=4.0,
    # Measured steady state is ~390 MB for the API and worker together. 2 GB is
    # headroom for a long or high-resolution clip without over-reserving, since
    # Modal bills provisioned memory by the second.
    memory=2048,
    # The job registry lives in this process and the client polls for progress,
    # so a second container would answer those polls with 404. One container
    # serves every request, exactly like the single-task deployment it replaces.
    max_containers=1,
    # Stay warm long enough to cover a reconstruction and the polling that
    # follows it, then scale to zero so credit is only spent on real use.
    scaledown_window=300,
    timeout=900,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="monocular-slam")
def fastapi_app():
    from api.main import app as fastapi

    return fastapi
