"""HTTP API for the monocular SLAM service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from slam import __version__

from .jobs import MAX_DURATION_S, MAX_UPLOAD_BYTES, JobManager
from .schemas import HealthResponse, JobCreated, JobStatus

logger = logging.getLogger("slam.api")

manager: JobManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = JobManager()
    logger.info("job manager started")
    yield
    if manager is not None:
        manager.shutdown()


app = FastAPI(
    title="Monocular RGB Sparse Point-Cloud SLAM",
    description="Estimate camera trajectory and sparse 3D structure from a single-lens "
                "RGB video, with loop-closure drift correction.",
    version=__version__,
    lifespan=lifespan)

# The API and the static frontend are served from the same origin in the
# container, so CORS only matters for local development against a Vite dev
# server on another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"])


def _manager() -> JobManager:
    if manager is None:
        raise HTTPException(503, "service is still starting")
    return manager


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness probe for the load balancer."""
    try:
        import cv2  # noqa: F401
        import gtsam  # noqa: F401
        ready = True
    except Exception:
        ready = False
    return HealthResponse(status="ok", version=__version__, slam_ready=ready)


@app.get("/api/limits")
def limits() -> dict:
    """Upload constraints, so the client can validate before sending bytes."""
    return {
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_duration_s": MAX_DURATION_S,
        "accepted_types": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"],
    }


@app.post("/api/jobs", response_model=JobCreated, status_code=202)
async def create_job(video: UploadFile = File(...),
                     focal_px: float | None = Form(default=None),
                     hfov_deg: float | None = Form(default=None)) -> JobCreated:
    """Accept a video and begin reconstruction.

    `focal_px` / `hfov_deg` are optional: an uploaded video carries no
    calibration, so the pipeline otherwise assumes a 60 degree horizontal field
    of view and reports which assumption it used.
    """
    data = await video.read()
    try:
        job = _manager().create(video.filename or "upload.mp4", data,
                                focal_px=focal_px, hfov_deg=hfov_deg)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    _manager().start(job.id)
    return JobCreated(job_id=job.id, state=job.state, filename=job.filename,
                      size_bytes=job.size_bytes)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = _manager().get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return JobStatus(
        job_id=job.id, state=job.state, progress=job.progress, error=job.error,
        elapsed_s=round(job.elapsed_s, 3),
        summary=(job.result or {}).get("summary") if job.result else None)


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> JSONResponse:
    """Full reconstruction: point cloud, trajectories, keyframes, telemetry."""
    job = _manager().get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.state.value == "failed":
        raise HTTPException(422, job.error or "reconstruction failed")
    if job.result is None:
        raise HTTPException(409, f"job is {job.state.value}; result not ready")
    return JSONResponse(job.result)


# The built frontend is copied here by the Docker image. Mounting it last means
# API routes always take precedence over the static catch-all.
_STATIC = Path(__file__).resolve().parent.parent / "web" / "dist"
if _STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/{path:path}")
    def spa_fallback(path: str) -> FileResponse:
        candidate = _STATIC / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_STATIC / "index.html")
