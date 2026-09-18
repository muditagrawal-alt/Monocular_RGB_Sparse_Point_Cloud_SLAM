"""Job manager: runs SLAM off the event loop and tracks progress.

SLAM is CPU-bound and takes seconds, so it runs in a separate *process*. A
thread would contend on the GIL with the event loop and make progress polling
unresponsive; a process keeps the API snappy while SLAM saturates a core.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slam.config import SlamConfig
from slam.io.exporters import result_to_viewer_json
from slam.pipeline import SlamPipeline

from .schemas import JobProgress, JobState

MAX_UPLOAD_BYTES = 120_000_000
MAX_DURATION_S = 60.0
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@dataclass
class Job:
    id: str
    filename: str
    size_bytes: int
    path: str
    state: JobState = JobState.QUEUED
    progress: JobProgress = field(default_factory=JobProgress)
    error: str | None = None
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    focal_px: float | None = None
    hfov_deg: float | None = None
    progress_path: str = ""

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.finished_at or time.time()) - self.started_at


def _run_slam(video_path: str, focal_px: float | None, hfov_deg: float | None,
              progress_path: str) -> dict:
    """Entry point executed in the worker process.

    Progress is written to a small JSON file rather than sent over a
    multiprocessing queue. A Manager queue needs a broker process that
    re-imports __main__, which is fragile under the "spawn" start method used
    on macOS and in containers; a file works with any start method, costs
    nothing at this update rate, and can be inspected directly when debugging.
    """

    def report(stage: str, fraction: float, extra: dict) -> None:
        try:
            tmp = f"{progress_path}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"stage": stage, "fraction": fraction, **extra}, fh)
            os.replace(tmp, progress_path)   # atomic: readers never see a partial file
        except OSError:
            pass   # progress reporting must never break the pipeline

    result = SlamPipeline(SlamConfig()).run(
        video_path, focal_px=focal_px, hfov_deg=hfov_deg, progress=report)
    return result_to_viewer_json(result)


class JobManager:
    """Owns job lifecycle, temp files, and the worker pool."""

    def __init__(self, max_workers: int = 1, retain_jobs: int = 40) -> None:
        # One worker by default: SLAM already uses the CPU heavily, and running
        # several jobs in parallel would push each one past its time budget.
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Any] = {}
        self._retain = retain_jobs
        self._root = Path(tempfile.mkdtemp(prefix="slam-uploads-"))

    # -- lifecycle --------------------------------------------------------
    def create(self, filename: str, data: bytes, *, focal_px: float | None = None,
               hfov_deg: float | None = None) -> Job:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(
                f"unsupported file type '{suffix or filename}'; "
                f"expected one of {', '.join(sorted(ALLOWED_SUFFIXES))}")
        if not data:
            raise ValueError("uploaded file is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"file is {len(data) / 1e6:.1f} MB; limit is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB")

        job_id = uuid.uuid4().hex[:16]
        job_dir = self._root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        path = job_dir / f"input{suffix}"
        path.write_bytes(data)

        self._validate_video(str(path))

        job = Job(id=job_id, filename=filename, size_bytes=len(data), path=str(path),
                  focal_px=focal_px, hfov_deg=hfov_deg,
                  progress_path=str(job_dir.parent / f"{job_id}.progress.json"))
        self._jobs[job_id] = job
        self._prune()
        return job

    @staticmethod
    def _validate_video(path: str) -> None:
        """Reject anything undecodable or over-long before committing a worker."""
        from slam.io.video import probe_video

        try:
            info = probe_video(path)
        except Exception as exc:
            raise ValueError(f"file is not a readable video: {exc}") from exc
        if not info.is_valid:
            raise ValueError("video has invalid dimensions or frame rate")
        if info.duration_s > MAX_DURATION_S:
            raise ValueError(
                f"video is {info.duration_s:.1f}s; limit is {MAX_DURATION_S:.0f}s")

    def start(self, job_id: str) -> None:
        job = self._jobs[job_id]
        job.state = JobState.RUNNING
        job.started_at = time.time()
        self._futures[job_id] = self._executor.submit(
            _run_slam, job.path, job.focal_px, job.hfov_deg, job.progress_path)

    def get(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is not None:
            self._poll(job)
        return job

    # -- internals --------------------------------------------------------
    def _poll(self, job: Job) -> None:
        """Read the latest progress snapshot and collect the result if finished."""
        try:
            with open(job.progress_path) as fh:
                msg = json.load(fh)
            job.progress = JobProgress(
                stage=str(msg.get("stage", job.progress.stage)),
                fraction=float(msg.get("fraction", job.progress.fraction)),
                frames=int(msg.get("frames", job.progress.frames)),
                keyframes=int(msg.get("keyframes", job.progress.keyframes)),
                landmarks=int(msg.get("landmarks", job.progress.landmarks)))
        except (OSError, ValueError):
            pass   # not written yet, or mid-replace

        future = self._futures.get(job.id)
        if future is None or not future.done() or job.state in (
                JobState.COMPLETED, JobState.FAILED):
            return

        job.finished_at = time.time()
        try:
            payload = future.result()
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = f"processing failed: {exc}"
            self._cleanup_input(job)
            return

        if not payload.get("success"):
            job.state = JobState.FAILED
            job.error = payload.get("reason") or "reconstruction failed"
        else:
            job.state = JobState.COMPLETED
            job.result = payload
            job.progress = JobProgress(
                stage="done", fraction=1.0,
                frames=payload["summary"].get("frames", 0),
                keyframes=payload["summary"].get("keyframes", 0),
                landmarks=payload["summary"].get("landmarks", 0))
        self._cleanup_input(job)

    @staticmethod
    def _cleanup_input(job: Job) -> None:
        """Delete the uploaded video once it is no longer needed."""
        try:
            parent = Path(job.path).parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)
            for leftover in (job.progress_path, f"{job.progress_path}.tmp"):
                if leftover and os.path.exists(leftover):
                    os.unlink(leftover)
        except OSError:
            pass

    def _prune(self) -> None:
        """Bound memory by dropping the oldest finished jobs."""
        if len(self._jobs) <= self._retain:
            return
        finished = sorted(
            (j for j in self._jobs.values()
             if j.state in (JobState.COMPLETED, JobState.FAILED)),
            key=lambda j: j.finished_at or j.created_at)
        for job in finished[: len(self._jobs) - self._retain]:
            self._jobs.pop(job.id, None)
            self._futures.pop(job.id, None)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self._root, ignore_errors=True)
