"""Request and response models for the HTTP API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobCreated(BaseModel):
    job_id: str
    state: JobState
    filename: str
    size_bytes: int


class JobProgress(BaseModel):
    stage: str = "queued"
    fraction: float = 0.0
    frames: int = 0
    keyframes: int = 0
    landmarks: int = 0


class JobStatus(BaseModel):
    job_id: str
    state: JobState
    progress: JobProgress
    error: str | None = None
    elapsed_s: float = 0.0
    summary: dict | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    slam_ready: bool = Field(description="Whether the SLAM stack imported successfully")
