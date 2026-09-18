# syntax=docker/dockerfile:1

# ---------- stage 1: build the frontend ----------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---------- stage 2: runtime ----------
# Python 3.11 specifically: gtsam publishes no wheels for 3.13+, and 3.12 works
# but 3.11 is what the dependency set was pinned and verified against.
FROM python:3.11-slim AS runtime

# libGL and libglib are required by OpenCV even in the headless build; ffmpeg
# provides the demuxers OpenCV uses to read uploaded video.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies are copied and installed before application code so the layer
# cache survives source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY slam/ ./slam/
COPY api/ ./api/
COPY --from=web /web/dist ./web/dist

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

# One worker: SLAM saturates the CPU for seconds at a time, so concurrent jobs
# would push each past its time budget. Scale out with more tasks, not more
# workers per task.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
