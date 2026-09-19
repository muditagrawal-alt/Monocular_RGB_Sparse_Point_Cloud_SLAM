"""Video decoding with a background producer thread.

Decode is pure I/O and releases the GIL inside OpenCV, so running it on a
separate thread overlaps it with the compute-bound front end and buys back
most of the decode cost from the runtime budget.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0 and self.fps > 0


@dataclass
class DecodedFrame:
    index: int
    timestamp: float
    gray: np.ndarray
    """Downscaled single-channel image the front end runs on."""
    color_small: np.ndarray | None = None
    """Downscaled BGR, kept only at keyframes for landmark colouring."""


def probe_video(path: str | Path) -> VideoInfo:
    """Read container metadata without decoding the whole file."""
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"cannot open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    # Containers frequently report nonsense here; fall back to something sane
    # rather than dividing by zero downstream.
    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 30.0
    if count < 0:
        count = 0
    return VideoInfo(path=path, width=width, height=height, fps=fps,
                     frame_count=count, duration_s=count / fps if fps > 0 else 0.0)


class VideoDecoder:
    """Iterates a video as downscaled grayscale frames.

    Supports frame-rate decimation (`max_fps`): 60 fps footage carries little
    extra geometric information over 30 fps but doubles the work.
    """

    def __init__(self, path: str | Path, target_width: int = 640,
                 max_fps: float = 30.0, max_frames: int | None = None,
                 queue_size: int = 8, keep_color: bool = True) -> None:
        self.info = probe_video(path)
        if not self.info.is_valid:
            raise ValueError(f"invalid video dimensions: {self.info}")
        self.target_width = target_width
        self.max_fps = max_fps
        self.max_frames = max_frames
        self.keep_color = keep_color
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: Exception | None = None

        self.scale = min(1.0, target_width / self.info.width) if self.info.width else 1.0
        self.out_width = max(1, int(round(self.info.width * self.scale)))
        self.out_height = max(1, int(round(self.info.height * self.scale)))

        # Keep every Nth frame to approximate max_fps.
        self.step = max(1, int(round(self.info.fps / max_fps))) if max_fps > 0 else 1
        self.effective_fps = self.info.fps / self.step

    @property
    def expected_frames(self) -> int:
        n = self.info.frame_count // self.step if self.info.frame_count else 0
        return min(n, self.max_frames) if self.max_frames else n

    def _produce(self) -> None:
        cap = cv2.VideoCapture(self.info.path)
        emitted = 0
        raw_index = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if raw_index % self.step != 0:
                    raw_index += 1
                    continue
                if self.max_frames is not None and emitted >= self.max_frames:
                    break

                if self.scale < 1.0:
                    frame = cv2.resize(frame, (self.out_width, self.out_height),
                                       interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self._queue.put(DecodedFrame(
                    index=emitted, timestamp=raw_index / self.info.fps, gray=gray,
                    color_small=frame if self.keep_color else None))
                emitted += 1
                raw_index += 1
        except Exception as exc:
            self._error = exc
        finally:
            cap.release()
            self._queue.put(None)   # sentinel

    def __iter__(self):
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        while True:
            item = self._queue.get()
            if item is None:
                break
            yield item
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class ImageSequenceDecoder:
    """Iterates a folder of images as if it were a video.

    Benchmark datasets ship image sequences with per-frame timestamps rather
    than encoded video, and feeding those directly avoids the compression
    artefacts a re-encode would introduce. It also lets a user point the
    pipeline at a folder of frames.

    `timestamps_file` is the TUM `rgb.txt` format: lines of `timestamp path`,
    with `#` comments. Without it, files are ordered by name at a fixed rate.
    """

    def __init__(self, folder: str | Path, target_width: int = 640,
                 timestamps_file: str | Path | None = None,
                 max_fps: float = 30.0, max_frames: int | None = None,
                 keep_color: bool = True) -> None:
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(str(folder))

        self.entries = self._load_entries(timestamps_file)
        if not self.entries:
            raise ValueError(f"no images found in {folder}")

        first = cv2.imread(str(self.entries[0][1]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"cannot read {self.entries[0][1]}")
        height, width = first.shape[:2]

        # Derive a nominal frame rate from the timestamps so downstream budget
        # calculations have a real duration to work with.
        if len(self.entries) > 1:
            span = self.entries[-1][0] - self.entries[0][0]
            fps = (len(self.entries) - 1) / span if span > 0 else 30.0
        else:
            fps = 30.0

        self.info = VideoInfo(path=str(self.folder), width=width, height=height,
                              fps=float(np.clip(fps, 1.0, 240.0)),
                              frame_count=len(self.entries),
                              duration_s=(self.entries[-1][0] - self.entries[0][0]
                                          if len(self.entries) > 1 else 0.0))
        self.target_width = target_width
        self.max_frames = max_frames
        self.keep_color = keep_color
        self.scale = min(1.0, target_width / width) if width else 1.0
        self.out_width = max(1, int(round(width * self.scale)))
        self.out_height = max(1, int(round(height * self.scale)))
        self.step = max(1, int(round(self.info.fps / max_fps))) if max_fps > 0 else 1
        self.effective_fps = self.info.fps / self.step

    def _load_entries(self, timestamps_file: str | Path | None
                      ) -> list[tuple[float, Path]]:
        if timestamps_file is not None:
            entries: list[tuple[float, Path]] = []
            for line in Path(timestamps_file).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                path = self.folder.parent / parts[1]
                if not path.exists():
                    path = self.folder / Path(parts[1]).name
                if path.exists():
                    entries.append((float(parts[0]), path))
            return entries

        suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        files = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in suffixes)
        return [(i / 30.0, p) for i, p in enumerate(files)]

    @property
    def expected_frames(self) -> int:
        n = len(self.entries) // self.step
        return min(n, self.max_frames) if self.max_frames else n

    def __iter__(self):
        emitted = 0
        t0 = self.entries[0][0]
        for index, (timestamp, path) in enumerate(self.entries):
            if index % self.step != 0:
                continue
            if self.max_frames is not None and emitted >= self.max_frames:
                break
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if self.scale < 1.0:
                frame = cv2.resize(frame, (self.out_width, self.out_height),
                                   interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield DecodedFrame(index=emitted, timestamp=timestamp - t0, gray=gray,
                               color_small=frame if self.keep_color else None)
            emitted += 1

    def close(self) -> None:
        """Present for parity with VideoDecoder; nothing to release."""
