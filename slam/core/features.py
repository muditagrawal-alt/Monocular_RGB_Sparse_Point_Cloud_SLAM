"""Feature seeding and KLT tracking -- the per-frame front end.

Design note
-----------
This tracks with pyramidal Lucas-Kanade optical flow every frame and computes
ORB descriptors only at keyframes. Benchmarked on this project
(see benchmarks/feasibility/) at 640x360, single core:

    ORB detect + describe + BFMatcher + RANSAC : 27.5 ms/frame
    Shi-Tomasi + pyramidal KLT (800 points)    :  4.3 ms/frame

Roughly 6x cheaper, which is what brings a 300-frame clip inside the 10 s
budget. Descriptors are still needed for loop closure, but only at the ~10% of
frames that become keyframes.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import FrontendConfig


class FeatureTracker:
    """Maintains a set of persistent point tracks across consecutive frames.

    Each live point carries an integer track id that survives as long as the
    track does, which is what lets the mapper accumulate multi-view
    observations of the same landmark.
    """

    def __init__(self, config: FrontendConfig) -> None:
        self.cfg = config
        self._prev_gray: np.ndarray | None = None
        self._points: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self._track_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._next_track_id = 0
        self._lk_params = {
            "winSize": (config.klt_window, config.klt_window),
            "maxLevel": config.klt_levels,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         config.klt_iters, config.klt_eps),
        }

    # -- state ------------------------------------------------------------
    @property
    def points(self) -> np.ndarray:
        """Live tracked points, Nx2 float32."""
        return self._points

    @property
    def track_ids(self) -> np.ndarray:
        return self._track_ids

    @property
    def n_tracks(self) -> int:
        return len(self._points)

    def reset(self) -> None:
        self._prev_gray = None
        self._points = np.zeros((0, 2), dtype=np.float32)
        self._track_ids = np.zeros(0, dtype=np.int64)

    # -- detection --------------------------------------------------------
    def _detect_grid(self, gray: np.ndarray, mask: np.ndarray | None,
                     budget: int) -> np.ndarray:
        """Shi-Tomasi corners distributed over a grid.

        Detecting globally lets every corner land in the highest-contrast
        region, which starves the rest of the image and makes pose estimation
        ill-conditioned. Per-cell budgets keep coverage even.
        """
        h, w = gray.shape[:2]
        cols, rows = max(1, self.cfg.grid_cols), max(1, self.cfg.grid_rows)
        per_cell = max(1, budget // (cols * rows))
        cell_w, cell_h = w / cols, h / rows
        out: list[np.ndarray] = []

        for r in range(rows):
            for c in range(cols):
                x0, y0 = int(c * cell_w), int(r * cell_h)
                x1, y1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                sub = gray[y0:y1, x0:x1]
                if sub.size == 0:
                    continue
                sub_mask = mask[y0:y1, x0:x1] if mask is not None else None
                if sub_mask is not None and not sub_mask.any():
                    continue
                corners = cv2.goodFeaturesToTrack(
                    sub, maxCorners=per_cell, qualityLevel=self.cfg.quality_level,
                    minDistance=self.cfg.min_distance, mask=sub_mask,
                    blockSize=self.cfg.block_size)
                if corners is not None and len(corners):
                    corners = corners.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
                    out.append(corners)

        if not out:
            return np.zeros((0, 2), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    def _occupancy_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Mask excluding a radius around existing tracks, so new corners fill gaps."""
        mask = np.full(shape[:2], 255, dtype=np.uint8)
        r = max(1, self.cfg.min_distance)
        for x, y in self._points:
            cv2.circle(mask, (int(round(x)), int(round(y))), r, 0, -1)
        return mask

    def _seed(self, gray: np.ndarray) -> int:
        """Top up tracks to the feature budget. Returns how many were added."""
        deficit = self.cfg.max_features - self.n_tracks
        if deficit <= 0:
            return 0
        mask = self._occupancy_mask(gray.shape) if self.n_tracks else None
        new_pts = self._detect_grid(gray, mask, deficit)
        if len(new_pts) == 0:
            return 0
        new_pts = new_pts[:deficit]
        new_ids = np.arange(self._next_track_id, self._next_track_id + len(new_pts),
                            dtype=np.int64)
        self._next_track_id += len(new_pts)
        self._points = (np.vstack([self._points, new_pts]).astype(np.float32)
                        if self.n_tracks else new_pts)
        self._track_ids = np.concatenate([self._track_ids, new_ids])
        return len(new_pts)

    # -- tracking ---------------------------------------------------------
    def track(self, gray: np.ndarray) -> TrackResult:
        """Advance all live tracks into `gray`, then top up the feature budget."""
        if gray.ndim != 2:
            raise ValueError("FeatureTracker expects a single-channel image")

        if self._prev_gray is None:
            added = self._seed(gray)
            self._prev_gray = gray
            return TrackResult(points=self._points.copy(), track_ids=self._track_ids.copy(),
                               prev_points=np.zeros((0, 2), np.float32),
                               n_tracked=0, n_added=added, is_first=True)

        prev_pts = self._points
        if len(prev_pts) == 0:
            added = self._seed(gray)
            self._prev_gray = gray
            return TrackResult(points=self._points.copy(), track_ids=self._track_ids.copy(),
                               prev_points=np.zeros((0, 2), np.float32),
                               n_tracked=0, n_added=added, is_first=False)

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts.reshape(-1, 1, 2), None, **self._lk_params)

        if nxt is None:
            self.reset()
            added = self._seed(gray)
            self._prev_gray = gray
            return TrackResult(points=self._points.copy(), track_ids=self._track_ids.copy(),
                               prev_points=np.zeros((0, 2), np.float32),
                               n_tracked=0, n_added=added, is_first=False)

        nxt = nxt.reshape(-1, 2)
        ok = status.reshape(-1).astype(bool)

        # Forward-backward check: track back and require the round trip to land
        # near the origin. This is the cheapest reliable way to drop the
        # drifting or occluded tracks that would otherwise corrupt the map.
        if ok.any():
            back, bstatus, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._prev_gray, nxt[ok].reshape(-1, 1, 2), None, **self._lk_params)
            if back is not None:
                fb_err = np.linalg.norm(back.reshape(-1, 2) - prev_pts[ok], axis=1)
                keep = (bstatus.reshape(-1).astype(bool)
                        & (fb_err < self.cfg.fb_error_threshold))
                idx = np.flatnonzero(ok)
                ok[idx[~keep]] = False

        h, w = gray.shape[:2]
        inside = ((nxt[:, 0] >= 0) & (nxt[:, 0] < w - 1)
                  & (nxt[:, 1] >= 0) & (nxt[:, 1] < h - 1))
        ok &= inside

        self._points = nxt[ok].astype(np.float32)
        prev_kept = prev_pts[ok].astype(np.float32)
        self._track_ids = self._track_ids[ok]
        n_tracked = int(ok.sum())

        added = 0
        if self.n_tracks < self.cfg.max_features * self.cfg.reseed_ratio:
            added = self._seed(gray)

        self._prev_gray = gray
        return TrackResult(points=self._points.copy(), track_ids=self._track_ids.copy(),
                           prev_points=prev_kept, n_tracked=n_tracked,
                           n_added=added, is_first=False)

    def drop(self, keep_mask: np.ndarray) -> None:
        """Drop tracks rejected downstream (e.g. by RANSAC in pose estimation)."""
        keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
        if len(keep_mask) != self.n_tracks:
            raise ValueError(f"mask length {len(keep_mask)} != {self.n_tracks} tracks")
        self._points = self._points[keep_mask]
        self._track_ids = self._track_ids[keep_mask]


class TrackResult:
    """Outcome of tracking one frame."""

    __slots__ = ("points", "track_ids", "prev_points", "n_tracked", "n_added", "is_first")

    def __init__(self, points: np.ndarray, track_ids: np.ndarray, prev_points: np.ndarray,
                 n_tracked: int, n_added: int, is_first: bool) -> None:
        self.points = points
        self.track_ids = track_ids
        self.prev_points = prev_points
        self.n_tracked = n_tracked
        self.n_added = n_added
        self.is_first = is_first

    def __repr__(self) -> str:
        return (f"TrackResult(n={len(self.points)}, tracked={self.n_tracked}, "
                f"added={self.n_added})")


def compute_orb(gray: np.ndarray, n_features: int = 500
                ) -> tuple[np.ndarray, np.ndarray | None]:
    """ORB keypoints and descriptors, for loop-closure retrieval at keyframes."""
    orb = cv2.ORB_create(nfeatures=n_features)
    kps, des = orb.detectAndCompute(gray, None)
    if not kps:
        return np.zeros((0, 2), dtype=np.float32), None
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    return pts, des


def compute_orb_at_points(gray: np.ndarray, points: np.ndarray,
                          patch_size: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """Describe the *tracked* points directly, rather than re-detecting.

    Returns ``(descriptors, kept)`` where ``kept`` indexes into `points`, so
    descriptor row i describes ``points[kept[i]]``.

    Why this instead of a fresh ORB detection: a separate detection produces
    keypoints at different locations than the tracked points, so linking a
    descriptor match back to a mapped landmark needs a proximity test. Measured
    on a synthetic orbit, the median ORB-keypoint to tracked-point distance is
    4.3 px, so a 3 px association test discarded 76% of otherwise-good matches
    and loop closure never fired. Describing the tracked points makes the
    association exact by construction.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 32), dtype=np.uint8), np.zeros(0, dtype=np.int64)

    orb = cv2.ORB_create()
    keypoints = [cv2.KeyPoint(float(x), float(y), float(patch_size)) for x, y in pts]
    # compute() drops keypoints whose patch falls outside the image, so the
    # surviving keypoints are matched back to their source index by position.
    kept_kps, descriptors = orb.compute(gray, keypoints)
    if descriptors is None or not kept_kps:
        return np.zeros((0, 32), dtype=np.uint8), np.zeros(0, dtype=np.int64)

    lookup: dict[tuple[int, int], int] = {}
    for i, (x, y) in enumerate(pts):
        lookup.setdefault((int(round(x * 8)), int(round(y * 8))), i)
    kept = np.array([lookup.get((int(round(kp.pt[0] * 8)), int(round(kp.pt[1] * 8))), -1)
                     for kp in kept_kps], dtype=np.int64)

    valid = kept >= 0
    return descriptors[valid], kept[valid]
