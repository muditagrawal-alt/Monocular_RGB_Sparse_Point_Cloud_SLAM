"""Loop-closure detection -- the retrieval half of drift-control layer L4.

Local BA (L3) suppresses how fast error grows but cannot remove error that has
already accumulated. Only recognising a revisited place supplies a constraint
between two distant keyframes, which is what lets the pose graph redistribute
accumulated drift.

Detection is deliberately staged cheap-to-expensive, because a false loop is
far more damaging than a missed one -- it warps the whole map:

1. **Bag-of-words retrieval** over ORB descriptors: a fast shortlist.
2. **Geometric verification**: descriptor matching plus RANSAC. A pair that
   cannot agree on a rigid geometry is not the same place.
3. **Temporal consistency**: the same region must be proposed across several
   consecutive keyframes, which rejects one-off perceptual aliasing.

The vocabulary is trained online with mini-batch k-means on descriptors from
the sequence itself, so no pretrained vocabulary file has to ship.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..camera import Camera
from ..config import LoopClosureConfig
from ..types import Keyframe, Pose, SlamMap


@dataclass
class LoopCandidate:
    query_id: int
    match_id: int
    similarity: float = 0.0
    n_matches: int = 0
    n_inliers: int = 0
    relative_pose: Pose | None = None
    verified: bool = False
    reason: str = ""


@dataclass
class LoopDetectionStats:
    n_queries: int = 0
    n_shortlisted: int = 0
    n_verified: int = 0
    n_rejected_similarity: int = 0
    n_rejected_matches: int = 0
    n_rejected_geometry: int = 0
    n_rejected_consistency: int = 0
    duration_ms: float = 0.0
    loops: list[LoopCandidate] = field(default_factory=list)


class BoWVocabulary:
    """Small visual vocabulary trained online by mini-batch k-means.

    ORB descriptors are binary, so distances use Hamming space. Cluster centres
    are maintained as bit-majority votes of their members, which keeps the
    centres themselves valid binary descriptors.
    """

    def __init__(self, size: int = 256, seed: int = 0) -> None:
        self.size = size
        self.centres: np.ndarray | None = None
        self._rng = np.random.default_rng(seed)
        self.idf: np.ndarray | None = None

    def train(self, descriptors: np.ndarray, iterations: int = 6) -> bool:
        """Train on an Nx32 uint8 descriptor matrix."""
        if descriptors is None or len(descriptors) < self.size:
            return False
        desc = np.asarray(descriptors, dtype=np.uint8)
        idx = self._rng.choice(len(desc), self.size, replace=False)
        centres = desc[idx].copy()
        bits = np.unpackbits(desc, axis=1).astype(np.float32)

        for _ in range(iterations):
            assign = self._assign(desc, centres)
            new_centres = centres.copy()
            for c in range(self.size):
                members = bits[assign == c]
                if len(members) == 0:
                    continue
                majority = (members.mean(axis=0) > 0.5).astype(np.uint8)
                new_centres[c] = np.packbits(majority)
            if np.array_equal(new_centres, centres):
                break
            centres = new_centres

        self.centres = centres
        return True

    @staticmethod
    def _hamming(desc: np.ndarray, centres: np.ndarray) -> np.ndarray:
        """Pairwise Hamming distances, NxK."""
        d = np.unpackbits(desc, axis=1).astype(np.int16)
        c = np.unpackbits(centres, axis=1).astype(np.int16)
        # (a - b)^2 == a XOR b for bits, so a matmul gives the distances
        return (d @ (1 - c).T) + ((1 - d) @ c.T)

    def _assign(self, desc: np.ndarray, centres: np.ndarray) -> np.ndarray:
        return np.argmin(self._hamming(desc, centres), axis=1)

    def describe(self, descriptors: np.ndarray | None) -> np.ndarray | None:
        """L2-normalised term-frequency histogram for one keyframe."""
        if self.centres is None or descriptors is None or len(descriptors) == 0:
            return None
        assign = self._assign(np.asarray(descriptors, dtype=np.uint8), self.centres)
        hist = np.bincount(assign, minlength=self.size).astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else None


class LoopDetector:
    """Detects and geometrically verifies revisited places."""

    def __init__(self, camera: Camera, config: LoopClosureConfig) -> None:
        self.camera = camera
        self.cfg = config
        self.vocab = BoWVocabulary(config.vocab_size)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._trained = False
        self._consistency: dict[int, int] = {}

    # -- vocabulary -------------------------------------------------------
    def build_vocabulary(self, keyframes: list[Keyframe]) -> bool:
        """Train the vocabulary from descriptors across the sequence."""
        pool = [kf.descriptors for kf in keyframes if kf.descriptors is not None]
        if not pool:
            return False
        desc = np.vstack(pool)
        if len(desc) > self.cfg.vocab_train_descriptors:
            sel = np.random.default_rng(0).choice(
                len(desc), self.cfg.vocab_train_descriptors, replace=False)
            desc = desc[sel]
        self._trained = self.vocab.train(desc)
        return self._trained

    def encode_all(self, keyframes: list[Keyframe]) -> None:
        for kf in keyframes:
            kf.bow = self.vocab.describe(kf.descriptors)

    # -- detection --------------------------------------------------------
    def detect(self, slam_map: SlamMap) -> LoopDetectionStats:
        """Find verified loop closures across the whole keyframe set.

        Runs once after tracking rather than incrementally: at a few hundred
        keyframes the whole search is milliseconds, and a single pass keeps the
        logic simple and deterministic.
        """
        started = time.perf_counter()
        stats = LoopDetectionStats()

        kfs = [slam_map.keyframes[k] for k in slam_map.keyframe_ids]
        if len(kfs) < self.cfg.min_keyframe_separation + 2:
            stats.duration_ms = (time.perf_counter() - started) * 1000
            return stats

        if not self._trained and not self.build_vocabulary(kfs):
            stats.duration_ms = (time.perf_counter() - started) * 1000
            return stats
        self.encode_all(kfs)

        bows = {kf.id: kf.bow for kf in kfs if kf.bow is not None}
        if len(bows) < 2:
            stats.duration_ms = (time.perf_counter() - started) * 1000
            return stats

        accepted: list[LoopCandidate] = []
        consistent: dict[int, int] = {}

        for kf in kfs:
            if kf.bow is None:
                continue
            stats.n_queries += 1

            # Only consider keyframes far enough back in time; nearby ones are
            # just ordinary tracking, not a loop.
            older = [(other_id, float(kf.bow @ bow))
                     for other_id, bow in bows.items()
                     if kf.id - other_id >= self.cfg.min_keyframe_separation]
            if not older:
                continue

            older.sort(key=lambda p: -p[1])
            shortlist = [(oid, sim) for oid, sim in older[: self.cfg.top_k_candidates]
                         if sim >= self.cfg.min_bow_similarity]
            if not shortlist:
                stats.n_rejected_similarity += 1
                continue
            stats.n_shortlisted += len(shortlist)

            for other_id, sim in shortlist:
                cand = self._verify(slam_map, kf, slam_map.keyframes[other_id], sim)
                if cand.verified:
                    # Temporal consistency: require the same region to be
                    # proposed by consecutive keyframes before trusting it.
                    bucket = other_id // 5
                    consistent[bucket] = consistent.get(bucket, 0) + 1
                    if consistent[bucket] >= self.cfg.consistency_required:
                        accepted.append(cand)
                        stats.n_verified += 1
                    else:
                        stats.n_rejected_consistency += 1
                    break
                if "matches" in cand.reason:
                    stats.n_rejected_matches += 1
                else:
                    stats.n_rejected_geometry += 1

        stats.loops = accepted
        stats.duration_ms = (time.perf_counter() - started) * 1000
        return stats

    def _verify(self, slam_map: SlamMap, query: Keyframe, match: Keyframe,
                similarity: float) -> LoopCandidate:
        """Geometrically verify a retrieval candidate.

        Estimates the relative pose from 3D landmarks in the matched keyframe
        against 2D observations in the query, so the resulting constraint is
        metrically consistent with the existing map rather than up-to-scale.
        """
        cand = LoopCandidate(query_id=query.id, match_id=match.id, similarity=similarity)

        if (query.descriptors is None or match.descriptors is None
                or query.keypoints is None or match.keypoints is None):
            cand.reason = "no descriptors"
            return cand

        raw = self._matcher.knnMatch(query.descriptors, match.descriptors, k=2)
        good = [m for pair in raw if len(pair) == 2
                for m, n in [pair] if m.distance < 0.75 * n.distance]
        cand.n_matches = len(good)
        if len(good) < self.cfg.min_match_count:
            cand.reason = f"too few matches ({len(good)})"
            return cand

        # Pair each matched keypoint in `match` with a mapped landmark.
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        match_pt_to_lm = {i: lm for i, lm in match.landmark_ids.items()}
        if not match_pt_to_lm:
            cand.reason = "matched keyframe has no landmarks"
            return cand

        match_kp = match.keypoints
        for m in good:
            # nearest mapped point in `match` to this keypoint
            kp = match_kp[m.trainIdx]
            best_idx, best_d = None, 1e9
            for pt_idx in match_pt_to_lm:
                if pt_idx >= len(match.points):
                    continue
                d = float(np.linalg.norm(match.points[pt_idx] - kp))
                if d < best_d:
                    best_d, best_idx = d, pt_idx
            if best_idx is None or best_d > 3.0:
                continue
            lm = slam_map.landmarks.get(match_pt_to_lm[best_idx])
            if lm is None or lm.is_outlier:
                continue
            obj_pts.append(lm.position)
            img_pts.append(query.keypoints[m.queryIdx])

        if len(obj_pts) < self.cfg.min_inlier_count:
            cand.reason = f"too few 2D-3D pairs for geometry ({len(obj_pts)})"
            return cand

        obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
        img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.camera.K, None, iterationsCount=200,
            reprojectionError=self.cfg.ransac_threshold_px, confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE)

        if not ok or inliers is None or len(inliers) < self.cfg.min_inlier_count:
            cand.n_inliers = 0 if inliers is None else len(inliers)
            cand.reason = f"geometry rejected ({cand.n_inliers} inliers)"
            return cand

        cand.n_inliers = len(inliers)
        query_pose = Pose.from_rvec_tvec(rvec, tvec)
        # Constraint expressed as the transform from match into query frame.
        cand.relative_pose = query_pose.inverse().compose(match.pose)
        cand.verified = True
        cand.reason = "ok"
        return cand
