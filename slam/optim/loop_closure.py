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
        self._assign_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
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

    def _assign(self, desc: np.ndarray, centres: np.ndarray) -> np.ndarray:
        """Nearest cluster centre for each descriptor, in Hamming space.

        Delegates to OpenCV's brute-force matcher, which has a SIMD popcount
        path. Only the argmin is needed, never the full distance matrix: an
        earlier version built the whole NxK matrix with np.unpackbits and a
        matmul, which profiled at 223 ms per call and dominated the entire
        pipeline runtime.
        """
        if len(desc) == 0:
            return np.zeros(0, dtype=np.int64)
        matches = self._assign_matcher.match(
            np.ascontiguousarray(desc, dtype=np.uint8),
            np.ascontiguousarray(centres, dtype=np.uint8))
        out = np.zeros(len(desc), dtype=np.int64)
        for m in matches:
            out[m.queryIdx] = m.trainIdx
        return out

    def term_frequency(self, descriptors: np.ndarray | None) -> np.ndarray | None:
        """Raw word counts for one keyframe."""
        if (self.centres is None or descriptors is None or len(descriptors) == 0
                or descriptors.shape[1] != self.centres.shape[1]):
            return None
        assign = self._assign(np.asarray(descriptors, dtype=np.uint8), self.centres)
        return np.bincount(assign, minlength=self.size).astype(np.float32)

    def fit_idf(self, term_frequencies: list[np.ndarray]) -> None:
        """Compute inverse document frequency over the keyframe set.

        Without IDF, words that appear in nearly every keyframe dominate the
        similarity and all keyframes look alike. Measured on a synthetic orbit,
        plain term frequency ranked the true loop partner 17th of 86 candidates
        -- outside any sane shortlist -- because common words swamped the
        distinctive ones. IDF is what makes retrieval discriminative.
        """
        if not term_frequencies:
            self.idf = None
            return
        n_docs = len(term_frequencies)
        df = np.zeros(self.size, dtype=np.float32)
        for tf in term_frequencies:
            df += (tf > 0).astype(np.float32)
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)).astype(np.float32) + 1.0

    def describe(self, descriptors: np.ndarray | None) -> np.ndarray | None:
        """L2-normalised TF-IDF histogram for one keyframe."""
        tf = self.term_frequency(descriptors)
        if tf is None:
            return None
        vec = tf * self.idf if self.idf is not None else tf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None


class LoopDetector:
    """Detects and geometrically verifies revisited places."""

    def __init__(self, camera: Camera, config: LoopClosureConfig) -> None:
        self.camera = camera
        self.cfg = config
        self.vocab = BoWVocabulary(config.vocab_size)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._trained = False

    # -- vocabulary -------------------------------------------------------
    def build_vocabulary(self, keyframes: list[Keyframe]) -> bool:
        """Train the vocabulary from descriptors across the sequence."""
        pool = [kf.descriptors for kf in keyframes
                if kf.descriptors is not None and len(kf.descriptors) > 0]
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
        """Encode every keyframe, fitting IDF over the sequence first."""
        tfs = [self.vocab.term_frequency(kf.descriptors) for kf in keyframes]
        self.vocab.fit_idf([t for t in tfs if t is not None])
        for kf, tf in zip(keyframes, tfs):
            if tf is None:
                kf.bow = None
                continue
            vec = tf * self.vocab.idf if self.vocab.idf is not None else tf
            norm = np.linalg.norm(vec)
            kf.bow = (vec / norm).astype(np.float32) if norm > 0 else None

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

        verified: list[LoopCandidate] = []
        centres = {kf.id: kf.pose.center for kf in kfs}

        # Build one global candidate list rather than spending a per-query
        # budget in sequence. Verifying in keyframe order exhausts the budget
        # on early keyframes, which cannot have loop partners yet, and starves
        # the later ones where loops actually close.
        pairs: dict[tuple[int, int], float] = {}
        for kf in kfs[:: max(1, self.cfg.query_stride)]:
            if kf.bow is None:
                continue
            stats.n_queries += 1
            eligible = [o for o in kfs
                        if kf.id - o.id >= self.cfg.min_keyframe_separation
                        and o.bow is not None]
            if not eligible:
                continue

            # Two independent signals, fused by reciprocal rank because their
            # scores are not on comparable scales: appearance similarity
            # (robust to drift, weak on repetitive texture) and proximity in
            # the current estimate (precise until drift exceeds the loop).
            bow_rank = {oid: r for r, (_, oid) in enumerate(sorted(
                ((float(kf.bow @ o.bow), o.id) for o in eligible), reverse=True))}
            dist_rank = {o.id: r for r, o in enumerate(sorted(
                eligible,
                key=lambda o: float(np.linalg.norm(centres[o.id] - centres[kf.id]))))}

            scored = []
            for o in eligible:
                rrf = 1.0 / (10 + bow_rank[o.id]) + 1.0 / (10 + dist_rank[o.id])
                scored.append((rrf, o.id))
            scored.sort(reverse=True)

            keep = self.cfg.top_k_candidates + self.cfg.spatial_candidates
            for rrf, oid in scored[:keep]:
                stats.n_shortlisted += 1
                pairs[(kf.id, oid)] = max(pairs.get((kf.id, oid), 0.0), rrf)

        # Spend the verification budget where a loop can plausibly be, rather
        # than spreading it evenly. A loop exists only where the trajectory
        # comes back near its own past, so queries are ordered by how close
        # they get to any temporally distant keyframe. Splitting the budget
        # evenly instead gave every query ~9 candidates and the true loop sat
        # at rank 10, so it was never verified at all.
        by_query: dict[int, list[tuple[float, int]]] = {}
        for (qid, oid), score in pairs.items():
            by_query.setdefault(qid, []).append((score, oid))

        def closest_approach(qid: int) -> float:
            return min((float(np.linalg.norm(centres[oid] - centres[qid]))
                        for _, oid in by_query[qid]), default=np.inf)

        query_order = sorted(by_query, key=closest_approach)

        budget = self.cfg.max_verifications
        per_query = max(1, self.cfg.candidates_per_query)

        ordered: list[tuple[int, int]] = []
        for qid in query_order:
            cands = sorted(by_query[qid], reverse=True)
            ordered.extend((qid, oid) for _, oid in cands[:per_query])

        best_per_query: dict[int, LoopCandidate] = {}
        for qid, oid in ordered[:budget]:
            cand = self._verify(slam_map, slam_map.keyframes[qid],
                                slam_map.keyframes[oid],
                                float(slam_map.keyframes[qid].bow
                                      @ slam_map.keyframes[oid].bow))
            if cand.verified:
                prev = best_per_query.get(qid)
                if prev is None or cand.n_inliers > prev.n_inliers:
                    best_per_query[qid] = cand
            elif "matches" in cand.reason:
                stats.n_rejected_matches += 1
            else:
                stats.n_rejected_geometry += 1
        verified = list(best_per_query.values())

        # Temporal consistency: a region must be proposed by several distinct
        # query keyframes. A single isolated match is far more likely to be
        # perceptual aliasing than a real revisit, and a false loop warps the
        # whole map.
        bucket_size = max(1, self.cfg.region_bucket)
        votes: dict[int, int] = {}
        for cand in verified:
            votes[cand.match_id // bucket_size] = votes.get(cand.match_id // bucket_size, 0) + 1

        accepted = []
        for cand in verified:
            corroborated = (votes[cand.match_id // bucket_size]
                            >= self.cfg.consistency_required)
            strong = cand.n_inliers >= self.cfg.strong_inlier_count
            if corroborated or strong:
                accepted.append(cand)
                stats.n_verified += 1
            else:
                stats.n_rejected_consistency += 1

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
                or query.desc_indices is None or match.desc_indices is None
                or len(query.descriptors) == 0 or len(match.descriptors) == 0):
            cand.reason = "no descriptors"
            return cand

        # Mutual-best matching with a Hamming cap. RANSAC below is the real
        # filter, so being slightly permissive here costs nothing and recovers
        # true loops that a strict ratio test discards.
        good = [m for m in self._matcher.match(query.descriptors, match.descriptors)
                if m.distance <= self.cfg.match_max_distance]
        cand.n_matches = len(good)
        if len(good) < self.cfg.min_match_count:
            cand.reason = f"too few matches ({len(good)})"
            return cand

        # Descriptors were computed at the tracked points, so a match maps
        # straight to a landmark with no proximity test and no association
        # error: descriptor row -> tracked point index -> landmark.
        match_pts = match.desc_indices[[m.trainIdx for m in good]]
        query_pts = query.desc_indices[[m.queryIdx for m in good]]
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        n_query_pts = len(query.points)
        for mp, qp in zip(match_pts.tolist(), query_pts.tolist()):
            lm_id = match.landmark_ids.get(mp)
            if lm_id is None or qp >= n_query_pts:
                continue
            lm = slam_map.landmarks.get(lm_id)
            if lm is None or lm.is_outlier:
                continue
            obj_pts.append(lm.position)
            img_pts.append(query.points[qp])

        if len(obj_pts) < self.cfg.min_inlier_count:
            cand.reason = f"too few 2D-3D pairs for geometry ({len(obj_pts)})"
            return cand

        obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
        img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, self.camera.K, None,
            iterationsCount=self.cfg.ransac_iterations,
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
