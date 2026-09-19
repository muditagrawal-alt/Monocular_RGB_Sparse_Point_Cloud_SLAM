"""Landmark creation and culling.

New structure is triangulated between the incoming keyframe and its covisible
neighbours, then gated on parallax, positive depth, reprojection error, and a
range sanity bound. Landmarks that later fail these checks are culled: this is
drift-control layer L1, removing bad structure before it can corrupt pose
estimates downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..camera import Camera
from ..config import MappingConfig
from ..geometry import filter_triangulated, reprojection_errors, triangulate
from ..types import Keyframe, SlamMap


@dataclass
class MappingResult:
    n_created: int = 0
    n_culled: int = 0
    n_merged: int = 0


class Mapper:
    """Grows and maintains the sparse landmark map."""

    def __init__(self, camera: Camera, config: MappingConfig) -> None:
        self.camera = camera
        self.cfg = config

    def create_landmarks(self, slam_map: SlamMap, keyframe: Keyframe,
                         track_to_landmark: dict[int, int],
                         reference_ids: list[int] | None = None) -> MappingResult:
        """Triangulate tracks not yet backed by a landmark.

        Only tracks seen in both the new keyframe and a reference keyframe can
        be triangulated, which the shared persistent track id identifies.
        """
        result = MappingResult()
        if reference_ids is None:
            candidates = [k for k in slam_map.keyframe_ids if k != keyframe.id]
            reference_ids = candidates[-3:]
        if not reference_ids:
            return result

        # Scale reference must come from a keyframe that already has landmarks.
        # Asking the incoming keyframe is a no-op, because its landmarks are
        # created below: median_depth then returns its 1.0 fallback and the
        # ratio silently becomes an absolute depth in slam units, which is
        # meaningless once the map settles at any other scale.
        median_depth = 0.0
        for ref_id in reversed(reference_ids):
            median_depth = slam_map.median_depth(ref_id)
            if median_depth > 0:
                break
        if median_depth <= 0:
            median_depth = 1.0
        max_depth = median_depth * self.cfg.max_depth_ratio

        # tracks in this keyframe still lacking 3D structure
        pending = {int(tid): idx for idx, tid in enumerate(keyframe.track_ids)
                   if int(tid) not in track_to_landmark}
        if not pending:
            return result

        for ref_id in reversed(reference_ids):
            if not pending:
                break
            ref = slam_map.keyframes.get(ref_id)
            if ref is None:
                continue

            ref_index = {int(tid): i for i, tid in enumerate(ref.track_ids)}
            shared = [(tid, idx, ref_index[tid]) for tid, idx in pending.items()
                      if tid in ref_index]
            if len(shared) < 2:
                continue

            cur_pts = np.array([keyframe.points[i] for _, i, _ in shared], dtype=np.float64)
            ref_pts = np.array([ref.points[j] for _, _, j in shared], dtype=np.float64)

            pts3d = triangulate(self.camera, ref.pose, keyframe.pose, ref_pts, cur_pts)
            good = filter_triangulated(
                self.camera, ref.pose, keyframe.pose, ref_pts, cur_pts, pts3d,
                min_parallax_deg=self.cfg.min_parallax_deg,
                max_reproj_error_px=self.cfg.max_reproj_error_px,
                min_depth=self.cfg.min_depth, max_depth=max_depth)

            for k, (tid, cur_idx, ref_idx) in enumerate(shared):
                if not good[k]:
                    continue
                lm = slam_map.add_landmark(pts3d[k], color=(210, 210, 210))
                slam_map.observe(lm.id, ref.id, ref_idx)
                slam_map.observe(lm.id, keyframe.id, cur_idx)
                track_to_landmark[tid] = lm.id
                pending.pop(tid, None)
                result.n_created += 1

        return result

    def add_observations(self, slam_map: SlamMap, keyframe: Keyframe,
                         track_to_landmark: dict[int, int]) -> int:
        """Attach this keyframe's observations of already-known landmarks.

        Multi-view observations are what make bundle adjustment able to
        constrain a landmark at all, so this matters as much as creation.
        """
        n = 0
        for idx, tid in enumerate(keyframe.track_ids):
            lm_id = track_to_landmark.get(int(tid))
            if lm_id is None or lm_id not in slam_map.landmarks:
                continue
            if keyframe.id in slam_map.landmarks[lm_id].observations:
                continue
            slam_map.observe(lm_id, keyframe.id, idx)
            n += 1
        return n

    def cull(self, slam_map: SlamMap, keyframe_ids: list[int] | None = None,
             max_error: float | None = None) -> int:
        """Remove landmarks whose reprojection error has grown too large.

        Run after bundle adjustment, where a landmark that cannot be reconciled
        with its observations is revealed as a bad triangulation.

        Errors are accumulated one keyframe at a time so each keyframe's
        observations project in a single vectorised call. Doing this per
        landmark-observation instead profiled at 110,969 separate calls and
        4.4 s of pipeline runtime.
        """
        threshold = max_error if max_error is not None else self.cfg.max_reproj_error_px
        ids = keyframe_ids if keyframe_ids is not None else slam_map.keyframe_ids
        active = [k for k in ids if k in slam_map.keyframes]
        if not active:
            return 0

        err_sum: dict[int, float] = {}
        err_count: dict[int, int] = {}

        for kf_id in active:
            kf = slam_map.keyframes[kf_id]
            if not kf.landmark_ids:
                continue

            lm_ids: list[int] = []
            positions: list[np.ndarray] = []
            pixels: list[np.ndarray] = []
            for pt_idx, lm_id in kf.landmark_ids.items():
                lm = slam_map.landmarks.get(lm_id)
                if lm is None or lm.is_outlier or pt_idx >= len(kf.points):
                    continue
                lm_ids.append(lm_id)
                positions.append(lm.position)
                pixels.append(kf.points[pt_idx])

            if not lm_ids:
                continue

            errors = reprojection_errors(
                self.camera, kf.pose,
                np.asarray(positions, dtype=np.float64).reshape(-1, 3),
                np.asarray(pixels, dtype=np.float64).reshape(-1, 2))

            for lm_id, err in zip(lm_ids, errors, strict=False):
                # A point behind the camera yields inf; treat it as a large
                # finite penalty so one bad view cannot poison the mean.
                e = float(err) if np.isfinite(err) else 1e6
                err_sum[lm_id] = err_sum.get(lm_id, 0.0) + e
                err_count[lm_id] = err_count.get(lm_id, 0) + 1

        # Depth sanity, measured against the settled map scale. A landmark
        # triangulated from near-zero parallax can satisfy its reprojection
        # error while its depth is essentially unconstrained, so it survives an
        # error-only cull and lands hundreds of units away. Those points carry
        # no information and visually shred the cloud, so they go here, where
        # the map scale is actually known.
        scale_depths: list[float] = []
        for kf_id in active:
            d = slam_map.median_depth(kf_id)
            if d > 0:
                scale_depths.append(d)
        depth_limit = (float(np.median(scale_depths)) * self.cfg.max_depth_ratio
                       if scale_depths else None)

        culled = 0
        if depth_limit is not None:
            for kf_id in active:
                kf = slam_map.keyframes[kf_id]
                for lm_id in list(kf.landmark_ids.values()):
                    lm = slam_map.landmarks.get(lm_id)
                    if lm is None or lm.is_outlier:
                        continue
                    z = kf.pose.world_to_camera(lm.position.reshape(1, 3))[0, 2]
                    if z <= 0 or z > depth_limit:
                        lm.is_outlier = True
                        culled += 1

        for lm_id, total in err_sum.items():
            lm = slam_map.landmarks.get(lm_id)
            if lm is None or lm.is_outlier:
                continue
            mean_err = total / max(err_count[lm_id], 1)
            lm.reproj_error = mean_err
            if mean_err > threshold or lm.n_observations < self.cfg.min_observations:
                lm.is_outlier = True
                culled += 1

        return culled
