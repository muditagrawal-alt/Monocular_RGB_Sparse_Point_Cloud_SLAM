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

        median_depth = slam_map.median_depth(keyframe.id) or 1.0
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
        """
        threshold = max_error if max_error is not None else self.cfg.max_reproj_error_px
        ids = keyframe_ids if keyframe_ids is not None else slam_map.keyframe_ids
        active = set(ids)
        culled = 0

        for lm in list(slam_map.landmarks.values()):
            if lm.is_outlier:
                continue
            obs = [(k, i) for k, i in lm.observations.items() if k in active]
            if not obs:
                continue

            errors = []
            for kf_id, pt_idx in obs:
                kf = slam_map.keyframes.get(kf_id)
                if kf is None or pt_idx >= len(kf.points):
                    continue
                err = reprojection_errors(self.camera, kf.pose,
                                          lm.position.reshape(1, 3),
                                          kf.points[pt_idx].reshape(1, 2))
                errors.append(float(err[0]))

            if not errors:
                continue
            mean_err = float(np.mean([e for e in errors if np.isfinite(e)])
                             ) if any(np.isfinite(errors)) else np.inf
            lm.reproj_error = mean_err if np.isfinite(mean_err) else 1e9

            too_bad = (not np.isfinite(mean_err)) or mean_err > threshold
            too_few = lm.n_observations < self.cfg.min_observations
            if too_bad or too_few:
                lm.is_outlier = True
                culled += 1

        return culled
