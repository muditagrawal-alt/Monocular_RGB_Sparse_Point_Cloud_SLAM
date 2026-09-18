"""Global pose-graph optimisation -- the correction half of drift-control L4.

Given sequential odometry constraints plus loop-closure constraints, this
redistributes accumulated drift over the whole trajectory so that the two
agree. A loop closure says "these two keyframes are the same place"; the
optimiser is what actually moves the intervening poses to honour that.

Landmarks are not variables here. Optimising poses alone is dramatically
cheaper than full bundle adjustment, and because a pose graph is almost always
well constrained it converges reliably -- which is why it is the standard way
to close large loops. Landmarks are then rigidly carried along by the
correction applied to the keyframe that observes them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import gtsam
import numpy as np

from ..config import PoseGraphConfig
from ..types import Pose, SlamMap
from .gtsam_utils import from_gtsam_pose, pose_key, pose_noise, robust_pose_noise, to_gtsam_pose


@dataclass
class PoseGraphResult:
    success: bool
    reason: str = ""
    n_poses: int = 0
    n_odometry_factors: int = 0
    n_loop_factors: int = 0
    initial_error: float = 0.0
    final_error: float = 0.0
    iterations: int = 0
    duration_ms: float = 0.0
    max_correction: float = 0.0
    """Largest distance any keyframe moved -- a direct read on drift removed."""
    corrections: dict[int, Pose] = field(default_factory=dict)


class PoseGraphOptimizer:
    """Optimises keyframe poses against odometry and loop constraints."""

    def __init__(self, config: PoseGraphConfig) -> None:
        self.cfg = config

    def optimize(self, slam_map: SlamMap,
                 loops: list[tuple[int, int, Pose]] | None = None) -> PoseGraphResult:
        """Run global optimisation.

        `loops` holds (query_id, match_id, relative_pose) triples, where the
        relative pose transforms the match keyframe into the query frame.
        """
        started = time.perf_counter()
        if not self.cfg.enabled:
            return PoseGraphResult(False, "disabled")

        kf_ids = slam_map.keyframe_ids
        if len(kf_ids) < 3:
            return PoseGraphResult(False, "fewer than 3 keyframes")

        loops = loops or []

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        for kf_id in kf_ids:
            initial.insert(pose_key(kf_id), to_gtsam_pose(slam_map.keyframes[kf_id].pose))

        # Anchor the first keyframe: a pose graph is otherwise free to drift as
        # a rigid body, leaving the solution gauge-ambiguous.
        graph.add(gtsam.PriorFactorPose3(
            pose_key(kf_ids[0]), to_gtsam_pose(slam_map.keyframes[kf_ids[0]].pose),
            gtsam.noiseModel.Diagonal.Sigmas(np.full(6, self.cfg.prior_sigma))))

        odo_noise = pose_noise(self.cfg.odometry_sigma_trans, self.cfg.odometry_sigma_rot)
        n_odo = 0
        for a, b in zip(kf_ids, kf_ids[1:], strict=False):
            rel = slam_map.keyframes[a].pose.inverse().compose(slam_map.keyframes[b].pose)
            graph.add(gtsam.BetweenFactorPose3(
                pose_key(a), pose_key(b), to_gtsam_pose(rel), odo_noise))
            n_odo += 1

        # Loop constraints get a robust kernel: one bad loop would otherwise
        # warp the entire trajectory, and false positives are never fully
        # eliminable by detection alone.
        loop_noise = (robust_pose_noise(self.cfg.loop_sigma_trans,
                                        self.cfg.loop_sigma_rot, self.cfg.huber_k)
                      if self.cfg.use_robust_kernel
                      else pose_noise(self.cfg.loop_sigma_trans, self.cfg.loop_sigma_rot))
        n_loops = 0
        for query_id, match_id, rel in loops:
            if query_id not in slam_map.keyframes or match_id not in slam_map.keyframes:
                continue
            graph.add(gtsam.BetweenFactorPose3(
                pose_key(query_id), pose_key(match_id), to_gtsam_pose(rel), loop_noise))
            n_loops += 1

        if n_loops == 0:
            # Without a loop the graph is a chain: optimisation reproduces the
            # input exactly, so skip the work and say so plainly.
            return PoseGraphResult(
                False, "no loop closures, pose graph would be a no-op",
                n_poses=len(kf_ids), n_odometry_factors=n_odo,
                duration_ms=(time.perf_counter() - started) * 1000)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.cfg.max_iterations)
        params.setVerbosityLM("SILENT")

        try:
            initial_error = float(graph.error(initial))
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = optimizer.optimize()
            final_error = float(graph.error(result))
            iterations = int(optimizer.iterations())
        except Exception as exc:
            return PoseGraphResult(False, f"optimiser failed: {exc}",
                                   duration_ms=(time.perf_counter() - started) * 1000)

        if not np.isfinite(final_error):
            return PoseGraphResult(False, "optimiser diverged",
                                   duration_ms=(time.perf_counter() - started) * 1000)

        # Apply corrections, carrying each keyframe's landmarks rigidly with it.
        corrections: dict[int, Pose] = {}
        max_corr = 0.0
        for kf_id in kf_ids:
            key = pose_key(kf_id)
            if not result.exists(key):
                continue
            old = slam_map.keyframes[kf_id].pose
            new = from_gtsam_pose(result.atPose3(key))
            if slam_map.keyframes[kf_id].odometry_pose is None:
                slam_map.keyframes[kf_id].odometry_pose = old.copy()
            # transform taking old frame to new frame
            corrections[kf_id] = new.compose(old.inverse())
            max_corr = max(max_corr, float(np.linalg.norm(new.center - old.center)))
            slam_map.keyframes[kf_id].pose = new

        self._correct_landmarks(slam_map, corrections)

        return PoseGraphResult(
            True, "ok", n_poses=len(kf_ids), n_odometry_factors=n_odo,
            n_loop_factors=n_loops, initial_error=initial_error, final_error=final_error,
            iterations=iterations, duration_ms=(time.perf_counter() - started) * 1000,
            max_correction=max_corr, corrections=corrections)

    @staticmethod
    def _correct_landmarks(slam_map: SlamMap, corrections: dict[int, Pose]) -> None:
        """Move each landmark by the correction of a keyframe that observes it.

        Using the landmark's reference keyframe keeps structure attached to the
        part of the trajectory it was triangulated from, so the map deforms
        consistently with the corrected poses.
        """
        for lm in slam_map.landmarks.values():
            if not lm.observations:
                continue
            ref_kf = min(lm.observations)
            corr = corrections.get(ref_kf)
            if corr is None:
                continue
            lm.position = corr.camera_to_world(lm.position.reshape(1, 3))[0]
