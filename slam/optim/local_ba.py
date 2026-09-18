"""Sliding-window bundle adjustment -- drift-control layer L3.

Jointly refines the most recent keyframe poses and the landmarks they observe.
The oldest keyframes in the window are held fixed to supply the gauge: a
monocular reconstruction is free up to a similarity transform, so without
fixed references the optimiser can drift or collapse the whole window.

Only a window is optimised rather than the full history, because full BA cost
grows with the sequence and would not fit the runtime budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import gtsam
import numpy as np

from ..camera import Camera
from ..config import LocalBAConfig
from ..types import SlamMap
from .gtsam_utils import (from_gtsam_pose, landmark_key, pose_key, robust_pixel_noise,
                          to_gtsam_calibration, to_gtsam_pose)


@dataclass
class BAResult:
    success: bool
    reason: str = ""
    n_poses: int = 0
    n_landmarks: int = 0
    n_factors: int = 0
    initial_error: float = 0.0
    final_error: float = 0.0
    iterations: int = 0
    duration_ms: float = 0.0
    updated_keyframes: list[int] = field(default_factory=list)

    @property
    def error_reduction(self) -> float:
        if self.initial_error <= 0:
            return 0.0
        return 1.0 - (self.final_error / self.initial_error)


class LocalBundleAdjuster:
    """Windowed bundle adjustment over keyframe poses and landmark positions."""

    def __init__(self, camera: Camera, config: LocalBAConfig) -> None:
        self.camera = camera
        self.cfg = config
        self._calibration = to_gtsam_calibration(camera)

    def optimize(self, slam_map: SlamMap, window_ids: list[int] | None = None) -> BAResult:
        started = time.perf_counter()
        if not self.cfg.enabled:
            return BAResult(False, "disabled")

        kf_ids = slam_map.keyframe_ids
        if window_ids is None:
            window_ids = kf_ids[-self.cfg.window_size:]
        window_ids = [k for k in window_ids if k in slam_map.keyframes]
        if len(window_ids) < 2:
            return BAResult(False, "fewer than 2 keyframes in window")

        window = set(window_ids)
        # Fix the oldest keyframes as the gauge. If the window covers the whole
        # map, at least one pose must still be anchored.
        n_fixed = max(1, min(self.cfg.fixed_count, len(window_ids) - 1))
        fixed_ids = set(window_ids[:n_fixed])

        # Landmarks worth optimising: seen at least twice inside the window,
        # otherwise they add an unconstrained variable and destabilise the solve.
        lm_obs: dict[int, list[tuple[int, int]]] = {}
        for kf_id in window_ids:
            kf = slam_map.keyframes[kf_id]
            for pt_idx, lm_id in kf.landmark_ids.items():
                lm = slam_map.landmarks.get(lm_id)
                if lm is None or lm.is_outlier:
                    continue
                if pt_idx >= len(kf.points):
                    continue
                lm_obs.setdefault(lm_id, []).append((kf_id, pt_idx))

        lm_obs = {k: v for k, v in lm_obs.items() if len(v) >= 2}
        if not lm_obs:
            return BAResult(False, "no landmarks with 2+ observations in window")

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        noise = robust_pixel_noise(self.cfg.pixel_sigma, self.cfg.huber_k)

        for kf_id in window_ids:
            initial.insert(pose_key(kf_id), to_gtsam_pose(slam_map.keyframes[kf_id].pose))
        for lm_id in lm_obs:
            initial.insert(landmark_key(lm_id),
                           gtsam.Point3(*slam_map.landmarks[lm_id].position))

        # Strong priors pin the gauge poses rather than using a constrained
        # elimination, which keeps the system well conditioned.
        gauge_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, 1e-6))
        for kf_id in fixed_ids:
            graph.add(gtsam.PriorFactorPose3(
                pose_key(kf_id), to_gtsam_pose(slam_map.keyframes[kf_id].pose), gauge_noise))

        n_factors = 0
        for lm_id, observations in lm_obs.items():
            for kf_id, pt_idx in observations:
                if kf_id not in window:
                    continue
                uv = slam_map.keyframes[kf_id].points[pt_idx]
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(float(uv[0]), float(uv[1])), noise,
                    pose_key(kf_id), landmark_key(lm_id), self._calibration))
                n_factors += 1

        if n_factors < 10:
            return BAResult(False, f"only {n_factors} projection factors")

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(self.cfg.max_iterations)
        params.setVerbosityLM("SILENT")

        try:
            initial_error = float(graph.error(initial))
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = optimizer.optimize()
            final_error = float(graph.error(result))
            iterations = int(optimizer.iterations())
        except Exception as exc:  # a diverged solve must not kill the pipeline
            return BAResult(False, f"optimiser failed: {exc}",
                            duration_ms=(time.perf_counter() - started) * 1000)

        # Reject a solve that made things worse rather than writing it back.
        if not np.isfinite(final_error) or final_error > initial_error:
            return BAResult(False, f"rejected: error {initial_error:.1f} -> {final_error:.1f}",
                            initial_error=initial_error, final_error=final_error,
                            duration_ms=(time.perf_counter() - started) * 1000)

        updated: list[int] = []
        for kf_id in window_ids:
            if kf_id in fixed_ids:
                continue
            key = pose_key(kf_id)
            if result.exists(key):
                slam_map.keyframes[kf_id].pose = from_gtsam_pose(result.atPose3(key))
                updated.append(kf_id)

        for lm_id in lm_obs:
            key = landmark_key(lm_id)
            if result.exists(key):
                pos = np.asarray(result.atPoint3(key)).reshape(3)
                if np.isfinite(pos).all():
                    slam_map.landmarks[lm_id].position = pos

        return BAResult(True, "ok", n_poses=len(window_ids), n_landmarks=len(lm_obs),
                        n_factors=n_factors, initial_error=initial_error,
                        final_error=final_error, iterations=iterations,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        updated_keyframes=updated)
