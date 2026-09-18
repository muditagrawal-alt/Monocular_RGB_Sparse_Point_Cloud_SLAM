"""The SLAM pipeline: decode -> track -> map -> optimise -> export.

Stage order and the reasoning behind each choice are documented in
IMPLEMENTATION_PLAN.md section 4. The pipeline is written to degrade rather
than fail: tracking loss triggers re-initialisation instead of aborting, a
diverged optimisation is rejected instead of corrupting the map, and the
adaptive-quality guard reduces work up front on long inputs rather than
overrunning the time budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .camera import Camera, resolve_intrinsics
from .config import SlamConfig
from .core.features import FeatureTracker, compute_orb_at_points
from .core.initializer import Initializer
from .core.keyframe_policy import KeyframePolicy
from .core.mapper import Mapper
from .core.tracker import Tracker
from .io.video import DecodedFrame, VideoDecoder, VideoInfo, probe_video
from .optim.local_ba import LocalBundleAdjuster
from .optim.loop_closure import LoopDetector
from .optim.pose_graph import PoseGraphOptimizer
from .types import Keyframe, Pose, SlamMap, TrajectoryPose

ProgressCallback = Callable[[str, float, dict], None]


@dataclass
class StageTimings:
    """Wall-clock milliseconds per stage, for the telemetry panel."""

    decode: float = 0.0
    frontend: float = 0.0
    initialization: float = 0.0
    tracking: float = 0.0
    mapping: float = 0.0
    descriptors: float = 0.0
    local_ba: float = 0.0
    loop_closure: float = 0.0
    pose_graph: float = 0.0
    export: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self.__dict__.items()}


@dataclass
class SlamResult:
    """Everything the API and viewer need."""

    success: bool
    reason: str = ""
    trajectory: list[TrajectoryPose] = field(default_factory=list)
    odometry_trajectory: list[TrajectoryPose] = field(default_factory=list)
    """Pre-optimisation trajectory, for the before/after drift comparison."""
    slam_map: SlamMap | None = None
    camera: Camera | None = None
    video_info: VideoInfo | None = None
    timings: StageTimings = field(default_factory=StageTimings)

    n_frames_processed: int = 0
    n_keyframes: int = 0
    n_landmarks: int = 0
    n_loop_closures: int = 0
    n_tracking_losses: int = 0
    n_reinitializations: int = 0
    mean_reproj_error: float = 0.0
    median_track_count: float = 0.0
    max_pose_correction: float = 0.0
    drift_correction_applied: bool = False

    processing_time_s: float = 0.0
    video_duration_s: float = 0.0
    realtime_factor: float = 0.0
    """processing_time / video_duration. <=1.0 satisfies requirement 6."""

    applied_width: int = 0
    applied_max_features: int = 0
    quality_reduced: bool = False

    def summary(self) -> dict:
        return {
            "success": self.success,
            "reason": self.reason,
            "frames": self.n_frames_processed,
            "keyframes": self.n_keyframes,
            "landmarks": self.n_landmarks,
            "loop_closures": self.n_loop_closures,
            "tracking_losses": self.n_tracking_losses,
            "reinitializations": self.n_reinitializations,
            "mean_reproj_error_px": round(self.mean_reproj_error, 3),
            "median_track_count": round(self.median_track_count, 1),
            "processing_time_s": round(self.processing_time_s, 3),
            "video_duration_s": round(self.video_duration_s, 3),
            "realtime_factor": round(self.realtime_factor, 3),
            "within_budget": self.realtime_factor <= 1.0,
            "drift_correction_applied": self.drift_correction_applied,
            "max_pose_correction": round(self.max_pose_correction, 4),
            "intrinsics_source": self.camera.source if self.camera else None,
            "applied_width": self.applied_width,
            "applied_max_features": self.applied_max_features,
            "quality_reduced": self.quality_reduced,
            "timings_ms": self.timings.as_dict(),
        }


class SlamPipeline:
    """Runs monocular SLAM over a video file."""

    def __init__(self, config: SlamConfig | None = None) -> None:
        self.cfg = config or SlamConfig()
        self._init_ref_gray: np.ndarray | None = None

    # -- runtime guard ----------------------------------------------------
    def _plan_quality(self, info: VideoInfo) -> tuple[int, int, bool]:
        """Choose resolution and feature budget to fit the time budget.

        Requirement 6 is a hard deadline, so for unusually long inputs it is
        better to reduce work deliberately up front than to miss it. Returns
        (width, max_features, reduced).
        """
        width = self.cfg.frontend.target_width
        features = self.cfg.frontend.max_features
        if not (self.cfg.budget.enabled and self.cfg.budget.adaptive_quality):
            return width, features, False

        step = max(1, int(round(info.fps / self.cfg.max_fps))) if self.cfg.max_fps > 0 else 1
        n_frames = (info.frame_count // step) if info.frame_count else 0
        # Reference: ~6 ms/frame front end at 640 px wide, measured.
        budget_s = max(info.duration_s, 1e-6) * self.cfg.budget.realtime_factor_target
        projected_s = n_frames * 0.0125   # front end + tracking + share of BA

        if n_frames == 0 or projected_s <= budget_s * 0.8:
            return width, features, False

        shrink = float(np.sqrt(max(budget_s * 0.8 / projected_s, 0.05)))
        new_width = max(self.cfg.budget.adaptive_min_width, int(width * shrink) // 2 * 2)
        new_features = max(self.cfg.budget.adaptive_min_features, int(features * shrink))
        return new_width, new_features, (new_width != width or new_features != features)

    # -- main entry point -------------------------------------------------
    def run(self, video_path: str | Path, *, focal_px: float | None = None,
            hfov_deg: float | None = None,
            progress: ProgressCallback | None = None) -> SlamResult:
        t_start = time.perf_counter()
        timings = StageTimings()

        def report(stage: str, frac: float, **extra) -> None:
            if progress is not None:
                progress(stage, float(np.clip(frac, 0.0, 1.0)), extra)

        try:
            info = probe_video(video_path)
        except Exception as exc:
            return SlamResult(False, f"cannot read video: {exc}")
        if not info.is_valid:
            return SlamResult(False, "video has invalid dimensions or frame rate")

        width, max_features, reduced = self._plan_quality(info)
        cfg = self.cfg
        cfg.frontend.target_width = width
        cfg.frontend.max_features = max_features

        report("decoding", 0.0, width=width)

        # Intrinsics for the *downscaled* frames the pipeline actually sees.
        full_cam = resolve_intrinsics(info.width, info.height,
                                      focal_px=focal_px, hfov_deg=hfov_deg)
        scale = min(1.0, width / info.width) if info.width else 1.0
        camera = full_cam.scaled(scale)

        tracker_frontend = FeatureTracker(cfg.frontend)
        initializer = Initializer(camera, cfg.init)
        tracker = Tracker(camera, cfg.tracking)
        mapper = Mapper(camera, cfg.mapping)
        kf_policy = KeyframePolicy(cfg.keyframe)
        local_ba = LocalBundleAdjuster(camera, cfg.local_ba)

        slam_map = SlamMap()
        track_to_landmark: dict[int, int] = {}
        trajectory: list[TrajectoryPose] = []

        initialized = False
        init_ref: tuple[np.ndarray, np.ndarray] | None = None
        init_attempts = 0
        current_pose = Pose()
        last_kf_pose = Pose()
        frames_since_kf = 0
        n_tracked_at_kf = 0
        n_losses = 0
        n_reinit = 0
        track_counts: list[int] = []
        reproj_errors: list[float] = []
        n_frames = 0
        expected = 0

        decoder = VideoDecoder(video_path, target_width=width, max_fps=cfg.max_fps,
                              max_frames=cfg.budget.max_frames)
        expected = max(decoder.expected_frames, 1)

        try:
            for frame in decoder:
                n_frames += 1
                t0 = time.perf_counter()
                result = tracker_frontend.track(frame.gray)
                timings.frontend += (time.perf_counter() - t0) * 1000
                track_counts.append(len(result.points))

                if not initialized:
                    t0 = time.perf_counter()
                    initialized, init_ref, init_attempts = self._try_init(
                        frame, result, tracker_frontend, initializer, slam_map,
                        track_to_landmark, init_ref, init_attempts, camera, cfg)
                    timings.initialization += (time.perf_counter() - t0) * 1000

                    if initialized:
                        kf = slam_map.keyframes[slam_map.keyframe_ids[-1]]
                        current_pose = kf.pose.copy()
                        last_kf_pose = current_pose.copy()
                        n_tracked_at_kf = len(result.points)
                        frames_since_kf = 0
                        trajectory.append(TrajectoryPose(
                            frame_index=frame.index, timestamp=frame.timestamp,
                            pose=current_pose.copy(), is_keyframe=True))
                        report("initialized", n_frames / expected,
                               keyframes=len(slam_map.keyframes))
                    else:
                        trajectory.append(TrajectoryPose(
                            frame_index=frame.index, timestamp=frame.timestamp,
                            pose=Pose(), tracking_ok=False))
                    continue

                # --- tracking ---
                t0 = time.perf_counter()
                tr = tracker.track(slam_map, result.points, result.track_ids,
                                   track_to_landmark, last_pose=current_pose)
                timings.tracking += (time.perf_counter() - t0) * 1000

                if tr.success and tr.pose is not None:
                    current_pose = tr.pose
                    reproj_errors.append(tr.mean_reproj_error)
                    frames_since_kf += 1
                    trajectory.append(TrajectoryPose(
                        frame_index=frame.index, timestamp=frame.timestamp,
                        pose=current_pose.copy(), n_inliers=tr.n_inliers))
                else:
                    # Tracking lost: restart initialisation rather than emit
                    # garbage poses. The map built so far is retained.
                    n_losses += 1
                    initialized = False
                    init_ref = None
                    init_attempts = 0
                    n_reinit += 1
                    tracker.motion.reset()
                    tracker_frontend.reset()
                    trajectory.append(TrajectoryPose(
                        frame_index=frame.index, timestamp=frame.timestamp,
                        pose=current_pose.copy(), tracking_ok=False))
                    continue

                # --- keyframe? ---
                decision = kf_policy.decide(
                    frames_since_keyframe=frames_since_kf,
                    n_tracked=len(result.points),
                    n_tracked_at_last_keyframe=n_tracked_at_kf,
                    current_pose=current_pose, last_keyframe_pose=last_kf_pose,
                    median_depth=slam_map.median_depth(slam_map.keyframe_ids[-1]))

                if decision.insert:
                    t0 = time.perf_counter()
                    kf = self._insert_keyframe(slam_map, frame, result, current_pose)
                    timings.descriptors += (time.perf_counter() - t0) * 1000

                    t0 = time.perf_counter()
                    mapper.add_observations(slam_map, kf, track_to_landmark)
                    refs = slam_map.covisible_keyframes(kf.id, min_shared=10)[:3]
                    if not refs:
                        refs = [k for k in slam_map.keyframe_ids if k != kf.id][-3:]
                    mapper.create_landmarks(slam_map, kf, track_to_landmark, refs)
                    self._colorize(slam_map, kf, frame)
                    timings.mapping += (time.perf_counter() - t0) * 1000

                    if (cfg.local_ba.enabled and len(slam_map.keyframes) >= 3
                            and kf.id % cfg.local_ba.run_every_n_keyframes == 0):
                        t0 = time.perf_counter()
                        local_ba.optimize(slam_map)
                        mapper.cull(slam_map, slam_map.keyframe_ids[-cfg.local_ba.window_size:])
                        timings.local_ba += (time.perf_counter() - t0) * 1000
                        current_pose = slam_map.keyframes[kf.id].pose.copy()
                        trajectory[-1].pose = current_pose.copy()

                    trajectory[-1].is_keyframe = True
                    last_kf_pose = current_pose.copy()
                    n_tracked_at_kf = len(result.points)
                    frames_since_kf = 0

                if n_frames % 15 == 0:
                    report("tracking", n_frames / expected,
                           frames=n_frames, keyframes=len(slam_map.keyframes),
                           landmarks=len(slam_map.active_landmarks()))
        finally:
            decoder.close()

        if not slam_map.keyframes:
            return SlamResult(
                False,
                "initialisation never succeeded: the video may lack texture, "
                "have too little camera translation, or be rotation-only",
                video_info=info, camera=camera, n_frames_processed=n_frames,
                processing_time_s=time.perf_counter() - t_start)

        # Snapshot the pre-optimisation trajectory for the before/after view.
        odometry_trajectory = [
            TrajectoryPose(frame_index=t.frame_index, timestamp=t.timestamp,
                           pose=t.pose.copy(), is_keyframe=t.is_keyframe,
                           n_inliers=t.n_inliers, tracking_ok=t.tracking_ok)
            for t in trajectory]

        # --- loop closure + global correction (drift layer L4) ---
        n_loops = 0
        max_correction = 0.0
        drift_applied = False
        if cfg.loop.enabled and len(slam_map.keyframes) >= 6:
            report("loop_closure", 0.9)
            t0 = time.perf_counter()
            detector = LoopDetector(camera, cfg.loop)
            stats = detector.detect(slam_map)
            timings.loop_closure += (time.perf_counter() - t0) * 1000
            n_loops = len(stats.loops)

            if n_loops and cfg.pose_graph.enabled:
                report("pose_graph", 0.94, loops=n_loops)
                t0 = time.perf_counter()
                pgo = PoseGraphOptimizer(cfg.pose_graph)
                pg = pgo.optimize(slam_map, [(c.query_id, c.match_id, c.relative_pose)
                                             for c in stats.loops
                                             if c.relative_pose is not None])
                timings.pose_graph += (time.perf_counter() - t0) * 1000
                if pg.success:
                    max_correction = pg.max_correction
                    drift_applied = True
                    self._rebase_trajectory(trajectory, slam_map, pg.corrections)

        t0 = time.perf_counter()
        mapper.cull(slam_map)
        timings.export += (time.perf_counter() - t0) * 1000

        timings.total = (time.perf_counter() - t_start) * 1000
        elapsed = timings.total / 1000.0
        duration = info.duration_s or (n_frames / max(decoder.effective_fps, 1e-6))

        report("done", 1.0)
        return SlamResult(
            success=True, reason="ok", trajectory=trajectory,
            odometry_trajectory=odometry_trajectory, slam_map=slam_map,
            camera=camera, video_info=info, timings=timings,
            n_frames_processed=n_frames, n_keyframes=len(slam_map.keyframes),
            n_landmarks=len(slam_map.active_landmarks()), n_loop_closures=n_loops,
            n_tracking_losses=n_losses, n_reinitializations=n_reinit,
            mean_reproj_error=float(np.mean(reproj_errors)) if reproj_errors else 0.0,
            median_track_count=float(np.median(track_counts)) if track_counts else 0.0,
            max_pose_correction=max_correction, drift_correction_applied=drift_applied,
            processing_time_s=elapsed, video_duration_s=duration,
            realtime_factor=elapsed / duration if duration > 0 else 0.0,
            applied_width=width, applied_max_features=max_features,
            quality_reduced=reduced)

    # -- helpers ----------------------------------------------------------
    def _try_init(self, frame: DecodedFrame, result, frontend: FeatureTracker,
                  initializer: Initializer, slam_map: SlamMap,
                  track_to_landmark: dict[int, int],
                  init_ref: tuple[np.ndarray, np.ndarray] | None,
                  attempts: int, camera: Camera, cfg: SlamConfig):
        """Attempt two-view initialisation against a stored reference frame."""
        if init_ref is None:
            if len(result.points) >= cfg.init.min_correspondences:
                init_ref = (result.points.copy(), result.track_ids.copy())
                self._init_ref_gray = frame.gray.copy()
            return False, init_ref, attempts

        ref_pts, ref_ids = init_ref
        # match by persistent track id, which the KLT front end maintains
        ref_index = {int(t): i for i, t in enumerate(ref_ids)}
        pairs = [(ref_index[int(t)], j) for j, t in enumerate(result.track_ids)
                 if int(t) in ref_index]
        if len(pairs) < cfg.init.min_correspondences:
            self._init_ref_gray = frame.gray.copy()
            return False, (result.points.copy(), result.track_ids.copy()), attempts + 1

        pa = np.array([ref_pts[i] for i, _ in pairs], dtype=np.float64)
        pb = np.array([result.points[j] for _, j in pairs], dtype=np.float64)
        shared_ids = np.array([int(result.track_ids[j]) for _, j in pairs])

        init = initializer.try_initialize(pa, pb)
        attempts += 1

        if not init.success:
            # Keep the same reference for a while: parallax grows with time, so
            # a later frame may succeed where this one failed.
            if attempts > cfg.init.max_init_frames:
                self._init_ref_gray = frame.gray.copy()
                return False, (result.points.copy(), result.track_ids.copy()), 0
            return False, init_ref, attempts

        kf_ref = Keyframe(id=slam_map.new_keyframe_id(), frame_index=0, timestamp=0.0,
                          pose=Pose(), points=pa.astype(np.float32), track_ids=shared_ids)
        # The reference keyframe needs descriptors too, otherwise the very
        # start of the sequence can never be recognised as a revisited place --
        # which is exactly where a loop closes.
        kf_ref.descriptors, kf_ref.desc_indices = compute_orb_at_points(
            self._init_ref_gray if self._init_ref_gray is not None else frame.gray,
            kf_ref.points)
        slam_map.add_keyframe(kf_ref)
        kf_cur = Keyframe(id=slam_map.new_keyframe_id(), frame_index=frame.index,
                          timestamp=frame.timestamp, pose=init.pose,
                          points=pb.astype(np.float32), track_ids=shared_ids)
        kf_cur.descriptors, kf_cur.desc_indices = compute_orb_at_points(
            frame.gray, kf_cur.points)
        slam_map.add_keyframe(kf_cur)

        mask = init.inlier_mask
        pts3d = init.points_world
        k = 0
        for idx in range(len(shared_ids)):
            if mask is None or not mask[idx]:
                continue
            lm = slam_map.add_landmark(pts3d[k])
            slam_map.observe(lm.id, kf_ref.id, idx)
            slam_map.observe(lm.id, kf_cur.id, idx)
            track_to_landmark[int(shared_ids[idx])] = lm.id
            k += 1

        return True, None, attempts

    @staticmethod
    def _insert_keyframe(slam_map: SlamMap, frame: DecodedFrame, result,
                         pose: Pose) -> Keyframe:
        kf = Keyframe(id=slam_map.new_keyframe_id(), frame_index=frame.index,
                      timestamp=frame.timestamp, pose=pose.copy(),
                      points=result.points.copy(), track_ids=result.track_ids.copy())
        # Descriptors are computed only here, not per frame -- the whole reason
        # the front end can afford to run at 6 ms/frame.
        # Describing the tracked points (rather than a fresh ORB detection)
        # keeps descriptor -> landmark association exact.
        kf.descriptors, kf.desc_indices = compute_orb_at_points(frame.gray, kf.points)
        slam_map.add_keyframe(kf)
        return kf

    @staticmethod
    def _colorize(slam_map: SlamMap, kf: Keyframe, frame: DecodedFrame) -> None:
        """Sample landmark colour from the frame, for the point-cloud render."""
        if frame.color_small is None:
            return
        img = frame.color_small
        h, w = img.shape[:2]
        for pt_idx, lm_id in kf.landmark_ids.items():
            lm = slam_map.landmarks.get(lm_id)
            if lm is None or pt_idx >= len(kf.points):
                continue
            x, y = kf.points[pt_idx]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                b, g, r = img[yi, xi]
                lm.color = (int(r), int(g), int(b))

    @staticmethod
    def _rebase_trajectory(trajectory: list[TrajectoryPose], slam_map: SlamMap,
                           corrections: dict[int, Pose]) -> None:
        """Propagate pose-graph corrections onto every frame.

        Only keyframes are variables in the pose graph, so non-keyframe poses
        inherit the correction of the nearest preceding keyframe.
        """
        if not corrections:
            return
        kf_by_frame = sorted((kf.frame_index, kf.id) for kf in slam_map.keyframes.values())
        if not kf_by_frame:
            return
        frame_idx = np.array([f for f, _ in kf_by_frame])
        kf_ids = [k for _, k in kf_by_frame]

        for tp in trajectory:
            pos = int(np.searchsorted(frame_idx, tp.frame_index, side="right")) - 1
            pos = max(0, min(pos, len(kf_ids) - 1))
            corr = corrections.get(kf_ids[pos])
            if corr is not None:
                tp.pose = corr.compose(tp.pose)
