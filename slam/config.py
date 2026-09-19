"""Tunable parameters for the SLAM pipeline.

Defaults are chosen to hold the performance budget described in
IMPLEMENTATION_PLAN.md section 7: a 10 s / 300-frame clip in under 10 s on CPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class FrontendConfig:
    """Feature seeding and optical-flow tracking."""

    target_width: int = 640
    """Frames are downscaled to this width. The single biggest runtime lever."""

    max_features: int = 800
    """Shi-Tomasi corner budget. 800 measured at 4.3 ms/frame for KLT."""

    quality_level: float = 0.01
    min_distance: int = 7
    block_size: int = 7

    grid_cols: int = 8
    grid_rows: int = 6
    """Corners are seeded per grid cell so features spread over the image
    instead of clumping on the highest-contrast region."""

    klt_window: int = 21
    klt_levels: int = 3
    klt_iters: int = 30
    klt_eps: float = 0.01

    fb_check_interval: int = 1
    """Run the forward-backward consistency check every Nth frame. The check
    runs a second optical-flow pass, so it doubles front-end cost, and the
    front end is the largest single stage. Bad tracks are also caught
    downstream by RANSAC in pose estimation, so checking less often trades a
    little robustness for a lot of time."""

    fb_error_threshold: float = 1.0
    """Forward-backward KLT consistency threshold in pixels. Tracks whose
    round trip lands further than this from the origin are dropped."""

    reseed_ratio: float = 0.6
    """Re-seed corners when live tracks fall below this fraction of the budget."""


@dataclass
class InitConfig:
    """Two-view initialisation."""

    min_correspondences: int = 100
    min_parallax_deg: float = 1.0
    ransac_confidence: float = 0.999
    ransac_threshold_px: float = 1.0
    homography_score_ratio: float = 0.45
    """If H_score / (H_score + E_score) exceeds this, the scene is treated as
    planar and initialisation waits for better-conditioned motion."""

    min_triangulated: int = 50
    max_init_frames: int = 60
    homography_fallback_after: int = 20
    """Failed attempts before the planar fallback is allowed. Until then only
    the essential matrix is tried, because it gives a better map on scenes that
    are merely near-planar. A scene that still has not initialised after this
    many attempts is genuinely planar, and refusing it outright would discard
    valid footage such as drone video over flat ground."""
    """Give up and restart initialisation if it has not succeeded by here."""


@dataclass
class TrackingConfig:
    """Frame-to-map pose estimation."""

    min_inliers: int = 15
    pnp_reproj_threshold_px: float = 3.0
    pnp_confidence: float = 0.99
    pnp_iterations: int = 100
    use_motion_prior: bool = True
    """Seed solvePnP with a constant-velocity prediction."""


@dataclass
class KeyframeConfig:
    """Keyframe insertion policy."""

    min_frame_gap: int = 5
    max_frame_gap: int = 20
    track_ratio_threshold: float = 0.75
    """Insert once tracked points drop below this fraction of the last keyframe."""
    min_translation_ratio: float = 0.04
    """Translation relative to median scene depth, to guarantee parallax."""


@dataclass
class MappingConfig:
    """Landmark triangulation and culling."""

    min_parallax_deg: float = 1.0
    max_reproj_error_px: float = 4.0
    min_depth: float = 1e-4
    max_depth_ratio: float = 20.0
    """Reject points beyond this multiple of the median scene depth."""
    min_observations: int = 2


@dataclass
class LocalBAConfig:
    """Sliding-window bundle adjustment."""

    enabled: bool = True
    window_size: int = 6
    """Keyframes optimised jointly. Older keyframes act as a fixed gauge.
    Measured better than 8 as well as cheaper: a shorter window keeps the
    solve close to well-constrained recent structure instead of dragging in
    older keyframes whose landmarks are thinly observed."""

    fixed_count: int = 2

    max_iterations: int = 5
    """Levenberg-Marquardt iterations. The solve converges well before 8, so
    the extra iterations cost time without improving the result."""
    huber_k: float = 1.345
    pixel_sigma: float = 1.5
    run_every_n_keyframes: int = 2
    """Local BA is the single largest cost; every other keyframe still keeps
    error growth in check because windows overlap."""
    min_observations: int = 3
    """Views a landmark needs before it may enter the solve. A point seen in
    only two views is exactly determined by those two observations: it adds
    three free parameters and no constraint, so instead of anchoring the poses
    it absorbs their error. On an orbiting sequence 62% of landmarks are
    two-view, and including them made bundle adjustment trebles the trajectory
    error rather than reduce it."""

    max_landmarks: int = 400
    """Cap on landmarks per solve. BA cost grows with landmark count, so the
    best-observed subset is optimised to keep the per-keyframe cost bounded as
    the map grows. Landmarks left out are still corrected by the pose graph."""


@dataclass
class LoopClosureConfig:
    """Loop detection and pose-graph correction."""

    enabled: bool = True
    orb_features: int = 500
    vocab_size: int = 512
    """Larger vocabularies quantise descriptors more finely, which sharpens
    retrieval; 512 words costs only a few ms more to train."""
    vocab_train_descriptors: int = 20000
    min_keyframe_separation: int = 15
    """Refuse loop candidates too close in time; those are just tracking."""
    top_k_candidates: int = 10
    """Bag-of-words ranking is only a shortlist and is weakly discriminative on
    repetitive scenes, so several candidates are passed to geometric
    verification, which is what actually decides."""
    min_bow_similarity: float = 0.10
    spatial_candidates: int = 4
    """Candidates proposed by proximity in the *current* pose estimate, in
    addition to appearance. Appearance retrieval degrades on scenes with
    homogeneous texture, while spatial proposal degrades once drift exceeds the
    loop size; taking the union is robust where either alone is not."""
    query_stride: int = 1
    """Attempt detection from every Nth keyframe. Consecutive keyframes see
    essentially the same place, so querying all of them multiplies cost without
    finding new loops."""
    time_aware_budget: bool = True
    """Size the verification budget from the time actually left, rather than
    degrading the map while it is being built. Loop detection runs after the
    frame loop, so by then the remaining budget is known exactly. This keeps
    full detection quality whenever there is time and gives up detection rather
    than the deadline when there is not, which is the right way round: the map
    is already built and correct at that point."""

    verification_cost_s: float = 0.025
    """Assumed cost of one geometric verification when sizing the time-aware
    budget. Measured at roughly 13 ms on an Apple Silicon core and 20 ms on a
    Fargate vCPU, and Fargate task placement varies enough that the same image
    and configuration measured 6.5 s on one host and 10.0 s on another. The
    figure here is deliberately pessimistic: underestimating it lets the stage
    overrun the deadline, while overestimating only costs some detection."""

    budget_headroom: float = 0.80
    """Fraction of the remaining time loop detection may consume. Without a
    margin the stage simply expands to fill whatever is left, so savings made
    earlier in the pipeline get spent here instead of shortening the run: at
    width 512 the front end got 950 ms cheaper and loop detection grew by
    1140 ms, leaving the total unchanged."""

    min_verifications: int = 40
    """Floor for the scaled budget. Below this, detection is so unlikely to
    succeed that the time is better not spent at all."""

    candidates_per_query: int = 14
    """Candidates verified per query keyframe, for queries that get a turn."""
    max_verifications: int = 200
    """Hard cap on geometric verifications, to bound worst-case runtime.
    Verification costs ~14 ms, so this is the main lever on loop-closure time.
    Because queries are tried in order of closest approach, the budget is spent
    on the keyframes where a loop can actually be, and a small cap suffices."""
    match_max_distance: int = 64
    """Hamming cap on accepted ORB matches (descriptors are 256-bit).
    Cross-check matching with a distance cap recovers far more true matches
    than a 0.75 ratio test on repetitive texture -- measured 46 vs 11 on a
    synthetic orbit -- and RANSAC remains the real filter."""
    min_match_count: int = 20
    min_inlier_count: int = 12
    ransac_threshold_px: float = 8.0
    """Deliberately looser than the 3 px used for frame-to-frame tracking.
    A loop constraint spans the whole accumulated drift, so the map geometry at
    the two ends has deformed relative to each other; the correspondences are
    still correct, they just do not reproject as tightly. Measured on a
    synthetic orbit: at 3 px verification found 5 inliers and the true loop was
    rejected, while at 8 px it found 17 -- all of them geometrically correct --
    and recovered the revisited pose to within 0.09 units."""
    ransac_iterations: int = 300
    consistency_required: int = 2
    """A region must be proposed by this many distinct query keyframes before
    it is trusted, which rejects one-off perceptual aliasing."""
    strong_inlier_count: int = 20
    """A candidate with at least this many geometric inliers is accepted
    without corroboration. The consistency rule guards against perceptual
    aliasing, but aliasing produces weak, marginal matches; a strongly
    supported pair is not aliasing, and requiring a second vote would discard
    the one real loop in a sequence that revisits a place only once."""
    region_bucket: int = 5
    """Keyframes are grouped into regions of this size for the consistency
    check, since consecutive keyframes see essentially the same place."""


@dataclass
class PoseGraphConfig:
    """Global pose-graph optimisation."""

    enabled: bool = True
    odometry_sigma_trans: float = 0.05
    odometry_sigma_rot: float = 0.02
    loop_sigma_trans: float = 0.10
    loop_sigma_rot: float = 0.05
    prior_sigma: float = 1e-4
    max_iterations: int = 30
    use_robust_kernel: bool = True
    huber_k: float = 1.345


@dataclass
class GlobalBAConfig:
    """Final full bundle adjustment. Off by default to protect the time budget."""

    enabled: bool = False
    max_iterations: int = 10
    huber_k: float = 1.345
    pixel_sigma: float = 1.5
    refine_focal_length: bool = False


@dataclass
class BudgetConfig:
    """Runtime guard. Requirement 6: <=10 s for a 10 s clip."""

    enabled: bool = False
    """Off by default. The guard sheds work based on measured throughput, which
    means the map it produces depends on how loaded the machine is: the same
    clip measured 0.76 ATE on an idle machine and 3.75 when the guard fired
    under load. Latency should come from the pipeline being fast enough, not
    from silently degrading the result, so this is now opt-in for deployments
    that would rather lose accuracy than miss a deadline."""

    realtime_factor_target: float = 1.0
    """Wall-clock seconds allowed per second of video."""
    max_frames: int = 1800
    adaptive_quality: bool = True
    """Downscale / reduce the feature budget up front for long inputs rather
    than overrunning the deadline."""
    adaptive_min_width: int = 384
    adaptive_min_features: int = 400

    runtime_check_fraction: float = 0.3
    """Fraction of the sequence after which measured throughput is projected to
    a total. The up-front estimate cannot know how fast the host actually is;
    measured on a 2 vCPU container the same clip runs 1.2x slower than on an
    unconstrained laptop, which is the difference between meeting the budget
    and missing it."""

    backend_reserve_fraction: float = 0.28
    """Correction applied to the mid-flight projection, which extrapolates
    per-frame cost only. Two effects make that extrapolation optimistic: loop
    detection and pose-graph optimisation run after the frame loop (~35% of
    total), while early frames are cheaper than late ones because the map is
    smaller. Measured directly on a 300-frame clip, naive extrapolation lands
    at 0.72x of the true total and is stable across checkpoints, so the
    projection is divided by (1 - 0.28) to correct it."""

    runtime_overrun_tolerance: float = 0.97
    """Degrade once the projection reaches this fraction of the budget. The
    projection is calibrated rather than pessimistic, so the margin is small on
    purpose: it separates a host that comfortably fits (projects ~9.1 s of a
    10 s budget) from one that marginally does not (~10.2 s), without shedding
    loop closure on hosts that never needed it."""


@dataclass
class SlamConfig:
    """Top-level configuration."""

    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    init: InitConfig = field(default_factory=InitConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    local_ba: LocalBAConfig = field(default_factory=LocalBAConfig)
    loop: LoopClosureConfig = field(default_factory=LoopClosureConfig)
    pose_graph: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    global_ba: GlobalBAConfig = field(default_factory=GlobalBAConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    max_fps: float = 30.0
    """Decimate input above this rate; 60 fps footage gains little accuracy."""

    seed: int = 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def config_from_env() -> SlamConfig:
    """Build a configuration, letting the environment override the few settings
    that differ between deployments.

    Processing width and the feature budget are the two levers that actually
    move runtime, and the right value depends on how fast the host is: the same
    clip runs about 2.5x slower on a Fargate vCPU than on an Apple Silicon
    core. Exposing them as environment variables means a deployment can be
    tuned without rebuilding the image, and the applied values are reported
    back in every result so a run is never ambiguous about what produced it.
    """
    cfg = SlamConfig()
    cfg.frontend.target_width = _env_int("SLAM_TARGET_WIDTH", cfg.frontend.target_width)
    cfg.frontend.max_features = _env_int("SLAM_MAX_FEATURES", cfg.frontend.max_features)
    cfg.frontend.fb_check_interval = _env_int("SLAM_FB_INTERVAL",
                                              cfg.frontend.fb_check_interval)
    cfg.local_ba.run_every_n_keyframes = _env_int("SLAM_BA_EVERY",
                                                  cfg.local_ba.run_every_n_keyframes)
    cfg.local_ba.window_size = _env_int("SLAM_BA_WINDOW", cfg.local_ba.window_size)
    cfg.loop.enabled = _env_bool("SLAM_LOOP_CLOSURE", cfg.loop.enabled)
    cfg.loop.max_verifications = _env_int("SLAM_LOOP_VERIFICATIONS",
                                          cfg.loop.max_verifications)
    cfg.loop.verification_cost_s = _env_float("SLAM_VERIFY_COST_S",
                                              cfg.loop.verification_cost_s)
    cfg.pose_graph.enabled = cfg.loop.enabled
    cfg.budget.enabled = _env_bool("SLAM_ADAPTIVE_BUDGET", cfg.budget.enabled)
    cfg.budget.realtime_factor_target = _env_float("SLAM_RTF_TARGET",
                                                   cfg.budget.realtime_factor_target)
    cfg.seed = _env_int("SLAM_SEED", cfg.seed)
    return cfg
