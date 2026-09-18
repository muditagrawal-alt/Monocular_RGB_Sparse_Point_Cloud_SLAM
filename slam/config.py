"""Tunable parameters for the SLAM pipeline.

Defaults are chosen to hold the performance budget described in
IMPLEMENTATION_PLAN.md section 7: a 10 s / 300-frame clip in under 10 s on CPU.
"""

from __future__ import annotations

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
    max_depth_ratio: float = 50.0
    """Reject points beyond this multiple of the median scene depth."""
    min_observations: int = 2


@dataclass
class LocalBAConfig:
    """Sliding-window bundle adjustment."""

    enabled: bool = True
    window_size: int = 8
    """Keyframes optimised jointly. Older keyframes act as a fixed gauge."""
    fixed_count: int = 2
    max_iterations: int = 8
    huber_k: float = 1.345
    pixel_sigma: float = 1.5
    run_every_n_keyframes: int = 2
    """Local BA is the single largest cost; every other keyframe still keeps
    error growth in check because windows overlap."""
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

    enabled: bool = True
    realtime_factor_target: float = 1.0
    """Wall-clock seconds allowed per second of video."""
    max_frames: int = 1800
    adaptive_quality: bool = True
    """Downscale / reduce the feature budget up front for long inputs rather
    than overrunning the deadline."""
    adaptive_min_width: int = 384
    adaptive_min_features: int = 400


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
