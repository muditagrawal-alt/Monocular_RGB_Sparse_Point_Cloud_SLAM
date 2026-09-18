"""When to promote a frame to a keyframe.

Keyframes are where the expensive work happens (descriptors, triangulation,
bundle adjustment), so this policy directly sets the accuracy/runtime
trade-off. Too few and the map starves; too many and the time budget blows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import KeyframeConfig
from ..types import Pose


@dataclass
class KeyframeDecision:
    insert: bool
    reason: str = ""


class KeyframePolicy:
    """Decides keyframe insertion from tracking health and camera motion."""

    def __init__(self, config: KeyframeConfig) -> None:
        self.cfg = config

    def decide(self, *, frames_since_keyframe: int, n_tracked: int,
               n_tracked_at_last_keyframe: int, current_pose: Pose,
               last_keyframe_pose: Pose, median_depth: float,
               tracking_ok: bool = True) -> KeyframeDecision:
        if not tracking_ok:
            return KeyframeDecision(False, "tracking lost")

        # Never insert two keyframes back to back: consecutive frames have
        # almost no parallax, so the triangulation would be degenerate.
        if frames_since_keyframe < self.cfg.min_frame_gap:
            return KeyframeDecision(False, "too soon after last keyframe")

        # Hard ceiling so slow motion still produces map coverage.
        if frames_since_keyframe >= self.cfg.max_frame_gap:
            return KeyframeDecision(True, "max frame gap reached")

        # Tracking quality is falling: insert before the map connection is lost.
        if n_tracked_at_last_keyframe > 0:
            ratio = n_tracked / n_tracked_at_last_keyframe
            if ratio < self.cfg.track_ratio_threshold:
                return KeyframeDecision(True, f"track ratio dropped to {ratio:.2f}")

        # Enough translation relative to scene depth to triangulate well.
        # Measuring against depth makes the threshold scale-invariant, which
        # matters because monocular scale is arbitrary.
        baseline = float(np.linalg.norm(current_pose.center - last_keyframe_pose.center))
        depth = max(median_depth, 1e-6)
        if baseline / depth > self.cfg.min_translation_ratio:
            return KeyframeDecision(True,
                                    f"baseline/depth {baseline / depth:.3f} sufficient")

        return KeyframeDecision(False, "insufficient motion and tracking still healthy")
