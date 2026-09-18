"""Core data structures: poses, keyframes, landmarks, and the map.

Pose convention throughout this codebase
----------------------------------------
A `Pose` holds the camera-to-world transform: `R` and `t` such that a point in
camera coordinates maps to the world by ``x_world = R @ x_cam + t``. So `t` is
the camera centre in world coordinates and the trajectory is just the sequence
of `t`. OpenCV's ``solvePnP`` returns the *inverse* (world-to-camera), so
conversions go through `Pose.from_world_to_camera`. Getting this backwards is
the classic source of mirrored trajectories, hence the explicit helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Pose:
    """Camera-to-world rigid transform."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @property
    def matrix(self) -> np.ndarray:
        """4x4 camera-to-world homogeneous transform."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return self.t

    def inverse(self) -> Pose:
        return Pose(self.R.T, -self.R.T @ self.t)

    def compose(self, other: Pose) -> Pose:
        """self * other."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """Transform Nx3 world points into this camera's frame."""
        pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
        return (self.R.T @ (pts - self.t).T).T

    def camera_to_world(self, points_cam: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
        return (self.R @ pts.T).T + self.t

    @property
    def rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        """OpenCV world-to-camera (rvec, tvec), as solvePnP/projectPoints want."""
        import cv2

        Rcw = self.R.T
        tcw = -Rcw @ self.t
        rvec, _ = cv2.Rodrigues(Rcw)
        return rvec.reshape(3), tcw.reshape(3)

    @staticmethod
    def from_world_to_camera(R_cw: np.ndarray, t_cw: np.ndarray) -> Pose:
        """Build from a world-to-camera transform (e.g. solvePnP output)."""
        R_cw = np.asarray(R_cw, dtype=np.float64).reshape(3, 3)
        t_cw = np.asarray(t_cw, dtype=np.float64).reshape(3)
        return Pose(R_cw.T, -R_cw.T @ t_cw)

    @staticmethod
    def from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> Pose:
        import cv2

        R_cw, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        return Pose.from_world_to_camera(R_cw, tvec)

    @staticmethod
    def from_matrix(T: np.ndarray) -> Pose:
        T = np.asarray(T, dtype=np.float64).reshape(4, 4)
        return Pose(T[:3, :3], T[:3, 3])

    def copy(self) -> Pose:
        return Pose(self.R.copy(), self.t.copy())


@dataclass
class Landmark:
    """A triangulated 3D map point."""

    id: int
    position: np.ndarray
    observations: dict[int, int] = field(default_factory=dict)
    """keyframe_id -> index into that keyframe's `points` array."""
    color: tuple[int, int, int] = (200, 200, 200)
    reproj_error: float = 0.0
    is_outlier: bool = False

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)

    @property
    def n_observations(self) -> int:
        return len(self.observations)


@dataclass
class Keyframe:
    """A frame retained for mapping and optimisation."""

    id: int
    frame_index: int
    timestamp: float
    pose: Pose
    points: np.ndarray
    """Nx2 tracked pixel locations."""
    track_ids: np.ndarray
    """Length-N persistent track ids, linking observations across frames."""
    landmark_ids: dict[int, int] = field(default_factory=dict)
    """point index -> landmark id."""
    descriptors: np.ndarray | None = None
    """ORB descriptors, computed lazily for loop-closure retrieval only."""
    keypoints: np.ndarray | None = None
    bow: np.ndarray | None = None
    odometry_pose: Pose | None = None
    """Pose before global optimisation, kept for the before/after drift view."""

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32).reshape(-1, 2)
        self.track_ids = np.asarray(self.track_ids, dtype=np.int64).reshape(-1)


@dataclass
class TrajectoryPose:
    """One estimated camera pose in the output trajectory."""

    frame_index: int
    timestamp: float
    pose: Pose
    is_keyframe: bool = False
    n_inliers: int = 0
    tracking_ok: bool = True


class SlamMap:
    """Keyframes and landmarks, with the covisibility relation between them."""

    def __init__(self) -> None:
        self.keyframes: dict[int, Keyframe] = {}
        self.landmarks: dict[int, Landmark] = {}
        self._next_landmark_id = 0
        self._next_keyframe_id = 0

    # -- keyframes --------------------------------------------------------
    def add_keyframe(self, kf: Keyframe) -> None:
        self.keyframes[kf.id] = kf

    def new_keyframe_id(self) -> int:
        kid = self._next_keyframe_id
        self._next_keyframe_id += 1
        return kid

    @property
    def keyframe_ids(self) -> list[int]:
        return sorted(self.keyframes)

    def latest_keyframe(self) -> Keyframe | None:
        return self.keyframes[self.keyframe_ids[-1]] if self.keyframes else None

    # -- landmarks --------------------------------------------------------
    def add_landmark(self, position: np.ndarray,
                     color: tuple[int, int, int] = (200, 200, 200)) -> Landmark:
        lm = Landmark(id=self._next_landmark_id, position=position, color=color)
        self.landmarks[lm.id] = lm
        self._next_landmark_id += 1
        return lm

    def observe(self, landmark_id: int, keyframe_id: int, point_index: int) -> None:
        """Record that a keyframe sees a landmark at one of its points."""
        lm = self.landmarks.get(landmark_id)
        if lm is None:
            return
        lm.observations[keyframe_id] = point_index
        kf = self.keyframes.get(keyframe_id)
        if kf is not None:
            kf.landmark_ids[point_index] = landmark_id

    def remove_landmark(self, landmark_id: int) -> None:
        lm = self.landmarks.pop(landmark_id, None)
        if lm is None:
            return
        for kf_id, pt_idx in lm.observations.items():
            kf = self.keyframes.get(kf_id)
            if kf is not None and kf.landmark_ids.get(pt_idx) == landmark_id:
                del kf.landmark_ids[pt_idx]

    def active_landmarks(self) -> list[Landmark]:
        return [lm for lm in self.landmarks.values() if not lm.is_outlier]

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """(Nx3 positions, Nx3 uint8 colors) for all inlier landmarks."""
        lms = self.active_landmarks()
        if not lms:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        return (np.array([lm.position for lm in lms], dtype=np.float64),
                np.array([lm.color for lm in lms], dtype=np.uint8))

    def median_depth(self, keyframe_id: int) -> float:
        """Median depth of landmarks seen by a keyframe; the natural scale unit."""
        kf = self.keyframes.get(keyframe_id)
        if kf is None or not kf.landmark_ids:
            return 1.0
        pts = np.array([self.landmarks[l].position for l in kf.landmark_ids.values()
                        if l in self.landmarks])
        if len(pts) == 0:
            return 1.0
        depths = kf.pose.world_to_camera(pts)[:, 2]
        depths = depths[depths > 0]
        return float(np.median(depths)) if len(depths) else 1.0

    def covisible_keyframes(self, keyframe_id: int, min_shared: int = 15) -> list[int]:
        """Keyframes sharing at least `min_shared` landmarks with this one."""
        kf = self.keyframes.get(keyframe_id)
        if kf is None:
            return []
        counts: dict[int, int] = {}
        for lm_id in kf.landmark_ids.values():
            lm = self.landmarks.get(lm_id)
            if lm is None:
                continue
            for other in lm.observations:
                if other != keyframe_id:
                    counts[other] = counts.get(other, 0) + 1
        return sorted([k for k, c in counts.items() if c >= min_shared],
                      key=lambda k: -counts[k])

    def stats(self) -> dict[str, int]:
        return {
            "keyframes": len(self.keyframes),
            "landmarks": len(self.landmarks),
            "active_landmarks": len(self.active_landmarks()),
        }
