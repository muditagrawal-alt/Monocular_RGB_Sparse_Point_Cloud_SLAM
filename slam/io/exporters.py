"""Serialisation of SLAM output: point cloud, trajectory, and viewer payload."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..types import SlamMap, TrajectoryPose


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None,
              binary: bool = True) -> str:
    """Write a point cloud as PLY, the standard interchange format."""
    path = str(path)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if colors is not None and len(colors):
        cols = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)[finite]
    else:
        cols = np.full((len(pts), 3), 200, dtype=np.uint8)

    header = (
        "ply\n"
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n")

    if binary:
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            for p, c in zip(pts, cols, strict=False):
                fh.write(struct.pack("<fffBBB", p[0], p[1], p[2], c[0], c[1], c[2]))
    else:
        with open(path, "w") as fh:
            fh.write(header)
            for p, c in zip(pts, cols, strict=False):
                fh.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
    return path


def write_tum_trajectory(path: str | Path, trajectory: list[TrajectoryPose]) -> str:
    """Write the trajectory in TUM format: `timestamp tx ty tz qx qy qz qw`.

    This is the format the standard evaluation tools (evo, the TUM benchmark
    scripts) expect, so results can be scored without conversion.
    """
    path = str(path)
    with open(path, "w") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw\n")
        for tp in trajectory:
            if not tp.tracking_ok:
                continue
            t = tp.pose.t
            qx, qy, qz, qw = _rotation_to_quaternion(tp.pose.R)
            fh.write(f"{tp.timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                     f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
    return path


def _rotation_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (qx, qy, qz, qw), using the numerically stable branch."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)


def _downsample(points: np.ndarray, colors: np.ndarray, limit: int,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample a cloud so the browser payload stays small."""
    if len(points) <= limit:
        return points, colors
    idx = np.random.default_rng(seed).choice(len(points), limit, replace=False)
    idx.sort()
    return points[idx], colors[idx]


def result_to_viewer_json(result: Any, max_points: int = 60000) -> dict:
    """Build the JSON payload the web viewer consumes.

    Coordinates are emitted as flat lists so the browser can load them straight
    into typed arrays without per-point object allocation.
    """
    slam_map: SlamMap | None = result.slam_map
    points = np.zeros((0, 3))
    colors = np.zeros((0, 3), dtype=np.uint8)
    if slam_map is not None:
        points, colors = slam_map.point_cloud()
        points, colors = _downsample(points, colors, max_points)

    def traj_payload(traj: list[TrajectoryPose]) -> dict:
        pts = [t.pose.center.tolist() for t in traj if t.tracking_ok]
        return {
            "positions": [c for p in pts for c in p],
            "count": len(pts),
            "keyframe_indices": [i for i, t in enumerate(
                [t for t in traj if t.tracking_ok]) if t.is_keyframe],
        }

    keyframes = []
    if slam_map is not None:
        for kf_id in slam_map.keyframe_ids:
            kf = slam_map.keyframes[kf_id]
            keyframes.append({
                "id": kf.id,
                "frame_index": kf.frame_index,
                "position": kf.pose.center.tolist(),
                "rotation": kf.pose.R.flatten().tolist(),
            })

    return {
        "success": result.success,
        "reason": result.reason,
        "summary": result.summary(),
        "cloud": {
            "positions": points.astype(np.float32).flatten().tolist(),
            "colors": colors.astype(np.uint8).flatten().tolist(),
            "count": int(len(points)),
            "truncated": bool(slam_map is not None
                              and len(slam_map.active_landmarks()) > len(points)),
        },
        "trajectory": traj_payload(result.trajectory),
        "odometry_trajectory": traj_payload(result.odometry_trajectory),
        "keyframes": keyframes,
        "camera": ({"fx": result.camera.fx, "fy": result.camera.fy,
                    "cx": result.camera.cx, "cy": result.camera.cy,
                    "width": result.camera.width, "height": result.camera.height,
                    "source": result.camera.source}
                   if result.camera is not None else None),
    }


def write_result_json(path: str | Path, result: Any, max_points: int = 60000) -> str:
    path = str(path)
    with open(path, "w") as fh:
        json.dump(result_to_viewer_json(result, max_points), fh)
    return path
