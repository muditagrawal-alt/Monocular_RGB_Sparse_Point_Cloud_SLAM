"""Evaluate against the TUM RGB-D benchmark.

The synthetic tests give exact ground truth but easy imagery. This measures the
same pipeline on real handheld footage with motion blur, rolling shutter, poor
texture and genuine tracking failures, scored against motion-capture ground
truth.

Monocular results are only defined up to a similarity transform, so trajectories
are aligned with a 7-DoF Umeyama fit before scoring, which is the standard
protocol for monocular entries on this benchmark.

    python benchmarks/tum.py --list
    python benchmarks/tum.py --sequence freiburg1_xyz
    python benchmarks/tum.py --sequence freiburg1_xyz --data-dir /path/to/extracted

The download host (webshare.cvg.cit.tum.de) is not reachable from every
network. When it is not, fetch the .tgz manually from
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download, extract it, and
pass --data-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slam.config import SlamConfig  # noqa: E402
from slam.geometry import align_sim3  # noqa: E402
from slam.pipeline import SlamPipeline  # noqa: E402

BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"

SEQUENCES = {
    "freiburg1_xyz": ("freiburg1", "Mostly translation, the gentlest sequence"),
    "freiburg1_desk": ("freiburg1", "Desk loop, revisits the start"),
    "freiburg1_room": ("freiburg1", "Large room loop, long trajectory"),
    "freiburg2_desk": ("freiburg2", "Slow desk loop, low drift expected"),
    "freiburg3_long_office_household": (
        "freiburg3", "Long office loop, the standard loop-closure sequence"),
}


@dataclass
class Evaluation:
    sequence: str
    n_gt: int
    n_est: int
    n_matched: int
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_max: float
    scale: float
    tracked_fraction: float
    processing_time_s: float
    duration_s: float
    realtime_factor: float
    loop_closures: int
    ate_before_correction: float | None = None

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def read_tum_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read `timestamp tx ty tz qx qy qz qw` into (timestamps, Nx3 positions)."""
    times: list[float] = []
    positions: list[list[float]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        times.append(float(parts[0]))
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(times), np.array(positions, dtype=np.float64).reshape(-1, 3)


def associate(est_times: np.ndarray, gt_times: np.ndarray,
              max_difference: float = 0.02) -> list[tuple[int, int]]:
    """Match estimated to ground-truth poses by nearest timestamp.

    Ground truth is logged at a different rate to the camera, so entries are
    paired by time rather than index, and unmatched frames are dropped rather
    than paired with a distant pose.
    """
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, t in enumerate(est_times):
        j = int(np.argmin(np.abs(gt_times - t)))
        if j in used or abs(gt_times[j] - t) > max_difference:
            continue
        used.add(j)
        pairs.append((i, j))
    return pairs


def evaluate(result, gt_path: Path, sequence: str, t_offset: float) -> Evaluation:
    gt_times, gt_pos = read_tum_trajectory(gt_path)

    tracked = [t for t in result.trajectory if t.tracking_ok]
    est_times = np.array([t.timestamp + t_offset for t in tracked])
    est_pos = np.array([t.pose.center for t in tracked], dtype=np.float64).reshape(-1, 3)

    pairs = associate(est_times, gt_times)
    if len(pairs) < 10:
        raise RuntimeError(f"only {len(pairs)} timestamp matches; check the offset")

    est = est_pos[[i for i, _ in pairs]]
    ref = gt_pos[[j for _, j in pairs]]

    # Monocular: solve the 7-DoF similarity before scoring.
    scale, R, t = align_sim3(est, ref, with_scale=True)
    aligned = (scale * (R @ est.T).T) + t
    errors = np.linalg.norm(aligned - ref, axis=1)

    before = None
    if result.odometry_trajectory and result.drift_correction_applied:
        odo = [t for t in result.odometry_trajectory if t.tracking_ok]
        odo_times = np.array([t.timestamp + t_offset for t in odo])
        odo_pos = np.array([t.pose.center for t in odo], dtype=np.float64).reshape(-1, 3)
        odo_pairs = associate(odo_times, gt_times)
        if len(odo_pairs) >= 10:
            oe = odo_pos[[i for i, _ in odo_pairs]]
            orf = gt_pos[[j for _, j in odo_pairs]]
            s2, r2, t2 = align_sim3(oe, orf, with_scale=True)
            before = float(np.sqrt(
                (((s2 * (r2 @ oe.T).T) + t2 - orf) ** 2).sum(axis=1).mean()))

    return Evaluation(
        sequence=sequence,
        n_gt=len(gt_times),
        n_est=len(tracked),
        n_matched=len(pairs),
        ate_rmse=float(np.sqrt((errors ** 2).mean())),
        ate_mean=float(errors.mean()),
        ate_median=float(np.median(errors)),
        ate_max=float(errors.max()),
        scale=float(scale),
        tracked_fraction=len(tracked) / max(result.n_frames_processed, 1),
        processing_time_s=result.processing_time_s,
        duration_s=result.video_duration_s,
        realtime_factor=result.realtime_factor,
        loop_closures=result.n_loop_closures,
        ate_before_correction=before)


def download(sequence: str, dest: Path) -> Path:
    group = SEQUENCES[sequence][0]
    name = f"rgbd_dataset_{sequence}"
    url = f"{BASE_URL}/{group}/{name}.tgz"
    archive = dest / f"{name}.tgz"
    target = dest / name

    if target.is_dir():
        return target
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
                open(archive, "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
    except Exception as exc:
        raise SystemExit(
            f"\nDownload failed: {exc}\n\n"
            f"The TUM download host is not reachable from every network. Fetch\n"
            f"{url}\nmanually, extract it, and re-run with --data-dir pointing at\n"
            f"the extracted folder.") from exc

    print("Extracting")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", default="freiburg1_xyz", choices=sorted(SEQUENCES))
    parser.add_argument("--data-dir", type=Path,
                        help="Folder of an already-extracted sequence")
    parser.add_argument("--cache", type=Path, default=Path("benchmarks/datasets"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/tum.json"))
    parser.add_argument("--list", action="store_true", help="List sequences and exit")
    args = parser.parse_args()

    if args.list:
        print("Available sequences:\n")
        for name, (_, description) in sorted(SEQUENCES.items()):
            print(f"  {name:36s} {description}")
        return

    root = args.data_dir or download(args.sequence, args.cache)
    rgb_dir = root / "rgb"
    rgb_txt = root / "rgb.txt"
    gt_txt = root / "groundtruth.txt"
    for required in (rgb_dir, rgb_txt, gt_txt):
        if not required.exists():
            raise SystemExit(f"missing {required}; is {root} a TUM sequence folder?")

    # TUM intrinsics per camera, from the dataset's calibration page. Supplying
    # the real focal length isolates SLAM accuracy from the field-of-view
    # guess the service falls back to for uncalibrated uploads.
    focal = {"freiburg1": 517.3, "freiburg2": 520.9, "freiburg3": 535.4}[
        SEQUENCES[args.sequence][0]]

    print(f"Running {args.sequence} from {root}")
    started = time.time()
    result = SlamPipeline(SlamConfig()).run(rgb_dir, focal_px=focal,
                                            timestamps_file=rgb_txt)
    if not result.success:
        raise SystemExit(f"reconstruction failed: {result.reason}")

    first_ts = float(next(
        line.split()[0] for line in rgb_txt.read_text().splitlines()
        if line.strip() and not line.startswith("#")))
    evaluation = evaluate(result, gt_txt, args.sequence, first_ts)

    print(f"\n{args.sequence}  ({time.time() - started:.1f}s wall)")
    print(f"  frames tracked      {evaluation.tracked_fraction:.1%} "
          f"({evaluation.n_est} of {result.n_frames_processed})")
    print(f"  poses scored        {evaluation.n_matched}")
    print(f"  ATE RMSE            {evaluation.ate_rmse:.4f} m")
    print(f"  ATE median          {evaluation.ate_median:.4f} m")
    print(f"  ATE max             {evaluation.ate_max:.4f} m")
    print(f"  recovered scale     {evaluation.scale:.4f}")
    print(f"  loop closures       {evaluation.loop_closures}")
    if evaluation.ate_before_correction is not None:
        print(f"  ATE before drift correction  {evaluation.ate_before_correction:.4f} m")
    print(f"  realtime factor     {evaluation.realtime_factor:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evaluation.as_dict(), indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
