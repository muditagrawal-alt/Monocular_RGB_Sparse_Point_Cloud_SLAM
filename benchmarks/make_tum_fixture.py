"""Write a synthetic sequence in exact TUM RGB-D layout.

Used to prove the TUM evaluation path end to end -- timestamp association,
Sim(3) alignment, ATE computation -- without needing the real dataset, which is
hundreds of megabytes and whose download host is not reachable everywhere. If
the harness scores a sequence whose ground truth it already knows, the harness
is correct and only the data is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slam.io.exporters import _rotation_to_quaternion  # noqa: E402
from slam.io.synthetic import make_sequence  # noqa: E402


def write_fixture(dest: Path, n_frames: int = 150, motion: str = "strafe",
                  fps: float = 30.0, seed: int = 0, t0: float = 1305031102.175304
                  ) -> Path:
    """Create `rgb/`, `rgb.txt` and `groundtruth.txt` under `dest`.

    `t0` mimics the Unix-epoch timestamps the real dataset uses, so the
    association code is exercised on realistic values rather than small
    integers.
    """
    sequence = make_sequence(n_frames=n_frames, motion=motion,
                             loop=(motion == "orbit"), seed=seed)
    rgb = dest / "rgb"
    rgb.mkdir(parents=True, exist_ok=True)

    rgb_lines = ["# color images", "# timestamp filename"]
    for i, frame in enumerate(sequence.frames):
        stamp = t0 + i / fps
        name = f"{stamp:.6f}.png"
        cv2.imwrite(str(rgb / name), frame)
        rgb_lines.append(f"{stamp:.6f} rgb/{name}")
    (dest / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")

    # Ground truth is logged faster than the camera in the real dataset, so
    # emit it at double rate to exercise nearest-timestamp association.
    gt_lines = ["# ground truth trajectory", "# timestamp tx ty tz qx qy qz qw"]
    for i, pose in enumerate(sequence.poses):
        for sub in (0.0, 0.5):
            if i + 1 >= len(sequence.poses) and sub > 0:
                break
            stamp = t0 + (i + sub) / fps
            if sub == 0.0:
                centre, rotation = pose.center, pose.R
            else:
                nxt = sequence.poses[i + 1]
                centre = 0.5 * (pose.center + nxt.center)
                rotation = pose.R
            qx, qy, qz, qw = _rotation_to_quaternion(rotation)
            gt_lines.append(
                f"{stamp:.6f} {centre[0]:.6f} {centre[1]:.6f} {centre[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
    (dest / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--motion", default="strafe",
                        choices=["strafe", "forward", "orbit"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = write_fixture(args.dest, args.frames, args.motion, seed=args.seed)
    n_gt = sum(1 for line in (out / "groundtruth.txt").read_text().splitlines()
               if not line.startswith("#"))
    print(f"Wrote {args.frames} frames and {n_gt} ground-truth poses to {out}")


if __name__ == "__main__":
    main()
