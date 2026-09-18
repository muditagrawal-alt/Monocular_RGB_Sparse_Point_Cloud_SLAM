"""Ablation study: how much does each drift-control layer actually buy?

Requirement 4 asks for an approach that minimises accumulated error. Asserting
that loop closure helps is cheap; measuring it is the point of this script.

Each configuration is run over several synthetic sequences with exact ground
truth, and absolute trajectory error is reported after 7-DoF Sim(3) alignment,
which monocular evaluation requires because scale is unobservable.

    python benchmarks/ablation.py --frames 300 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running as a plain script from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slam.config import SlamConfig
from slam.geometry import ate_rmse
from slam.io.synthetic import make_sequence
from slam.pipeline import SlamPipeline


@dataclass
class Variant:
    name: str
    description: str
    local_ba: bool
    loop_closure: bool

    def config(self) -> SlamConfig:
        cfg = SlamConfig()
        cfg.local_ba.enabled = self.local_ba
        cfg.loop.enabled = self.loop_closure
        cfg.pose_graph.enabled = self.loop_closure
        return cfg


VARIANTS = [
    Variant("odometry", "Frame-to-map PnP only (L1 + L2)", False, False),
    Variant("local_ba", "+ sliding-window bundle adjustment (L3)", True, False),
    Variant("full", "+ loop closure and pose graph (L4)", True, True),
]


def trajectory_ate(result, sequence) -> float:
    gt = np.array([p.center for p in sequence.poses])
    est = np.array([t.pose.center for t in result.trajectory])
    ref = gt[[t.frame_index for t in result.trajectory]]
    ok = np.array([t.tracking_ok for t in result.trajectory])
    if ok.sum() < 3:
        return float("nan")
    return ate_rmse(est[ok], ref[ok])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--motion", default="orbit", choices=["orbit", "strafe", "forward"])
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/ablation.json"))
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="slam-ablation-"))
    videos = []
    for seed in range(args.seeds):
        seq = make_sequence(n_frames=args.frames, motion=args.motion,
                            loop=(args.motion == "orbit"), seed=seed)
        path = tmp / f"{args.motion}-{seed}.mp4"
        seq.write_video(path)
        videos.append((seed, seq, str(path)))

    print(f"Ablation: {args.motion}, {args.frames} frames, {args.seeds} seeds\n")
    header = f"{'configuration':<44} {'ATE':>9} {'loops':>6} {'RTF':>7}"
    print(header)
    print("-" * len(header))

    records: list[dict] = []
    for variant in VARIANTS:
        ates, rtfs, loops = [], [], []
        for _, seq, path in videos:
            result = SlamPipeline(variant.config()).run(path)
            if not result.success:
                continue
            ates.append(trajectory_ate(result, seq))
            rtfs.append(result.realtime_factor)
            loops.append(result.n_loop_closures)

        if not ates:
            print(f"{variant.description:<44} {'failed':>9}")
            continue

        mean_ate = statistics.fmean(ates)
        record = {
            "variant": variant.name,
            "description": variant.description,
            "ate_mean": mean_ate,
            "ate_all": ates,
            "loops_mean": statistics.fmean(loops),
            "realtime_factor_mean": statistics.fmean(rtfs),
        }
        records.append(record)
        print(f"{variant.description:<44} {mean_ate:>9.4f} "
              f"{statistics.fmean(loops):>6.1f} {statistics.fmean(rtfs):>7.2f}")

    if len(records) >= 2:
        baseline = records[0]["ate_mean"]
        best = records[-1]["ate_mean"]
        print(f"\nEnd-to-end improvement over odometry: {baseline / max(best, 1e-9):.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"motion": args.motion, "frames": args.frames, "seeds": args.seeds,
         "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "results": records},
        indent=2))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
