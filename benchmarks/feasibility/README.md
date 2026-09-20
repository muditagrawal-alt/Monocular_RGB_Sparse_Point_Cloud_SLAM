# Feasibility benchmarks

Scripts that produced the measured numbers quoted in `IMPLEMENTATION_PLAN.md`.
They were run *before* the architecture was chosen, to validate the two riskiest
assumptions in the plan.

Run with the pinned stack (Python 3.11, **not** the host `python3`, which is 3.14):

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install "numpy==1.26.4" "opencv-python-headless==4.11.0.86" "gtsam==4.2.2"
.venv/bin/python benchmarks/feasibility/bench_orb_frontend.py
.venv/bin/python benchmarks/feasibility/bench_klt_frontend.py
```

## What they established

**`bench_orb_frontend.py`**: per-frame cost of an ORB detect+describe+BFMatch+RANSAC front end.
Result at 640x360/1000 features, single core: **27.5 ms/frame**. The BFMatcher dominates and
scales badly with feature count (9.5 ms at 1000 features -> 36.7 ms at 2000).

**`bench_klt_frontend.py`**: per-frame cost of a Shi-Tomasi + pyramidal KLT front end.
Result at 640x360/800 points, single core: **4.3 ms/frame**, ~580 points tracked.
ORB detect+describe alone (paid only at keyframes) is 11.1 ms.

=> KLT is ~6x cheaper than per-frame descriptor matching. This is why the plan tracks with KLT
every frame and computes descriptors only at keyframes, where they are needed for loop closure.

The GTSAM pose-graph drift experiment (4.757 m -> 0.539 m ATE, 8.8x reduction, 2.9 ms solve for
60 poses) is described in plan section 4.6; it will be reimplemented as a proper regression test
in Phase 2 rather than kept as a throwaway script.

Note: measured on Apple Silicon (arm64). AWS x86_64 Fargate figures are to be re-measured in
Phase 3, see the plan's performance section for the headroom levers held in reserve.
