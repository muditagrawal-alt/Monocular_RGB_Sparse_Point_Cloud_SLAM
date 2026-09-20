# Monocular RGB Sparse Point-Cloud SLAM

Recover a camera trajectory and a sparse 3D point cloud from a single-lens RGB
video, on CPU, with accumulated drift corrected wherever the camera revisits a
place.

**Live: https://mo-1693468beaff4294af01115326a65b4f.ecs.us-east-1.on.aws**

Upload a video, watch the pipeline run, and inspect the reconstruction in 3D.
O-Hive take-home, Assignment 2, by Mudit Agrawal.

![Reconstruction](docs/media/reconstruction.gif)

*A 10 second clip, reconstructed in 5.9 seconds on CPU. The ring is the
recovered camera path with a frustum at every keyframe; the cloud inside it is
the 3,853 landmarks triangulated along the way.*

### Demo

[**Watch the 50 second walkthrough**](docs/media/demo.mp4) (upload, tracking,
reconstruction and the telemetry that backs the timing claim).

| Upload to finished map | |
|---|---|
| ![Pipeline](docs/media/pipeline.gif) | Corners are tracked with optical flow, each pose is solved against the map so far, and bundle adjustment refines poses and structure together. No neural network is involved at any point. |

---

## Contents

- [Measured processing time and test environment](#measured-processing-time-and-test-environment)
- [Accuracy](#accuracy)
- [Setup](#setup)
- [Deployment](#deployment)
- [Architecture and major technical decisions](#architecture-and-major-technical-decisions)
- [Libraries, frameworks and external components](#libraries-frameworks-and-external-components)
- [Known limitations and what I would improve with more time](#known-limitations-and-what-i-would-improve-with-more-time)
- [Repository layout](#repository-layout)

---

## Measured processing time and test environment

The requirement is that a 10-second video processes in 10 seconds or less.

### Deployed service (the number that counts)

| | |
|---|---|
| Input | 300 frames, 640x360, 30 fps, 10.0 s |
| **Processing time** | **5.77 to 5.92 s** |
| **Realtime factor** | **0.58x** |
| Consistency | 5 consecutive runs, all within budget |

**Test environment:** AWS ECS Express Mode on Fargate, `linux/amd64`,
8 vCPU / 16 GB, `us-east-1`. Python 3.11, OpenCV 4.11 headless, GTSAM 4.2.2.
Deployed settings: 448 px processing width, 560 features, loop-verification cap
100. No GPU is present or used.

### Other clips on the same deployment

| Clip | Length | Processing | Realtime factor |
|---|---:|---:|---:|
| Synthetic orbit (closes a loop) | 10.0 s | 5.85 s | 0.58x |
| Real museum walkthrough | 10.0 s | 8.53 s | 0.85x |
| Real street footage, forward motion | 10.0 s | 5.88 s | 0.59x |

### Development machine, for comparison

Apple M-series, macOS 26, Python 3.11: **0.42x to 0.65x realtime** at the
development defaults (512 px). A Fargate vCPU is roughly **2.5x slower** than an
Apple Silicon core on this workload, and Fargate task placement varies enough
that the same image measured 6.5 s on one host and 10.0 s on another. The
deployed configuration is tuned for the slower case.

### Where the time goes (deployed, 10 s clip)

| Stage | Time | Share |
|---|---:|---:|
| Front end (corner seeding, KLT tracking) | ~2.2 s | 38% |
| Pose estimation (PnP against the map) | ~1.5 s | 26% |
| Bundle adjustment | ~1.2 s | 21% |
| Loop detection and pose graph | ~0.9 s | 15% |

Asserted in CI by `tests/performance/test_budget.py`, so a regression past
realtime fails the build.

---

## Accuracy

Absolute trajectory error against exact synthetic ground truth, after the 7-DoF
Sim(3) alignment that monocular evaluation requires. Five seeds per
configuration, via `benchmarks/ablation.py`.

**Orbit, 300 frames, camera returns to its starting point**

| Configuration | ATE | Loops found |
|---|---:|---:|
| Frame-to-map PnP only (L1 + L2) | 3.0657 | 0.0 |
| + sliding-window bundle adjustment (L3) | 0.7343 | 0.0 |
| + loop closure and pose graph (L4) | **0.6416** | 0.2 |

**End to end: 4.78x**

**Strafe, 240 frames, camera never revisits**

| Configuration | ATE | Loops found |
|---|---:|---:|
| Frame-to-map PnP only (L1 + L2) | 1.3277 | 0.0 |
| + sliding-window bundle adjustment (L3) | **0.0987** | 0.0 |
| + loop closure and pose graph (L4) | 0.0987 | 0.0 |

**End to end: 13.45x**

Read together: bundle adjustment does most of the work on any sequence, and
loop closure adds more only where the camera genuinely returns somewhere. On
the non-looping clip it correctly finds nothing and changes nothing, which is
what you want from a component whose failure mode is warping the whole map.

### Real footage

45 arbitrary clips from Wikimedia Commons (walking tours, cathedral interiors,
cycling, drone flights, driving, hiking, tunnels, palaces), via
`benchmarks/wild.py`:

| | |
|---|---|
| Reconstructed | **42 of 44 valid clips** |
| Median landmarks | 1,052 |
| Median reprojection error | 1.02 px |
| Median realtime factor | 0.44x |

The two failures are both severely low-texture: an art installation offering 99
trackable corners and degraded 1906 archival film offering 125, against the 700
to 800 a healthy clip provides. Declining to build a map from those is correct
behaviour, not a gap.

**Results are reproducible.** Four separate processes produce identical output.
This was not true earlier in development and is discussed below.

---

## Setup

### Docker (recommended)

```bash
docker compose -f docker/compose.yaml up --build
# http://localhost:8000
```

### Local development

Requires **Python 3.11** specifically: GTSAM publishes no wheels for 3.13+, and
the pinned set is verified against 3.11.

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/python -m uvicorn api.main:app --reload    # API on :8000
cd web && npm install && npm run dev                  # UI on :5173
```

### Tests and checks

```bash
.venv/bin/pytest tests/ -q          # 78 tests
.venv/bin/ruff check slam/ api/ tests/ benchmarks/
.venv/bin/mypy slam/ api/
```

### Benchmarks

```bash
python benchmarks/ablation.py --frames 300 --seeds 5 --motion orbit
python benchmarks/wild.py --collect 30 && python benchmarks/wild.py --run
python benchmarks/tum.py --sequence freiburg1_xyz      # needs the dataset
```

---

## Deployment

The container is the deployment unit and runs unchanged on AWS or Azure.

```bash
./infra/bootstrap.sh     # one-time: ECR, IAM roles, log group, service
./infra/deploy.sh        # build, push, roll out, verify
./infra/deploy.sh --config-only   # change tuning without rebuilding
```

**Target: AWS ECS Express Mode**, which provisions a Fargate service, load
balancer, TLS and a public HTTPS hostname from a container image. Note that
**AWS App Runner stopped accepting new customers on 30 April 2026**, so the
obvious choice from older documentation is unavailable; Express Mode is its
successor.

### Runtime tuning

The right settings depend on how fast the host is, so they are environment
variables rather than rebuilt constants. Every applied value is reported back
in each result, so a run is never ambiguous about what produced it.

| Variable | Deployed | Purpose |
|---|---|---|
| `SLAM_TARGET_WIDTH` | 448 | Processing width. The main runtime lever |
| `SLAM_MAX_FEATURES` | 560 | Corner budget per frame |
| `SLAM_LOOP_VERIFICATIONS` | 100 | Cap on geometric verifications |
| `SLAM_VERIFY_COST_S` | 0.030 | Assumed cost per verification, for budgeting |
| `SLAM_RTF_TARGET` | 0.80 | Target realtime factor, leaving margin |
| `SLAM_LOOP_CLOSURE` | on | Loop detection and pose-graph correction |
| `SLAM_ADAPTIVE_BUDGET` | off | Mid-flight quality degradation; see limitations |

### Two deployment gotchas worth recording

**Buildx attestation breaks the Fargate pull.** The default build produces an
OCI image index with attestation manifests, which Fargate cannot pull. Build
with `--provenance=false --sbom=false`.

**The service URL is not in the API response.** `serviceUrl` came back null.
The AWS-provided hostname is on the load balancer's listener rule; `deploy.sh`
reads it from there.

---

## Architecture and major technical decisions

```
video ─► decode & downscale ─► intrinsics resolution
           │
           ▼
   FRONT END (every frame)
   Shi-Tomasi corners seeded per grid cell  ─►  pyramidal KLT optical flow
   forward-backward consistency check, re-seed as tracks expire
           │
           ▼
   TWO-VIEW INITIALISATION
   homography vs essential model selection, parallax gate,
   scene depth normalised to 1.0
           │
           ▼
   TRACKING (every frame)
   solvePnPRansac against the existing map, constant-velocity prior
           │
           ▼
   keyframe decision ─► triangulate new landmarks ─► cull outliers
           │
           ▼
   BACK END
   sliding-window bundle adjustment  (GTSAM, Huber robust kernel)
   loop detection                    (bag of words + geometry + consistency)
   global pose-graph optimisation    (GTSAM, robust loop constraints)
           │
           ▼
   trajectory (TUM + JSON) │ point cloud (PLY + packed binary) │ telemetry
```

### Classical geometry, not a learned model

The 2025-26 headline monocular systems (MASt3R-SLAM, VGGT-SLAM, SLAM3R) are
dense, transformer-based and need a CUDA GPU to run at real time. They are the
wrong tool here for three independent reasons: the assignment asks for a
**sparse** point cloud, a GPU task costs far more and complicates deployment,
and the 10-second budget is achievable on CPU classically.

So this uses **no machine-learning model at all**: no neural network, no
weights, no inference. Shi-Tomasi corners, Lucas-Kanade optical flow, essential
and homography matrices, PnP with RANSAC, linear triangulation, bundle
adjustment and pose-graph optimisation. Even the loop-closure vocabulary is
built by k-means over the video's own descriptors at runtime, so nothing
pretrained ships with it.

### Track with optical flow, describe only at keyframes

The single largest performance decision, benchmarked before committing
(`benchmarks/feasibility/`), single core at 640x360:

| Front end | Cost per frame |
|---|---:|
| ORB detect + describe + BFMatcher + RANSAC | 27.5 ms |
| Shi-Tomasi + pyramidal KLT | **4.3 ms** |

Roughly 6x cheaper. Descriptors are computed only at keyframes, where they are
genuinely needed for loop-closure retrieval.

### Drift control is layered

No single mechanism is sufficient:

| Layer | Mechanism | What it addresses |
|---|---|---|
| **L1** | RANSAC everywhere, Huber kernels, forward-backward track check, landmark culling | Outliers, before they corrupt the map |
| **L2** | Frame-to-map PnP rather than frame-to-frame chaining | Pose error compounding through integration |
| **L3** | Sliding-window bundle adjustment | Local error growth in poses and structure |
| **L4** | Loop detection plus global pose-graph optimisation | Error already accumulated over the sequence |

Loop detection runs cheap-to-expensive, because a false loop warps the whole
map and is far worse than a missed one: bag-of-words retrieval with TF-IDF
weighting, fused by reciprocal rank with proximity in the current estimate,
then descriptor matching with RANSAC PnP against mapped landmarks, then a
consistency requirement that strong geometric support can override.

### Latency is handled where it does no harm

An earlier design degraded the map mid-stream when it projected an overrun.
That made accuracy depend on machine load: the same clip scored 0.76 ATE on an
idle machine and 3.75 when the guard fired under contention. That is now
**off by default**.

Instead, loop detection sizes its own budget from the time actually remaining.
It runs after the frame loop, so by then the map is already built and correct,
and the remaining budget is known exactly. Full detection quality when there is
time; detection is given up rather than the deadline when there is not.

### Process isolation for the worker

SLAM is CPU-bound for seconds at a time, so it runs in a separate **process**,
not a thread: a thread would contend on the GIL with the event loop and make
progress polling unresponsive. Progress is reported by atomically replacing a
small JSON file rather than over a multiprocessing queue, because a `Manager`
queue needs a broker process that re-imports `__main__`, which is fragile under
the `spawn` start method used on macOS and in containers.

---

## Libraries, frameworks and external components

### Runtime

| Component | Version | Role |
|---|---|---|
| Python | 3.11 | GTSAM publishes no wheels for 3.13+ |
| OpenCV (headless) | 4.11.0.86 | Corners, optical flow, PnP, epipolar geometry, video I/O |
| GTSAM | 4.2.2 | Factor-graph optimisation: bundle adjustment and pose graph |
| NumPy | 1.26.4 | Array maths throughout |
| SciPy | 1.14.1 | Spatial helpers |
| FastAPI | 0.115.6 | HTTP API |
| uvicorn | 0.34.0 | ASGI server |
| Pydantic | 2.10.4 | Request and response models |

**The pins are load-bearing.** GTSAM 4.2.2 requires `numpy<2`, which forces
`opencv<=4.11.x`. Bumping either alone breaks the environment. The same
constraint rules out `rerun-sdk` (needs `numpy>=2`), which is why the viewer is
custom rather than an off-the-shelf visualiser.

### Frontend

React 18, TypeScript 5.7, Vite 6, Tailwind CSS v4, react-three-fiber and drei
over three.js, Phosphor icons. Built as a static bundle and served by the same
container, so there is one origin and no CORS in production.

### Development

pytest, ruff, mypy, httpx (test client), evo (trajectory metrics),
Playwright (browser verification), Docker, GitHub Actions.

### Pretrained models

**None.** No neural network, no weights file, no downloaded checkpoint, no
inference of any kind. This is stated explicitly because it is unusual for a
2026 vision project and is the reason the service needs no GPU.

### External data

Wikimedia Commons video (CC BY-SA and public domain) for the wild benchmark;
each clip's source and licence is recorded in `samples/README.md`. The TUM
RGB-D benchmark is supported but its dataset is not redistributed here.

---

## Known limitations and what I would improve with more time

### Inherent to monocular SLAM

**Scale is unobservable.** One moving camera cannot recover absolute size: a
real room and a perfect dollhouse produce identical images. Output is in
consistent but arbitrary "slam units", labelled as such in the UI. *Fix:* show
the camera one object of known size, or add any second source (stereo, depth,
IMU, GPS).

**Uploaded video carries no calibration.** Focal length comes from container
metadata when present, otherwise a 60-degree horizontal field of view is
assumed, with a UI override. A wrong guess warps the geometry. *Fix:* estimate
the focal length as a free variable in a final global bundle adjustment.

**Degenerate motion is refused, not guessed.** Pure rotation gives no baseline,
so no depth information exists; the system declines rather than inventing a
map. *Fix:* none possible. This is geometry.

**The world is assumed static.** Crowds and traffic violate that. *Fix:* mask
moving regions with a segmentation model, which would reintroduce a GPU
dependency.

### Things I would fix first with more time

**Loop closure is under-sensitive.** It fires on roughly 1 in 5 synthetic
sequences and 8 of 42 real clips. When it fires it helps (ATE 0.73 to 0.64 on
the orbit set); when it does not, layers L1 to L3 carry the result alone. The
bias is deliberate, since a false loop warps the entire map, but the retrieval
stage is the weak link. *Fix:* a proper DBoW2-style vocabulary tree trained
offline instead of online k-means, which would sharpen retrieval considerably.

**Tracking robustness on real footage.** Only 20 of 42 real clips track with no
loss at all. Losses trigger re-initialisation and are handled, but the resulting
trajectory is segmented. *Fix:* relocalisation against the existing map rather
than starting fresh.

**Accuracy validation is synthetic.** The TUM RGB-D harness is built and tested
end to end against a synthetic sequence in exact TUM layout, but the dataset
host was unreachable from the development network, so no real-benchmark ATE
numbers exist. *Fix:* run `benchmarks/tum.py` from a network that can reach it.

**Sparse means sparse.** About 1 point per 400 pixels: enough to localise, not
enough to look like a scene. *Fix:* a densification pass after the poses are
known, as an optional high-quality mode, since it would not fit the budget.

**Forward motion is the weak geometry.** Moving along the optical axis gives
little parallax, so those clips yield thinner maps. This is inherent, but a
wider baseline keyframe policy would help.

**Fargate placement varies.** The same image measured 6.5 s on one host and
10.0 s on another. The deployment is tuned for the slow case, which leaves
headroom unused on a fast one. *Fix:* calibrate per-task at startup with a short
synthetic benchmark.

### A process note

Several published figures in this repository turned out not to reproduce,
because OpenCV's RANSAC was unseeded and the adaptive guard reacted to machine
load. Both are fixed and results are now deterministic, but the numbers above
were re-measured from scratch afterwards rather than carried over. Single
measurements were treated as facts earlier in development; they should not have
been.

---

## Repository layout

```
slam/                 core library, no web or cloud dependencies
  camera.py           pinhole model and intrinsics resolution
  geometry.py         triangulation, parallax, Sim(3) alignment
  types.py            poses, keyframes, landmarks, the map
  config.py           all tunable parameters, and env overrides
  pipeline.py         orchestration
  core/               feature tracking, initialisation, PnP tracking, mapping
  optim/              bundle adjustment, loop closure, pose graph
  io/                 video decode, image sequences, synthetic fixture, exporters
api/                  FastAPI service and job manager
web/                  React + TypeScript viewer
tests/                unit, integration, performance
benchmarks/           feasibility, ablation, wild-video and TUM harnesses
docker/               image and compose file
infra/                bootstrap and deploy scripts
docs/                 implementation plan, design brief, capture guide
```

Further reading: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the
original plan and where it met reality, [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md)
for how to record a video that reconstructs well, and
[docs/DESIGN_BRIEF.md](docs/DESIGN_BRIEF.md) for the UI design derivation.
