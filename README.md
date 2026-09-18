# Monocular RGB Sparse Point-Cloud SLAM

Recover a camera trajectory and a sparse 3D point cloud from a single-lens RGB
video, on CPU, with accumulated drift corrected wherever the camera revisits a
place.

Built for O-HIVE take-home assignment 2. Implementation plan:
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

---

## Results

Measured on 640x360 synthetic sequences with exact ground truth (Apple Silicon,
single process). Absolute trajectory error is reported after 7-DoF Sim(3)
alignment, which monocular evaluation requires because scale is unobservable.

### Drift reduction (requirement 4)

Each layer is measured separately, three seeds per configuration, via
`benchmarks/ablation.py`:

**Orbit, 300 frames, camera returns to its starting point**

| Configuration | ATE | Loops found | Realtime factor |
|---|---:|---:|---:|
| Frame-to-map PnP only (L1 + L2) | 3.1271 | 0.0 | 0.38 |
| + sliding-window bundle adjustment (L3) | 0.7798 | 0.0 | 0.59 |
| + loop closure and pose graph (L4) | **0.6127** | 0.3 | 0.88 |

**End-to-end improvement: 5.10x**

**Strafe, 240 frames, camera never revisits a place**

| Configuration | ATE | Loops found | Realtime factor |
|---|---:|---:|---:|
| Frame-to-map PnP only (L1 + L2) | 1.3235 | 0.0 | 0.30 |
| + sliding-window bundle adjustment (L3) | **0.1376** | 0.0 | 0.36 |
| + loop closure and pose graph (L4) | 0.1376 | 0.0 | 0.36 |

**End-to-end improvement: 9.62x**

Read together these say something useful: bundle adjustment does most of the
work on any sequence, and loop closure adds more only when the camera actually
returns somewhere. On the non-looping clip it correctly finds nothing and
changes nothing, which is the behaviour you want from a component whose failure
mode is warping the entire map.

### Processing time (requirement 6)

Target: a 10-second clip in 10 seconds or less.

| | |
|---|---|
| Video length | 10.0 s (300 frames at 30 fps) |
| Processing time | **8.5 s** |
| Realtime factor | **0.85x** |

Asserted in CI by `tests/performance/test_budget.py`, so a regression past
realtime fails the build.

---

## Running it

Requires Docker.

```bash
docker compose -f docker/compose.yaml up --build
# open http://localhost:8000
```

Local development without Docker needs **Python 3.11** specifically (gtsam
publishes no wheels for 3.13+):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m uvicorn api.main:app --reload    # API on :8000

cd web && npm install && npm run dev                  # UI on :5173
```

Tests, lint, types:

```bash
.venv/bin/pytest tests/ -q
.venv/bin/ruff check slam/ api/ tests/
.venv/bin/mypy slam/ api/
```

---

## How it works

```
video ─► decode & downscale ─► intrinsics resolution
           │
           ▼
   FRONT END (every frame)
   Shi-Tomasi seeding on a grid  ─►  pyramidal KLT optical flow
   forward-backward consistency check, re-seed as tracks expire
           │
           ▼
   two-view initialisation
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
   sliding-window bundle adjustment  (GTSAM, Huber)
   loop detection                    (bag of words + geometry + consistency)
   global pose-graph optimisation    (GTSAM, robust loop constraints)
           │
           ▼
   trajectory (TUM + JSON) │ point cloud (PLY + packed binary) │ telemetry
```

### The front end tracks, it does not match

The single biggest performance decision. Benchmarked both, single core at
640x360 (`benchmarks/feasibility/`):

| Front end | Cost per frame |
|---|---:|
| ORB detect + describe + BFMatcher + RANSAC | 27.5 ms |
| Shi-Tomasi + pyramidal KLT | **4.3 ms** |

So the pipeline tracks with optical flow every frame and computes ORB
descriptors only at keyframes, where they are genuinely needed for loop-closure
retrieval. That is roughly 6x cheaper and is what makes the 10 second budget
comfortable rather than marginal.

### Drift control is layered

No single mechanism is sufficient, so there are four:

| Layer | Mechanism | What it addresses |
|---|---|---|
| L1 | RANSAC everywhere, Huber kernels, forward-backward track check, landmark culling | Outliers, before they corrupt the map |
| L2 | Frame-to-map PnP rather than frame-to-frame chaining | Pose error compounding through integration |
| L3 | Sliding-window bundle adjustment | Local error growth in poses and structure |
| L4 | Loop detection plus global pose-graph optimisation | Error already accumulated over the sequence |

Loop detection runs cheap-to-expensive, because a false loop warps the whole
map and is far worse than a missed one: bag-of-words retrieval with TF-IDF
weighting, fused with proximity in the current pose estimate, then descriptor
matching with RANSAC PnP against mapped landmarks, then a consistency
requirement that a strongly supported match can override.

---

## What this cannot do

Stating these plainly, because they are properties of monocular SLAM rather
than gaps in the implementation.

**Scale is unobservable.** One moving camera cannot recover absolute size: a
2 m room and a 4 m room produce identical images if the camera motion scales
too. Output is in consistent but arbitrary "slam units", labelled as such in
the UI. Benchmarks align to ground truth with a 7-DoF Sim(3) fit, the standard
protocol for monocular results.

**An uploaded video carries no calibration.** Focal length is taken from file
metadata when present, otherwise a 60 degree horizontal field of view is
assumed, and the UI accepts an override and always reports which path was used.

**Degenerate motion is refused, not guessed.** Pure rotation gives no
triangulation baseline and a planar scene makes the essential matrix
ill-conditioned. Initialisation scores a homography against an essential matrix
and waits for better-conditioned motion rather than building an unusable map.

**Loop closure needs an actual revisit.** On a clip that never returns
anywhere, layers L1 to L3 carry the result alone.

---

## Layout

```
slam/           core library: camera, geometry, front end, mapper, optimisers
  core/         feature tracking, initialisation, PnP tracking, mapping
  optim/        bundle adjustment, loop closure, pose graph
  io/           video decode, synthetic ground-truth fixture, exporters
api/            FastAPI service and job manager
web/            React + TypeScript + Vite viewer
tests/          unit, integration, performance
benchmarks/     feasibility measurements and the ablation study
docker/         image and compose file
docs/           design brief and notes
```

## Stack

Python 3.11, OpenCV 4.11 (headless), GTSAM 4.2.2, NumPy, SciPy, FastAPI,
uvicorn. React 18, TypeScript, Vite, Tailwind v4, react-three-fiber.

The pins are load-bearing: gtsam 4.2.2 requires `numpy<2`, which forces
`opencv<=4.11.x`. Bumping either without the other breaks the environment.

## Deployment

The container is the deployment unit and runs unchanged on AWS ECS Express Mode
or Azure Container Apps. Note that **AWS App Runner stopped accepting new
customers on 30 April 2026**; ECS Express Mode is its successor and provisions
a Fargate service, load balancer, TLS and a public HTTPS hostname from a
container image.

Deployment is deliberately the last step, after local verification.
