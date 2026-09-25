# Monocular RGB Sparse Point-Cloud SLAM: Implementation Plan

**Project:** O-Hive Take-Home Assignment 2
**Repo:** `muditagrawal-alt/Monocular_RGB_Sparse_Point_Cloud_SLAM`
**Author:** Mudit Agrawal
**Plan date:** 2026-09-18
**Status:** All phases complete. The service is deployed and live at
<https://muditagrawal-alt--monocular-slam.modal.run>. Measured results are in
[README.md](README.md); this document is kept as the original plan of record,
with a note below on where reality diverged from it.

> **Section 9 is superseded.** The plan targeted AWS ECS Express Mode and that
> is what shipped first, at 8.4 s for a 10 second clip. The service later moved
> to Modal when the AWS credits ran out, which turned out to be faster rather
> than a compromise: Modal allocates physical cores where Fargate vCPUs are
> hyperthreads. The AWS sizing, cost and rollout detail below is kept as the
> reasoning that was applied at the time. The current deployment is described
> in the README.

---

## 1. Assignment Restated

| # | Requirement | Where it is handled |
|---|---|---|
| 1 | Single-lens RGB sparse point-cloud SLAM tool | §4 SLAM Core |
| 2 | User can upload a video | §5 API + §6 Frontend |
| 3 | Estimate camera pose + trajectory | §4.3 to 4.5 |
| 4 | Minimize accumulated error and drift | §4.6, **the differentiator** |
| 5 | Visualize sparse 3D point cloud + trajectory | §6 Frontend viewer |
| 6 | 10 s video → ≤10 s processing | §7 Performance budget (**measured**) |
| 7 | Deploy to AWS, public URL | §9 Deployment |

### 1.1 What "sparse" rules out

The 2025 to 26 headline monocular SLAM systems (MASt3R-SLAM, VGGT-SLAM, SLAM3R, EC3R-SLAM) are
**dense**, transformer-based, and need a CUDA GPU to hit real time. They are the wrong tool here
for three independent reasons:

1. The assignment explicitly asks for a **sparse point cloud**.
2. They need GPU inference; a GPU task on AWS costs ~20 to 40× a CPU task and complicates deployment.
3. The 10 s budget is achievable on CPU with a classical pipeline (proven in §7).

So this is a **classical feature-based SLAM** build: ORB/Shi-Tomasi features, KLT tracking,
epipolar initialization, PnP tracking, triangulation, bundle adjustment, loop closure, and
pose-graph optimization. This is also the approach that best demonstrates understanding of the
underlying geometry rather than wrapping someone else's network.

---

## 2. Three Honest Technical Constraints

These are inherent to monocular SLAM. Stating and handling them explicitly is a strength, not a
weakness, because a reviewer who knows the field will look for exactly this.

### 2.1 Scale is unobservable
A single moving camera cannot recover absolute metric scale. A 2 m room and a 4 m room produce
identical images if the camera motion also doubles. **Handling:** fix the initial two-view baseline
to unit length, report the trajectory in explicit "SLAM units", and label it as such in the UI. For
benchmark evaluation, align to ground truth with a 7-DoF Sim(3) Umeyama fit (`evo --align
--correct_scale`), the standard protocol for monocular results.

### 2.2 Intrinsics of an uploaded video are unknown
There is no calibration for an arbitrary upload. **Handling, in priority order:**
1. Read focal length from container/EXIF metadata when present.
2. Otherwise assume a horizontal FOV of ~60°, i.e. `f ≈ 0.9 · max(W, H)`, `(cx, cy) = (W/2, H/2)`.
3. Expose focal length as an optional UI override.
4. Optionally refine `f` as a free variable in the final global bundle adjustment.

The UI will state which path was taken, so results are never silently mis-scaled.

### 2.3 Degenerate motion breaks naive pipelines
Pure rotation gives zero triangulation baseline; a planar scene makes the essential matrix
ill-conditioned. Both are common in casual handheld video. **Handling:** ORB-SLAM-style automatic
model selection, score a homography and an essential matrix in parallel during initialization and
pick by score ratio; refuse to initialize until parallax passes a threshold; gate keyframe
insertion on sufficient translation.

---

## 3. Tech Stack

### 3.1 Verified dependency lockfile

I resolved and installed this set before writing the plan. Two genuine conflicts exist in the
obvious stack and this pinning avoids both:

- `gtsam==4.2.2` requires `numpy<2`, but `opencv-python-headless>=4.12` requires `numpy>=2`.
  → pin OpenCV to **4.11.0.86**, which accepts numpy 1.x.
- `rerun-sdk` requires `numpy>=2`, so it cannot coexist with GTSAM 4.2.2.
  → **drop Rerun**, build the viewer in Three.js (better for a public URL regardless, since Rerun's
  viewer is a separate application rather than an embeddable web scene).

```
python == 3.11            # pinned in Docker; wheels exist for every dep below
numpy == 1.26.4
opencv-python-headless == 4.11.0.86
gtsam == 4.2.2
scipy == 1.14.1
fastapi == 0.115.6
uvicorn[standard] == 0.34.0
python-multipart == 0.0.20
pydantic == 2.10.4
evo == 1.37.1             # dev only: ATE/RPE metrics
```

**Verified:** installed together on Python 3.11.15, `pip check` → "No broken requirements found",
all imports succeed, GTSAM↔NumPy interop confirmed.

> **Note on the local machine:** `python3` on this host resolves to Python **3.14.7** (via
> Homebrew), with pyenv shims and a Homebrew 3.11 also present. Python 3.14 has no wheels for
> scipy/gtsam and falls back to a Fortran source build that fails. All development must therefore
> run inside Docker or an explicit 3.11 venv, never bare `python3`. Docker Desktop is also
> currently **not running** and will need to be started.

### 3.2 Why GTSAM for the optimizer

| Option | Verdict |
|---|---|
| **GTSAM 4.2.2** | **Chosen.** pip-installable with cp311/cp312 manylinux wheels, excellent pose-graph and BA support, robust noise models, iSAM2 for incremental solving, well documented. |
| g2o-python | Rejected, no cp312 wheels (cp311 max), less maintained, smaller community. |
| pyceres | Good solver, but a lower-level building block; more glue code for factor graphs. |
| pycolmap | Mature Ceres BA, but COLMAP's incremental SfM is minutes-per-sequence, far over budget. |
| ORB-SLAM3 + bindings | Rejected, GPLv3, painful build (Pangolin, old OpenCV), flaky third-party bindings, and it would hide the engineering being assessed. |

### 3.3 Application stack

- **Backend:** FastAPI + uvicorn. SLAM runs in a `ProcessPoolExecutor` worker so the async event
  loop is never blocked and the GIL is not contended.
- **Frontend:** React 18 + TypeScript + Vite, `@react-three/fiber` + `@react-three/drei` for the
  3D scene, Tailwind CSS. Point cloud as a single `THREE.Points` buffer geometry; trajectory as a
  `Line`; camera frustums at keyframes.
- **Container:** multi-stage Docker (Node build → Python slim runtime), `linux/amd64`.
- **Quality:** pytest, ruff, mypy, GitHub Actions CI.

---

## 4. SLAM Core Architecture

```
video ──► decode & downscale ──► intrinsics resolution
            │
            ▼
      ┌─────────────────────────── FRONT END (every frame) ──────────────────────────┐
      │  Shi-Tomasi seeding (grid-bucketed)  →  pyramidal KLT optical flow           │
      │  bidirectional consistency check     →  re-seed when track count drops       │
      └──────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
      two-view initialization  (homography vs essential model selection, parallax gate)
            │                   baseline normalized to unit scale
            ▼
      ┌──────────────────────── TRACKING (every frame) ──────────────────────────────┐
      │  solvePnPRansac: tracked 2D observations ↔ existing 3D map points            │
      │  constant-velocity motion prior as initial guess                             │
      └──────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
      keyframe decision  (track-ratio drop │ parallax │ max frame gap)
            │
            ▼
      ┌────────────────────────── MAPPING (keyframes) ───────────────────────────────┐
      │  triangulate new landmarks (parallax + positive-depth + reprojection gates)  │
      │  ORB descriptors computed here only, for loop-closure retrieval             │
      │  outlier landmark culling                                                    │
      └──────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
      ┌──────────────────── BACK END, drift control (§4.6) ─────────────────────────┐
      │  sliding-window local BA  (GTSAM, Huber robust kernel)                       │
      │  loop closure detection   (BoW retrieval + geometric verification)           │
      │  global pose-graph optimization  (GTSAM Pose3 between-factors)               │
      │  optional final global BA if time budget permits                             │
      └──────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
      export: trajectory (TUM + JSON) │ point cloud (PLY + packed binary) │ timings
```

### 4.1 Decode and downscale
`cv2.VideoCapture` in a producer thread, overlapping I/O with compute. Downscale to a target width
(default 640), resolution is the single biggest lever on runtime (§7). Convert to grayscale once
and reuse.

### 4.2 Front-end: KLT, not per-frame descriptor matching

This is the **key performance decision**, and I benchmarked both before committing:

| Front-end | Cost/frame @640×360, 1 core |
|---|---|
| ORB detect+describe + BF-matcher + RANSAC | **27.5 ms** |
| Shi-Tomasi + pyramidal KLT (800 pts) | **4.3 ms** |

KLT is **~6× cheaper** and gives sub-pixel correspondence on consecutive frames, where appearance
change is small. Descriptors are then computed **only at keyframes** (~11 ms each, roughly every
10th frame) where they are genuinely needed, for loop-closure retrieval. This one choice is what
makes the 10 s budget comfortable rather than marginal.

### 4.3 Initialization
Collect frames until parallax is sufficient. Score homography and essential matrix; select by
score ratio; `recoverPose`; triangulate the inlier set; normalize median scene depth (or baseline)
to 1.0. Retry with later frames on failure rather than proceeding with a bad map.

### 4.4 Tracking
Frame-to-**map** PnP (`solvePnPRansac`, iterative refinement) against the existing landmark set, **not** frame-to-frame pose chaining. This matters for drift (§4.6, layer 2).

### 4.5 Keyframes and mapping
Insert a keyframe when tracked-point ratio falls below a threshold, or translation/parallax exceeds
a minimum, or a maximum frame gap is reached. Triangulate new landmarks between the new keyframe
and its co-visible neighbours, applying parallax, positive-depth, and reprojection-error gates.
Cull landmarks that fail observation-count or error checks.

### 4.6 Drift minimization: Requirement #4

Presented as a **layered defense**, since no single mechanism is sufficient:

| Layer | Mechanism | What it fixes |
|---|---|---|
| **L1** | Robust estimation everywhere: RANSAC on E/H/PnP, Huber kernels in every optimization, bidirectional KLT check, landmark culling | Outliers before they poison the map |
| **L2** | Frame-to-map PnP rather than frame-to-frame chaining | Removes pure integration drift, each pose is anchored to the map, not to its predecessor |
| **L3** | Sliding-window local bundle adjustment (GTSAM `GenericProjectionFactorCal3_S2`, oldest keyframes fixed as gauge) | Jointly refines recent poses + structure; suppresses local error growth |
| **L4** | **Loop closure detection + global pose-graph optimization** | Removes *accumulated* drift when a place is revisited |
| **L5** | Optional final global BA if budget permits | Last-mile polish |

**Loop closure detection** (three-stage, cheap→expensive):
1. **Retrieve**: compact global descriptor per keyframe from its ORB descriptors (BoW histogram
   over a small vocabulary trained online via mini-batch k-means); shortlist by similarity.
2. **Verify geometrically**: ORB descriptor matching between query and candidate, then RANSAC PnP
   / essential check; require a minimum inlier count.
3. **Confirm temporally**: require consistency across consecutive keyframes to reject one-off
   perceptual aliasing.

Only surviving candidates become `BetweenFactorPose3` loop constraints.

**Proof this works, already measured.** I built a 60-keyframe synthetic loop with realistic
odometry noise and ran the real GTSAM backend:

```
ATE before optimization:  4.757 m
ATE after  optimization:  0.539 m     →  8.8× drift reduction
solve time:               2.9 ms      (60 poses + 1 loop closure)
```

The mechanism for requirement #4 is therefore validated end-to-end *before* committing to the
architecture, and it is effectively free within the time budget. The final deliverable will report
this same before/after comparison on real sequences, which is far more convincing than asserting
that drift is handled.

### 4.7 Outputs
Trajectory in TUM format (`timestamp tx ty tz qx qy qz qw`) plus JSON for the viewer; point cloud
as PLY plus a packed `Float32Array` for fast web transfer; per-stage timing breakdown; quality
metrics (landmark count, mean reprojection error, loop closures found, tracking-loss events).

---

## 5. Backend API

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | Multipart video upload → `{job_id}`; validates type, size, duration |
| `GET` | `/api/jobs/{id}` | Status + progress + stage timings |
| `GET` | `/api/jobs/{id}/result` | Trajectory + point cloud (packed binary) |
| `GET` | `/api/jobs/{id}/preview` | Optional annotated tracking preview |
| `GET` | `/healthz` | Container health check |

Guards: MIME/extension allowlist, size cap (~100 MB), duration cap (~60 s), ffprobe validation
before decode, per-job timeout, temp-file cleanup, bounded job queue. Progress is polled (simpler
and more robust behind a load balancer than WebSockets for this workload).

---

## 6. Frontend

Single-page flow: **upload → progress → interactive 3D result**.

- Drag-and-drop upload with client-side duration/size validation and a bundled demo video so a
  reviewer can see results in one click without hunting for footage.
- Live progress with per-stage timings and a visible total-runtime figure (evidences requirement 6).
- **3D viewer:** sparse point cloud (`THREE.Points`, depth/height-colored), camera trajectory
  polyline, keyframe frustums, orbit/pan/zoom, grid and axes for scale reference.
- **Drift panel:** toggle between the pre-optimization (odometry-only) and post-optimization
  trajectories in the same scene. This makes requirement #4 *visible* rather than merely claimed, the single highest-value UI element for a reviewer.
- Metrics panel: frames, keyframes, landmarks, loop closures, mean reprojection error, runtime.
- Clear "scale is relative (SLAM units)" annotation, per §2.1.

---

## 7. Performance Budget: Requirement #6

Target: 10 s video (300 frames @ 30 fps) in ≤10 s. Projection from **measured** single-core numbers
at 640×360 (Apple Silicon; AWS x86_64 figures to be re-measured in Phase 3):

| Stage | Measured unit cost | ×300 frames | Subtotal |
|---|---|---|---|
| Decode + downscale | ~1 ms/frame | 300 | 0.3 s |
| KLT tracking | **4.3 ms/frame** | 300 | 1.3 s |
| PnP pose estimation | ~3 ms/frame | 300 | 0.9 s |
| ORB at keyframes | **11.1 ms** | ~30 | 0.3 s |
| Local BA windows | ~50 ms | ~30 | 1.5 s |
| Loop closure + global PGO | ~3 ms measured |, | 0.3 s |
| Export / serialize |, |, | 0.2 s |
| | | **Total** | **≈ 4.8 s** |

**≈2× headroom** against the 10 s target, single-core, before any parallelism.

Levers held in reserve if the AWS instance measures slower:
1. **Adaptive quality**: estimate frame count up front and auto-reduce resolution/feature count
   to stay inside budget (graceful degradation rather than a blown SLA).
2. Overlap decode with compute in a producer thread.
3. Tune `cv2.setNumThreads`; parallelize BA and front-end across processes.
4. Cap keyframe count and local-BA window size.
5. Frame decimation for high-fps input (process 30 fps effective from 60 fps source).

A performance regression test will assert the budget in CI so it cannot silently regress.

---

## 8. Validation Strategy

Accuracy claims need evidence, so this gets its own phase.

1. **Unit tests**: geometry primitives: projection/unprojection round-trip, triangulation against
   closed-form, pose composition/inversion, Sim(3) alignment.
2. **Synthetic integration test**: generated scene with *exact* known camera poses and 3D points.
   Asserts ATE below a tight threshold. This is the fastest, most deterministic correctness signal
   and needs no dataset download.
3. **Real-sequence benchmark**: TUM RGB-D (`fr1/xyz`, `fr1/desk`, `fr2/desk`, `fr3/long_office`,
   which contains a genuine loop) and optionally a KITTI sequence. ATE/RPE via `evo` with 7-DoF
   Sim(3) alignment per §2.1. `fr3/long_office` is the key sequence for demonstrating loop closure.
4. **Ablation table**: the deliverable's headline accuracy evidence:

   | Configuration | ATE (m) |
   |---|---|
   | Odometry only (no BA, no loop closure) | baseline |
   | + local bundle adjustment | ↓ |
   | + loop closure & global PGO | ↓↓ |

5. **Performance test**: asserts the 10 s budget on a 10 s clip.
6. **Robustness cases**: pure rotation, planar scene, motion blur, low texture, tracking loss and
   recovery, very short clip, corrupt upload.

---

## 9. AWS Deployment

> Per your instruction, this phase begins **only after** local verification passes (§10, Gate).

### 9.1 Service choice

**Important finding:** **AWS App Runner stopped accepting new customers on 30 April 2026.** It is
the obvious choice in older tutorials and is no longer available for this project. Its successor is
**Amazon ECS Express Mode**.

| Option | Verdict |
|---|---|
| **ECS Express Mode** | **Chosen.** Provide a container image; it provisions a Fargate service, load balancer, SSL/TLS, auto-scaling, monitoring, and an **AWS-provided HTTPS domain**: satisfying the "publicly accessible URL" requirement with no DNS or certificate work. No extra charge beyond the underlying resources. |
| ECS Fargate (manual) | Fallback. Full control, but ALB + target groups + listeners + certs by hand. |
| EC2 | Cheaper at steady state, but I'd own patching, TLS, and scaling. Not worth it here. |
| Lambda | Rejected, 15 min cap is fine, but the container size, CPU profile, and cold starts fit poorly. |

**Sizing:** 2 vCPU / 4 GB Fargate task to start (~$0.04/vCPU-hr, ~$0.0044/GB-hr). Scale-to-low at
idle to keep the review-period cost to a few dollars.

### 9.2 Pipeline
`docker build --platform linux/amd64` → ECR → ECS Express Mode service → public HTTPS URL.
GitHub Actions on push to `main`: lint → typecheck → test → build → push → deploy.

### 9.3 Pre-deployment checklist
Cost guardrail (budget alarm + task cap) · CloudWatch logs and metrics · `/healthz` wired to the
load balancer · upload size/duration caps enforced server-side · per-job timeout · temp-file
cleanup · no secrets in the image · `linux/amd64` verified · README with the live URL, an
architecture diagram, the benchmark tables, and a 60-second reviewer walkthrough.

---

## 10. Phased Plan

**Local build and verification first; deployment only after the gate.**

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0, Scaffold** | Repo layout, pinned lockfile, Dockerfile, compose, CI, ruff/mypy/pytest config | `docker compose up` serves a stub; CI green |
| **1, SLAM front end** | Decode, KLT tracking, initialization, PnP tracking, keyframes, triangulation | Synthetic sequence tracks end-to-end; trajectory plausible |
| **2, Back end / drift** | Local BA, loop-closure detection, global PGO | **Ablation shows measurable ATE reduction** on a looping sequence |
| **3, Validation** | Unit + synthetic + TUM benchmarks, `evo` metrics, perf test | ATE tables produced; **10 s budget met and asserted in CI** |
| **4, API + viewer** | FastAPI endpoints, React/Three.js viewer, drift toggle, demo clip | Upload → 3D result works in the browser locally |
| **5, GATE** | Full local end-to-end verification | **All tests pass, budget met, results reviewed by you** ✋ |
| **6, AWS** | ECR image, ECS Express Mode service, CI/CD, cost guardrails, README | Public HTTPS URL live and stable; cost alarm set |

### Proposed repo layout
```
slam/            core library, decoder, frontend, initializer, tracker, mapper,
                 optimizer (local BA), loop_closure, pose_graph, exporters, config
api/             FastAPI app, job manager, schemas
web/             React + TS + Vite + react-three-fiber
tests/           unit, synthetic integration, benchmark, performance
benchmarks/      dataset fetch scripts, ablation runner, results tables
docker/          Dockerfile, compose, entrypoint
infra/           ECR + ECS Express Mode deployment scripts
docs/            architecture, design decisions, benchmark results
```

---

## 11. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| AWS CPU slower than local benchmark | Medium | 2× headroom already; adaptive quality; re-measure in Phase 3 on x86_64 |
| Tracking loss on difficult upload | Medium | Relocalization via loop-closure retrieval; multi-segment trajectory; honest UI reporting |
| Unknown intrinsics distort geometry | High | §2.2 fallback chain + UI override + optional BA refinement |
| Degenerate motion (pure rotation / planar) | Medium | H-vs-E model selection, parallax gates, explicit user feedback |
| No loop in the reviewer's test video | High | Loop closure can't fire without a revisit, so L1-L3 must carry the result alone, and the bundled demo clip **will** contain a loop to showcase L4 |
| Dependency conflicts at deploy time | Low | Already resolved and verified (§3.1); Docker pins Python 3.11 |
| `python3` = 3.14 on this host breaks builds | High | All work inside Docker or an explicit 3.11 venv; documented in README |
| AWS cost overrun during review | Low | Budget alarm, task cap, scale-to-low at idle |

---

## 12. Open Questions for You

1. **AWS credentials**: `aws sts get-caller-identity` currently fails with `NoCredentials`. Which
   account/region should I target, and will you provide credentials or run the deploy step yourself?
   (Not needed until Phase 6.)
2. **Region**: recommend `ap-south-1` (Mumbai) for latency if O-Hive's team is in India;
   `us-east-1` is the cheapest and has the widest feature availability.
3. **Deploy target**: confirm **AWS** (you mentioned Azure as an alternative). The plan assumes
   AWS per the assignment text; Azure Container Apps would be the equivalent choice there.
4. **Demo video**: I'll source a small CC-licensed clip containing a loop. Do you have preferred
   footage, or shall I record/source one?
5. **Scope of L5**: is a final global BA worth the extra runtime, or should I keep the headroom?
   My recommendation: implement it behind a flag, off by default.

---

## 13. Why This Plan Should Score Well

- **Every requirement is mapped** to a specific component, with the hard one (#4) given a layered,
  measurable treatment rather than a single hand-waved mechanism.
- **Claims are pre-validated, not asserted.** The 6× front-end speedup, the 8.8× drift reduction,
  and the dependency lockfile were all measured *before* this plan was written, so the
  architecture rests on evidence, and two conflicts and one discontinued AWS service were caught
  before they could cost implementation time.
- **The hard parts of monocular SLAM are named honestly** (§2) and handled, rather than quietly
  ignored, which is what distinguishes a real SLAM implementation from a tutorial port.
- **Drift reduction is made visible** in the UI via the before/after toggle and in the docs via the
  ablation table.
- **Sensible engineering throughout:** pinned reproducible builds, CI, tests at three levels, cost
  guardrails, and graceful degradation instead of a blown time budget.


---

## 14. Post-Build Note: Where This Plan Met Reality

Kept deliberately, because the differences are the interesting part.

**Held up.** The KLT-over-descriptor-matching decision was the right call and
delivered the predicted speedup. GTSAM was the right optimiser and never became
a bottleneck. The dependency pinning survived contact with the container. The
layered approach to drift (L1 to L4) is what the ablation actually measures.

**Cost more than expected.** Getting loop closure to fire took five distinct
fixes, none of which were visible from the plan: descriptors had to be computed
at the tracked points rather than re-detected, cross-check matching replaced the
ratio test, retrieval needed TF-IDF plus a spatial signal, the verification
RANSAC threshold had to be loosened because a loop constraint spans accumulated
drift, and the verification budget had to be spent in order of closest
trajectory approach. The plan treated "loop closure detection" as one line item.

**The estimate was optimistic in the wrong place.** The plan projected 4.8 s for
a 300-frame clip. The real figure is 8.4 s in a 4 vCPU container. Two reasons:
the projection assumed local BA would cost ~50 ms per window when it costs more
as the map grows, and it ignored that loop detection is ~35% of total runtime.
The budget still holds, but with less margin than predicted.

**Missed entirely.** The plan never considered how many cores the pipeline
needs. A 2 vCPU task straddles the deadline while 4 vCPU has margin, which is
the single most important deployment fact and only appeared once the container
was actually run under a CPU limit. Nor did it anticipate that the synthetic
fixture itself would twice produce results that looked like algorithm failures:
a scene layout that put structure at near-zero depth for two of the three
motions, and texture too repetitive for any appearance-based loop closure to
work. Both were fixture defects, and both cost real debugging time.

**Deliberately deferred.** Global BA (L5) is implemented as configuration but
left off by default, per the recommendation in section 12. Real-dataset
benchmarking against TUM RGB-D was scoped in section 8 and has not been run;
the accuracy evidence is currently synthetic ground truth only, which is exact
but easier than real imagery.
