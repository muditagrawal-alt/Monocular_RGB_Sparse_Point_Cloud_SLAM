<div align="center">

<img src="docs/media/banner.png" alt="Monocular RGB Sparse Point-Cloud SLAM" width="100%">

<br>

[![Live Demo](https://img.shields.io/badge/Live_Demo-Open_App-d08a2c?style=flat-square&logo=amazonaws&logoColor=white)](https://mo-a4ee3c36d4e04673b2fbfe9eac1c73ae.ecs.us-east-1.on.aws)
[![Realtime](https://img.shields.io/badge/10s_clip-8.4s_on_CPU-2ea043?style=flat-square)](#-measured-processing-time-and-test-environment)
[![Tests](https://img.shields.io/badge/tests-78_passing-2ea043?style=flat-square&logo=pytest&logoColor=white)](#-setup)
[![No GPU](https://img.shields.io/badge/GPU-not_required-8a8a8a?style=flat-square&logo=nvidia&logoColor=white)](#-architecture-and-major-technical-decisions)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](requirements.txt)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](requirements.txt)
[![GTSAM](https://img.shields.io/badge/GTSAM-4.2.2-1f6feb?style=flat-square)](requirements.txt)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](api/)
[![React](https://img.shields.io/badge/React-18_+_TS-61DAFB?style=flat-square&logo=react&logoColor=black)](web/)
[![AWS](https://img.shields.io/badge/AWS-ECS_Fargate-FF9900?style=flat-square&logo=amazonecs&logoColor=white)](infra/)

**Recover a camera trajectory and a sparse 3D point cloud from a single-lens RGB video.**<br>
No depth sensor. No GPS. No neural network. No GPU.

[**Open the live app**](https://mo-a4ee3c36d4e04673b2fbfe9eac1c73ae.ecs.us-east-1.on.aws) · [Watch the demo](docs/media/demo.mp4) · [Architecture](#-architecture-and-major-technical-decisions) · [Benchmarks](#-accuracy)

<sub>O-Hive take-home, Assignment 2 · **Mudit Agrawal**</sub>

</div>

---

<div align="center">

<img src="docs/media/reconstruction.gif" alt="3D reconstruction" width="88%">

<sub>A 10 second clip, reconstructed in **8.4 seconds on CPU**. The ring is the recovered camera path<br>with a frustum at every keyframe; the cloud inside it is the 3,853 landmarks triangulated along the way.</sub>

</div>

---

## ⚡ What this does

You give it an ordinary video from an ordinary camera. It works out two things
at once, from nothing but the flat 2D frames:

- **Where the camera went**: its full path through 3D space
- **What the scene looks like in 3D**: a sparse cloud of triangulated points

That is circular by nature: knowing where the camera is requires knowing where
the scene is, and vice versa. Solving both together is what SLAM means.

> The whole pipeline is **classical multi-view geometry**. There is no model
> file, no inference, and no GPU anywhere in it. That is what makes a 10 second
> clip finish in under 9 seconds on a CPU container.

| | |
|---|---|
| <img src="docs/media/pipeline.gif" alt="Pipeline" width="460"> | **Upload → track → reconstruct.**<br><br>Corners are tracked frame to frame with optical flow, each camera pose is solved against the map built so far, and bundle adjustment refines poses and structure together.<br><br>Wherever the camera revisits a place, loop closure recognises it and redistributes the accumulated drift. |

---

## 📊 Measured processing time and test environment

The requirement: a 10-second video must process in **10 seconds or less**.

<div align="center">

| | Deployed result |
|---|---|
| Input | 300 frames, 640×360, 30 fps, **10.0 s** |
| **Processing time** | **8.40 to 8.52 s** |
| **Realtime factor** | **0.84×** |
| Consistency | 3 consecutive runs, all within budget |

</div>

**Test environment**: AWS ECS Express Mode on Fargate, `linux/amd64`,
**4 vCPU / 8 GB**, `us-east-1`. Sized to fit the account's default 8 vCPU
Fargate quota, which needs headroom for a rolling deployment. Python 3.11, OpenCV 4.11 headless, GTSAM 4.2.2.
Deployed settings: 448 px processing width, 560 features, loop-verification cap
100. **No GPU is present or used.**

<details>
<summary><b>Other clips, same deployment</b></summary>

<br>

| Clip | Length | Processing | Realtime factor |
|---|---:|---:|---:|
| Synthetic orbit (closes a loop) | 10.0 s | 8.45 s | 0.84× |
| Real museum walkthrough | 10.0 s | 7.93 s | 0.79× |
| Real street footage, forward motion | 10.0 s | 5.89 s | 0.59× |

</details>

<details>
<summary><b>Where the time goes, and how it got there</b></summary>

<br>

| Stage | Time | Share |
|---|---:|---:|
| Front end (corner seeding, KLT tracking) | ~2.2 s | 38% |
| Pose estimation (PnP against the map) | ~1.5 s | 26% |
| Bundle adjustment | ~1.2 s | 21% |
| Loop detection and pose graph | ~0.9 s | 15% |

The first deployed build took **12.3 s** and missed the budget. Four changes,
each kept only because it was measured:

| Change | Effect |
|---|---|
| Loop detection sized from the time *remaining*, after the map is built | 3135 ms → ~900 ms |
| Bundle-adjustment window 8 → 6, iterations 8 → 5 | 39% cheaper **and more accurate** |
| Processing width 448 | front end −35% |
| Honest per-verification cost estimate | stopped the stage overrunning |

</details>

Asserted in CI by `tests/performance/test_budget.py`, so a regression past
realtime fails the build.

---

## 🎯 Accuracy

Absolute trajectory error against exact synthetic ground truth, after the 7-DoF
Sim(3) alignment monocular evaluation requires. **Five seeds per configuration.**

<table>
<tr><th>Orbit, 300 frames, camera returns to its start</th><th>Strafe, 240 frames, never revisits</th></tr>
<tr><td>

| Configuration | ATE |
|---|---:|
| Frame-to-map PnP only (L1+L2) | 3.0657 |
| + bundle adjustment (L3) | 0.7343 |
| + loop closure & pose graph (L4) | **0.6416** |
| | **4.78× better** |

</td><td>

| Configuration | ATE |
|---|---:|
| Frame-to-map PnP only (L1+L2) | 1.3277 |
| + bundle adjustment (L3) | **0.0987** |
| + loop closure & pose graph (L4) | 0.0987 |
| | **13.45× better** |

</td></tr>
</table>

Read together: bundle adjustment does most of the work on any sequence, and
loop closure adds more **only where the camera genuinely returns somewhere**.
On the non-looping clip it correctly finds nothing and changes nothing, which is
exactly what you want from a component whose failure mode is warping the whole
map.

### Real footage

**45 arbitrary clips** from Wikimedia Commons: walking tours, cathedral
interiors, cycling, drone flights, driving, hiking, tunnels and palaces.

<div align="center">

| Reconstructed | Median landmarks | Median reprojection | Median realtime factor |
|:---:|:---:|:---:|:---:|
| **42 of 44 valid clips** | 1,052 | 1.02 px | 0.44× |

</div>

The two failures are both severely low-texture: an art installation offering 99
trackable corners and degraded 1906 archival film offering 125, against the 700
to 800 a healthy clip provides. Declining to build a map from those is correct
behaviour, not a gap.

> **Results are reproducible.** Four separate processes produce identical
> output. This was not true earlier in development, and fixing it is described
> under [AI Usage](#-ai-usage).

---

## 🚀 Setup

<table>
<tr><th>Prerequisite</th><th>Version</th><th>Check</th></tr>
<tr><td>Docker</td><td>any recent</td><td><code>docker --version</code></td></tr>
<tr><td>Python <sub>(local dev only)</sub></td><td><b>3.11</b> exactly</td><td><code>python3.11 --version</code></td></tr>
<tr><td>Node <sub>(local dev only)</sub></td><td>20+</td><td><code>node --version</code></td></tr>
</table>

> **Python 3.11 specifically.** GTSAM publishes no wheels for 3.13+, and the
> pinned dependency set is verified against 3.11.

### Run with Docker

```bash
docker compose -f docker/compose.yaml up --build
# open http://localhost:8000
```

### Run locally

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt

.venv/bin/python -m uvicorn api.main:app --reload    # API  → :8000
cd web && npm install && npm run dev                  # UI   → :5173
```

### Tests and checks

```bash
.venv/bin/pytest tests/ -q                            # 78 tests
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

## ☁️ Deployment

The container is the deployment unit and runs unchanged on AWS or Azure.

```bash
./infra/bootstrap.sh              # one-time: ECR, IAM roles, log group, service
./infra/deploy.sh                 # build, push, roll out, verify
./infra/deploy.sh --config-only   # retune without rebuilding
```

**Target: AWS ECS Express Mode**, which provisions a Fargate service, load
balancer, TLS and a public HTTPS hostname from a container image.

> ⚠️ **AWS App Runner stopped accepting new customers on 30 April 2026**, so the
> obvious choice from older documentation is unavailable. Express Mode is its
> successor.

<details>
<summary><b>Runtime tuning</b> (the right values depend on how fast the host is)</summary>

<br>

| Variable | Deployed | Purpose |
|---|---|---|
| `SLAM_TARGET_WIDTH` | 448 | Processing width. The main runtime lever |
| `SLAM_MAX_FEATURES` | 560 | Corner budget per frame |
| `SLAM_LOOP_VERIFICATIONS` | 100 | Cap on geometric verifications |
| `SLAM_VERIFY_COST_S` | 0.030 | Assumed cost per verification, for budgeting |
| `SLAM_RTF_TARGET` | 0.80 | Target realtime factor, leaving margin |
| `SLAM_LOOP_CLOSURE` | on | Loop detection and pose-graph correction |
| `SLAM_ADAPTIVE_BUDGET` | off | Mid-flight degradation; see limitations |

Every applied value is reported back in each result, so a run is never
ambiguous about what produced it.

</details>

<details>
<summary><b>Two deployment gotchas worth recording</b></summary>

<br>

**Buildx attestation breaks the Fargate pull.** The default build produces an
OCI image index with attestation manifests that Fargate cannot pull. Build with
`--provenance=false --sbom=false`.

**The service URL is not in the API response.** `serviceUrl` came back null.
The AWS-provided hostname lives on the load balancer's listener rule, and
`deploy.sh` reads it from there.

</details>

---

## 🏗 Architecture and major technical decisions

```mermaid
%%{init: {'theme':'base','themeVariables':{
'primaryColor':'#1f1f1f','primaryTextColor':'#ededed','primaryBorderColor':'#4a4a4a',
'lineColor':'#8a8a8a','secondaryColor':'#262626','tertiaryColor':'#191919',
'clusterBkg':'#161616','clusterBorder':'#3a3a3a','fontSize':'14px'}}}%%
flowchart TD
    vid["Video or image folder"] --> dec["Decode and downscale<br/>threaded producer"]
    dec --> intr["Resolve intrinsics<br/>metadata, else 60° FOV"]

    intr --> fe

    subgraph fe["Front end · every frame"]
        seed["Shi-Tomasi corners<br/>seeded per grid cell"]
        klt["Pyramidal KLT optical flow"]
        fb["Forward-backward check<br/>drop drifting tracks"]
        seed --> klt --> fb
    end

    fb --> init{"Map<br/>initialised?"}

    init -->|"no"| two

    subgraph two["Two-view initialisation"]
        score["Score homography<br/>against essential matrix"]
        ess["Essential matrix<br/>general scene"]
        hom["Homography<br/>planar fallback"]
        par["Parallax gate<br/>refuse pure rotation"]
        score -->|"general"| ess
        score -->|"planar, after retries"| hom
        ess --> par
        hom --> par
    end

    par -->|"normalise depth to 1.0"| map[("Map<br/>keyframes + landmarks")]

    init -->|"yes"| track["solvePnPRansac against the map<br/>constant-velocity prior"]
    track --> kf{"Insert<br/>keyframe?"}
    kf -->|"no"| fe
    kf -->|"yes"| tri["Triangulate new landmarks<br/>parallax, depth and reprojection gates"]
    tri --> cull["Cull outliers"]
    cull --> map

    map --> back

    subgraph back["Back end · drift control"]
        ba["L3 · sliding-window<br/>bundle adjustment<br/>GTSAM, Huber kernel"]
        loop["L4 · loop detection<br/>bag of words + TF-IDF,<br/>then RANSAC PnP"]
        pgo["L4 · pose-graph optimisation<br/>robust loop constraints"]
        ba --> loop --> pgo
    end

    back --> out["Trajectory (TUM, JSON)<br/>Point cloud (PLY, packed binary)<br/>Telemetry"]
```

### Deployment shape

```mermaid
%%{init: {'theme':'base','themeVariables':{
'primaryColor':'#1f1f1f','primaryTextColor':'#ededed','primaryBorderColor':'#4a4a4a',
'lineColor':'#8a8a8a','secondaryColor':'#262626','tertiaryColor':'#191919',
'clusterBkg':'#161616','clusterBorder':'#3a3a3a','fontSize':'14px'}}}%%
flowchart TD
    spa["React SPA<br/>three.js point-cloud viewer"]

    spa -->|"HTTPS · multipart upload"| alb

    subgraph aws["AWS ECS Express Mode · Fargate 4 vCPU / 8 GB · us-east-1"]
        direction TB
        alb["Application Load Balancer<br/>TLS · /healthz · autoscaling"]
        api["FastAPI<br/>upload validation · job registry"]
        worker["SLAM worker<br/>separate OS process"]
        core["SLAM core · pipeline.run()<br/>Python · OpenCV · GTSAM · no GPU"]

        alb --> api
        api -->|"submit to process pool"| worker
        worker -->|"runs"| core
    end

    core -.->|"trajectory · point cloud · telemetry"| api
    worker -.->|"progress, by atomically replacing a JSON file"| api
    api -.->|"client polls until done"| spa
```

### 1. Classical geometry, not a learned model

The 2025-26 headline monocular systems, MASt3R-SLAM, VGGT-SLAM, SLAM3R, are
dense, transformer-based, and need a CUDA GPU to run at real time. Wrong tool
here for three independent reasons: the assignment asks for a **sparse** cloud,
a GPU task costs far more and complicates deployment, and the 10-second budget
is achievable on CPU classically.

So: **no machine-learning model at all.** Shi-Tomasi corners, Lucas-Kanade
optical flow, essential and homography matrices, PnP with RANSAC, linear
triangulation, bundle adjustment, pose-graph optimisation. Even the
loop-closure vocabulary is built by k-means over the video's own descriptors at
runtime, so nothing pretrained ships with it.

### 2. Track with optical flow, describe only at keyframes

The single largest performance decision, benchmarked *before* committing
(`benchmarks/feasibility/`), single core at 640×360:

| Front end | Cost per frame |
|---|---:|
| ORB detect + describe + BFMatcher + RANSAC | 27.5 ms |
| **Shi-Tomasi + pyramidal KLT** | **4.3 ms** |

Roughly **6× cheaper**. Descriptors are computed only at keyframes, where they
are genuinely needed for loop-closure retrieval.

### 3. Drift control is layered

No single mechanism is sufficient:

| Layer | Mechanism | What it addresses |
|:---:|---|---|
| **L1** | RANSAC everywhere, Huber kernels, forward-backward track check, landmark culling | Outliers, before they corrupt the map |
| **L2** | Frame-to-map PnP rather than frame-to-frame chaining | Pose error compounding through integration |
| **L3** | Sliding-window bundle adjustment | Local error growth in poses and structure |
| **L4** | Loop detection + global pose-graph optimisation | Error already accumulated over the sequence |

Loop detection runs cheap-to-expensive, because a false loop warps the whole
map and is far worse than a missed one: bag-of-words retrieval with TF-IDF
weighting, fused by reciprocal rank with proximity in the current estimate,
then descriptor matching with RANSAC PnP against mapped landmarks, then a
consistency requirement that strong geometric support can override.

### 4. Latency is handled where it does no harm

An earlier design degraded the map mid-stream when it projected an overrun.
That made accuracy depend on machine load. The same clip scored **0.76 ATE on
an idle machine and 3.75 when the guard fired under contention**, so it is now
**off by default**.

Instead, loop detection sizes its own budget from the time actually remaining.
It runs *after* the frame loop, so the map is already built and correct, and the
remaining budget is known exactly. Full detection quality when there is time;
detection is given up rather than the deadline when there is not.

### 5. Process isolation for the worker

SLAM is CPU-bound for seconds at a time, so it runs in a separate **process**,
not a thread: a thread would contend on the GIL with the event loop and make
progress polling unresponsive. Progress is reported by atomically replacing a
small JSON file rather than over a multiprocessing queue, because a `Manager`
queue needs a broker process that re-imports `__main__`, which is fragile under
the `spawn` start method used on macOS and in containers.

---

## 📦 Libraries, frameworks and external components

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

> **The pins are load-bearing.** GTSAM 4.2.2 requires `numpy<2`, which forces
> `opencv<=4.11.x`. Bumping either alone breaks the environment. The same
> constraint rules out `rerun-sdk` (needs `numpy>=2`), which is why the viewer
> is custom rather than an off-the-shelf visualiser.

### Frontend

React 18 · TypeScript 5.7 · Vite 6 · Tailwind CSS v4 · react-three-fiber and
drei over three.js · Phosphor icons. Built as a static bundle and served by the
same container, so there is one origin and no CORS in production.

### Development

pytest · ruff · mypy · httpx · evo (trajectory metrics) · Playwright · Docker ·
GitHub Actions.

### Pretrained models

**None.** No neural network, no weights file, no downloaded checkpoint, no
inference of any kind. Stated explicitly because it is unusual for a 2026
vision project, and it is the reason the service needs no GPU.

### External data

Wikimedia Commons video (CC BY-SA and public domain) for the wild benchmark;
each clip's source and licence is recorded in [`samples/README.md`](samples/README.md).
The TUM RGB-D benchmark is supported but its dataset is not redistributed here.
Demo music: *"Placid Ambient"* by MusicLFiles, CC BY 4.0.

---

## 🔍 Known limitations and what I would improve with more time

### Inherent to monocular SLAM

| Limitation | Why | How it could be lifted |
|---|---|---|
| **Scale is unobservable** | One moving camera cannot recover absolute size; a real room and a perfect dollhouse produce identical images | Show one object of known size, or add stereo / depth / IMU / GPS |
| **No calibration on an upload** | Focal length comes from metadata when present, else a 60° FOV assumption with a UI override | Estimate focal length as a free variable in a final global bundle adjustment |
| **Degenerate motion is refused** | Pure rotation gives no baseline, so no depth information exists | Not possible. This is geometry, not implementation |
| **The world is assumed static** | Crowds and traffic violate the rigid-scene assumption | Mask moving regions with segmentation, reintroduces a GPU dependency |

### What I would fix first

- **Loop closure is under-sensitive.** Fires on roughly 1 in 5 synthetic
  sequences and 8 of 42 real clips. The bias is deliberate, since a false loop
  warps the entire map, but retrieval is the weak link. *Fix:* a proper DBoW2-style
  vocabulary tree trained offline instead of online k-means.
- **Tracking robustness on real footage.** Only 20 of 42 real clips track with
  no loss at all. Losses trigger re-initialisation, so the trajectory becomes
  segmented. *Fix:* relocalisation against the existing map rather than
  starting fresh.
- **Accuracy validation is synthetic.** The TUM RGB-D harness is built and
  tested end to end against a synthetic sequence in exact TUM layout, but the
  dataset host was unreachable from the development network, so no
  real-benchmark ATE numbers exist. *Fix:* run `benchmarks/tum.py` from a
  network that can reach it.
- **Sparse means sparse.** About 1 point per 400 pixels: enough to localise,
  not enough to look like a scene. *Fix:* a densification pass once poses are
  known, as an optional high-quality mode.
- **Forward motion is weak geometry.** Moving along the optical axis gives
  little parallax, so those clips yield thinner maps. Inherent, but a wider
  baseline keyframe policy would help.
- **Fargate placement varies.** The same image measured 6.5 s on one host and
  10.0 s on another. *Fix:* calibrate per-task at startup with a short
  synthetic benchmark.

---

## 🤖 AI Usage

### Tools

**Claude Code** (Anthropic) was used throughout as a development assistant, for
implementation, benchmark tooling, and documentation. Everything it produced
was checked against measurement before being kept.

### How it was used

The method was to treat the assistant as a fast implementer and a tireless
measurer, and to keep engineering judgement, direction and acceptance criteria
on my side:

- **I set the constraints and the order of work**: build and verify locally
  before touching any cloud; deployment as a gated step rather than something
  interleaved; benchmark against real footage, not only synthetic fixtures;
  polish last, after the substance was proven.
- **I required claims to be reproducible before accepting them.** This is what
  exposed the most important defect in the project, below.
- **I chose the deployment target** after reviewing the CPU evidence, and
  directed the latency work when the deployed service missed the budget.
- The assistant did the implementation, ran the sweeps, and drafted the docs.

### Recommendations adopted, after they survived measurement

| Decision | Why it was kept |
|---|---|
| Classical geometry over a learned dense model | The assignment asks for a *sparse* cloud, and a CPU pipeline hits the 10 s budget where MASt3R-SLAM or VGGT-SLAM would need a GPU |
| KLT optical flow every frame, descriptors only at keyframes | 4.3 ms/frame against 27.5 ms for ORB matching. This one decision is why the budget is met at all |
| GTSAM for bundle adjustment and pose graph | pip-installable with the right wheels, and never became a bottleneck |
| Layered drift control (L1-L4) | The ablation shows each layer earning its place: 4.78× on a looping sequence, 13.45× on a straight one |
| Homography initialisation as a *gated fallback* | Recovered drone and other planar footage that was previously refused outright |

### Recommendations rejected or modified

The more informative half. Each was rejected on evidence, not taste:

| What was proposed | What happened to it |
|---|---|
| **Adaptive mid-flight quality degradation** to protect the deadline | **Rejected, disabled by default.** It made accuracy depend on machine load: 0.76 ATE on an idle machine against 3.75 when it fired under contention. Latency is now handled by sizing loop detection from the time remaining *after* the map is built |
| **Capping bundle-adjustment landmarks at 250** | **Rejected.** Degraded the orbit sequence to 4.9054 ATE, far worse than the time it saved |
| **Halving the forward-backward track check** for 25% off the front end | **Rejected.** Looked free on one seed, then cost 17% accuracy across five. A reminder that one measurement is not a result |
| **Bundle adjustment every third keyframe** for the deployed build | **Rejected.** 88% worse on the orbit sequence. A shorter BA window was found instead, faster *and* more accurate |
| An early ablation reporting **"5.10× drift improvement"** | **Rejected as unreproducible.** Re-running the exact commit gave a different figure. Cause: unseeded RANSAC plus the adaptive guard reacting to load. Both fixed; every number here was re-measured from scratch afterwards |

> Insisting that results reproduce before accepting them is what exposed the
> reproducibility defect. Until it was fixed, the headline accuracy figures were
> not trustworthy, and no amount of further implementation would have made them
> so. Four separate processes now produce identical output, which is the
> precondition for any measurement here meaning anything.

---

## 📁 Repository layout

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
docs/                 implementation plan, design brief, capture guide, media
```

### Further reading

| Document | What is in it |
|---|---|
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | The original plan, and a note on where it met reality |
| [docs/CAPTURE_GUIDE.md](docs/CAPTURE_GUIDE.md) | How to record a video that reconstructs well |
| [docs/DESIGN_BRIEF.md](docs/DESIGN_BRIEF.md) | How the UI design was derived |
| [samples/README.md](samples/README.md) | Sample clips, their sources and licences |

---

<div align="center">

**[Open the live app →](https://mo-a4ee3c36d4e04673b2fbfe9eac1c73ae.ecs.us-east-1.on.aws)**

<sub>Built by Mudit Agrawal · O-Hive take-home, Assignment 2</sub>

</div>
