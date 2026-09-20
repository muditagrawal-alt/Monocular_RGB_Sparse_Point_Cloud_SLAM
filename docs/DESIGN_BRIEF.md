# UI Design Brief: derived from O-HIVE's own design system

Researched 2026-09-18 by extracting O-HIVE's compiled CSS/JS bundle from https://o-hive.ai/.
This is not a guess at their taste; these are their actual tokens and their actual copy.

## Who they are

O-HIVE builds **spatial vision intelligence for physical AI**: Edge VLM inspection and a VLM
robotics chip for humanoids, AR glasses, and autonomous machines. Delaware, US; 2 to 10 employees.

Verbatim positioning:
- **"Capture. Detect. Execute.: with zero latency."**
- "Spatial Vision Intelligence at Edge"
- "Spatial intelligence, delivered."
- "Capture Anywhere, Inspect Instantly"
- "From camera feed to confident action: the new inspection stack"
- "Explore why on-site inference matters for low-latency decisions…"

**Why this matters for our UI:** their entire thesis is *low-latency, on-device 3D understanding
from cost-efficient cameras*. Our deliverable is monocular SLAM on CPU with a hard 10 s budget, the same thesis. So the UI must foreground **latency and real-time factor as first-class metrics**,
and make clear this runs **CPU-only, no GPU**. That is what makes it read as theirs rather than as
a generic upload form.

## Their actual design tokens (extracted)

**Palette, fully achromatic.** Every core color is `oklch(L 0 0)`, i.e. zero chroma:

| Token | Value | Role |
|---|---|---|
| near-black | `oklch(14.5% 0 0)` | primary surface / text |
| dark | `oklch(20.5% 0 0)` | elevated dark surface |
| dark-2 | `oklch(26.9% 0 0)` | borders in dark |
| mid | `oklch(43.9% 0 0)` | secondary text |
| muted | `oklch(70.8% 0 0)` | tertiary text |
| off-white | `oklch(98.5% 0 0)` | light surface |
| white | `oklch(100% 0 0)` | light surface |

Accents used sparingly: blue `oklch(48.8% .243 264.376)`, amber `oklch(76.9% .188 70.08)`.
Hex fallbacks present: `#030213` (deep navy-black primary), `#717182` (muted text),
`#d4183d` (destructive), `#f3f3f5` / `#ececf0` (muted surfaces).

**Typography:** no custom webfont, system stack
(`ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, …`),
with a `--font-mono` for numeric/technical content.

**Stack:** Tailwind CSS v4 (`@theme`, oklch tokens) + shadcn/ui + Radix primitives
(`data-slot` attributes, 30 radix references), Vite build, 77 `.dark` rules → dark mode is
first-class. Subtle `backdrop-filter: blur(8px)` for layering. No heavy gradients or glows.

## Direction for our app

1. **Match their stack exactly**: React + TypeScript + Vite + Tailwind v4 + shadcn/ui + Radix.
   The app should feel native to their codebase, not like a foreign artifact.
2. **Dark-first, achromatic.** Reuse their neutral ramp verbatim. A sparse point cloud reads far
   better on near-black than on white, so this is both on-brand and functionally correct.
3. **Colour carries meaning, never decoration.** The neutral ramp is the entire UI; the accent is
   reserved for the one thing that matters most, the drift before/after comparison and the
   loop-closure markers. Point cloud depth-coloring is the only place a ramp appears.
4. **Mono for all telemetry.** Runtime, real-time factor, frame/keyframe/landmark counts, ATE,
   reprojection error, all in `--font-mono`, tabular, right-aligned. Instrument panel, not
   marketing page.
5. **Latency is the hero metric.** A prominent, honest readout: total runtime vs video duration,
   and the resulting real-time factor. This directly answers requirement #6 and speaks their
   language.
6. **Anti-slop guardrails:** no emoji in UI chrome, no purple/blue SaaS gradient, no glassmorphism
   for its own sake, no rounded-3xl pill everything, no stock hero illustration, no "✨ AI-powered"
   copy. Tone is instrument-panel terse, matching "Capture. Detect. Execute."

## Copy skeleton (mirrors their three-beat cadence)

Their cadence is `Capture. Detect. Execute.` Ours, stated plainly:

> **Upload. Track. Reconstruct.**
> Sparse 3D structure and camera trajectory from a single RGB camera. CPU-only.

Section labels: `INPUT` · `PIPELINE` · `TRAJECTORY` · `STRUCTURE` · `DRIFT` · `TELEMETRY`.
