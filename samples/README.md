# Sample clips

Not committed (`*.mp4` is gitignored); regenerate the synthetic ones with
`benchmarks/make_tum_fixture.py` or the snippet in the project README.

## Synthetic, with exact ground truth

| File | What it exercises |
|---|---|
| `loop-10s.mp4` | 10 s orbit returning to its start. Triggers loop closure, so the before/after drift toggle appears. The requirement-6 timing case. |
| `strafe-4s.mp4` | Sideways pass, no revisit. Drift correction correctly stays idle. |

## Real footage, from Wikimedia Commons

| File | Source | Licence |
|---|---|---|
| `real-museum-walk-loops.mp4` | [PAF Aerospace Museum Outdoor Exhibit Walkthrough](https://commons.wikimedia.org/wiki/File:PAF_Aerospace_Museum_Outdoor_Exhibit_Walkthrough.webm) (50 s in, 10 s) | CC BY-SA |
| `real-museum-walk.mp4` | same source, 20 s in | CC BY-SA |
| `real-street-forward.mp4` | [A Trip Down Market Street (1906)](https://commons.wikimedia.org/wiki/File:A_Trip_down_market_street_(1906).webm) (3 min in, 10 s) | Public domain |

Measured on these clips:

| Clip | Keyframes | Landmarks | Loops | Tracking losses | Reprojection |
|---|---:|---:|---:|---:|---:|
| museum @50s | 30 | 562 | 4 | 0 | 1.37 px |
| museum @20s | 28 | 889 | 0 | 0 | 1.05 px |
| market street | 12 | 468 | 0 | 0 | 0.99 px |

The street clip yields far fewer keyframes because the camera moves straight
down its own optical axis. Forward motion produces little parallax, which is
the weakest geometry for a single camera, and the map is correspondingly
sparser even though tracking never fails.
