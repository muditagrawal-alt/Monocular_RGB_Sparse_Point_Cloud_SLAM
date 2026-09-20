# Sample clips

Video files are not committed (`*.mp4` is gitignored). Regenerate the synthetic
ones from the repository root:

```python
from slam.io.synthetic import make_sequence
make_sequence(n_frames=300, motion="orbit", loop=True, seed=0).write_video("samples/loop-10s.mp4")
make_sequence(n_frames=120, motion="strafe", seed=0).write_video("samples/strafe-4s.mp4")
```

## Synthetic, with exact ground truth

| File | What it exercises |
|---|---|
| `loop-10s.mp4` | 10 s orbit returning to its start. The requirement-6 timing case, and the clip used for every deployed measurement. |
| `strafe-4s.mp4` | Sideways pass, no revisit. Drift correction correctly stays idle. |

## Real footage, from Wikimedia Commons

| File | Source | Licence |
|---|---|---|
| `real-museum-walk-loops.mp4` | [PAF Aerospace Museum Outdoor Exhibit Walkthrough](https://commons.wikimedia.org/wiki/File:PAF_Aerospace_Museum_Outdoor_Exhibit_Walkthrough.webm), 50 s in | CC BY-SA |
| `real-museum-walk.mp4` | same source, 20 s in | CC BY-SA |
| `real-street-forward.mp4` | [A Trip Down Market Street (1906)](https://commons.wikimedia.org/wiki/File:A_Trip_down_market_street_(1906).webm), 3 min in | Public domain |

The street clip yields a far sparser map because the camera moves straight down
its own optical axis. Forward motion produces little parallax, which is the
weakest geometry for a single camera, even though tracking never fails.

## Demo video credits

`docs/media/demo.mp4` uses "Placid Ambient" by MusicLFiles, CC BY 4.0, from
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Placid_Ambient_by_MusicLFiles.ogg).
