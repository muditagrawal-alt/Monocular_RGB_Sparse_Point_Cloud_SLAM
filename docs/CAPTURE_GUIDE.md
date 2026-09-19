# Recording a video that reconstructs well

A single camera recovers 3D structure from **parallax**: the way nearby things
shift more than distant things when the camera moves. Everything below follows
from that one fact. Where a rule exists because of geometry rather than
implementation, it is marked as such, because no amount of engineering will
work around it.

## The 30-second version

Walk a full circle around an object, keeping it centred, taking about 15
seconds to get all the way round. Record in landscape, in good light, on a
scene with visible surface detail.

## The rules that matter

### 1. Move the camera through space. Do not pivot.

**This is geometry, not a limitation.** Standing still and rotating gives every
point the same apparent motion regardless of distance, so no depth information
exists in the footage at all. The system detects this and declines to build a
map rather than inventing one.

Do: walk sideways past a scene, or around an object.
Do not: stand on one spot and pan, or rotate on your heel.

### 2. Move slowly and smoothly.

Tracks are followed frame to frame, and they survive roughly **1.2 degrees of
rotation per frame**, degrade by 2.4, and break by 4. At 30 fps that is a full
circle in **10 seconds or slower**. Walking pace is right; sweeping the phone
around is not.

Walk heel to toe, hold the phone in both hands, and keep your elbows in. If
your phone has stabilisation, leave it on.

### 3. Point at something with texture.

The tracker locks onto corners and edges. A cluttered bookshelf, a brick wall,
a desk with objects on it, foliage, patterned carpet: all excellent. A plain
painted wall, a clear sky, or a polished floor give it nothing to hold.

### 4. Close a loop if you want to see drift correction.

Drift correction only engages when the camera **returns to somewhere it has
already been**. Finish where you started, framing roughly what you framed at
the beginning. Without a revisit the trajectory and point cloud are still
produced, the drift comparison simply has nothing to show.

### 5. Keep the scene still.

The geometry assumes a rigid world. A few people walking through is survivable;
a crowd, heavy traffic, or leaves thrashing in wind is not.

### 6. Mind the light.

Dim light means longer exposures, which means motion blur, which destroys
tracks. Outdoors in daylight or a well-lit room. Avoid pointing into the sun or
a bright window.

## Settings

| Setting | Use | Why |
|---|---|---|
| Orientation | **Landscape** | Wider field of view holds more of the scene |
| Resolution | 1080p | 4K is downscaled anyway; it only slows the upload |
| Frame rate | 30 or 60 fps | 60 is decimated to 30 automatically |
| Stabilisation | On | Reduces blur; the slight warping it adds costs less than the blur would |
| Duration | **10 seconds** | The assignment's timing case. 60 s is the hard limit |
| Format | .mp4 or .mov | Straight off any phone is fine |

## Three captures worth trying

**The object orbit (best first test).** Put something interesting on a table:
a plant, a stack of books, a bag. Walk a full circle around it at arm's length
plus a metre, keeping it roughly centred, taking about 15 seconds. This gives
strong parallax *and* closes a loop, so you should see loop closure fire and
the before/after drift toggle appear.

**The desk sweep.** Stand at one end of a cluttered desk and walk slowly along
it, keeping the camera pointed at the clutter rather than straight ahead. Good
parallax, no loop, so expect a solid map and no drift correction.

**The doorway pass.** Walk slowly through a doorway into another room, camera
level and facing forwards. This is the hard case, included to show what weak
geometry looks like: moving along your own line of sight produces very little
parallax, so expect fewer points and a thinner map. Useful for understanding
the system's limits.

## Reading the result

| Panel value | What it tells you |
|---|---|
| Realtime factor | Processing time against video length. Below 1.0 meets the target |
| Landmarks | Size of the 3D map. Hundreds is normal for 10 s, thousands is good |
| Mean reprojection error | Internal consistency. Under about 2 px is healthy |
| Tracking losses | How often tracking broke. 0 is what you want |
| Loop closures | Revisits recognised. 0 simply means you did not return anywhere |

## If it does not work

| What you see | Most likely cause |
|---|---|
| "initialisation never succeeded" | Rotation without translation, or a textureless scene |
| Very few landmarks | Forward motion along the view direction, or low texture |
| Tracking losses above zero | Moved too fast, or motion blur from low light |
| Scattered, noisy cloud | Weak parallax; move more sideways relative to the scene |
| No drift toggle | No loop was detected, usually because the path did not revisit anywhere |
