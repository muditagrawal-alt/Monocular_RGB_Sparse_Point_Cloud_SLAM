"""Run the pipeline over a batch of real videos and report how it behaves.

The synthetic tests measure accuracy against known truth. This measures
something different and equally important: how often the system copes with
arbitrary real footage it was never tuned on, and how it fails when it does
not. There is no ground truth here, so the output is a behaviour profile
(tracking survival, map size, reprojection error, runtime), not an error score.

Clips are sourced from Wikimedia Commons, which is openly licensed and does not
gate automated access the way the stock-video sites do.

    python benchmarks/wild.py --collect 30      # find and cache clips
    python benchmarks/wild.py --run             # run whatever is cached
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slam.config import SlamConfig  # noqa: E402
from slam.pipeline import SlamPipeline  # noqa: E402

# Wikimedia asks automated clients to identify themselves and to back off when
# throttled. Ignoring either gets you 429s within a handful of requests.
UA = {"User-Agent": "monocular-slam-benchmark/0.1 (research evaluation; contact via repo)"}
REQUEST_SPACING_S = 1.5
API = "https://commons.wikimedia.org/w/api.php"

QUERIES = [
    "walking tour city", "museum walkthrough", "drone flight over", "driving street view",
    "hiking trail", "bicycle ride", "castle interior", "cathedral interior", "garden walk",
    "train window view", "corridor walk", "aerial city", "park walk", "forest path",
    "old town walk", "campus tour", "harbour walk", "market walk",
]


@dataclass
class Outcome:
    name: str
    query: str
    success: bool
    reason: str = ""
    frames: int = 0
    keyframes: int = 0
    landmarks: int = 0
    loop_closures: int = 0
    tracking_losses: int = 0
    reinitializations: int = 0
    reproj_px: float = 0.0
    realtime_factor: float = 0.0
    within_budget: bool = False
    tracked_fraction: float = 0.0


_last_request = 0.0


def api_get(params: dict, retries: int = 5) -> dict:
    """Query the Commons API, pacing requests and backing off on throttling."""
    global _last_request
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        gap = time.time() - _last_request
        if gap < REQUEST_SPACING_S:
            time.sleep(REQUEST_SPACING_S - gap)
        _last_request = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            time.sleep(4 * (attempt + 1))      # linear backoff on throttling
    return {}


def collect(limit: int, cache: Path, max_mb: int = 250) -> list[tuple[str, str]]:
    """Find candidate videos and download a 10 s excerpt of each."""
    cache.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    for query in QUERIES:
        if len(found) >= limit * 3:
            break
        try:
            data = api_get({"action": "query", "list": "search",
                            "srsearch": f"{query} filetype:video",
                            "srnamespace": 6, "srlimit": 6, "format": "json"})
            for item in data.get("query", {}).get("search", []):
                found.setdefault(item["title"], query)
        except Exception as exc:
            print(f"  search failed for {query!r}: {str(exc)[:60]}")

    saved: list[tuple[str, str]] = []
    for title, query in sorted(found.items()):
        if len(saved) >= limit:
            break
        slug = "".join(c if c.isalnum() else "-" for c in title[5:])[:48].strip("-")
        out = cache / f"{slug}.mp4"
        if out.exists():
            saved.append((out.name, query))
            continue
        try:
            info = api_get({"action": "query", "titles": title, "prop": "imageinfo",
                            "iiprop": "url|size", "format": "json"})
            page = next(iter(info["query"]["pages"].values()))
            image = (page.get("imageinfo") or [{}])[0]
            url, size = image.get("url"), image.get("size", 0)
            if not url or size > max_mb * 1_000_000:
                continue

            raw = cache / f"{slug}.src"
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=240) as response, open(raw, "wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)

            # Take a 10 s excerpt from a few seconds in, past any title card,
            # and normalise to 720p mp4 so every clip enters on equal terms.
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "5", "-t", "10",
                 "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                 "-an", "-vf", "scale='min(1280,iw)':-2", str(out), "-y"],
                check=True, timeout=300)
            raw.unlink(missing_ok=True)
            if out.exists() and out.stat().st_size > 50_000:
                saved.append((out.name, query))
                print(f"  [{len(saved):2d}] {out.name[:58]}")
        except Exception as exc:
            print(f"  skipped {title[5:40]}: {str(exc)[:60]}")
            continue
    return saved


def run(cache: Path, index: dict[str, str]) -> list[Outcome]:
    outcomes: list[Outcome] = []
    clips = sorted(cache.glob("*.mp4"))
    for i, clip in enumerate(clips, 1):
        query = index.get(clip.name, "")
        try:
            result = SlamPipeline(SlamConfig()).run(clip)
        except Exception as exc:
            outcomes.append(Outcome(clip.stem, query, False, f"crashed: {exc}"[:90]))
            print(f"  [{i:2d}/{len(clips)}] {clip.stem[:42]:44s} CRASH")
            continue

        if not result.success:
            outcomes.append(Outcome(clip.stem, query, False, result.reason[:90]))
            print(f"  [{i:2d}/{len(clips)}] {clip.stem[:42]:44s} no map")
            continue

        tracked = sum(1 for t in result.trajectory if t.tracking_ok)
        s = result.summary()
        outcomes.append(Outcome(
            clip.stem, query, True, "ok", s["frames"], s["keyframes"], s["landmarks"],
            s["loop_closures"], s["tracking_losses"], s["reinitializations"],
            s["mean_reproj_error_px"], s["realtime_factor"], s["within_budget"],
            tracked / max(result.n_frames_processed, 1)))
        print(f"  [{i:2d}/{len(clips)}] {clip.stem[:42]:44s} "
              f"kf={s['keyframes']:3d} pts={s['landmarks']:5d} "
              f"loops={s['loop_closures']} rtf={s['realtime_factor']:.2f}")
    return outcomes


def report(outcomes: list[Outcome], out: Path) -> None:
    ok = [o for o in outcomes if o.success]
    print(f"\n{'=' * 78}\nReconstructed {len(ok)} of {len(outcomes)} clips\n")
    if ok:
        import statistics as st
        print(f"  median landmarks        {st.median(o.landmarks for o in ok):.0f}")
        print(f"  median keyframes        {st.median(o.keyframes for o in ok):.0f}")
        print(f"  median reprojection     {st.median(o.reproj_px for o in ok):.2f} px")
        print(f"  median realtime factor  {st.median(o.realtime_factor for o in ok):.2f}")
        print(f"  within budget           {sum(o.within_budget for o in ok)}/{len(ok)}")
        print(f"  clean tracking (no loss){sum(o.tracking_losses == 0 for o in ok):>3}/{len(ok)}")
        print(f"  found a loop closure    {sum(o.loop_closures > 0 for o in ok):>3}/{len(ok)}")
    failed = [o for o in outcomes if not o.success]
    if failed:
        print(f"\n  did not reconstruct ({len(failed)}):")
        for o in failed:
            print(f"    {o.name[:44]:46s} {o.reason[:52]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(o) for o in outcomes], indent=2))
    print(f"\nWritten to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect", type=int, metavar="N", help="download N clips")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path("benchmarks/datasets/wild"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/wild.json"))
    args = parser.parse_args()

    index_path = args.cache / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    if args.collect:
        saved = collect(args.collect, args.cache)
        index.update(dict(saved))
        index_path.write_text(json.dumps(index, indent=2))
        print(f"\ncached {len(saved)} clips in {args.cache}")

    if args.run:
        report(run(args.cache, index), args.out)


if __name__ == "__main__":
    main()
