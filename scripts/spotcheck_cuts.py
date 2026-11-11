"""Contact sheets of random cuts (frame before | frame after) for hand-checking detection.
Usage: uv run python scripts/spotcheck_cuts.py --film night_of_the_living_dead --threshold 12 --n 50
Output: data/interim/<film>/spotcheck_t<thr>/sheet_*.jpg (never committed)."""
import argparse
import json

import cv2
import numpy as np

from cgt.p1_shots_color.keyframes import read_frames
from cgt.p1_shots_color.pipeline import ROOT, film_entry

ap = argparse.ArgumentParser()
ap.add_argument("--film", required=True); ap.add_argument("--threshold", type=float, default=27.0)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
video = str(ROOT / film_entry(a.film)["path"])
shots = json.loads((ROOT / f"data/interim/{a.film}/cuts_t{a.threshold:g}.json").read_text())
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
rng = np.random.default_rng(a.seed)
idx = sorted(rng.choice(np.arange(1, len(shots)), size=min(a.n, len(shots) - 1), replace=False))
out = ROOT / f"data/interim/{a.film}/spotcheck_t{a.threshold:g}"; out.mkdir(parents=True, exist_ok=True)
W, H = 320, 240
tiles = []
for i in idx:
    t = shots[i][0]
    before, after = read_frames(video, [t - 2 / fps, t + 1 / fps])
    pair = []
    for f in (before, after):
        f = np.zeros((H, W, 3), np.uint8) if f is None else cv2.resize(f, (W, H))
        pair.append(f)
    tile = np.hstack(pair)
    cv2.putText(tile, f"#{i} t={t:.1f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    tiles.append(tile)
per = 10
for s in range(0, len(tiles), per):
    chunk = tiles[s:s + per]
    rows = [np.hstack(chunk[r:r + 2] + [np.zeros_like(chunk[0])] * (2 - len(chunk[r:r + 2]))) for r in range(0, len(chunk), 2)]
    cv2.imwrite(str(out / f"sheet_{s // per}.jpg"), np.vstack(rows))
print(out)
