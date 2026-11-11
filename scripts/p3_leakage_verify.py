"""Second opinion on dHash near-duplicates: pixel-level comparison of EVERY candidate pair with hamming <= 8
(analysis only; ShotBench images are read here ONLY to verify leakage, never for training).
Needs results/phase3/near_pairs.json from scripts/p3_leakage.py.
MAD = mean absolute difference of 32x32 grey thumbnails (0 identical .. 1 black vs white); 'same frame' = MAD < 0.05.
Writes results/phase3/leakage_verified.md and leak_ids_pixel.json (ShotQA ids with any pair at hamming <= 8 and MAD < 0.05)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cgt.p3_static_attrs.data import image_path

ROOT = Path(__file__).resolve().parents[1]
MAD_SAME = 0.05
near = json.loads((ROOT / "results/phase3/near_pairs.json").read_text())


def thumb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32) / 255.0


pairs = []
for qid, lst in near.items():
    a = thumb(image_path(qid))
    for sb_id, d in lst:
        b = thumb(ROOT / "data/external/shotbench/image" / sb_id)
        pairs.append({"qa": qid, "sb": sb_id, "d": d, "mad": float(np.abs(a - b).mean()), "std": float(a.std())})
(ROOT / "results/phase3/near_pairs_verified.json").write_text(json.dumps(pairs))
same_ids = sorted({p["qa"] for p in pairs if p["mad"] < MAD_SAME})
(ROOT / "results/phase3/leak_ids_pixel.json").write_text(json.dumps(same_ids))
leak = set(json.loads((ROOT / "results/phase3/leak_ids.json").read_text()))
lines = ["# dHash near-duplicates verified at pixel level", "",
         (f"All candidate pairs (ShotQA image, ShotBench image) with hamming <= 8: {len(pairs)} pairs over {len(near)} ShotQA images. "
          f"MAD = mean absolute difference of 32x32 grey thumbnails; 'same frame' = MAD < {MAD_SAME}; "
          "'flat' = ShotQA thumbnail std < 0.05 (hash collisions on near-uniform frames)."), "",
         "| hamming of the pair | pairs | same frame | flat collisions | textured, different | median MAD |", "|---|---|---|---|---|---|"]
for d in range(9):
    g = [p for p in pairs if p["d"] == d]
    if g:
        s = sum(p["mad"] < MAD_SAME for p in g)
        f = sum(p["mad"] >= MAD_SAME and p["std"] < 0.05 for p in g)
        lines.append(f"| {d} | {len(g)} | {s} | {f} | {len(g) - s - f} | {np.median([p['mad'] for p in g]):.3f} |")
by_d = {d: len({p["qa"] for p in pairs if p["mad"] < MAD_SAME and p["d"] <= d}) for d in range(9)}
bounds = ", ".join(f"<={d}: {n}" for d, n in by_d.items())
lines += [
    "",
    f"ShotQA images with at least one same-frame pair: {len(same_ids)} (by hamming bound: {bounds}).",
    (f"Excluded by the hamming <= 4 rule alone: {len(leak)}. "
     f"Same-frame images that the <= 4 rule would MISS: {len(set(same_ids) - leak)}."),
    (f"Final exclusion (union of both): {len(leak | set(same_ids))} images, used by p3_extract_features.py and p3_train.py."),
    ("Visual check (10 pairs): MAD ~ 0 pairs are the same still (some are adjacent frames of the same shot); pairs with "
     "MAD >= 0.09 are unrelated dark/flat images that collide in the 64-bit hash."),
]
(ROOT / "results/phase3/leakage_verified.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
