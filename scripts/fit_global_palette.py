"""Fit the global k=32 palette on ShotQA film-split train stills (CPU, run with nice -n 19). Usage: nice -n 19 uv run python scripts/fit_global_palette.py"""
import time

import numpy as np

from cgt.p3_static_attrs.data import image_path
from cgt.p3b_vlm.data import film_split_ids
from cgt.tokens import palette_global as pg


def main() -> None:
    train, _, _ = film_split_ids()
    ids = sorted(train)
    t0 = time.time()
    x = pg.palette_vectors([str(image_path(i)) for i in ids])
    ok = ~np.isnan(x).any(axis=1)
    print(f"palette vectors: {ok.sum()}/{len(ids)} readable, {time.time() - t0:.0f}s", flush=True)
    pg.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(pg.CACHE / "palette_vectors_train.npz", ids=np.array(ids)[ok], x=x[ok])
    c = pg.fit_centroids(x[ok])
    pg.save(pg.CACHE / "palette_global_k32.npz", c, ids, int(ok.sum()), pg.fit_provenance(x[ok]))
    used = np.bincount(pg.assign(x[ok], c), minlength=pg.K)
    print("cluster sizes:", used.tolist(), f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
