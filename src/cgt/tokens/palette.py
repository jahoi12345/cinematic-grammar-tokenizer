"""Provisional per-film palette ids: k-means (k=32, seed 0) over the shots' 5-colour LAB palettes. Ids are FILM-SPECIFIC (not comparable across films)
and ordered by centroid lightness (p00 darkest). Not the Phase 5 palette token; no claim beyond 'shots with similar tonal palettes share an id'."""
import numpy as np
from sklearn.cluster import KMeans

K = 32
SEED = 0


def fit_palette_ids(palettes: list, k: int = K, seed: int = SEED) -> np.ndarray:
    """palettes: per shot a flat (5*3,) LAB vector (colours ordered by pixel share, as Phase 1 writes it). Returns int ids in 0..k-1, p00 = darkest centroid."""
    x = np.asarray([np.asarray(p, dtype=float).ravel() for p in palettes])
    if x.ndim != 2 or x.shape[1] % 3:
        raise ValueError("palettes must be flat LAB triples")
    k = min(k, len(np.unique(x, axis=0)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x)
    lightness = km.cluster_centers_[:, 0::3].mean(axis=1)
    rank = np.empty(k, dtype=int)
    rank[np.argsort(lightness, kind="stable")] = np.arange(k)
    return rank[km.labels_]
