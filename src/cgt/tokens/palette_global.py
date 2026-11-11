"""Global k=32 palette: fitted on the LAB palettes of the ShotQA film-split TRAIN stills only (never NOTLD, the owned film, ShotBench, gold or dev).
Palette vector of a frame = Phase 1's `palette_lab` (5 LAB colours, most common first, 160 px working size) flattened to 15 numbers. Centroids are ShotQA-derived (CC-BY-NC-ND data):
they stay under data/cache/tokens/ and are never committed. CPU only."""
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache/tokens"
K = 32
SEED = 0


def lightness_rank(centers: np.ndarray) -> np.ndarray:
    """rank[c] = position of cluster c when clusters are ordered by the mean L of their five colours (g00 = darkest)."""
    light = centers[:, 0::3].mean(axis=1)
    rank = np.empty(len(centers), dtype=int)
    rank[np.argsort(light, kind="stable")] = np.arange(len(centers))
    return rank


def fit_centroids(x: np.ndarray, k: int = K, seed: int = SEED) -> np.ndarray:
    """k-means (n_init=10, seed) on (N, 15) palette vectors; returns centroids sorted by lightness, shape (k, 15)."""
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(x)
    order = np.argsort(lightness_rank(km.cluster_centers_))
    return km.cluster_centers_[order]


def assign(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid (Euclidean, ties to the lowest id) for each (N, 15) palette vector."""
    x = np.asarray(x, dtype=float)
    d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return d.argmin(axis=1)


def _one(path: str):
    import cv2

    from cgt.p1_shots_color import color
    try:
        os.nice(19)
    except OSError:
        pass
    im = cv2.imread(path)
    return None if im is None else color.palette_lab(im).ravel()


def palette_vectors(paths: list[str], workers: int = 4) -> np.ndarray:
    """Palette vector per image path (unreadable images get NaN rows)."""
    with Pool(workers) as p:
        out = p.map(_one, paths, chunksize=64)
    return np.stack([np.full(15, np.nan) if o is None else o for o in out])


def ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def save(path: Path, centroids: np.ndarray, ids: list[str], n_used: int, provenance: dict | None = None) -> None:
    import sklearn
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=centroids)
    path.with_suffix(".json").write_text(json.dumps({
        "k": len(centroids), "seed": SEED, "n_images": n_used, "train_ids_sha256": ids_digest(ids), "sklearn": sklearn.__version__,
        "source": "ShotQA film-split train stills (cgt.p3b_vlm.data.film_split_ids, leakage ids excluded)",
        "centroid_sha256": centroid_digest(centroids), "fit_provenance": provenance or "not recorded",
        "reproducibility": "ids are defined by this frozen centroid array; a refit with seed 0 is not guaranteed to be bit-identical across thread counts or dtypes"}, indent=1))


def centroid_digest(c: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(c, dtype=np.float64).tobytes()).hexdigest()


class CentroidMismatch(RuntimeError):
    """The centroid array does not match the digest recorded next to it (the frozen artifact was replaced or refitted)."""


def load(path: Path = CACHE / "palette_global_k32.npz") -> np.ndarray:
    """The FROZEN centroid artifact is the definition of the global palette ids. A refit with the same seed is not guaranteed to reproduce it bit for bit
    (it depends on the thread configuration and input dtype), so the array is verified against the digest recorded in its sidecar before use."""
    c = np.load(path)["centroids"]
    meta = path.with_suffix(".json")
    if meta.exists():
        want = json.loads(meta.read_text()).get("centroid_sha256")
        if want and centroid_digest(c) != want:
            raise CentroidMismatch(f"{path.name}: centroid digest {centroid_digest(c)[:12]} != recorded {want[:12]}")
    return c


def fit_provenance(x: np.ndarray) -> dict:
    """Environment of a fit that can change the result: input dtype, library versions and the BLAS/OpenMP thread pools actually in use."""
    import sklearn
    try:
        from threadpoolctl import threadpool_info
        pools = [{"api": p.get("user_api"), "lib": p.get("internal_api"), "threads": p.get("num_threads")} for p in threadpool_info()]
    except ImportError:
        pools = "threadpoolctl not installed"
    return {"input_dtype": str(x.dtype), "n_vectors": len(x), "numpy": np.__version__, "sklearn": sklearn.__version__, "threadpools": pools,
            "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}}
