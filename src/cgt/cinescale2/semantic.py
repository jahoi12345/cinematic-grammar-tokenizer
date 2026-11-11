"""Gate G1: semantic same-scene check of CineScale2 vs ShotBench test IMAGES.

Image embeddings only (no ShotBench questions, labels, answers or accuracy). Embedder: the Phase 3A frozen SigLIP
(`google/siglip-base-patch16-224`, revision 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed, `get_image_features` pooler output),
run on the CPU ONLY (never MPS: the GPU belongs to the lock holder), loaded from the local Hugging Face cache (no network).
Cosine similarity of L2-normalised embeddings.

Threshold: calibrated on ShotQA alone (cgt.cinescale2.semantic.calibrate) BEFORE it is applied to ShotBench:
for every ShotQA validation image, the maximum cosine to a seeded random reference set of ShotQA TRAIN images the size of the
ShotBench image set (the film split guarantees DIFFERENT films); the threshold is the 99th percentile of that
different-film maximum, i.e. a nominal 1% false-flag rate for a CineScale2 image unrelated to every ShotBench film."""
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

MODEL = "google/siglip-base-patch16-224"
REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
WEIGHTS_SHA256 = "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"
QUANTILE = 0.99
MEMORY_FLOOR = 25


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n > 0, n, 1)


def max_cos(q: np.ndarray, ref: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """(max cosine, argmax into ref) per row of q; inputs need not be normalised. Empty ref -> (-1, -1)."""
    q, ref = l2n(q), l2n(ref)
    if len(ref) == 0:
        return np.full(len(q), -1.0, np.float32), np.full(len(q), -1, np.int64)
    best = np.empty(len(q), np.float32)
    arg = np.empty(len(q), np.int64)
    for s in range(0, len(q), chunk):
        sim = q[s:s + chunk] @ ref.T
        arg[s:s + chunk] = sim.argmax(axis=1)
        best[s:s + chunk] = sim[np.arange(len(sim)), arg[s:s + chunk]]
    return best, arg


def calibrate(val_feats: np.ndarray, train_feats: np.ndarray, ref_size: int, seed: int = 0,
              quantile: float = QUANTILE) -> dict:
    """Different-film null distribution: max cosine of each val image to `ref_size` seeded random train images (no replacement).
    Returns {threshold, quantile, ref_size, seed, n_val, null_percentiles}."""
    rng = np.random.default_rng(seed)
    ref = train_feats[np.sort(rng.choice(len(train_feats), size=min(ref_size, len(train_feats)), replace=False))]
    m, _ = max_cos(val_feats, ref)
    thr = float(np.quantile(m, quantile))
    return {"threshold": round(thr, 4), "quantile": quantile, "ref_size": len(ref), "seed": seed, "n_val": len(m),
            "null_percentiles": {str(p): round(float(np.percentile(m, p)), 4) for p in (50, 90, 95, 99, 99.9, 100)}}


def same_film_sensitivity(feats: np.ndarray, films: Sequence[str], threshold: float, k: int, seed: int = 0,
                          exclude_pairs: set[tuple[int, int]] | None = None) -> dict:
    """For each image whose film has > k other images: max cosine to k seeded random OTHER images of the same film (pairs in
    `exclude_pairs`, e.g. hash near-duplicates, are not drawn); fraction above `threshold` = how often a film represented by
    k frames in the reference set would be caught."""
    rng = np.random.default_rng(seed)
    f = l2n(feats)
    by: dict[str, list[int]] = {}
    for i, film in enumerate(films):
        by.setdefault(film, []).append(i)
    ex = exclude_pairs or set()
    vals = []
    for i, film in enumerate(films):
        others = [j for j in by[film] if j != i and (min(i, j), max(i, j)) not in ex]
        if len(others) < k:
            continue
        pick = rng.choice(others, size=k, replace=False)
        vals.append(float((f[pick] @ f[i]).max()))
    v = np.array(vals)
    return {"k": k, "n_images": len(v), "fraction_above_threshold": round(float((v > threshold).mean()), 4) if len(v) else None,
            "percentiles": {str(p): round(float(np.percentile(v, p)), 4) for p in (10, 50, 90)} if len(v) else None}


def memory_free_percent() -> int | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return None


def wait_for_memory(floor: int = MEMORY_FLOOR, log: Callable = print, poll_s: int = 30,
                    probe: Callable[[], int | None] = memory_free_percent, sleep: Callable = time.sleep) -> int:
    """Block until free memory >= floor percent (pause, never abort). Returns the number of pauses."""
    pauses = 0
    while True:
        free = probe()
        if free is None or free >= floor:
            return pauses
        if pauses % 10 == 0:
            log(f"[mem] free {free}% < {floor}%: paused")
        pauses += 1
        sleep(poll_s)


def embed_cpu(paths: Sequence[Path], batch: int = 16, threads: int = 4, log: Callable = print,
              mem_floor: int = MEMORY_FLOOR) -> tuple[list[int], np.ndarray, dict]:
    """(indices of readable inputs, float32 [n, 768] raw embeddings, stats). CPU only; checks the weights' sha256 first;
    waits (pauses) whenever free memory drops below `mem_floor` percent, checked before every batch."""
    import hashlib

    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image, UnidentifiedImageError
    from transformers import AutoImageProcessor, SiglipModel

    snap = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    h = hashlib.sha256()
    with open(snap / "model.safetensors", "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    if h.hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError(f"SigLIP weights sha256 {h.hexdigest()} != pinned {WEIGHTS_SHA256}")
    torch.set_num_threads(threads)
    dev = torch.device("cpu")
    proc = AutoImageProcessor.from_pretrained(snap)
    model = SiglipModel.from_pretrained(snap).to(dev).eval()
    ok, feats, pauses = [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(paths), batch):
            pauses += wait_for_memory(mem_floor, log)
            ims, idx = [], []
            for k in range(s, min(s + batch, len(paths))):
                try:
                    with Image.open(paths[k]) as im:
                        ims.append(im.convert("RGB"))
                    idx.append(k)
                except (OSError, UnidentifiedImageError):
                    continue
            if not ims:
                continue
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            out = model.get_image_features(pixel_values=px)
            emb = getattr(out, "pooler_output", out)
            feats.append(emb.float().numpy())
            ok += idx
            if (s // batch) % 50 == 0:
                el = time.time() - t0
                log(f"[embed] {len(ok)}/{len(paths)} {len(ok) / max(el, 1e-9):.1f} img/s")
    el = time.time() - t0
    stats = {"model": MODEL, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256, "device": "cpu", "threads": threads,
             "batch": batch, "n": len(ok), "seconds": round(el, 1), "img_per_s": round(len(ok) / max(el, 1e-9), 2),
             "memory_pauses": pauses}
    return ok, (np.concatenate(feats) if feats else np.zeros((0, 768), np.float32)), stats
