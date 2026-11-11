"""Frozen SigLIP image embeddings with a resumable shard cache (docs/phase3_interface.md section 3).
Generic: takes image paths. The ShotQA-only guard is applied by the entry-point scripts (they pass `allowed_root`);
the cache records its provenance (model, revision, preprocessing, source paths) and `verify_cache` audits it."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

MODEL = "google/siglip-base-patch16-224"
SHARD = 1024


def slug(model: str) -> str:
    return model.replace("/", "__")


def list_cached(cache_dir: Path) -> set[str]:
    done = set()
    for f in sorted(Path(cache_dir).glob("shard_*.npz")):
        with np.load(f) as z:
            done |= set(z["ids"].tolist())
    return done | list_unreadable(cache_dir)


def list_unreadable(cache_dir: Path) -> set[str]:
    """Images that failed to decode once; remembered so a re-run does not retry (or reload the model) for them."""
    f = Path(cache_dir) / "unreadable.txt"
    return set(f.read_text().split("\n")) - {""} if f.exists() else set()


class CacheProvenanceError(RuntimeError):
    """The feature cache was not produced by the expected model/preprocessing, or holds images from outside the allowed root."""


COMPARE_KEYS = ("model_name", "model_revision", "processor_sha256", "feature", "dtype")


def _inside(path: Path, root: Path) -> bool:
    p, r = Path(path).resolve(), Path(root).resolve()
    return p == r or r in p.parents


def current_provenance(model_name: str = MODEL, proc=None) -> dict:
    """What a cache produced NOW would claim: model id + git revision of the cached weights, SHA-256 of the image-processor
    configuration (resize, mean/std, resample...), the feature definition and storage dtype (no model weights are loaded)."""
    import transformers
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoImageProcessor
    proc = proc or AutoImageProcessor.from_pretrained(model_name)
    try:
        rev = Path(snapshot_download(model_name, local_files_only=True)).name
    except LocalEntryNotFoundError:
        rev = "unresolved"
    return {"model_name": model_name, "model_revision": rev,
            "processor_sha256": hashlib.sha256(json.dumps(proc.to_dict(), sort_keys=True, default=str).encode()).hexdigest(),
            "feature": "SiglipModel.get_image_features pooler_output, unnormalised", "dtype": "float16",
            "transformers_version (informational)": transformers.__version__}


def _check_provenance(cache_dir: Path, expected: dict) -> None:
    f = Path(cache_dir) / "provenance.json"
    if not f.exists():
        raise CacheProvenanceError(f"{cache_dir} has shards but no provenance.json (legacy cache): rebuild it")
    stored = json.loads(f.read_text())
    bad = {k: (stored.get(k), expected[k]) for k in COMPARE_KEYS if stored.get(k) != expected[k]}
    if bad:
        raise CacheProvenanceError(f"cache provenance mismatch (stored, expected): {bad}")


def verify_cache(cache_dir: Path, model_name: str = MODEL, allowed_root: Path | None = None) -> dict:
    """Prove a cache hit is valid: provenance.json matches the CURRENT model/revision/preprocessing, and (if `allowed_root`)
    every cached image was read from inside it. Returns {provenance, n_images, source_roots}. Raises CacheProvenanceError."""
    cache_dir = Path(cache_dir)
    shards = sorted(cache_dir.glob("shard_*.npz"))
    if not shards:
        raise CacheProvenanceError(f"{cache_dir} holds no shards")
    _check_provenance(cache_dir, current_provenance(model_name))
    n, roots = 0, set()
    for f in shards:
        with np.load(f) as z:
            if "src" not in z.files:
                raise CacheProvenanceError(f"{f.name} lacks source paths (legacy shard): rebuild the cache")
            for src in z["src"].tolist():
                if allowed_root is not None and not _inside(Path(src), allowed_root):
                    raise CacheProvenanceError(f"{f.name}: cached image outside {allowed_root}: {src}")
                roots.add(str(Path(src).parent))
            n += len(z["ids"])
    return {"provenance": json.loads((cache_dir / "provenance.json").read_text()), "n_images": n,
            "source_roots": sorted(roots)}


def load_features(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """All cached (ids, feats float16 N x D), shard order."""
    ids, feats = [], []
    for f in sorted(Path(cache_dir).glob("shard_*.npz")):
        with np.load(f) as z:
            ids.append(z["ids"])
            feats.append(z["feats"])
    return np.concatenate(ids), np.concatenate(feats)


class _Imgs(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return i, Image.open(self.paths[i]).convert("RGB")
        except (OSError, UnidentifiedImageError):
            return i, None


class _Collate:
    """Module-level (picklable under macOS spawn): decode results -> processor pixel tensor."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, items):
        ok = [(i, im) for i, im in items if im is not None]
        bad = [i for i, im in items if im is None]
        if not ok:
            return [], None, bad
        px = self.proc(images=[im for _, im in ok], return_tensors="pt")["pixel_values"]
        return [i for i, _ in ok], px, bad


def embed_images(paths: list[Path], cache_dir: Path, model_name: str = MODEL, batch: int = 64,
                 workers: int = 4, log=print, allowed_root: Path | None = None) -> dict:
    """Embed every path whose basename is not cached yet; append shards. Returns stats (cache hit => n_new == 0).
    `allowed_root`: every requested path must resolve inside it (ValueError otherwise). A cache that already holds shards is
    only reused if its provenance matches the current model/revision/preprocessing (CacheProvenanceError otherwise);
    the check does not load the model weights, so a valid cache hit stays cheap."""
    from transformers import AutoImageProcessor, SiglipModel
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if allowed_root is not None:
        outside = [str(p) for p in paths if not _inside(Path(p), allowed_root)]
        if outside:
            raise ValueError(f"{len(outside)} paths outside {allowed_root}, e.g. {outside[0]}")
    proc = AutoImageProcessor.from_pretrained(model_name)
    prov = current_provenance(model_name, proc)
    if list(cache_dir.glob("shard_*.npz")):
        _check_provenance(cache_dir, prov)
    else:
        (cache_dir / "provenance.json").write_text(json.dumps(prov, indent=1, sort_keys=True))
    done = list_cached(cache_dir)
    todo = [p for p in paths if Path(p).name not in done]
    stats = {"requested": len(paths), "cached": len(paths) - len(todo), "n_new": 0, "skipped_unreadable": 0,
             "seconds": 0.0, "img_per_s": None}
    if not todo:
        return stats
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SiglipModel.from_pretrained(model_name).to(dev).eval()

    dl = DataLoader(_Imgs(todo), batch_size=batch, num_workers=workers, collate_fn=_Collate(proc))
    shard_idx = len(list(cache_dir.glob("shard_*.npz")))
    buf_ids, buf_src, buf_feats, bad_names = [], [], [], []
    t0 = time.time()

    def flush():
        nonlocal shard_idx, buf_ids, buf_src, buf_feats
        if not buf_ids:
            return
        tmp = cache_dir / f".tmp_shard_{shard_idx:05d}.npz"
        np.savez(tmp, ids=np.array(buf_ids), src=np.array(buf_src), feats=np.concatenate(buf_feats).astype(np.float16))
        tmp.rename(cache_dir / f"shard_{shard_idx:05d}.npz")
        shard_idx += 1
        buf_ids, buf_src, buf_feats = [], [], []

    with torch.no_grad():
        for idx, px, bad in dl:
            stats["skipped_unreadable"] += len(bad)
            bad_names += [Path(todo[i]).name for i in bad]
            if px is None:
                continue
            out = model.get_image_features(pixel_values=px.to(dev))
            emb = getattr(out, "pooler_output", out)  # transformers 5.x returns a ModelOutput; SigLIP embedding = pooler_output
            f = emb.float().cpu().numpy()
            buf_ids += [Path(todo[i]).name for i in idx]
            buf_src += [str(Path(todo[i]).resolve()) for i in idx]
            buf_feats.append(f)
            stats["n_new"] += len(idx)
            if len(buf_ids) >= SHARD:
                flush()
                el = time.time() - t0
                log(f"[feat] {stats['n_new']}/{len(todo)}  {stats['n_new'] / el:.1f} img/s")
        flush()
    if bad_names:
        (cache_dir / "unreadable.txt").write_text("\n".join(sorted(list_unreadable(cache_dir) | set(bad_names))))
    stats["seconds"] = time.time() - t0
    stats["img_per_s"] = stats["n_new"] / max(stats["seconds"], 1e-9)
    return stats
