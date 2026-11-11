"""Frame-embedding baseline features (docs/phase4_interface.md section 5): frozen SigLIP-base-patch16-224 (the Phase 3A model and feature definition:
`SiglipModel.get_image_features`, unnormalised) of 8 uniformly sampled frames per clip, pooled by mean, signed half difference and mean absolute consecutive difference.
CPU/GPU agnostic (device argument), no training. Never reads ShotBench or CameraBench."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

MODEL = "google/siglip-base-patch16-224"
N_FRAMES = 8
PARAMS = {"model": MODEL, "n_frames": N_FRAMES, "feature": "get_image_features unnormalised", "dtype": "float16"}
_cache: dict = {}


def cache_key() -> str:
    from huggingface_hub import snapshot_download
    rev = Path(snapshot_download(MODEL, local_files_only=True)).name
    return hashlib.sha256(Path(__file__).read_bytes() + json.dumps(PARAMS | {"revision": rev}, sort_keys=True).encode()).hexdigest()[:16]


def sample_frames(path: str | Path, n: int = N_FRAMES) -> list[Image.Image]:
    """n frames at the centres of n equal time bins of the clip, as RGB PIL images (full resolution; the processor squashes them to 224 x 224)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = [int((i + 0.5) * total / n) for i in range(n)]
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i, total - 1))
        ok, fr = cap.read()
        if not ok:
            raise ValueError(f"{path}: frame {i} unreadable")
        out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def embed_clip(path: str | Path, device: str = "cpu") -> np.ndarray:
    """(8, 768) float16 embeddings of the sampled frames."""
    if "model" not in _cache:
        from transformers import AutoImageProcessor, SiglipModel
        _cache["proc"] = AutoImageProcessor.from_pretrained(MODEL)
        _cache["model"] = SiglipModel.from_pretrained(MODEL).eval().to(device)
    px = _cache["proc"](images=sample_frames(path), return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        e = _cache["model"].get_image_features(pixel_values=px)
    e = e.pooler_output if hasattr(e, "pooler_output") else e
    return e.float().cpu().numpy().astype(np.float16)


def pooled(e: np.ndarray) -> np.ndarray:
    """Mean (768), signed half difference mean(second half) - mean(first half) (768) and mean absolute consecutive difference (768): 2,304 values."""
    e = e.astype(np.float32)
    h = len(e) // 2
    return np.concatenate([e.mean(0), e[h:].mean(0) - e[:h].mean(0), np.abs(np.diff(e, axis=0)).mean(0)])


def extract_clip(path: str | Path, cache_dir: Path, key: str, device: str = "cpu") -> Path:
    dst = Path(cache_dir) / f"{Path(path).name}.{key}.npy"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".tmp_" + dst.name)
    with open(tmp, "wb") as fh:
        np.save(fh, embed_clip(path, device))
    tmp.replace(dst)
    return dst
