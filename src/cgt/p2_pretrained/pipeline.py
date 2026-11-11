"""Phase 2 pipeline: adds pretrained-model columns to the Phase 1 shot table.
Output: data/processed/<film>_p2.parquet (joined to the Phase 1 table on shot_id) plus <film>_p2.meta.json.

Cache key per stage = sha256(film_id, stage, source code of the stage modules, model versions, shot fingerprint).
A stage is a cache hit only if the output already has its columns for every requested shot AND the stored key matches;
if every requested stage hits, the output file is not rewritten."""
import hashlib
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..p1_shots_color.pipeline import ROOT, film_entry

STAGES = ("blocking", "depth", "audio", "subtitles")
COLUMNS = {
    "blocking": ["people", "faces", "subject_col", "subject_row", "headroom", "facing", "light_dir", "light_asym",
                 "face_bg_ratio"],
    "depth": ["dof_log_ratio", "dof"],
    "audio": ["sound", "sound_db", "sound_top", "sound_top_p", "sound_p_speech", "sound_p_music"],
    "subtitles": ["subtitle_text"],
}
DTYPES = {  # documented dtypes (docs/phase2_interface.md); enforced so an all-null column does not degrade to object
    "shot_id": "int64", "people": "int64", "faces": "int64",
    "subject_col": "float64", "subject_row": "float64", "headroom": "float64", "light_asym": "float64",
    "face_bg_ratio": "float64", "dof_log_ratio": "float64", "sound_db": "float64", "sound_top_p": "float64",
    "sound_p_speech": "float64", "sound_p_music": "float64",
    "facing": "str", "light_dir": "str", "dof": "str", "sound": "str", "sound_top": "str", "subtitle_text": "str",
}


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every known column to its documented dtype (None -> NaN for float, missing stays missing for str)."""
    return df.astype({c: t for c, t in DTYPES.items() if c in df.columns})


MARKERS = {"blocking": ["people", "facing"], "depth": ["dof"], "audio": ["sound"], "subtitles": ["subtitle_text"]}
HF_MODELS = {"depth": "depth-anything/Depth-Anything-V2-Small-hf", "audio": "MIT/ast-finetuned-audioset-10-10-0.4593"}
LOCAL_MODELS = {"blocking": ["models/yolo11n.pt", "models/face_detection_yunet_2023mar.onnx"]}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _source_hash(stage: str) -> str:
    from . import audio, blocking, depth, subtitles
    mod = {"blocking": blocking, "depth": depth, "audio": audio, "subtitles": subtitles}[stage]
    return _sha(inspect.getsource(mod).encode())


def _model_versions(stage: str) -> list[str]:
    out = []
    if stage in HF_MODELS:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        try:  # snapshot directory name = git revision of the cached weights
            out.append(f"{HF_MODELS[stage]}@{Path(snapshot_download(HF_MODELS[stage], local_files_only=True)).name}")
        except LocalEntryNotFoundError:
            out.append(f"{HF_MODELS[stage]}@unresolved")
    for rel in LOCAL_MODELS.get(stage, []):
        p = ROOT / rel
        out.append(f"{rel}@{_sha(p.read_bytes())[:16]}" if p.exists() else f"{rel}@missing")
    return out


def stage_key(film_id: str, stage: str, shots: pd.DataFrame, film: dict) -> str:
    fp = _sha(shots[["shot_id", "start_s", "end_s", "kf_mid"]].to_csv(index=False).encode())
    extra = []
    if stage == "audio":  # depends on the video file itself
        vp = ROOT / film["path"]
        extra.append(f"video:{vp.stat().st_size}" if vp.exists() else "video:missing")
    if stage == "subtitles":
        sp = film.get("subtitles")
        extra.append(_sha((ROOT / sp).read_bytes())[:16] if sp and (ROOT / sp).exists() else "srt:none")
    blob = json.dumps([film_id, stage, _source_hash(stage), _model_versions(stage), fp, extra])
    return _sha(blob.encode())


def run_film(film_id: str, stages=STAGES, limit: int | None = None, max_seconds: float | None = None,
             force: bool = False) -> pd.DataFrame:
    film = film_entry(film_id)
    shots = pd.read_parquet(ROOT / f"data/processed/{film_id}_shots.parquet")
    if max_seconds:
        shots = shots[shots.end_s <= max_seconds]
    if limit:
        shots = shots.head(limit)
    out_path = ROOT / f"data/processed/{film_id}_p2.parquet"
    meta_path = ROOT / f"data/processed/{film_id}_p2.meta.json"
    prev = pd.read_parquet(out_path) if out_path.exists() and not force else pd.DataFrame({"shot_id": []})
    meta = json.loads(meta_path.read_text()) if meta_path.exists() and not force else {}
    ids = set(shots.shot_id)
    stages = [s for s in stages if s != "subtitles" or film.get("subtitles")]
    keys = {s: stage_key(film_id, s, shots, film) for s in stages}

    def hit(s: str) -> bool:
        return (all(c in prev.columns for c in MARKERS[s]) and ids <= set(prev.shot_id) and meta.get(s) == keys[s])

    todo = [s for s in stages if not hit(s)]
    for s in stages:
        print(f"[p2] {s}: {'cache hit' if s not in todo else 'compute'}")
    if not todo:
        return prev[prev.shot_id.isin(ids)].reset_index(drop=True)

    res = {s: {} for s in todo}
    if "blocking" in todo:
        from .blocking import blocking_features
        for r in tqdm(shots.itertuples(), total=len(shots), desc="blocking"):
            res["blocking"][r.shot_id] = blocking_features(cv2.imread(str(ROOT / r.kf_mid)))
    if "depth" in todo:
        from .depth import depth_map, dof_features
        ddir = ROOT / "data/interim" / film_id / "depth"
        ddir.mkdir(parents=True, exist_ok=True)
        for r in tqdm(shots.itertuples(), total=len(shots), desc="depth"):
            img = cv2.imread(str(ROOT / r.kf_mid))
            d = depth_map(img)
            np.save(ddir / f"{r.shot_id:05d}.npy", (d * 255).astype("uint8"))
            res["depth"][r.shot_id] = dof_features(img, d)
    if "audio" in todo:
        from .audio import CACHE_NAME, SR, classify_clip, decode_audio
        audio = decode_audio(str(ROOT / film["path"]), ROOT / f"data/interim/{film_id}/{CACHE_NAME}")
        for r in tqdm(shots.itertuples(), total=len(shots), desc="audio"):
            res["audio"][r.shot_id] = classify_clip(audio[int(r.start_s * SR): int(r.end_s * SR)])
    if "subtitles" in todo:
        from .subtitles import parse_srt, text_for_shot
        cues = parse_srt(str(ROOT / film["subtitles"]))
        for r in shots.itertuples():
            res["subtitles"][r.shot_id] = {"subtitle_text": text_for_shot(cues, r.start_s, r.end_s)}

    new = pd.DataFrame({"shot_id": shots.shot_id.values}).set_index("shot_id")
    for s in todo:
        new = new.join(pd.DataFrame.from_dict(res[s], orient="index"))
    new = new.reset_index()
    kept = [c for c in prev.columns if c not in new.columns]
    if kept:  # columns from stages that were cache hits: carry over for the requested shots
        old = prev[prev.shot_id.isin(ids)][["shot_id"] + kept]
        new = new.merge(old, on="shot_id", how="left")
    order = ["shot_id"] + [c for s in STAGES for c in COLUMNS[s] if c in new.columns]
    new = enforce_dtypes(new[order]).sort_values("shot_id").reset_index(drop=True)  # canonical order => byte-stable
    new.to_parquet(out_path, index=False)
    meta.update({s: keys[s] for s in todo})
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return new
