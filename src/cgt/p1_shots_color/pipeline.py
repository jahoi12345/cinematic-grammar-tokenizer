"""Phase 1 pipeline: video -> shot table (Parquet) with color, key and length columns."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from cgt.pipeline import registry

from . import color, keyframes, plots
from .detection import detect_shots

ROOT = Path(__file__).resolve().parents[3]


def film_entry(film_id: str) -> dict:
    """The film's entry in the merged registry.
    A films.yaml film returns its films.yaml entry unchanged; an overlay film keeps `registry: overlay`. An unknown id raises StopIteration."""
    films = registry.load_films(ROOT)
    if film_id not in films:
        raise StopIteration(f"film '{film_id}' is not registered in configs/films.yaml or configs/films.local.yaml")
    e = films[film_id]
    return e if registry.is_overlay(e) else {k: v for k, v in e.items() if k != "registry"}


def figures_dir(film: dict) -> Path | None:
    """Where the barcode and shot-length figures go: results/figures for a public-domain films.yaml film, gitignored data/interim/<id>/figures for
    an overlay film (local only whatever its flag), none for an owned films.yaml film."""
    if registry.is_overlay(film):
        return ROOT / "data/interim" / film["id"] / "figures"
    return ROOT / "results/figures" if film.get("public_domain") else None


DEFAULT_CUT_THRESHOLD = 27.0


def resolve_threshold(film_id: str, threshold: float | None = None) -> float:
    """Explicit value wins, else the film's `cut_threshold` from the merged registry (configs/films.yaml + overlay), else the library default."""
    if threshold is not None:
        return float(threshold)
    return float(film_entry(film_id).get("cut_threshold", DEFAULT_CUT_THRESHOLD))


def cuts_cache_path(film_id: str, threshold: float) -> Path:
    return ROOT / "data/interim" / film_id / f"cuts_t{threshold:g}.json"


def run_film(film_id: str, threshold: float | None = None, k: int = 5, force: bool = False) -> pd.DataFrame:
    film = film_entry(film_id)
    threshold = resolve_threshold(film_id, threshold)
    video = ROOT / film["path"]
    interim = ROOT / "data/interim" / film_id
    (interim / "keyframes").mkdir(parents=True, exist_ok=True)
    cuts_file = cuts_cache_path(film_id, threshold)

    if cuts_file.exists() and not force:
        shots = [tuple(x) for x in json.loads(cuts_file.read_text())]
    else:
        shots = detect_shots(str(video), threshold)
        cuts_file.write_text(json.dumps(shots))

    rows = []
    for i, (s, e) in enumerate(tqdm(shots, desc=film_id)):
        times = keyframes.pick_keyframe_times(s, e)
        frames = keyframes.read_frames(str(video), list(times))
        good = [f for f in frames if f is not None]
        if not good:
            continue
        paths = []
        for name, f in zip(("start", "mid", "end"), frames):
            if f is None:
                paths.append(None)
                continue
            p = interim / "keyframes" / f"{i:05d}_{name}.jpg"
            cv2.imwrite(str(p), f)
            paths.append(str(p.relative_to(ROOT)))
        stats = [color.color_stats(f) for f in good]
        mid = frames[1] if frames[1] is not None else good[0]
        rows.append({
            "film_id": film_id, "shot_id": i, "start_s": s, "end_s": e, "len_s": e - s,
            "kf_start": paths[0], "kf_mid": paths[1], "kf_end": paths[2],
            "palette_lab": color.palette_lab(mid, k).ravel().tolist(),
            "palette_shares": color.palette_shares(mid, k).tolist(),
            **{key: float(np.mean([st[key] for st in stats])) for key in stats[0]},
            "key": color.lighting_key(mid),
        })

    df = pd.DataFrame(rows)
    out = ROOT / "data/processed"
    df.to_parquet(out / f"{film_id}_shots.parquet", index=False)
    fig = figures_dir(film)
    if fig is not None:
        if registry.is_overlay(film):
            fig.mkdir(parents=True, exist_ok=True)
        plots.color_barcode(df, str(fig / f"{film_id}_barcode.png"))
        plots.shot_length_plot(df, str(fig / f"{film_id}_shot_length.png"))
    return df
