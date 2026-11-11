"""descriptive GeoCalib FoV probe of the ShotQA lens classes on TRAIN only (docs/plans/geocalib_lens_probe.md). CPU only.
  run:    PYTHONPATH=src nice -n 19 <venv>/bin/python -I scripts/geocalib_lens_probe.py run [--per-class K]
  report: PYTHONPATH=src nice -n 19 <venv>/bin/python -I scripts/geocalib_lens_probe.py report
(-I ignores PYTHONPATH, so the script adds src/ and the GeoCalib clone to sys.path itself.)
Per-image rows go to gitignored data/cache/geocalib_probe/; only aggregates are written to results/geocalib_lens_probe/."""
import argparse
import json
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from cgt.error_analysis import core as E
from cgt.geocalib_probe import core as G
from cgt.p3b_vlm import guards
from cgt.stepa_4b.core import shotqa_film_of
from cgt.stepc import config as SC
from cgt.worktrees import main_checkout

CODE = ROOT / "data/external/geocalib_code/GeoCalib"
WEIGHTS = main_checkout() / "data/external/geocalib/geocalib-pinhole.tar"
WEIGHTS_SIZE = 116_074_121
CACHE = ROOT / "data/cache/geocalib_probe"
ROWS = CACHE / "fov.jsonl"
OUT = ROOT / "results/geocalib_lens_probe"
PAIRS = [("Wide", "Medium"), ("Medium", "Long Lens"), ("Ultra Wide / Fisheye", "Wide"), ("Ultra Wide / Fisheye", "Long Lens"),
         ("Ultra Wide / Fisheye", "Medium"), ("Wide", "Long Lens")]


def pair_name(a: str, b: str) -> str:
    return f"{G.SHORT[a]}_vs_{G.SHORT[b]}"


def population() -> dict[str, list[str]]:
    if guards.sha256_file(SC.TRAIN_JSONL) != SC.TRAIN_SHA:
        raise SystemExit("train.jsonl does not match its recorded sha256")
    lens = [r for r in map(json.loads, SC.TRAIN_JSONL.read_text().splitlines()) if r["attribute"] == "lens"]
    per_img = Counter(r["img_id"] for r in lens)
    by: dict[str, list[str]] = {c: [] for c in G.CLASSES}
    for r in lens:
        g = r["options"][r["answer"]]
        if g in by and per_img[r["img_id"]] == 1:
            by[g].append(r["img_id"])
    return by


def stub_kornia() -> None:
    """kornia is not installed; it is only used by GeoCalib's kornia resize backend, which this script does not use."""
    m = types.ModuleType("kornia")

    def _fail(name):
        raise AttributeError(f"kornia stub: {name} is not available (resize_backend must be torchvision)")
    m.__getattr__ = _fail
    sys.modules["kornia"] = m


def load_model():
    import torch
    stub_kornia()
    sys.path.insert(0, str(CODE))
    from geocalib.geocalib import GeoCalib
    from geocalib.utils import ImagePreprocessor
    if WEIGHTS.stat().st_size != WEIGHTS_SIZE:
        raise SystemExit("unexpected checkpoint size")
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    model = GeoCalib()
    model.flexible_load(sd["model"])
    model.eval()
    prep = ImagePreprocessor({"resize": 320, "edge_divisible_by": 32, "resize_backend": "torchvision"})
    return model, prep


def calibrate(model, prep, rgb: np.ndarray) -> dict:
    """The extractor's calibrate() path (pinhole, no priors), minus the field interpolation we do not need."""
    import torch
    img = torch.tensor(rgb.transpose(2, 0, 1) / 255.0, dtype=torch.float)[None]
    data = prep(img)
    model.optimizer.set_camera_model("pinhole")
    model.optimizer.shared_intrinsics = False
    with torch.no_grad():
        out = model(data | {"camera_R_rig": None})
    cam = out["camera"].undo_scale_crop(data)
    return {"hfov": float(torch.rad2deg(cam.hfov)[0]), "vfov": float(torch.rad2deg(cam.vfov)[0]),
            "vfov_unc": float(torch.rad2deg(out["vfov_uncertainty"])[0]), "w": int(rgb.shape[1]), "h": int(rgb.shape[0])}


def cmd_run(per_class: int) -> None:
    import torch
    torch.set_num_threads(4)
    sample = G.balanced_sample(population())
    order = [(c, sample[c][k]) for k in range(per_class) for c in G.CLASSES]   # interleaved: a cut stays balanced
    CACHE.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["img_id"] for x in ROWS.read_text().splitlines()} if ROWS.exists() else set()
    model, prep = load_model()
    with ROWS.open("a") as fh:
        for c, i in order:
            if i in done:
                continue
            t0 = time.time()
            rec = {"img_id": i, "gold": c}
            try:
                with Image.open(guards.require_under_image_root(SC.SHOTQA_ROOT / i, SC.SHOTQA_ROOT)) as im:
                    rgb = np.asarray(im.convert("RGB"))
                crop, info = G.crop_black_bars(rgb)
                rec |= {"crop": info, **calibrate(model, prep, np.ascontiguousarray(crop))}
            except Exception as e:  # noqa: BLE001 (failure cases are counted, plan 4.5)
                rec["error"] = f"{type(e).__name__}: {e}"[:200]
            rec["sec"] = time.time() - t0
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


def cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=G.SEED).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression()).fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return G.auc(s, y.astype(bool))


def sharpness(img_id: str) -> float:
    with Image.open(SC.IMG_CACHE / img_id) as im:
        return E.image_stats(np.asarray(im.convert("RGB")))["sharpness"]


def capped(r: dict) -> bool:
    """A side hit the 30% cap (recomputed from the original size: early rows were written before the field existed)."""
    if "capped" in r["crop"]:
        return r["crop"]["capped"]
    with Image.open(SC.SHOTQA_ROOT / r["img_id"]) as im:
        w, h = im.size
    c, lh, lw = r["crop"], int(G.BAR_MAX_SIDE * h), int(G.BAR_MAX_SIDE * w)
    return bool(c["top"] == lh or c["bottom"] == lh or c["left"] == lw or c["right"] == lw)


def one_sided(c: dict) -> bool:
    return any((c[a] >= 10 and c[b] == 0) or (c[b] >= 10 and c[a] == 0) for a, b in (("top", "bottom"), ("left", "right")))


def cmd_report() -> None:
    from scipy.stats import spearmanr
    rows = [json.loads(x) for x in ROWS.read_text().splitlines()]
    ok = [r for r in rows if "error" not in r and np.isfinite(r["hfov"])]
    films = shotqa_film_of({r["img_id"] for r in ok})
    for r in ok:
        r["film"] = films.get(r["img_id"]) or f"untitled:{r['img_id']}"
        r["logsharp"] = float(np.log1p(sharpness(r["img_id"])))
    unc_med = float(np.median([r["vfov_unc"] for r in ok]))
    rep = {"plan": "docs/plans/geocalib_lens_probe.md", "seed": G.SEED, "split": "ShotQA train (Phase 3B train.jsonl), single-label lens gold",
           "geocalib_commit": subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
           "weights": {"file": WEIGHTS.name, "bytes": WEIGHTS.stat().st_size, "sha256": guards.sha256_file(WEIGHTS)},
           "n_rows": len(rows), "n_ok": len(ok), "n_errors": len(rows) - len(ok),
           "per_class_n": dict(Counter(G.SHORT[r["gold"]] for r in ok)), "n_films": len({r["film"] for r in ok}),
           "sec_per_image": {"median": float(np.median([r["sec"] for r in rows])), "mean": float(np.mean([r["sec"] for r in rows]))},
           "letterbox": {"cropped": sum(r["crop"]["cropped"] for r in ok), "refused": sum(r["crop"]["refused"] for r in ok),
                         "cropped_by_class": dict(Counter(G.SHORT[r["gold"]] for r in ok if r["crop"]["cropped"])),
                         "capped": sum(capped(r) for r in ok),
                         "rows_removed_median_when_cropped": float(np.median([r["crop"]["top"] + r["crop"]["bottom"] for r in ok
                                                                               if r["crop"]["cropped"]] or [0])),
                         "cols_removed_median_when_cropped": float(np.median([r["crop"]["left"] + r["crop"]["right"] for r in ok
                                                                               if r["crop"]["cropped"]] or [0])),
                         # one side >= 10 px and the opposite side 0: likely dark scene content, not a true letterbox
                         "one_sided_ge10px": sum(one_sided(r["crop"]) for r in ok),
                         "one_sided_by_class": dict(Counter(G.SHORT[r["gold"]] for r in ok if one_sided(r["crop"])))},
           "aspect_by_class": {G.SHORT[c]: G.quantiles([r["w"] / r["h"] for r in ok if r["gold"] == c]) for c in G.CLASSES},
           "fov_by_class": {G.SHORT[c]: {k: G.quantiles([r[k] for r in ok if r["gold"] == c]) for k in ("hfov", "vfov", "vfov_unc", "logsharp")}
                            for c in G.CLASSES},
           "spearman_hfov_logsharp": float(spearmanr([r["hfov"] for r in ok], [r["logsharp"] for r in ok])[0]),
           "vfov_unc_median": unc_med}
    pairs = {}
    for a, b in PAIRS:   # a = wider class = positive
        rs = [r for r in ok if r["gold"] in (a, b)]
        y = np.array([r["gold"] == a for r in rs])
        hf, vf = np.array([r["hfov"] for r in rs]), np.array([r["vfov"] for r in rs])
        ls, fl = np.array([r["logsharp"] for r in rs]), [r["film"] for r in rs]
        asp = np.array([r["w"] / r["h"] for r in rs])
        lo_unc = np.array([r["vfov_unc"] <= unc_med for r in rs])
        terc = np.quantile(ls, [1 / 3, 2 / 3])
        tb = np.digitize(ls, terc)
        blur_cv, both_cv = cv_auc(ls[:, None], y), cv_auc(np.c_[ls, hf], y)
        pairs[pair_name(a, b)] = {
            "n_pos": int(y.sum()), "n_neg": int((~y).sum()),
            "hfov_auc": G.auc(hf, y), "hfov_ci": G.cluster_bootstrap_auc(hf, y, fl),
            "vfov_auc": G.auc(vf, y), "vfov_ci": G.cluster_bootstrap_auc(vf, y, fl),
            # blur: sharper = wider
            "blur_auc": G.auc(ls, y), "cv_auc_blur": blur_cv, "cv_auc_blur_hfov": both_cv, "cv_gain": both_cv - blur_cv,
            "cv_auc_hfov": cv_auc(hf[:, None], y),
            # added after the first report (descriptive): hfov = f(vfov, aspect), and aspect differs by class
            "aspect_auc": G.auc(asp, y), "cv_auc_blur_aspect": cv_auc(np.c_[ls, asp], y),
            "cv_auc_blur_vfov": cv_auc(np.c_[ls, vf], y), "cv_auc_blur_aspect_vfov": cv_auc(np.c_[ls, asp, vf], y),
            "hfov_auc_low_uncertainty": G.auc(hf[lo_unc], y[lo_unc]), "n_low_uncertainty": int(lo_unc.sum()),
            "hfov_auc_by_sharpness_tertile": [G.auc(hf[tb == t], y[tb == t]) for t in range(3)],
            "hfov_auc_no_one_sided_crop": G.auc(hf[[not one_sided(r["crop"]) for r in rs]], y[[not one_sided(r["crop"]) for r in rs]]),
            "hfov_auc_uncropped": G.auc(hf[[not r["crop"]["cropped"] for r in rs]], y[[not r["crop"]["cropped"] for r in rs]]),
        }
    rep["pairs"] = pairs
    tab = {G.SHORT[c]: dict(Counter(G.fov_band(r["hfov"]) for r in ok if r["gold"] == c)) for c in G.CLASSES}
    rep["band_table"] = {"bands_deg_hfov": G.BANDS, "gold_x_band": tab,
                         "own_band_share": {k: v.get(k, 0) / max(1, sum(v.values())) for k, v in tab.items()}}
    rep["reading"] = G.reading(pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("n_ok", "per_class_n", "sec_per_image", "letterbox", "reading")}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--per-class", type=int, default=G.PER_CLASS)
    a = ap.parse_args()
    if a.per_class > G.PER_CLASS:
        raise SystemExit("per-class above the plan's sample")
    cmd_run(a.per_class) if a.cmd == "run" else cmd_report()


if __name__ == "__main__":
    main()
