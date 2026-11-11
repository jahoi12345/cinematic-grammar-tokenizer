"""Bounded style (RQ-VAE) feasibility run. CPU ONLY (torch threads capped, nice 19); no GPU, no MPS.
Fits on ShotQA film-split TRAIN stills only; NOTLD (inference target, never fitted on) is used for evaluation: same-shot agreement and code concentration.
Writes results/phase5/style_feasibility.{md,json}; derived artefacts (embeddings, codes, model) stay under data/cache/tokens/ (ShotQA-derived, never committed).
Usage: nice -n 19 uv run python scripts/style_feasibility.py"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cgt.p3_static_attrs import features, heads
from cgt.p3b_vlm.data import film_split_ids
from cgt.p3c_tokens.vocab import ATTRS, labels
from cgt.tokens import palette_global as pg
from cgt.tokens import style
from cgt.worktrees import sibling

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/tokens"
P3_ROOT = sibling("phase2")  # read-only: Phase 3A feature cache and heads
FEATS = P3_ROOT / "data/cache/p3/google__siglip-base-patch16-224"
TAG = "siglip_base_filmsplit_lr1e-2_ep300"
FILM = "night_of_the_living_dead"
KFS = ("kf_start", "kf_mid", "kf_end")
BARS = {"level1_used_min": 200, "perplexity_frac_min": 0.60, "max_share_max": 0.03, "same_shot_min": 0.50, "same_shot_weakest_attr": 0.42, "notld_max_share_max": 0.10}


def head_primaries(X: np.ndarray) -> np.ndarray:
    """(N, 7) primary label indices of the Phase 3A film-split heads (seed 0, the chosen head type per attribute); stand-ins for the Phase 3C attribute tokens
    because the 3B model cannot run on CPU over 45k stills. Used identically on ShotQA and NOTLD."""
    chosen = json.loads((P3_ROOT / f"results/phase3/runs/{TAG}/chosen_heads_{TAG}.json").read_text())
    x = torch.from_numpy(X.astype(np.float32))
    out = []
    for a in ATTRS:
        v, kn = labels(a), chosen[a]
        if a == "size" and kn in {"softmax", "coral", "softmax_mlp", "coral_mlp"}:
            kind, hid = kn.split("_")[0], 512 if kn.endswith("_mlp") else 0
        else:
            kind, hid = "multihot", 512 if kn == "mlp" else 0
        h = heads.Head(x.shape[1], len(v), kind, hid)
        h.load_state_dict(torch.load(P3_ROOT / f"models/p3/{TAG}/{a}_{kn}_s0.pt"))
        h.eval()
        out.append(h.predict(x).argmax(1))
    return np.stack(out, 1)


def design(prim: np.ndarray, pal: np.ndarray) -> np.ndarray:
    return np.concatenate([style.one_hot(prim[:, j], len(labels(a))) for j, a in enumerate(ATTRS)] + [style.one_hot(pal, pg.K)], axis=1)


def embed_cpu(paths: list[str], batch: int = 32) -> np.ndarray:
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipModel
    proc = AutoImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to("cpu").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            o = model.get_image_features(pixel_values=proc(images=ims, return_tensors="pt")["pixel_values"])
            out.append(getattr(o, "pooler_output", o).float().numpy())
    return np.concatenate(out)


def main() -> None:
    try:
        os.nice(19)
    except OSError:
        pass  # already at the lowest priority (run under nice -n 19)
    torch.set_num_threads(3)
    t0 = time.time()
    train_ids, val_ids, _ = film_split_ids()
    ids, feats = features.load_features(FEATS)
    ids = np.array(ids)
    X = heads.l2norm(feats)
    tr, va = np.array([i in train_ids for i in ids]), np.array([i in val_ids for i in ids])
    print(f"features {X.shape}, train {tr.sum()}, val {va.sum()}", flush=True)

    cents = pg.load()
    pv = np.load(CACHE / "palette_vectors_train.npz")
    pal_tr_map = dict(zip(pv["ids"], pg.assign(pv["x"], cents), strict=True))
    val_list = [i for i in ids[va]]
    from cgt.p3_static_attrs.data import image_path
    xv = pg.palette_vectors([str(image_path(i)) for i in val_list])
    ok_v = ~np.isnan(xv).any(axis=1)
    pal_va_map = dict(zip(np.array(val_list)[ok_v], pg.assign(xv[ok_v], cents), strict=True))
    keep_tr = np.array([i in pal_tr_map for i in ids[tr]])
    keep_va = np.array([i in pal_va_map for i in ids[va]])
    Xtr, Xva = X[tr][keep_tr], X[va][keep_va]
    ptr = np.array([pal_tr_map[i] for i in ids[tr][keep_tr]])
    pva = np.array([pal_va_map[i] for i in ids[va][keep_va]])
    prim_tr, prim_va = head_primaries(Xtr), head_primaries(Xva)
    Ttr, Tva = design(prim_tr, ptr), design(prim_va, pva)

    w, b = style.ridge_fit(Ttr, Xtr, alpha=1.0)
    Rtr, Rva = style.residual(Ttr, Xtr, w, b), style.residual(Tva, Xva, w, b)
    r2_removed = float(1 - (Rva ** 2).sum() / ((Xva - Xva.mean(0)) ** 2).sum())
    mu, sd = Rtr.mean(0), Rtr.std() + 1e-8  # one global scale keeps the residual's relative geometry
    ztr, zva = ((Rtr - mu) / sd).astype(np.float32), ((Rva - mu) / sd).astype(np.float32)
    print(f"residual: variance explained by attribute tokens on val {r2_removed:.3f}", flush=True)

    model, info = style.train_rqvae(ztr, epochs=30, max_seconds=1500, log=lambda m: print(m, flush=True))
    ctr, cva = model.encode(ztr), model.encode(zva)
    with torch.no_grad():
        rec = model(torch.from_numpy(zva))[0].numpy()
    r2_rec = float(1 - ((rec - zva) ** 2).sum() / ((zva - zva.mean(0)) ** 2).sum())
    use_tr, use_va = style.usage_stats(ctr), style.usage_stats(cva)

    # NOTLD (evaluation only): embed the three keyframes on CPU, same pipeline
    shots = pd.read_parquet(ROOT / f"data/processed/{FILM}_shots.parquet").sort_values("shot_id")
    nf = CACHE / f"{FILM}_siglip_kf.npz"
    if nf.exists():
        E = {k: np.load(nf)[k] for k in KFS}
    else:
        E = {k: embed_cpu([str(ROOT / p) for p in shots[k]]) for k in KFS}
        np.savez(nf, **E)
    codes_n = {}
    import cv2

    from cgt.p1_shots_color import color
    for k in KFS:
        Xn = heads.l2norm(E[k])
        pn = pg.assign(np.stack([color.palette_lab(cv2.imread(str(ROOT / p))).ravel() for p in shots[k]]), cents)
        Tn = design(head_primaries(Xn), pn)
        zn = ((style.residual(Tn, Xn, w, b) - mu) / sd).astype(np.float32)
        codes_n[k] = model.encode(zn)
    agree = {lv: style.same_shot_agreement(codes_n, lv) for lv in range(3)}
    use_n = style.usage_stats(codes_n["kf_mid"])

    # probe: attributes (head primaries) from codes versus from the raw embedding; train subsample, evaluate on held-out films
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    sub = rng.permutation(len(ctr))[:15000]

    def codes_oh(c):
        return np.concatenate([style.one_hot(c[:, lv], 256) for lv in range(3)], axis=1)
    probe = {}
    for j, a in enumerate(ATTRS):
        res = {"majority": float(np.bincount(prim_va[:, j]).max() / len(prim_va))}
        for name, ftr, fva in (("raw_embedding", Xtr[sub], Xva), ("style_codes", codes_oh(ctr[sub]), codes_oh(cva))):
            clf = LogisticRegression(max_iter=300, C=1.0).fit(ftr, prim_tr[sub, j])
            res[name] = float((clf.predict(fva) == prim_va[:, j]).mean())
        probe[a] = res
    mean_raw = float(np.mean([p["raw_embedding"] for p in probe.values()]))
    mean_codes = float(np.mean([p["style_codes"] for p in probe.values()]))
    mean_maj = float(np.mean([p["majority"] for p in probe.values()]))

    perp_ok = all(l["perplexity"] >= BARS["perplexity_frac_min"] * 256 for l in use_tr["levels"])
    bars = {
        "level1_used>=200": (use_tr["levels"][0]["used"], use_tr["levels"][0]["used"] >= BARS["level1_used_min"]),
        "perplexity>=60%_of_256_all_levels": ([round(l["perplexity"], 1) for l in use_tr["levels"]], perp_ok),
        "no_code_above_3%_all_levels": ([round(l["max_share"], 4) for l in use_tr["levels"]], all(l["max_share"] <= BARS["max_share_max"] for l in use_tr["levels"])),
        "same_shot_level1_agreement>=50%_NOTLD": (round(agree[0]["all_three"], 4), agree[0]["all_three"] >= BARS["same_shot_min"]),
        "same_shot_level1_agreement>=42%_weakest_attribute": (round(agree[0]["all_three"], 4), agree[0]["all_three"] >= BARS["same_shot_weakest_attr"]),
        "no_code_above_10%_NOTLD_level1": (round(use_n["levels"][0]["max_share"], 4), use_n["levels"][0]["max_share"] <= BARS["notld_max_share_max"]),
        "probe_codes_clearly_worse_than_raw (not numerically pre-registered; shown as mean accuracy)": ((round(mean_codes, 4), round(mean_raw, 4), round(mean_maj, 4)), mean_codes <= mean_raw - 0.10),
    }
    res = {"n_train": len(ztr), "n_val": len(zva), "attribute_variance_explained_val": r2_removed, "rqvae": {k: v for k, v in info.items() if k != "loss"} | {"final_loss": info["loss"][-1]},
           "val_reconstruction_r2": r2_rec, "usage_train": use_tr, "usage_val": use_va, "usage_notld_mid": use_n, "same_shot_agreement_notld": agree,
           "probe_attribute_accuracy_val": probe, "probe_means": {"raw": mean_raw, "codes": mean_codes, "majority": mean_maj}, "bars": {k: {"value": v[0], "pass": bool(v[1])} for k, v in bars.items()},
           "seconds": time.time() - t0, "attribute_tokens": "Phase 3A film-split SigLIP heads (seed 0) on ShotQA and NOTLD alike", "human_check": "deferred (30 random level-1 codes, 20-shot grids, rater = user)"}
    out = ROOT / "results/phase5"
    out.mkdir(parents=True, exist_ok=True)
    (out / "style_feasibility.json").write_text(json.dumps(res, indent=1))
    torch.save(model.state_dict(), CACHE / "rqvae_style.pt")
    np.savez(CACHE / "style_codes.npz", train=ctr, val=cva, ids_train=ids[tr][keep_tr], ids_val=ids[va][keep_va])
    print(json.dumps(res["bars"], indent=1), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
