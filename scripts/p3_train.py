"""Train per-attribute heads on cached ShotQA features; validate on the ShotQA val split only (no ShotBench here).
Usage: uv run python scripts/p3_train.py [--cache-dir data/cache/p3/<model>] [--tag siglip_base] [--out results/phase3]"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cgt.eval import metrics as M
from cgt.p3_static_attrs import features, heads
from cgt.p3_static_attrs.data import (
    SHOTQA_ROOT,
    assert_shotqa_path,
    image_films,
    load_leak_ids,
    split_films,
    split_images,
)
from cgt.p3_static_attrs.labels import build_examples, vocab

ROOT = Path(__file__).resolve().parents[1]
ATTRS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
HEADS = {"linear": ("multihot", 0), "mlp": ("multihot", 512)}


def build_xy(exs, X, fidx, v):
    """Features (L2-normalized rows) and multi-hot targets for a list of Examples."""
    vi = {c: i for i, c in enumerate(v)}
    x = X[[fidx[e.img_id] for e in exs]]
    y = np.zeros((len(exs), len(v)), dtype=np.float32)
    for r, e in enumerate(exs):
        y[r, [vi[c] for c in e.target]] = 1
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(ROOT / "data/cache/p3" / features.slug(features.MODEL)))
    ap.add_argument("--tag", default="siglip_base")
    ap.add_argument("--out", default=None, help="default results/phase3/runs/<tag>")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--split", choices=["image", "film"], default="image",
                    help="image: 10%% of images; film: 10%% of films held out (untitled images dropped)")
    a = ap.parse_args()
    a.out = a.out or str(ROOT / "results/phase3/runs" / a.tag)
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mdir = ROOT / "models/p3" / a.tag  # gitignored; ShotQA is CC-BY-NC-ND: weights are not redistributed
    mdir.mkdir(parents=True, exist_ok=True)

    examples, _ = build_examples(assert_shotqa_path(SHOTQA_ROOT / "sft.json"), strict=True)
    audit = features.verify_cache(Path(a.cache_dir), features.MODEL, SHOTQA_ROOT)  # provenance + every source inside ShotQA
    ids, feats = features.load_features(Path(a.cache_dir))
    fidx = {i: n for n, i in enumerate(ids)}
    leak = load_leak_ids()
    have = [e.img_id for e in examples if e.img_id in fidx]
    if a.split == "film":
        img_film, minfo = image_films(assert_shotqa_path(SHOTQA_ROOT / "meta.jsonl"), set(have))
        train_ids, val_ids, split_info = split_films(img_film, 0.1, 0, exclude=leak)
        split_info["meta_join"] = minfo
    else:
        train_ids, val_ids = split_images(have, 0.1, 0, exclude=leak)
        split_info = {"train_images": len(train_ids), "val_images": len(val_ids)}
    split_info["split"] = a.split
    (out / f"split_{a.tag}.json").write_text(json.dumps(split_info, indent=1))
    tr_set, va_set = set(train_ids), set(val_ids)
    X = heads.l2norm(feats)
    rows, size_rows, chosen = [], [], {}

    for attr in ATTRS:
        v = vocab(attr)
        vi = {c: i for i, c in enumerate(v)}
        ex = [e for e in examples if e.attribute == attr and e.img_id in fidx]
        ex_tr = [e for e in ex if e.img_id in tr_set]
        ex_va = [e for e in ex if e.img_id in va_set]

        xtr, ytr = build_xy(ex_tr, X, fidx, v)
        xva, yva = build_xy(ex_va, X, fidx, v)
        print(f"[{attr}] train {len(ex_tr)} val {len(ex_va)}")
        per_kind = defaultdict(list)
        for kind_name, (kind, hidden) in HEADS.items():
            for seed in seeds:
                h, info = heads.fit_head(kind, xtr, ytr, xva, yva, seed, hidden, a.epochs, a.lr, patience=a.patience)
                torch.save(h.state_dict(), mdir / f"{attr}_{kind_name}_s{seed}.pt")
                logits = h.predict(torch.from_numpy(xva))
                f1 = M.multilabel_f1(logits, yva)
                hits = M.option_restricted_hits(ex_va, logits, v, "multihot")
                acc = float(hits.mean())
                comp = np.array([len(e.target) > 1 for e in ex_va])
                rows.append({"attribute": attr, "head": kind_name, "seed": seed, "n_val": len(ex_va),
                             "hit_at_1": M.hit_at_1(logits, yva), "option_restricted_acc": acc,
                             "n_compound_answers": int(comp.sum()),
                             "acc_single_answer": float(hits[~comp].mean()) if (~comp).any() else None,
                             "acc_compound_answer": float(hits[comp].mean()) if comp.any() else None, **f1,
                             "epochs": info["epochs"], "val_loss": info["best_val_loss"]})
                per_kind[kind_name].append(acc)
        chosen[attr] = max(per_kind, key=lambda k: np.mean(per_kind[k]))

        if attr == "size":  # ordinal vs softmax on the single-label rows
            s_tr = [e for e in ex_tr if len(e.target) == 1]
            s_va = [e for e in ex_va if len(e.target) == 1]
            ytr_i = np.array([vi[next(iter(e.target))] for e in s_tr])
            yva_i = np.array([vi[next(iter(e.target))] for e in s_va])
            xs_tr, xs_va = X[[fidx[e.img_id] for e in s_tr]], X[[fidx[e.img_id] for e in s_va]]
            for name, kind, hidden in [("softmax", "softmax", 0), ("coral", "coral", 0),
                                       ("softmax_mlp", "softmax", 512), ("coral_mlp", "coral", 512)]:
                for seed in seeds:
                    h, info = heads.fit_head(kind, xs_tr, ytr_i, xs_va, yva_i, seed, hidden, a.epochs, a.lr, patience=a.patience)
                    torch.save(h.state_dict(), mdir / f"size_{name}_s{seed}.pt")
                    p = h.predict(torch.from_numpy(xs_va))
                    size_rows.append({"head": name, "seed": seed, "n_val_single": len(s_va),
                                      "top1_acc": float((p.argmax(1) == yva_i).mean()),
                                      "mae_steps": M.ordinal_mae(p.argmax(1), yva_i),
                                      "option_restricted_acc": M.option_restricted_accuracy(s_va, p, v, "categorical")})
                    if name == "coral" and seed == seeds[0]:
                        cm = M.confusion(p.argmax(1), yva_i, len(v))
                        pd.DataFrame(cm, index=v, columns=v).to_csv(out / "confusion_size_val_coral.csv")
            # multi-hot heads on the same single-label rows for a like-for-like comparison
            xm, _ = build_xy(s_va, X, fidx, v)
            for kind_name, (kind, hidden) in HEADS.items():
                for seed in seeds:
                    hh = heads.Head(xm.shape[1], len(v), "multihot", hidden)
                    hh.load_state_dict(torch.load(mdir / f"{attr}_{kind_name}_s{seed}.pt"))
                    hh.eval()
                    lg = hh.predict(torch.from_numpy(xm))
                    size_rows.append({"head": f"multihot_{kind_name}", "seed": seed, "n_val_single": len(s_va),
                                      "top1_acc": float((lg.argmax(1) == yva_i).mean()),
                                      "mae_steps": M.ordinal_mae(lg.argmax(1), yva_i),
                                      "option_restricted_acc": M.option_restricted_accuracy(s_va, lg, v, "multihot")})

    if size_rows:  # final size head = best mean option-restricted acc on the single-label VAL rows (ShotQA val only)
        sdf = pd.DataFrame(size_rows).groupby("head").option_restricted_acc.mean()
        best = sdf.idxmax()
        chosen["size"] = best.removeprefix("multihot_")
        chosen["size_selection"] = {k: round(float(v_), 4) for k, v_ in sdf.items()}
    pd.DataFrame(rows).to_csv(out / "val_metrics.csv", index=False)
    pd.DataFrame(size_rows).to_csv(out / "size_ordinal_vs_softmax.csv", index=False)
    (out / f"chosen_heads_{a.tag}.json").write_text(json.dumps(chosen, indent=1))
    (out / f"data_roots_{a.tag}.json").write_text(json.dumps(
        {"model": a.tag, "training_and_validation_roots": ["data/external/shotqa"],
         "verified_cache_source_dirs": audit["source_roots"], "verified_cache_images": audit["n_images"],
         "cache_provenance": audit["provenance"],
         "feature_cache": str(Path(a.cache_dir).relative_to(ROOT)) if Path(a.cache_dir).is_relative_to(ROOT) else a.cache_dir,
         "excluded_leak_ids": len(leak), "seeds": seeds, "split": a.split,
         "hyper_parameters": {"lr": a.lr, "epochs_max": a.epochs, "patience": a.patience, "weight_decay": 1e-4,
                              "selected_on": "ShotQA validation split only"}}, indent=1))
    print(pd.DataFrame(rows).groupby(["attribute", "head"])[["hit_at_1", "option_restricted_acc"]].mean().round(3))
    print(pd.DataFrame(size_rows).groupby("head")[["top1_acc", "mae_steps", "option_restricted_acc"]].mean().round(3))
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
