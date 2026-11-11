"""Image-free control on the ShotQA VALIDATION split (no ShotBench): score each option by the training frequency of its exact
label set as a correct answer. Shows how much option-set priors alone give, and writes the val confusion matrix of the frozen
size head. Usage: uv run python scripts/p3_prior_baseline.py --tag <tag> [--split film]"""
import argparse
import json
from collections import Counter
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
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--split", choices=["image", "film"], default="image")
a = ap.parse_args()
ex, _ = build_examples(assert_shotqa_path(SHOTQA_ROOT / "sft.json"), strict=True)
ids, feats = features.load_features(ROOT / "data/cache/p3/google__siglip-base-patch16-224")  # train.py already verified this cache
fidx = {i: n for n, i in enumerate(ids)}
have = [e.img_id for e in ex if e.img_id in fidx]
if a.split == "film":
    img_film, _ = image_films(assert_shotqa_path(SHOTQA_ROOT / "meta.jsonl"), set(have))
    tr, va, _ = split_films(img_film, 0.1, 0, exclude=load_leak_ids())
else:
    tr, va = split_images(have, 0.1, 0, exclude=load_leak_ids())
tr, va = set(tr), set(va)
chosen = json.loads((ROOT / f"results/phase3/runs/{a.tag}/chosen_heads_{a.tag}.json").read_text())
X = heads.l2norm(feats)
rows = []
for attr in ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]:
    v = vocab(attr)
    tr_ex = [e for e in ex if e.attribute == attr and e.img_id in tr]
    va_ex = [e for e in ex if e.attribute == attr and e.img_id in va]
    freq = Counter(e.target for e in tr_ex)
    hits = 0
    for e in va_ex:
        letters = sorted(e.options)
        sc = np.array([np.log(freq.get(e.options[ltr], 0) + 0.5) for ltr in letters])
        hits += M.pick_option(sc, letters) == e.answer_letter
    # frozen image model, validation (mean of 3 seeds), for the same questions
    kind_name = chosen[attr]
    x = torch.from_numpy(X[[fidx[e.img_id] for e in va_ex]])
    preds = []
    cat = kind_name.split("_")[0] in {"softmax", "coral"}
    for s in (0, 1, 2):
        h = heads.Head(x.shape[1], len(v), kind_name.split("_")[0] if cat else "multihot",
                       512 if kind_name.endswith("mlp") else 0)
        h.load_state_dict(torch.load(ROOT / f"models/p3/{a.tag}/{attr}_{kind_name}_s{s}.pt"))
        h.eval()
        preds.append(h.predict(x))
    p = np.mean(preds, axis=0)
    img_acc = M.option_restricted_accuracy(va_ex, p, v, "categorical" if cat else "multihot")
    rows.append({"attribute": attr, "n_val": len(va_ex), "prior_only_acc": round(hits / len(va_ex), 3),
                 "frozen_model_val_acc": round(img_acc, 3)})
    if attr == "size":
        single = [i for i, e in enumerate(va_ex) if len(e.target) == 1]
        vi = {c: i for i, c in enumerate(v)}
        true = np.array([vi[next(iter(va_ex[i].target))] for i in single])
        cm = M.confusion(p[single].argmax(1), true, len(v))
        pd.DataFrame(cm, index=v, columns=v).to_csv(ROOT / f"results/phase3/runs/{a.tag}/confusion_size_val_frozen_head.csv")
        print("size confusion (rows true, cols predicted), val single-label rows:\n", pd.DataFrame(cm, index=v, columns=v))
out = pd.DataFrame(rows)
out.to_csv(ROOT / f"results/phase3/runs/{a.tag}/prior_only_baseline_val.csv", index=False)
print(out.to_string(index=False))
