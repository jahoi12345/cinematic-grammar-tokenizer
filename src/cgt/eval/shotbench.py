"""ShotBench evaluation (TEST ONLY). Imported only by scripts/p3_eval_shotbench.py, never by training code.
Each model tag may be evaluated exactly once (ledger results/phase3/shotbench_evaluated.json)."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cgt.eval import metrics as M
from cgt.p3_static_attrs import features, heads
from cgt.p3_static_attrs.labels import UnmappedLabel, parse_label_set, vocab

ROOT = Path(__file__).resolve().parents[3]
SB = ROOT / "data/external/shotbench"
CATEGORY_ATTR = {"shot size": "size", "lens size": "lens", "shot framing": "framing", "camera angle": "angle",
                 "lighting type": "light_type", "lighting": "light_cond", "composition": "composition"}
# reference numbers from docs/research/shotbench_paper_notes.md (paper Table 2), % option-restricted accuracy
REFERENCE = {
    "GPT-4o": {"size": 69.3, "framing": 83.1, "angle": 58.2, "lens": 48.9, "light_type": 63.2, "light_cond": 48.0, "composition": 55.2},
    "ShotVL-3B": {"size": 77.9, "framing": 85.6, "angle": 68.8, "lens": 59.3, "light_type": 65.7, "light_cond": 53.1, "composition": 57.4},
    "Qwen2.5-VL-3B (base)": {"size": 54.6, "framing": 56.6, "angle": 43.1, "lens": 36.6, "light_type": 59.3, "light_cond": 45.1, "composition": 41.5},
}
LEDGER = ROOT / "results/phase3/shotbench_evaluated.json"


def ledger_read(path: Path = LEDGER) -> dict[str, str]:
    """{model_tag: note}; empty dict if the ledger file does not exist."""
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def ledger_assert_unused(tag: str, path: Path = LEDGER) -> None:
    """Raise RuntimeError if `tag` was already evaluated on ShotBench (each model tag may be evaluated exactly once)."""
    led = ledger_read(path)
    if tag in led:
        raise RuntimeError(f"model '{tag}' was already evaluated on ShotBench ({led[tag]}); use a new tag for a new model")


def ledger_record(tag: str, note: str, path: Path = LEDGER) -> None:
    """Add `tag` to the ledger (creates the file/dirs). Raises RuntimeError if the tag is already present."""
    ledger_assert_unused(tag, path)
    led = ledger_read(path)
    led[tag] = note
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(path).with_name(".tmp_" + Path(path).name)  # temp file + atomic replace: an interrupted write never leaves a truncated ledger
    with open(tmp, "w") as f:
        f.write(json.dumps(led, indent=1))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_questions() -> pd.DataFrame:
    """Image questions of the 7 static categories with parsed options. Rows with unmappable options are counted, not dropped silently."""
    sb = pd.read_csv(SB / "test.tsv", sep="\t")
    sb = sb[sb.type.str.contains("image") & sb.category.isin(CATEGORY_ATTR)].copy()
    rows, unmapped = [], 0
    totals = sb.category.map(CATEGORY_ATTR).value_counts().to_dict()
    dropped: dict[str, int] = {}
    for r in sb.itertuples():
        attr = CATEGORY_ATTR[r.category]
        raw = json.loads(r.options)
        try:
            opts = {k: parse_label_set(attr, v) for k, v in raw.items()}
        except UnmappedLabel:
            unmapped += 1
            dropped[attr] = dropped.get(attr, 0) + 1
            continue
        rows.append({"attribute": attr, "img": Path(ast.literal_eval(r.path)[0]).name, "options": opts,
                     "answer": r.answer, "path": SB / ast.literal_eval(r.path)[0]})
    df = pd.DataFrame(rows)
    df.attrs["unmapped_questions"] = unmapped
    df.attrs["rows_in_category"] = totals
    df.attrs["dropped_unparsed"] = dropped
    return df


def parse_coverage() -> dict:
    """Dry run: how many ShotBench options map to the schema (uses option text only)."""
    q = load_questions()
    return {"questions": len(q), "unmapped_questions": q.attrs["unmapped_questions"]}


def evaluate(tag: str, chosen: dict, model_dir: Path, seeds: list[int], cache_dir: Path, out_csv: Path) -> pd.DataFrame:
    ledger_assert_unused(tag)
    q = load_questions()
    paths = sorted(set(q.path))
    features.embed_images(paths, cache_dir)
    ids, feats = features.load_features(cache_dir)
    fidx = {i: n for n, i in enumerate(ids)}
    X = heads.l2norm(feats)
    rows = []
    for attr, g in q.groupby("attribute"):
        v = vocab(attr)
        vi = {c: i for i, c in enumerate(v)}
        n_total = int(q.attrs["rows_in_category"].get(attr, 0))
        n_unparsed = int(q.attrs["dropped_unparsed"].get(attr, 0))
        n_before = len(g)
        g = g[g.img.isin(fidx)]
        n_no_feat = n_before - len(g)
        x = torch.from_numpy(X[[fidx[i] for i in g.img]])
        kind_name = chosen[attr]
        preds = []
        for s in seeds:
            if attr == "size" and kind_name in {"softmax", "coral", "softmax_mlp", "coral_mlp"}:
                kind, hidden = kind_name.split("_")[0], 512 if kind_name.endswith("_mlp") else 0
                n_out = len(v)
            else:
                kind, hidden, n_out = "multihot", (512 if kind_name == "mlp" else 0), len(v)
            h = heads.Head(x.shape[1], n_out, kind, hidden)
            h.load_state_dict(torch.load(model_dir / f"{attr}_{kind_name}_s{s}.pt"))
            h.eval()
            preds.append(h.predict(x))
        p = np.mean(preds, axis=0)
        kind_eval = "multihot" if kind == "multihot" else "categorical"

        class _Ex:  # minimal Example-like holder for the metric
            def __init__(self, opts, ans):
                self.options, self.answer_letter = opts, ans
        exs = [_Ex(o, a) for o, a in zip(g.options, g.answer, strict=True)]
        hits = M.option_restricted_hits(exs, p, v, kind_eval)
        acc = float(hits.mean())
        comp = np.array([len(o[a]) > 1 for o, a in zip(g.options, g.answer, strict=True)])
        tgt = np.zeros((len(g), len(v)), dtype=np.float32)
        for r, (o, a) in enumerate(zip(g.options, g.answer, strict=True)):
            tgt[r, [vi[c] for c in o[a]]] = 1
        rows.append({"model": tag, "attribute": attr, "n": len(g), "option_restricted_acc_pct": round(100 * acc, 1),
                     "full_vocab_hit_at_1_pct": round(100 * M.hit_at_1(p, tgt), 1), "head": kind_name,
                     "seeds": len(seeds), "rows_in_category": n_total, "dropped_unparsed": n_unparsed,
                     "dropped_no_image_feature": n_no_feat, "n_compound_answers": int(comp.sum()),
                     "acc_single_answer_pct": round(100 * float(hits[~comp].mean()), 1) if (~comp).any() else None,
                     "acc_compound_answer_pct": round(100 * float(hits[comp].mean()), 1) if comp.any() else None})
    res = pd.DataFrame(rows)
    ref = [{"model": m, "attribute": a, "n": None, "option_restricted_acc_pct": val, "full_vocab_hit_at_1_pct": None,
            "head": "reference (paper)", "seeds": None} for m, d in REFERENCE.items() for a, val in d.items()]
    pd.concat([res, pd.DataFrame(ref)]).to_csv(out_csv, index=False)
    ledger_record(tag, f"evaluated once; n={int(res.n.sum())}")
    return res
