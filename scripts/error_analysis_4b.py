"""descriptive error analysis of the Step C 4B (ckpt_100, adapter aba56883...), the 4B + C5 blend and the frozen 2B on the
dedup_v2 ShotQA film-held-out validation set (5,822 questions, 71 films). CPU only. Nothing is fitted, tuned or selected: C5 uses
the per-attribute weights already fitted by Step A-4B (results/stepa_4b/val.json, calibration half) and the cached Phase 3A
SigLIP option scores; image statistics are computed once per validation image from the Phase 3B budget-shrunk copies and only
described. Writes AGGREGATES only (no per-question rows, no images) to results/error_analysis_4b/report.json.
  PYTHONPATH=src nice -n 19 <venv>/bin/python scripts/error_analysis_4b.py"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image

from cgt.error_analysis import core as E
from cgt.eval.metrics import pick_option
from cgt.p3_static_attrs.labels import parse_label_set, vocab
from cgt.p3b_vlm.guards import sha256_file
from cgt.p3b_vlm.runstatus import now_iso
from cgt.p3s_stepa import combine as C
from cgt.p3s_stepa.split import ATTRS
from cgt.stepa_4b import config as K
from cgt.stepa_4b import core as S4
from cgt.stepc import config as SC

LET = ["A", "B", "C", "D"]
OUT = ROOT / "results/error_analysis_4b"
FOCUS = ["light_cond", "light_type", "composition", "lens"]
STATS = ["mean_luma", "rms_contrast", "p95_p5", "dark_frac", "bright_frac", "saturation", "warmth", "sharpness", "grad_cx",
         "lr_balance", "aspect"]


def load_preds(path: Path, sha: str) -> list[dict]:
    if sha256_file(path) != sha:
        raise SystemExit(f"{path} does not match its recorded sha256")
    return [json.loads(x) for x in path.read_text().splitlines()]


def load_sig() -> dict:
    sig = {}
    for f, sha in ((K.SIGLIP, K.SIGLIP_SHA), (K.SIGLIP_ADDED, K.SIGLIP_ADDED_SHA)):
        if sha256_file(f) != sha:
            raise SystemExit(f"{f} does not match its recorded sha256")
        for r in map(json.loads, f.read_text().splitlines()):
            sig[r["key"]] = r["scores"]
    return sig


def img_stats(img_ids: set[str]) -> tuple[dict, dict]:
    out, src = {}, Counter()
    for i in sorted(img_ids):
        p = SC.IMG_CACHE / i
        if not p.exists():
            p, s = SC.SHOTQA_ROOT / i, "original"
        else:
            s = "shrunk_147456"
        with Image.open(p) as im:
            out[i] = E.image_stats(np.asarray(im.convert("RGB")))
        src[s] += 1
    return out, dict(src)


def main() -> None:
    rows = S4.load_val_rows()
    qs = S4.v2_split(rows, S4.shotqa_film_of)["questions"]
    p4 = {r["line"]: r for r in load_preds(K.STEPC_PRED, K.STEPC_PRED_SHA)}
    p2 = load_preds(SC.FROZEN_PRED, SC.FROZEN_PRED_SHA)
    assert len(p2) == len(rows)
    sig = load_sig()
    vj = json.loads((ROOT / "results/stepa_4b/val.json").read_text())
    w = vj["results"]["C5"]["fit"]["ensemble_w"]

    s4 = {q["key"]: np.asarray(p4[q["line"]]["logits"], float) for q in qs}
    s2 = {q["key"]: np.asarray(p2[q["line"]]["logits"], float) for q in qs}
    for q in qs:
        assert p4[q["line"]]["id"] == q["id"] == p2[q["line"]]["id"]
    s5 = C.mix(s4, sig, w, qs)
    models = {"2B": s2, "4B": s4, "4B+C5": s5}
    acc = {m: C.accuracy(qs, s) for m, s in models.items()}
    checks = {"4B_full_mean_equals_stepc_primary": abs(100 * acc["4B"]["mean"] - K.STEPC_PRIMARY) < 1e-9,
              "4B+C5_full_mean_equals_stepa_4b_C5": abs(acc["4B+C5"]["mean"] - vj["results"]["C5"]["full"]["mean"]) < 1e-12,
              "2B_full_mean": 100 * acc["2B"]["mean"]}

    stats, stat_src = img_stats({q["img_id"] for q in qs})

    # per-question working records (never written)
    rec = {m: [] for m in models}
    for q in qs:
        gl = LET.index(q["answer"])
        gset = parse_label_set(q["attribute"], q["options"][q["answer"]])
        for m, s in models.items():
            pl = pick_option(s[q["key"]], LET)
            rec[m].append({"attr": q["attribute"], "film": q["film"], "img": q["img_id"], "gold": gset,
                           "pred": parse_label_set(q["attribute"], q["options"][pl]), "gold_text": q["options"][q["answer"]],
                           "pred_text": q["options"][pl], "correct": pl == q["answer"], "rank": E.gold_rank(s[q["key"]], gl),
                           "scores": s[q["key"]], "gl": gl, "compound": len(gset) > 1})

    report = {"time": now_iso(), "task": "descriptive error analysis (no fitting, tuning or selection)",
              "val": "dedup_v2 ShotQA film-held-out validation", "questions": len(qs), "films": len({q["film"] for q in qs}),
              "inputs": {"4B_predictions_sha256": K.STEPC_PRED_SHA, "4B_adapter_sha256": K.ADAPTER_SHA,
                         "2B_predictions_sha256": SC.FROZEN_PRED_SHA, "val_jsonl_sha256": SC.VAL_SHA,
                         "c5_weights_from": "results/stepa_4b/val.json results.C5.fit.ensemble_w (fitted on the calibration half by Step A-4B)",
                         "c5_weights": w, "siglip_sha256": [K.SIGLIP_SHA, K.SIGLIP_ADDED_SHA]},
              "checks": checks, "accuracy": {m: {"per_attribute": {a: 100 * v for a, v in r["per_attribute"].items()},
                                                 "mean": 100 * r["mean"]} for m, r in acc.items()},
              "image_stats_method": {"images": len(stats), "source": stat_src, "stats": STATS,
                                     "how": "cgt.error_analysis.core.image_stats on the Phase 3B budget-shrunk copy (147,456 px) "
                                            "converted to RGB; one value per image, joined to its questions"},
              "attributes": {}}

    for a in ATTRS:
        A = {}
        labs = vocab(a)
        rs = {m: [r for r in rec[m] if r["attr"] == a] for m in models}
        r4 = rs["4B"]
        A["n"] = len(r4)
        A["compound_gold_share"] = float(np.mean([r["compound"] for r in r4]))
        A["accuracy"] = {m: 100 * float(np.mean([r["correct"] for r in rs[m]])) for m in models}
        A["accuracy_single_vs_compound_gold_4B"] = {
            k: {"n": len(v), "acc": 100 * float(np.mean([r["correct"] for r in v])) if v else None}
            for k, v in (("single", [r for r in r4 if not r["compound"]]), ("compound", [r for r in r4 if r["compound"]]))}
        A["per_label"] = {m: E.component_prf(rs[m], labs) for m in models}
        A["top_label_confusions"] = {m: E.component_confusions(rs[m], 12) for m in ("4B", "4B+C5")}
        A["top_option_confusions_4B"] = E.option_confusions(r4, 12)
        # margin analysis (4B and 4B+C5)
        for m in ("4B", "4B+C5"):
            rr = rs[m]
            wrong = [r for r in rr if not r["correct"]]
            ranks = Counter(r["rank"] for r in wrong)
            gap = [float(np.max(r["scores"]) - r["scores"][r["gl"]]) for r in wrong]
            A[f"margin_{m}"] = {"errors": len(wrong), "gold_rank_among_errors": {str(k): ranks.get(k, 0) for k in (2, 3, 4)},
                                "gold_second_share_of_errors": ranks.get(2, 0) / len(wrong) if wrong else None,
                                "top2_accuracy": 100 * float(np.mean([r["rank"] <= 2 for r in rr])),
                                "median_score_gap_top_minus_gold_on_errors": float(np.median(gap)) if gap else None,
                                "score_units": "logits (4B)" if m == "4B" else "blended log-probs (C5)"}
        # how close the errors are: share of 4B errors with a gap < 1 logit
        wrong4 = [r for r in r4 if not r["correct"]]
        A["margin_4B"]["errors_gap_lt_1_logit_share"] = (float(np.mean([np.max(r["scores"]) - r["scores"][r["gl"]] < 1 for r in wrong4]))
                                                       if wrong4 else None)
        # complementarity
        c2, c4, c5 = (np.array([r["correct"] for r in rs[m]]) for m in ("2B", "4B", "4B+C5"))
        A["agreement"] = {"4B_right_2B_wrong": int((c4 & ~c2).sum()), "4B_wrong_2B_right": int((~c4 & c2).sum()),
                          "both_wrong": int((~c4 & ~c2).sum()), "C5_fixes_4B": int((c5 & ~c4).sum()),
                          "C5_breaks_4B": int((~c5 & c4).sum())}
        A["per_film_4B"] = E.film_concentration([r["film"] for r in r4], [r["correct"] for r in r4])
        A["per_film_4B+C5"] = E.film_concentration([r["film"] for r in rs["4B+C5"]], [r["correct"] for r in rs["4B+C5"]])
        # image statistics vs 4B errors (descriptive)
        ok = [r["correct"] for r in r4]
        img = {}
        for st in STATS:
            x = [stats[r["img"]][st] for r in r4]
            img[st] = {"auc_error_vs_correct": E.auc(x, [not c for c in ok]),
                       "median_correct": float(np.median([v for v, c in zip(x, ok, strict=True) if c])),
                       "median_wrong": float(np.median([v for v, c in zip(x, ok, strict=True) if not c])) if not all(ok) else None}
        A["image_stats_vs_4B_error"] = img
        # per gold label: median stat (does the stat separate the classes?) and per-label AUC one-vs-rest
        per_lab = {}
        for st in STATS:
            per_lab[st] = {}
            for c in labs:
                pos = [c in r["gold"] for r in r4]
                if any(pos):
                    xs = [stats[r["img"]][st] for r in r4]
                    per_lab[st][c] = {"n": int(sum(pos)), "median": float(np.median([v for v, p in zip(xs, pos, strict=True) if p])),
                                      "auc_label_vs_rest": E.auc(xs, pos)}
        A["image_stats_by_gold_label"] = per_lab
        A["accuracy_by_quintile_4B"] = {st: E.accuracy_by_quantile([stats[r["img"]][st] for r in r4], ok)
                                        for st in ("mean_luma", "rms_contrast", "sharpness", "lr_balance")}
        report["attributes"][a] = A

    # lens: pairwise separability of single-label gold classes by each image statistic (descriptive AUC)
    lens1 = [r for r in rec["4B"] if r["attr"] == "lens" and not r["compound"]]
    pairs = {}
    for c1, c2 in (("long", "medium"), ("medium", "wide"), ("wide", "ultra-wide"), ("long", "wide")):
        sub = [r for r in lens1 if next(iter(r["gold"])) in (c1, c2)]
        pairs[f"{c1}_vs_{c2}"] = {st: E.auc([stats[r["img"]][st] for r in sub], [next(iter(r["gold"])) == c2 for r in sub])
                                  for st in ("sharpness", "rms_contrast", "mean_luma", "aspect")}
    report["lens_pairwise_stat_auc"] = pairs
    # top confusions vs one image statistic: among 4B questions whose gold contains `c`, those answered with an option holding
    # c (correct) vs those answered without c but with d (confused); medians and AUC(confused vs correct). Descriptive.
    conf_pairs = [("lens", "medium", "long", "sharpness"), ("lens", "medium", "long", "aspect"), ("lens", "wide", "medium", "sharpness"),
                  ("lens", "wide", "ultra-wide", "sharpness"), ("light_cond", "hard", "soft", "rms_contrast"),
                  ("light_cond", "hard", "soft", "p95_p5"), ("light_cond", "soft", "hard", "rms_contrast"),
                  ("light_cond", "high-contrast", "soft", "rms_contrast"), ("composition", "left-heavy", "balanced", "lr_balance"),
                  ("composition", "left-heavy", "center", "lr_balance"), ("composition", "right-heavy", "balanced", "lr_balance"),
                  ("composition", "right-heavy", "center", "lr_balance"), ("light_type", "daylight", "artificial", "mean_luma"),
                  ("light_type", "overcast", "artificial", "mean_luma"), ("light_type", "daylight", "fluorescent", "mean_luma")]
    cs = {}
    for a, c, d, st in conf_pairs:
        rr = [r for r in rec["4B"] if r["attr"] == a and c in r["gold"]]
        good = [stats[r["img"]][st] for r in rr if r["correct"]]
        bad = [stats[r["img"]][st] for r in rr if c not in r["pred"] and d in r["pred"]]
        cs[f"{a}:{c}->{d}:{st}"] = {"n_correct": len(good), "n_confused": len(bad),
                                    "median_correct": float(np.median(good)) if good else None,
                                    "median_confused": float(np.median(bad)) if bad else None,
                                    "auc_confused_vs_correct": E.auc(good + bad, [False] * len(good) + [True] * len(bad))}
    report["confusion_vs_image_stat_4B"] = cs
    # lens vs shot size on the same image (single-label gold on both): counts and 4B lens accuracy per gold size
    size_of = {r["img"]: next(iter(r["gold"])) for r in rec["4B"] if r["attr"] == "size" and not r["compound"]}
    xt, acc_by = Counter(), {}
    for r in lens1:
        s = size_of.get(r["img"])
        if s:
            xt[(next(iter(r["gold"])), s)] += 1
            acc_by.setdefault(s, []).append(r["correct"])
    report["lens_by_gold_size_4B"] = {"pairs": len([r for r in lens1 if r["img"] in size_of]),
                                      "crosstab_lens_x_size": {f"{a}|{b}": n for (a, b), n in sorted(xt.items())},
                                      "lens_accuracy_by_size": {s: {"n": len(v), "acc": 100 * float(np.mean(v))} for s, v in sorted(acc_by.items())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"checks": checks, "accuracy": report["accuracy"]}, indent=1))


if __name__ == "__main__":
    main()
