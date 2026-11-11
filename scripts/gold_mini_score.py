"""SEALED scorer of the gold mini evaluation, exactly as pre-registered in docs/gold_mini_eval_plan.md (section 5). Run ONCE by the author after all 60
labels exist:
  cd <main checkout> && PYTHONPATH=src .venv/bin/python scripts/gold_mini_score.py
Refuses unless the label file holds exactly the 60 pre-registered shots (--partial scores what exists and writes score_v1_PARTIAL.*, which is not the result).
Refuses to overwrite an existing score. Writes results/gold_mini/score_v1.{md,json}. Nothing here tunes anything."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app/label"))
sys.path.insert(0, str(ROOT / "src"))
import gold_mini as GM

from cgt.tokens import v1 as T

Z = 1.959963984540054
N_BOOT = 10_000
MIN_POS = 5
PLAN = "docs/gold_mini_eval_plan.md"
# Amendment 2 (A1): the accepted NOTLD token table (docs/sound_refresh_2026-10-15.md); any other table is refused. Not a CLI option.
EXPECTED_TOKENS_SHA256 = "0e818c9d48546fc087609c539e9b9dd007b74100fd58b467c42f1350eaee9fb7"


class Refused(Exception):
    pass


def wilson(k: int, n: int):
    if n == 0:
        return None
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def slot_pred(value: str):
    """(value or None when missing, starred)."""
    starred = value.endswith("*")
    v = value[:-1] if starred else value
    return (None if v in (T.v0.MISSING, T.v0.NOT_AVAILABLE) else v), starred


def move_pred(value: str) -> tuple[frozenset, bool]:
    """(atom set, missing): `?`/`n/a` is the empty set."""
    v = value.rstrip("*")
    if v in (T.v0.MISSING, T.v0.NOT_AVAILABLE, ""):
        return frozenset(), True
    return frozenset(v.split("+")), False


def predictions(tokens: pd.DataFrame, shot_ids: list[int]) -> dict[int, dict[str, str]]:
    df = tokens.set_index("shot_id")
    missing = [s for s in shot_ids if s not in df.index]
    if missing:
        raise Refused(f"token table has no row for shots {missing}")
    return {s: T.parse_row(df.loc[s, "token_row"]) for s in shot_ids}


def acc_block(pairs, values):
    """pairs: list of (label or None, prediction slot string). Accuracy over non-null labels; starred/unstarred split; confusion; missing predictions."""
    out = {"n_labels": len(pairs), "n_cant_tell": sum(lab is None for lab, _ in pairs)}
    rows = [(lab, *slot_pred(p)) for lab, p in pairs if lab is not None]
    n = len(rows)
    k = sum(pred == lab for lab, pred, _ in rows)
    out.update(n_scored=n, n_correct=k, accuracy=(k / n if n else None), wilson95=wilson(k, n),
               n_pred_missing=sum(pred is None for _, pred, _ in rows))
    for name, sel in (("starred", True), ("unstarred", False)):
        sub = [(lab, pred) for lab, pred, st in rows if st is sel]
        ks = sum(pred == lab for lab, pred in sub)
        out[name] = {"n": len(sub), "n_correct": ks, "accuracy": (ks / len(sub) if sub else None), "wilson95": wilson(ks, len(sub))}
    conf: dict = {}
    for lab, pred, _ in rows:
        conf.setdefault(lab, {}).setdefault(pred if pred is not None else "?", 0)
        conf[lab][pred if pred is not None else "?"] += 1
    out["confusion_label_to_pred"] = conf
    if values is not None:  # ordinal: share within one step (descriptive)
        w = sum(pred is not None and pred in values and abs(values.index(pred) - values.index(lab)) <= 1 for lab, pred, _ in rows)
        out["within_one_step"] = {"n": w, "share": (w / n if n else None)}
    return out


def ci(boot: np.ndarray):
    ok = boot[~np.isnan(boot)]
    return ([float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))] if ok.size else None), int(boot.size - ok.size)


def move_block(pairs, atoms):
    """pairs: list of (label atom list or None, prediction slot string)."""
    out = {"n_labels": len(pairs), "n_cant_tell": sum(lab is None for lab, _ in pairs)}
    rows = [(frozenset(lab), *move_pred(p)) for lab, p in pairs if lab is not None]
    out["n_pred_missing"] = sum(miss for _, _, miss in rows)
    out["n_pred_static_plus_direction"] = sum("static" in move_pred(p)[0] and len(move_pred(p)[0]) > 1 for _, p in pairs)
    n = len(rows)
    out["n_scored"] = n
    if n == 0:
        return out
    idx = np.random.default_rng(GM.SEED).integers(0, n, size=(N_BOOT, n))
    exact = np.array([lab == pred for lab, pred, _ in rows], float)
    jac = np.array([len(lab & pred) / len(lab | pred) for lab, pred, _ in rows], float)
    for name, arr in (("exact_set_match", exact), ("jaccard_mean", jac)):
        interval, _ = ci(arr[idx].mean(axis=1))
        out[name] = {"value": float(arr.mean()), "bootstrap95": interval}
    per, few = {}, {}
    for a in atoms:
        y = np.array([a in lab for lab, _, _ in rows])
        p = np.array([a in pred for _, pred, _ in rows])
        pos = int(y.sum())
        if pos < MIN_POS:
            few[a] = {"n_pos_labels": pos, "n_pos_pred": int(p.sum())}
            continue
        tp_b, p_b, y_b = (y & p)[idx].sum(1), p[idx].sum(1), y[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prec_b = np.where(p_b > 0, tp_b / np.maximum(p_b, 1), np.nan)
            rec_b = np.where(y_b > 0, tp_b / np.maximum(y_b, 1), np.nan)
        tp = int((y & p).sum())
        pi, pd_ = ci(prec_b)
        ri, rd = ci(rec_b)
        per[a] = {"n_pos_labels": pos, "n_pos_pred": int(p.sum()), "tp": tp,
                  "precision": (tp / int(p.sum()) if p.sum() else None), "precision_bootstrap95": pi, "precision_dropped_resamples": pd_,
                  "recall": tp / pos, "recall_bootstrap95": ri, "recall_dropped_resamples": rd}
    out["per_atom"] = per
    out["atoms_below_min_positives"] = few
    return out


def score(records: list[dict], preds: dict[int, dict[str, str]], voc: dict) -> dict:
    """The pre-registered metrics. records: validated label records; preds: shot_id -> parsed token row."""
    return {"size": acc_block([(r["size"], preds[r["shot_id"]]["size"]) for r in records], voc["size"]),
            # Amendment 1: PRIMARY over every definite angle label (an eye-level label is a model error: the model has no eye-level class);
            # SECONDARY excludes eye-level labels
            "angle": acc_block([(r["angle"], preds[r["shot_id"]]["angle"]) for r in records], None)
            | {"n_eye_level": sum(r["angle"] == GM.EYE_LEVEL for r in records)},
            "angle_excl_eye_level": acc_block([(r["angle"], preds[r["shot_id"]]["angle"]) for r in records if r["angle"] != GM.EYE_LEVEL], None),
            "movement": move_block([(r["movement"], preds[r["shot_id"]]["move"]) for r in records], voc["movement"])}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt_iv(iv):
    return "n/a" if iv is None else f"[{100 * iv[0]:.1f}, {100 * iv[1]:.1f}]"


def fmt_p(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def to_markdown(res: dict) -> str:
    L = [f"# Gold mini evaluation: score v1{' (PARTIAL: NOT THE RESULT)' if res['partial'] else ''}",
         f"Pre-registered in `{PLAN}`. {res['n_records']} of {res['n_selected']} labeled NOTLD gold shots; predictions = slot values of `token_row` in the token table.",
         "A 60-shot descriptive check of one film, not a benchmark result. Nothing was tuned on these labels.", "",
         "| field | scored | correct | accuracy | 95% interval | can't tell | starred preds (acc) | missing preds |", "|---|---|---|---|---|---|---|---|"]
    for f, title in (("size", "size"), ("angle", "angle (PRIMARY: eye-level labels count as errors)"),
                     ("angle_excl_eye_level", "angle (secondary: eye-level labels excluded)")):
        b = res["metrics"][f]
        L.append(f"| {title} | {b['n_scored']} | {b['n_correct']} | {fmt_p(b['accuracy'])} | {fmt_iv(b['wilson95'])} (Wilson) | {b['n_cant_tell']} | "
                 f"{b['starred']['n']} ({fmt_p(b['starred']['accuracy'])}) | {b['n_pred_missing']} |")
    m = res["metrics"]["movement"]
    L += ["", (f"Size within one ordinal step: {fmt_p(res['metrics']['size']['within_one_step']['share'])}. "
               f"Eye-level angle labels: {res['metrics']['angle']['n_eye_level']} (the model has no eye-level class)."), "",
          (f"Movement: {m['n_scored']} scored, {m['n_cant_tell']} can't tell, {m['n_pred_missing']} missing predictions (of the {m['n_scored']} scored), "
           f"{m['n_pred_static_plus_direction']} predictions combine static with a direction atom (of all {m['n_labels']} labeled shots, "
           "including can't tell).")]
    if m["n_scored"]:
        L += [f"- exact-set match {fmt_p(m['exact_set_match']['value'])} {fmt_iv(m['exact_set_match']['bootstrap95'])} (bootstrap)",
              f"- mean Jaccard {m['jaccard_mean']['value']:.3f} [{m['jaccard_mean']['bootstrap95'][0]:.3f}, {m['jaccard_mean']['bootstrap95'][1]:.3f}] (bootstrap)", "",
              "| atom | label positives | predicted | precision (95%) | recall (95%) |", "|---|---|---|---|---|"]
        for a, b in m["per_atom"].items():
            L.append(f"| {a} | {b['n_pos_labels']} | {b['n_pos_pred']} | {fmt_p(b['precision'])} {fmt_iv(b['precision_bootstrap95'])} | "
                     f"{fmt_p(b['recall'])} {fmt_iv(b['recall_bootstrap95'])} |")
        if m["atoms_below_min_positives"]:
            L += ["", "Atoms with fewer than 5 positive labels (counts only): " +
                  ", ".join(f"{a} {b['n_pos_labels']}/{b['n_pos_pred']}" for a, b in m["atoms_below_min_positives"].items()) + " (labels/predicted)."]
    L += ["", "Denominators (Amendment 2, A3):",
          "- scored = labeled shots whose label for the field is not can't tell (angle secondary: also not eye-level); accuracy = correct / scored;",
          "  the Wilson interval is over scored. Starred preds and missing preds are counted among the scored shots; starred accuracy = correct starred /",
          "  starred. Can't tell is counted among all labeled shots (angle secondary: among the non-eye-level ones). Within one step is over size-scored shots.",
          "- movement: exact-set match, mean Jaccard and per-atom precision/recall are over the movement-scored shots; precision = TP / scored shots",
          "  predicting the atom, recall = TP / scored shots labeled with it; bootstrap intervals resample the scored shots.",
          "- eye-level count: among all labeled shots.", "",
          f"Token table sha256 `{res['inputs_sha256']['tokens']}` (pinned `{res['tokens_sha256_expected']}`).",
          "Inputs (sha256): " + ", ".join(f"{k} `{v[:12]}`" for k, v in res["inputs_sha256"].items()), ""]
    return "\n".join(L)


def run(root: Path, labels_path: Path, tokens_path: Path, out_dir: Path, partial: bool) -> Path:
    sel = GM.load_selection(root)
    order = [s["shot_id"] for s in sel]
    if not Path(labels_path).exists():
        raise Refused(f"no label file at {labels_path}")
    voc = GM.vocab(root)
    try:
        labels = {k: GM.validate(v, voc) for k, v in GM.read_labels(labels_path).items()}
    except (ValueError, KeyError) as e:
        raise Refused(f"invalid label file: {e}") from e
    extra = sorted(set(labels) - set(order))
    if extra:
        raise Refused(f"labels for shots outside the pre-registered selection: {extra}")
    if len(labels) != GM.N_SHOTS and not partial:
        raise Refused(f"{len(labels)} of {GM.N_SHOTS} shots labeled: the scorer runs only on the complete set (or --partial, which is not the result)")
    records = [labels[s] for s in order if s in labels]
    if not records:
        raise Refused("no label records")
    name = "score_v1_PARTIAL" if partial and len(labels) != GM.N_SHOTS else "score_v1"
    out_dir = Path(out_dir)
    if (out_dir / f"{name}.json").exists() or (out_dir / f"{name}.md").exists():
        raise Refused(f"{out_dir / name}.* exists: the sealed score is written once")
    if not Path(tokens_path).is_file():
        raise Refused(f"no token table at {tokens_path}")
    tokens_sha = sha(tokens_path)
    if tokens_sha != EXPECTED_TOKENS_SHA256:
        raise Refused(f"token table sha256 {tokens_sha} != pinned {EXPECTED_TOKENS_SHA256} (Amendment 2)")
    preds = predictions(pd.read_parquet(tokens_path, columns=["shot_id", "token_row"]), [r["shot_id"] for r in records])
    res = {"plan": PLAN, "partial": name != "score_v1", "n_records": len(records), "n_selected": GM.N_SHOTS, "film": GM.FILM,
           "selection_shot_ids": order, "scored_shot_ids": [r["shot_id"] for r in records],
           "tokens_sha256_expected": EXPECTED_TOKENS_SHA256,
           "inputs_sha256": {"labels": sha(labels_path), "tokens": tokens_sha, "gold_shots": sha(Path(root) / "data/gold/shots.json")},
           "bootstrap": {"n": N_BOOT, "seed": GM.SEED, "percentiles": [2.5, 97.5]}, "wilson_z": Z, "min_positives_per_atom": MIN_POS,
           "metrics": score(records, preds, voc)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    (out_dir / f"{name}.md").write_text(to_markdown(res))
    return out_dir / f"{name}.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT, help="checkout holding data/gold/shots.json and configs/schema.yaml")
    ap.add_argument("--labels", type=Path, default=None, help=f"default <root>/{GM.LABELS_REL}")
    ap.add_argument("--tokens", type=Path, default=None, help=f"default <root>/data/processed/{GM.FILM}_tokens_v1.parquet")
    ap.add_argument("--out-dir", type=Path, default=None, help="default <root>/results/gold_mini")
    ap.add_argument("--partial", action="store_true", help="score an incomplete label file; output is marked PARTIAL and is not the result")
    a = ap.parse_args(argv)
    labels = a.labels or a.root / GM.LABELS_REL
    tokens = a.tokens or a.root / f"data/processed/{GM.FILM}_tokens_v1.parquet"
    try:
        out = run(a.root, labels, tokens, a.out_dir or a.root / "results/gold_mini", a.partial)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"wrote {out} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
