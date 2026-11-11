"""Baseline multi-label ATOM classifier with strict validation-label isolation (docs/phase4_interface.md section 4). CPU only.
Order enforced by `run`: (1) split from clip/source ids only; (2) parse TRAIN labels only; vocabulary, priors, majority set from TRAIN; (3) the regularisation C (and every
other choice) by source-grouped 5-fold CV on TRAIN sources only; (4) the configuration is FROZEN and written to disk; (5) a final-result guard and a look record are written;
(6) only then the validation labels are read (`final_evaluation=True`) and the validation metrics computed. Validation numbers are DESCRIPTIVE: feature sets are never selected on
them. Never reads ShotBench movement or CameraBench."""
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cgt.p4_camera_move.labels import MovementLabelStore, schema_moves

CS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
PAIRS = [("push-in", "zoom-in"), ("zoom-in", "push-in"), ("pull-out", "zoom-out"), ("zoom-out", "pull-out"), ("push-in", "pull-out"), ("pull-out", "push-in"),
         ("tilt-up", "boom-up"), ("boom-up", "tilt-up"), ("tilt-down", "boom-down"), ("boom-down", "tilt-down"), ("tilt-up", "tilt-down"), ("tilt-down", "tilt-up"),
         ("pan-left", "move-left"), ("move-left", "pan-left"), ("pan-right", "move-right"), ("move-right", "pan-right"), ("pan-left", "pan-right"), ("pan-right", "pan-left")]
# Validation passes (looks) that happened BEFORE the look log existed. Reconstructed from the session record, NOT from retained artifacts; declared for honesty, not proven.
HISTORICAL_LOOKS_DECLARED = {"flow": 4, "siglip": 1}


ROOT = Path(__file__).resolve().parents[3]


def code_sha() -> str:
    """HEAD commit of the repository, with `+dirty` if src or scripts have uncommitted changes (a record with `+dirty` cannot support clean-tree wording)."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return git("rev-parse", "HEAD") + ("+dirty" if git("status", "--porcelain", "--", "src", "scripts") else "")


def record_look(looks_path: Path, *, kind: str, tag: str, feature_sets: list[str], config_hash: str = "", code_sha_: str | None = None, overwrite: bool = False) -> dict:
    """Append ONE validation look to the ledger BEFORE any validation label is read (contract section 4: every invocation that reads validation labels is a look, final prediction passes and
    comparison/report scripts alike). Returns the record. `n_validation_looks_so_far` = declared historical (unverified) + logged lines + this one."""
    total, logged = _looks_so_far(looks_path, feature_sets)
    rec = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), "code_sha": code_sha_ or code_sha(), "kind": kind, "feature_set": "+".join(feature_sets), "tag": tag,
           "config_hash": config_hash, "n_validation_looks_so_far": total + 1, "n_logged_looks_so_far": logged + 1,
           "historical_looks_declared_unverified": HISTORICAL_LOOKS_DECLARED, "overwrite_of_prior_final": overwrite}
    Path(looks_path).parent.mkdir(parents=True, exist_ok=True)
    with open(looks_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


class FinalResultExists(RuntimeError):
    """A final result for this tag already exists; overwriting needs the explicit (and itself logged) allow_overwrite_final flag."""


def fit_scores(Xtr, Ytr, Xte, C):
    """Per-atom one-vs-rest logistic regression probabilities (balanced class weights); an atom with a single class in training scores its prior."""
    out = np.zeros((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        if Ytr[:, j].sum() == 0 or Ytr[:, j].sum() == len(Ytr):
            out[:, j] = Ytr[:, j].mean()
            continue
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
        m.fit(Xtr, Ytr[:, j])
        out[:, j] = m.predict_proba(Xte)[:, 1]
    return out


def prf(y, pred):
    tp, fp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn


def ap_score(y, s):
    return float(average_precision_score(y, s)) if y.sum() else float("nan")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def option_scores(row, p, atoms, prior_only: np.ndarray | None = None) -> dict[str, float]:
    """Bernoulli set likelihood of the SCHEMA atoms of each offered option (flags and unmapped parts ignored)."""
    ai = {a: i for i, a in enumerate(atoms)}
    pp = np.clip(prior_only if prior_only is not None else p, 1e-4, 1 - 1e-4)
    out = {}
    for letter, at in row["option_atoms"].items():
        t = np.zeros(len(atoms))
        for a in at:
            if a in ai:
                t[ai[a]] = 1
        out[letter] = float((t * np.log(pp) + (1 - t) * np.log(1 - pp)).sum())
    return out


def option_hits(rows, P, atoms, prior=None):
    """Hit per question; TIE CONVENTION: among equal scores the lowest letter (A before B ...) wins."""
    return np.array([max(sorted(sc), key=lambda k: sc[k]) == r["gold_letter"]
                     for r, p in zip(rows, P, strict=True) for sc in [option_scores(r, p, atoms, prior)]])


def n_collapsed_options(rows) -> int:
    """Questions in which at least two offered options have the identical schema atom set (their scores tie exactly)."""
    return sum(len(set(r["option_atoms"].values())) < len(r["option_atoms"]) for r in rows)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _looks_so_far(looks_path: Path, feature_sets: list[str]) -> tuple[int, int]:
    """(declared historical looks + logged looks overall, logged looks overall)."""
    logged = sum(1 for line in Path(looks_path).read_text().splitlines() if line.strip()) if Path(looks_path).exists() else 0
    return sum(HISTORICAL_LOOKS_DECLARED.values()) + logged, logged


def run(*, feats: pd.DataFrame, manifest: dict, store: MovementLabelStore, feature_sets: list[str], tag: str, out_dir: Path, looks_path: Path, code_sha: str,
        allow_overwrite_final: bool = False, events: list | None = None, cs: list[float] | None = None) -> dict:
    cs = cs or CS
    ev = events if events is not None else store.events
    out_dir = Path(out_dir)
    train_clips, val_clips = list(manifest["clips"]["train"]), list(manifest["clips"]["val"])
    final_path = out_dir / f"baseline_atoms_{tag}.json"
    if final_path.exists() and not allow_overwrite_final:  # BEFORE any label is parsed and before the frozen config is written (a refused run changes nothing)
        raise FinalResultExists(f"{final_path.name} exists: pass allow_overwrite_final (logged) to replace a prior final result")
    # ---- (2) TRAIN labels only
    tr_all = store.rows(train_clips)
    tr = [r for r in tr_all if r["atoms"]]  # clips with no schema atom (tracking/rack only) are excluded in v1; for validation the same rule is applied at the final step
    atoms = [a for a in schema_moves() if any(a in r["atoms"] for r in tr)]
    Xtr = feats.loc[[r["clip"] for r in tr]].to_numpy(dtype=np.float64)
    Ytr = np.array([[a in r["atoms"] for a in atoms] for r in tr], dtype=int)
    gtr = np.array([r["source"] for r in tr])
    # ---- (3) selection by source-grouped CV on TRAIN sources only
    elig = [j for j in range(len(atoms)) if Ytr[:, j].sum() >= 8]
    cv, oof_by_c = {}, {}
    for C in cs:
        oof = np.zeros(Ytr.shape)
        for a_idx, b_idx in GroupKFold(n_splits=5).split(Xtr, groups=gtr):
            oof[b_idx] = fit_scores(Xtr[a_idx], Ytr[a_idx], Xtr[b_idx], C)
        oof_by_c[C] = oof
        cv[C] = float(np.nanmean([ap_score(Ytr[:, j], oof[:, j]) for j in elig]))
    best_c = max(cv, key=cv.get)
    oof = oof_by_c[best_c]
    prior = Ytr.mean(0)
    maj_set = pd.Series(map(tuple, Ytr)).value_counts().index[0]
    # ---- (4) FREEZE
    cfg = {"tag": tag, "feature_sets": feature_sets, "n_features": int(Xtr.shape[1]), "atoms": atoms, "chosen_C": best_c, "c_grid": cs, "cv_macro_ap_by_C": {str(k): round(v, 4) for k, v in cv.items()},
           "cv_macro_ap_at_chosen_C": round(cv[best_c], 4), "threshold": 0.5, "majority_set": [atoms[j] for j in range(len(atoms)) if maj_set[j]], "n_train_clips": len(tr),
           "n_train_excluded_no_schema_atom": len(tr_all) - len(tr), "manifest_sha256": manifest["sha256"], "code_sha": code_sha}
    cfg["config_hash"] = config_hash({k: v for k, v in cfg.items() if k != "code_sha"})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"frozen_config_{tag}.json").write_text(json.dumps(cfg, indent=1))
    ev.append(("freeze", tag))
    # model fitted on TRAIN with the frozen C; probabilities for the features of ALL validation clips (no label is involved)
    Pva_all = fit_scores(Xtr, Ytr, feats.loc[val_clips].to_numpy(dtype=np.float64), best_c)
    # ---- (5) final-result guard and look record, BEFORE any validation label is read
    total_looks = record_look(looks_path, kind="baseline_final", tag=tag, feature_sets=feature_sets, config_hash=cfg["config_hash"], code_sha_=code_sha, overwrite=final_path.exists())["n_validation_looks_so_far"]
    # ---- (6) FINAL EVALUATION: validation labels are read here and nowhere else
    va_all = store.rows(val_clips, final_evaluation=True)
    keep = [i for i, r in enumerate(va_all) if r["atoms"]]
    va = [va_all[i] for i in keep]
    Pva = Pva_all[keep]
    unseen = sorted({a for r in va for a in r["atoms"]} - set(atoms))
    Yva = np.array([[a in r["atoms"] for a in atoms] for r in va], dtype=int)
    res: dict = {"config": cfg, "atoms": atoms, "atoms_without_examples": [a for a in schema_moves() if a not in atoms], "validation_atoms_unseen_in_train": unseen,
                 "n_excluded_no_schema_atom": {"train": len(tr_all) - len(tr), "val": len(va_all) - len(va)}, "n_train_clips": len(tr), "n_val_clips": len(va), "n_features": int(Xtr.shape[1]),
                 "cv_macro_ap_by_C": cfg["cv_macro_ap_by_C"], "chosen_C": best_c, "manifest_sha256": manifest["sha256"], "feature_sets": feature_sets, "tag": tag,
                 "c_grid": cs, "majority_set": cfg["majority_set"], "n_validation_looks_so_far": total_looks}
    table = []
    for j, a in enumerate(atoms):
        y = Yva[:, j]
        p_, r_, f_, tp, fp, fn = prf(y, (Pva[:, j] >= 0.5).astype(int))
        mp, mr, mf, *_ = prf(y, np.full(len(y), maj_set[j]))
        table.append({"atom": a, "val_support": int(y.sum()), "train_support": int(Ytr[:, j].sum()), "precision": round(p_, 3), "recall": round(r_, 3), "f1": round(f_, 3), "tp": tp, "fp": fp, "fn": fn,
                      "ap": round(ap_score(y, Pva[:, j]), 3) if y.sum() else None, "prior_ap": round(ap_score(y, np.full(len(y), prior[j])), 3) if y.sum() else None,
                      "majority_precision": round(mp, 3), "majority_recall": round(mr, 3), "majority_f1": round(mf, 3), "prevalence_val": round(float(y.mean()), 3)})
    res["validation_per_atom"] = table
    sup = [t for t in table if t["val_support"] > 0]
    res["validation_macro"] = {"atoms_with_support": len(sup), **{k: round(float(np.mean([t[k] for t in sup])), 3) for k in ("precision", "recall", "f1", "ap", "prior_ap", "majority_f1")}}
    res["validation_exact_set_accuracy"] = {"model_thr0.5": round(float(((Pva >= 0.5).astype(int) == Yva).all(1).mean()), 3), "majority_set": round(float((Yva == np.array(maj_set)).all(1).mean()), 3), "n": len(va)}
    dev = []
    for j, a in enumerate(atoms):
        p_, r_, f_, *_ = prf(Ytr[:, j], (oof[:, j] >= 0.5).astype(int))
        dev.append({"atom": a, "support": int(Ytr[:, j].sum()), "precision": round(p_, 3), "recall": round(r_, 3), "f1": round(f_, 3), "ap": round(ap_score(Ytr[:, j], oof[:, j]), 3), "prior_ap": round(float(Ytr[:, j].mean()), 3)})
    res["train_oof_per_atom"] = dev
    h_m, h_p = option_hits(va, Pva, atoms), option_hits(va, Pva, atoms, prior)
    single = np.array([r["kind"] == "single" for r in va])
    res["validation_option_accuracy"] = {"n": len(va), "chance_pct": 25.0, "tie_convention": "equal scores: lowest letter wins", "n_questions_with_options_collapsing_to_the_same_atom_set": n_collapsed_options(va)}
    for name, h in (("model", h_m), ("label_prior", h_p)):
        lo, hi = wilson(int(h.sum()), len(h))
        res["validation_option_accuracy"][name] = {"acc_pct": round(100 * float(h.mean()), 1), "ci95": [round(lo, 1), round(hi, 1)], "correct": int(h.sum()),
                                                   "single_label_acc_pct": round(100 * float(h[single].mean()), 1), "n_single": int(single.sum()),
                                                   "compound_acc_pct": round(100 * float(h[~single].mean()), 1), "n_compound": int((~single).sum())}
    ai = {a: i for i, a in enumerate(atoms)}

    def conf(Yg, Pp):
        out = []
        for a, b in PAIRS:
            if a in ai and b in ai:
                ga, gb, pb = Yg[:, ai[a]], Yg[:, ai[b]], (Pp[:, ai[b]] >= 0.5)
                out.append({"gold": a, "predicted": b, "n_gold": int(ga.sum()), "count": int(((ga == 1) & (gb == 0) & pb).sum()),
                            "share_of_gold_pct": round(100 * float(((ga == 1) & (gb == 0) & pb).sum() / max(1, ga.sum())), 1)})
        return out
    res["confusions_validation"], res["confusions_train_oof"] = conf(Yva, Pva), conf(Ytr, oof)
    final_path.write_text(json.dumps(res, indent=1))
    (out_dir / f"val_probs_{tag}.json").write_text(json.dumps({"atoms": atoms, "clips": [r["clip"] for r in va], "probs": np.round(Pva, 5).tolist()}))
    (out_dir / f"baseline_atoms_{tag}.md").write_text(render_markdown(res, len(va)))
    return res


def render_markdown(res: dict, n_val: int) -> str:
    names, tag, m = res["feature_sets"], res["tag"], res["validation_macro"]
    cv = res["cv_macro_ap_by_C"]
    L = [f"# Phase 4 baseline `{tag}`: atom classifier on feature sets {names} (CPU, ShotQA source-grouped split)", "",
         (f"Features: {res['n_features']} per clip (sets: {', '.join(names)}). Split: `results/phase4/split_manifest.json` (sha256 `{res['manifest_sha256'][:16]}`), {n_val} validation clips of 19 validation sources and "
         f"{res['n_train_clips']} train clips (clips with only tracking/rack labels excluded: {res['n_excluded_no_schema_atom']}). Model: per-atom logistic regression (balanced), threshold 0.5 (not tuned)."), "",
         "**Procedure and what the numbers are.** Labels, vocabulary, priors and the majority set come from TRAIN clips only; C = " + f"{res['chosen_C']}" + " was chosen by source-grouped 5-fold CV on TRAIN sources only "
         "(macro AP by C: " + ", ".join(f"C={k}: {v:.3f}" for k, v in cv.items()) + f"; at the chosen C: {res['config']['cv_macro_ap_at_chosen_C']:.3f}); the configuration (`frozen_config_{tag}.json`, config hash "
         f"`{res['config']['config_hash']}`) was written to disk before the validation labels were read, and the script computes ONE held-out prediction pass per invocation. "
         f"Validation looks so far (all feature sets, including {sum(HISTORICAL_LOOKS_DECLARED.values())} declared from the session record before the look log existed, not proven by retained artifacts): "
         f"{res['n_validation_looks_so_far']}; see `results/phase4/validation_looks.jsonl`. **The validation numbers below are descriptive; feature sets are selected by the train-source CV macro AP only, never by them.** "
         "The retained artifacts do not independently establish the historical number of validation passes of earlier runs.", "",
         "## Validation (held-out sources), per atom", "", "| atom | val support | precision | recall | F1 | AP | prior AP | majority-set F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in res["validation_per_atom"]:
        apc = "-" if t["ap"] is None else f"{t['ap']:.2f}"
        pac = "-" if t["prior_ap"] is None else f"{t['prior_ap']:.2f}"
        L.append(f"| {t['atom']} | {t['val_support']} | {t['precision']:.2f} | {t['recall']:.2f} | {t['f1']:.2f} | {apc} | {pac} | {t['majority_f1']:.2f} |")
    es = res["validation_exact_set_accuracy"]
    L += [f"| **macro over {m['atoms_with_support']} atoms with support** | | {m['precision']:.2f} | {m['recall']:.2f} | **{m['f1']:.2f}** | **{m['ap']:.2f}** | {m['prior_ap']:.2f} | {m['majority_f1']:.2f} |", "",
          (f"Majority baseline (the most frequent training atom set `{'+'.join(res['majority_set'])}` for every clip) and label-prior baseline (train prevalence as the score; at threshold 0.5 it predicts nothing, "
           f"so its AP equals the validation prevalence). **Exact atom-set accuracy on validation: model {es['model_thr0.5']:.3f}** (majority set {es['majority_set']:.3f}, n={es['n']}); "
           f"atoms without any training example: {', '.join(res['atoms_without_examples']) or 'none'}; validation atoms unseen in train: {', '.join(res['validation_atoms_unseen_in_train']) or 'none'}."), "",
          "## Option-restricted accuracy on the ShotQA validation questions (4-way MC, atom-set likelihood; NOT ShotBench)", "",
          "| scorer | accuracy % (95% Wilson) | correct/n | single-label answers | compound answers |", "|---|---|---|---|---|"]
    oa = res["validation_option_accuracy"]
    for name in ("model", "label_prior"):
        o = oa[name]
        L.append(f"| {name} | {o['acc_pct']} ({o['ci95'][0]}-{o['ci95'][1]}) | {o['correct']}/{oa['n']} | {o['single_label_acc_pct']} (n={o['n_single']}) | {o['compound_acc_pct']} (n={o['n_compound']}) |")
    L += ["| chance | 25.0 | | | |", "",
          (f"Scoring convention: each option is scored on its schema atoms only (flags and unmapped parts ignored) by the Bernoulli set likelihood; **ties go to the lowest letter**. "
           f"{oa['n_questions_with_options_collapsing_to_the_same_atom_set']} of {oa['n']} validation questions have at least two offered options that collapse to the same atom set (exact ties). "
           f"The option-restricted accuracy is a four-option restricted-choice figure on ShotQA validation questions; **it must not be read as unrestricted movement-labeling accuracy** "
           f"(the exact atom-set accuracy of {es['model_thr0.5']:.1%} is the unrestricted analogue) **and it cannot be compared with the paper's ShotBench movement figures** (different data, options and protocol)."), "",
          "## Confusions that matter: clips whose gold has atom A, whose prediction has atom B and whose gold lacks B", "",
          "| gold A -> predicted B | validation: n gold A | count | % of gold A | train out-of-fold: n gold A | count | % of gold A |", "|---|---:|---:|---:|---:|---:|---:|"]
    for cv_, ct in zip(res["confusions_validation"], res["confusions_train_oof"], strict=True):
        L.append(f"| {cv_['gold']} -> {cv_['predicted']} | {cv_['n_gold']} | {cv_['count']} | {cv_['share_of_gold_pct']} | {ct['n_gold']} | {ct['count']} | {ct['share_of_gold_pct']} |")
    L += ["", "zoom-in / zoom-out have 3 training clips each and none in validation: the push-vs-zoom rows say nothing about zoom recall. The out-of-fold columns use the TRAIN sources only.", "",
          "## Development estimate (train sources, source-grouped 5-fold out-of-fold, chosen C)", "", "| atom | support | precision | recall | F1 | AP | prevalence (prior AP) |", "|---|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {d['atom']} | {d['support']} | {d['precision']:.2f} | {d['recall']:.2f} | {d['f1']:.2f} | {d['ap']:.2f} | {d['prior_ap']:.2f} |" for d in res["train_oof_per_atom"]]
    return "\n".join(L) + "\n"
