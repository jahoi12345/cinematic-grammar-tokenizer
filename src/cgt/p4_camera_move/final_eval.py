"""Pure CPU logic of the SEALED Phase 4 final evaluations (docs/phase4_interface.md section 10, docs/phase4_video_plan.md Amendments 2 item 4-5 and 9): the six pre-registered
evaluations are {geometry, video, combined} x {ShotBench movement, CameraBench public test}. Nothing here reads a label file, loads a model or touches a ledger; every function is a pure
function of its arguments (the stores, the ledger and the sealed protocol live in `final_stores`, `final_seal` and `final_run`).
Contents: the registered model tags and the go-file text; the generated four-option CameraBench questions (exact distractor algorithm, fixed before any label is read); option scoring
(geometry set likelihood, video letter logits, their combination with the frozen weights), the tie rule; the metrics (option accuracy with Wilson intervals, single/compound strata,
atom-level precision/recall/F1 and AP, confusions, label-prior baseline, source-disjoint subset), the paired bootstrap by source video; the transductive recalibration variant."""
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cgt.p3b_vlm import guards
from cgt.p4_camera_move.baseline import (
    PAIRS,
    ap_score,
    option_scores,
    prf,
    wilson,
)
from cgt.p4_camera_move.camerabench import (
    PRIMITIVE_TO_ATOM,
    STATIC_VARIANT,
    source_id_of,
)
from cgt.p4_camera_move.labels import parse_move_label, schema_moves

LETTERS = "ABCD"
SEED = 0
CHANCE = 0.25
DATASETS = ("shotbench_movement", "camerabench_public_test")
MODELS = ("geometry", "video", "combined")
MODEL_TAGS = {"geometry": "move_geom_flowdepth_v1", "video": "move_vlm_q3vl2b_lora_v1", "combined": "move_combined_v1"}
# mapping variants of the CameraBench primitives (docs/phase4_video_plan.md Amendment 2 item 4): the main mapping does NOT turn `no-motion` into a static atom; the sensitivity variant does
MAPPINGS = {"main": PRIMITIVE_TO_ATOM, "no_motion_as_static": STATIC_VARIANT}
OPTION_WORDING = {"push-in": "Push in", "pull-out": "Pull out", "boom-up": "Boom up", "boom-down": "Boom down", "pan-left": "Pan left", "pan-right": "Pan right", "tilt-up": "Tilt up",
                  "tilt-down": "Tilt down", "zoom-in": "Zoom in", "zoom-out": "Zoom out", "move-left": "Move to the left", "move-right": "Move to the right", "arc": "Arc",
                  "roll": "Camera roll", "static": "Static shot"}
GO_HEADER = "PHASE4_FINAL_GO"


def key_of(dataset: str, tag: str) -> str:
    """The name of one evaluation in every state and result file: `<dataset>__<model tag>`."""
    return f"{dataset}__{tag}"


# ---------------------------------------------------------------------------------------------------- go file and registry
def go_text(dataset: str, tag: str, adapter_sha256: str | None, config_hash: str | None, freeze_sha256: str, geometry_inputs: str | None = None) -> str:
    """The exact content of the author's go-ahead file for ONE evaluation: bound to the dataset, the model tag, the adapter hash, the configuration hash, the SHA-256 of the committed freeze record the independent verification
    verified and, for geometry and combined, `geometry_inputs` (the committed geometry configuration bytes, the split manifest and the training feature tables; Amendment 12 item 3). A different dataset,
    tag, hash, freeze record or geometry input authorizes nothing."""
    return (f"{GO_HEADER}\ndataset={dataset}\ntag={tag}\nadapter_sha256={adapter_sha256 or '-'}\nconfig_hash={config_hash or '-'}\nfreeze_sha256={freeze_sha256}\n"
            f"geometry_inputs={geometry_inputs or '-'}\n")


def require_go(path, expected: str) -> None:
    """NotAuthorized (exit 13) unless the go file exists and its content equals `expected` exactly."""
    p = Path(path)
    if not p.is_file():
        raise guards.NotAuthorized(f"go-ahead file missing: {p} (the operator writes it with scripts/p4_write_go.py after the freeze is verified)")
    if p.read_text() != expected:
        raise guards.NotAuthorized(f"go-ahead file {p} is not bound to this dataset, tag, hashes and committed freeze record")


def combined_config_hash(epoch: int, w_s: float, w_g: float, adapter_sha256: str) -> str:
    """The configuration hash registered for `move_combined_v1` (same formula as scripts/p4_video_freeze.py)."""
    return hashlib.sha256(json.dumps({"E": epoch, "w_s": w_s, "w_g": w_g, "adapter_sha256": adapter_sha256}, sort_keys=True).encode()).hexdigest()[:16]


def freeze_digest(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


# ---------------------------------------------------------------------------------------------------- options and CameraBench questions
def _order() -> dict[str, int]:
    return {a: i for i, a in enumerate(schema_moves())}


def sort_atoms(atoms) -> tuple[str, ...]:
    o = _order()
    return tuple(sorted(set(atoms), key=lambda a: o[a]))


def option_text(atoms) -> str:
    """The ShotQA option wording of an atom set: the wordings joined by ' and ' in schema order (docs/phase4_video_plan.md Amendment 2 item 4)."""
    return " and ".join(OPTION_WORDING[a] for a in sort_atoms(atoms))


def gold_sets(names: list[str], labels_by_stem: dict[str, list[str]], mapping: dict[str, str]) -> list[tuple[str, ...] | None]:
    """Per GIF (in the order of `names`): the mapped atom set in schema order, or None when the clip has no label row or no mapped atom under `mapping`."""
    out = []
    for n in names:
        tags = labels_by_stem.get(n.rsplit(".", 1)[0])
        atoms = sorted({mapping[x] for x in (tags or []) if x in mapping})
        out.append(sort_atoms(atoms) if atoms else None)
    return out


def build_camerabench_questions(names: list[str], labels_by_stem: dict[str, list[str]], mapping: dict[str, str], seed: int = SEED) -> tuple[list[dict], dict]:
    """The generated four-option CameraBench questions, EXACT algorithm (fixed before any label is read; Amendment 9 spells out what Amendment 2 item 4 states):
    `names` is the sorted GIF file list, i = the clip's index in it. A clip with no mapped atom is excluded (counted). For clip i: gold = its atom set; pool = the gold sets of all OTHER clips
    that have one, in index order; `rng = numpy.random.default_rng([seed, i])`; the pool positions are visited in the order `rng.permutation(len(pool))` and a set is drawn unless it equals the
    gold set or was already drawn, until 3 are drawn (fewer than 3 distinct distractors: the question is dropped and counted); the four options [gold, d1, d2, d3] are placed on A-D by
    `rng.permutation(4)` of the same generator (option at position r of the result goes to letter r)."""
    sets = gold_sets(names, labels_by_stem, mapping)
    questions: list[dict] = []
    dropped = {"no_label_row": sum(1 for n in names if n.rsplit(".", 1)[0] not in labels_by_stem),
               "no_mapped_atom": sum(1 for n, s in zip(names, sets, strict=True) if s is None and n.rsplit(".", 1)[0] in labels_by_stem), "fewer_than_3_distinct_distractors": 0}
    for i, (name, g) in enumerate(zip(names, sets, strict=True)):
        if g is None:
            continue
        rng = np.random.default_rng([seed, i])
        pool = [sets[j] for j in range(len(names)) if j != i and sets[j] is not None]
        drawn: list[tuple[str, ...]] = []
        for k in rng.permutation(len(pool)):
            if pool[k] != g and pool[k] not in drawn:
                drawn.append(pool[k])
                if len(drawn) == 3:
                    break
        if len(drawn) < 3:
            dropped["fewer_than_3_distinct_distractors"] += 1
            continue
        opts = [g, *drawn]
        perm = rng.permutation(4)
        placed = [opts[int(r)] for r in perm]
        gold_letter = LETTERS[int(np.flatnonzero(perm == 0)[0])]
        questions.append({"id": f"cb{i}", "index": i, "clip": name, "source": source_id_of(name), "gold_letter": gold_letter, "gold_atoms": g,
                          "options": {L: option_text(s) for L, s in zip(LETTERS, placed, strict=True)}, "option_atoms": dict(zip(LETTERS, placed, strict=True))})
    dropped["n_questions"] = len(questions)
    return questions, dropped


def questions_digest(questions: list[dict], with_gold: bool = False) -> str:
    """SHA-256 over the question texts (id, clip, options in letter order; the gold letter only when `with_gold`): the same questions are scored by the three models, which the paired comparison checks."""
    rows = [[q["id"], q["clip"], [q["options"][L] for L in LETTERS], *([q["gold_letter"]] if with_gold else [])] for q in questions]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def parsed_option_atoms(options: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Schema atoms of every offered option text (flags and unmapped parts are ignored by the scoring, as in the baselines)."""
    return {k: parse_move_label(v).atoms for k, v in options.items()}


# ---------------------------------------------------------------------------------------------------- scoring
def lsm(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def geometry_option_scores(question: dict, p_atoms: np.ndarray, atoms: list[str]) -> np.ndarray:
    """Bernoulli set log-likelihood of each offered option under the clip's atom probabilities (atoms outside the TRAIN vocabulary are ignored), in letter order."""
    sc = option_scores(question, p_atoms, atoms)
    return np.array([sc[L] for L in LETTERS])


def combine(model: str, video_logits, geometry_ll, w_s: float, w_g: float) -> np.ndarray:
    """The option scores of a model: video = log-softmax of the four letter logits; geometry = the set log-likelihoods; combined = w_s * video + w_g * geometry (frozen weights)."""
    if model == "geometry":
        return np.asarray(geometry_ll, dtype=np.float64)
    if model == "video":
        return lsm(video_logits)
    if model == "combined":
        return w_s * lsm(video_logits) + w_g * np.asarray(geometry_ll, dtype=np.float64)
    raise ValueError(f"unknown model {model!r}")


def choose(scores4) -> str:
    """The predicted letter; TIE CONVENTION (as in the baselines): among equal scores the lowest letter wins."""
    s = np.asarray(scores4, dtype=np.float64)
    return max(LETTERS, key=lambda L: s[LETTERS.index(L)])


def fit_atom_probs(Xtr, Ytr, Xte, C: float, *, recalibrate: bool = False) -> np.ndarray:
    """Per-atom one-vs-rest logistic regression (balanced, as `baseline.fit_scores`) fitted on the TRAIN rows only; `recalibrate=False` standardises the test features with the TRAIN mean and standard
    deviation (the primary protocol); `recalibrate=True` is the declared transductive SECONDARY variant: the test features are imputed and standardised with the mean and standard deviation of the
    TEST FEATURES themselves (no label involved) and the train-fitted coefficients are applied. An atom with a single class in training scores its prior. With Xte == Xtr both variants agree."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr = sc.transform(imp.transform(Xtr))
    if recalibrate:
        imp_te = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xte)
        Zte = StandardScaler().fit(imp_te.transform(Xte)).transform(imp_te.transform(Xte))
    else:
        Zte = sc.transform(imp.transform(Xte))
    out = np.zeros((len(Xte), Ytr.shape[1]))
    for j in range(Ytr.shape[1]):
        if Ytr[:, j].sum() == 0 or Ytr[:, j].sum() == len(Ytr):
            out[:, j] = Ytr[:, j].mean()
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Ztr, Ytr[:, j])
        out[:, j] = clf.predict_proba(Zte)[:, 1]
    return out


# ---------------------------------------------------------------------------------------------------- metrics
def _hits(questions: list[dict], chosen: list[str]) -> np.ndarray:
    return np.array([c == q["gold_letter"] for q, c in zip(questions, chosen, strict=True)], dtype=bool)


def option_accuracy(questions: list[dict], chosen: list[str]) -> dict:
    """Option-restricted accuracy with the 95% Wilson interval (percent), strata by the size of the gold atom set (single / compound), exact-set accuracy (the chosen option's atom set equals the
    gold set; differs from the letter hit only where two offered options collapse to one atom set) and the number of questions with collapsed options."""
    h = _hits(questions, chosen)
    n = len(h)
    if n == 0:
        return {"n": 0}
    single = np.array([len(q["gold_atoms"]) <= 1 for q in questions])
    lo, hi = wilson(int(h.sum()), n)
    exact = np.array([set(q["option_atoms"][c]) == set(q["gold_atoms"]) for q, c in zip(questions, chosen, strict=True)])
    return {"n": n, "correct": int(h.sum()), "acc": round(float(h.mean()), 4), "ci95_pct": [round(lo, 1), round(hi, 1)], "chance": CHANCE, "n_single": int(single.sum()),
            "acc_single": round(float(h[single].mean()), 4) if single.any() else None, "n_compound": int((~single).sum()),
            "acc_compound": round(float(h[~single].mean()), 4) if (~single).any() else None, "exact_set_accuracy": round(float(exact.mean()), 4),
            "n_questions_with_collapsed_options": sum(len(set(q["option_atoms"].values())) < 4 for q in questions)}


def atom_level(questions: list[dict], chosen: list[str]) -> dict:
    """Atom-level precision / recall / F1 of the atom set of the CHOSEN option against the gold atom set, per atom (support > 0 only in the macro mean)."""
    vocab = sorted({a for q in questions for a in q["gold_atoms"]} | {a for q, c in zip(questions, chosen, strict=True) for a in q["option_atoms"][c]}, key=_order().get)
    rows = []
    for a in vocab:
        y = np.array([a in q["gold_atoms"] for q in questions], dtype=int)
        p = np.array([a in q["option_atoms"][c] for q, c in zip(questions, chosen, strict=True)], dtype=int)
        pr, rc, f1, tp, fp, fn = prf(y, p)
        rows.append({"atom": a, "support": int(y.sum()), "predicted": int(p.sum()), "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn})
    sup = [r for r in rows if r["support"] > 0]
    macro = {"atoms_with_support": len(sup), **({k: round(float(np.mean([r[k] for r in sup])), 3) for k in ("precision", "recall", "f1")} if sup else {})}
    return {"per_atom": rows, "macro": macro}


def confusions(questions: list[dict], chosen: list[str]) -> list[dict]:
    """Gold atom A (and not B), predicted atom B: the pairs of `baseline.PAIRS` (tilt vs boom, pan vs move, push/pull vs zoom, opposite directions), from the chosen option."""
    out = []
    for a, b in PAIRS:
        ga = np.array([a in q["gold_atoms"] and b not in q["gold_atoms"] for q in questions])
        pb = np.array([b in q["option_atoms"][c] for q, c in zip(questions, chosen, strict=True)])
        out.append({"gold": a, "predicted": b, "n_gold": int(sum(a in q["gold_atoms"] for q in questions)), "count": int((ga & pb).sum())})
    return out


def geometry_atom_metrics(gold_sets_: list[tuple[str, ...]], P: np.ndarray, atoms: list[str]) -> dict:
    """Atom-level AP and F1 at the fixed threshold 0.5 of the geometry model's atom probabilities against the gold atom sets (one row per clip; atoms without a positive are reported with ap None)."""
    rows = []
    for j, a in enumerate(atoms):
        y = np.array([a in s for s in gold_sets_], dtype=int)
        pr, rc, f1, *_ = prf(y, (P[:, j] >= 0.5).astype(int))
        rows.append({"atom": a, "support": int(y.sum()), "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3), "ap": round(ap_score(y, P[:, j]), 3) if y.sum() else None})
    sup = [r for r in rows if r["support"] > 0]
    return {"per_atom": rows, "macro": {"atoms_with_support": len(sup), **({k: round(float(np.mean([r[k] for r in sup])), 3) for k in ("precision", "recall", "f1", "ap")} if sup else {})},
            "threshold": 0.5}


def unseen_gold_atoms(questions: list[dict], atoms: list[str]) -> list[str]:
    """Gold atoms that have no TRAIN example (outside the geometry vocabulary): the geometry model cannot predict them (zoom, roll, ... on ShotQA)."""
    return sorted({a for q in questions for a in q["gold_atoms"]} - set(atoms))


def prior_option_accuracy(questions: list[dict], prior: np.ndarray, atoms: list[str]) -> dict:
    """Option accuracy of the label-prior baseline (the train prevalence of every atom as the atom probability of every clip)."""
    chosen = [choose(geometry_option_scores(q, prior, atoms)) for q in questions]
    return option_accuracy(questions, chosen)


def source_disjoint(questions: list[dict], reference_sources: set[str]) -> list[dict]:
    """The pre-declared secondary subset: questions whose filename-derived source id has no clip in `reference_sources` (the ShotQA pool)."""
    return [q for q in questions if q["source"] not in reference_sources]


def variant_report(questions: list[dict], chosen: list[str], *, geometry_probs: np.ndarray | None = None, atoms: list[str] | None = None, prior: np.ndarray | None = None) -> dict:
    """All metrics of one scored question set. `geometry_probs` (n_questions x n_atoms, row i = the clip of question i) adds the geometry model's atom AP/F1; `prior` adds the label-prior baseline."""
    rep = {"option_accuracy": option_accuracy(questions, chosen), "atom_level_chosen_option": atom_level(questions, chosen), "confusions": confusions(questions, chosen)}
    if atoms is not None:
        rep["gold_atoms_without_train_examples"] = unseen_gold_atoms(questions, atoms)
        if prior is not None:
            rep["label_prior_option_accuracy"] = prior_option_accuracy(questions, prior, atoms)
        if geometry_probs is not None:
            rep["geometry_atom_level"] = geometry_atom_metrics([q["gold_atoms"] for q in questions], geometry_probs, atoms)
    return rep


def paired_bootstrap(hits_a, hits_b, sources, n_boot: int = 1000, seed: int = 0) -> dict:
    """Accuracy difference a - b with a 95% percentile interval, resampling SOURCE VIDEOS (all questions of a source move together), on the SAME questions."""
    a, b, src = np.asarray(hits_a, dtype=float), np.asarray(hits_b, dtype=float), np.asarray(sources)
    usrc = sorted(set(src.tolist()))
    index = {s: np.where(src == s)[0] for s in usrc}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        ix = np.concatenate([index[s] for s in rng.choice(usrc, len(usrc), replace=True)])
        d.append(a[ix].mean() - b[ix].mean())
    return {"diff": round(float(a.mean() - b.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], "n_sources": len(usrc), "n_boot": n_boot}
