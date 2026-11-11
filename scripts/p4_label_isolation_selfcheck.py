"""CPU self-check of validation-label isolation (Phase 4 review F2) on SYNTHETIC data: a grpo-style file whose VALIDATION clip carries a planted marker in its gold label.
Drives the real `cgt.p4_camera_move.baseline.run` and proves (a) no validation label is parsed before the configuration is frozen and the look is recorded, (b) the locked store
raises when validation rows are requested without final_evaluation=True, (c) the metadata loader parses no label, (d) the selection (C, CV table) is identical whether or not the
validation labels are readable, (e) a prior final result is not overwritten without the logged flag. Writes results/phase4/label_isolation_selfcheck.md.
Usage: uv run python scripts/p4_label_isolation_selfcheck.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.p4_camera_move import baseline as B
from cgt.p4_camera_move import labels as L
from cgt.p4_camera_move.split import write_split_manifest

ROOT = Path(__file__).resolve().parents[1]
MARK = "ZZMARKERQQ"
rows = []


def check(name, ok, detail=""):
    rows.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


rng = np.random.default_rng(0)
ATOMS = ["push-in", "pan-left", "tilt-up", "static"]
TEXT = {"push-in": "Push in", "pan-left": "Pan left", "tilt-up": "Tilt up", "static": "Static shot"}
tmp = Path(tempfile.mkdtemp())
entries, feats, meta = [], {}, []
for s in range(24):
    for j in range(4):
        clip = f"SRC{s:02d}.webm_{j}.mp4"
        a = ATOMS[(s + j) % 4]
        gold = TEXT[a]
        letter = "ABCD"[(s * 3 + j) % 4]
        others = [TEXT[x] for x in ATOMS if x != a]
        opts, k = {}, 0
        for ch in "ABCD":
            if ch == letter:
                opts[ch] = gold
            else:
                opts[ch] = others[k]
                k += 1
        text = "<video>Question: What's the camera movement in this movie shot?\nOptions:\n" + "\n".join(f"{c}. {t}" for c, t in opts.items()) + "\nPlease select the most likely answer from the options above. \n"
        entries.append({"videos": [f"data/{clip}"], "messages": [{"role": "user", "content": text}], "solution": f"<answer> {letter} </answer>"})
        f = rng.normal(size=12)
        f[ATOMS.index(a)] += 3.0
        feats[clip] = f
        meta.append({"clip": clip, "source": f"SRC{s:02d}"})
grpo = tmp / "grpo.json"
man = write_split_manifest(meta, tmp / "m.json")
val_clips = man["clips"]["val"]
planted = val_clips[0]
for e in entries:  # plant the marker in the gold option text of ONE validation clip
    if e["videos"][0] == f"data/{planted}":
        letter = e["solution"].split()[1]
        e["messages"][0]["content"] = e["messages"][0]["content"].replace(f"{letter}. ", f"{letter}. {MARK} ", 1)
grpo.write_text(json.dumps(entries))
F = pd.DataFrame.from_dict(feats, orient="index", columns=[f"f{i}" for i in range(12)])


def spied_run(out: Path, events: list, poison: bool = False, **kw):
    real = L._row_from_entry

    def spy(x):
        if MARK in x["messages"][0]["content"]:
            events.append(("marker_parsed", L._clip_of(x), "looks_lines=" + str(len(Path(out / "looks.jsonl").read_text().splitlines()) if (out / "looks.jsonl").exists() else 0)))
            if poison:
                raise ValueError("poisoned validation label")
        return real(x)
    L._row_from_entry = spy
    try:
        return B.run(feats=F, manifest=man, store=L.MovementLabelStore(man["clips"]["val"], grpo, events=events), feature_sets=["syn"], tag="syn", out_dir=out, looks_path=out / "looks.jsonl",
                     code_sha="test", events=events, **kw)
    finally:
        L._row_from_entry = real


# (c) metadata loader parses no label
calls = []
real_row = L._row_from_entry
L._row_from_entry = lambda x: calls.append(1) or real_row(x)
meta_rows = L.load_shotqa_movement_meta(grpo)
L._row_from_entry = real_row
check("metadata loader returns clip/source for every row and parses no label", len(meta_rows) == 96 and not calls and set(meta_rows[0]) == {"clip", "source"}, f"{len(meta_rows)} rows, {len(calls)} label parses")
# (b) locked store
st = L.MovementLabelStore(val_clips, grpo)
for label, fn in (("rows(validation clip)", lambda: st.rows([planted])), ("validation_rows()", lambda: st.validation_rows()), ("rows(train + validation mix)", lambda: st.rows([man["clips"]["train"][0], planted]))):
    try:
        fn()
        check(f"locked store refuses {label} without final_evaluation=True", False)
    except L.ValidationLabelsLocked:
        check(f"locked store refuses {label} without final_evaluation=True", True)
check("train rows are readable and the store logged only train parses", len(st.rows(man["clips"]["train"][:3])) == 3 and all(e[2] == "train" for e in st.events), str(st.events[:1]))
# (a) full run with the planted marker
out1, ev1 = tmp / "o1", []
res = spied_run(out1, ev1)
i_freeze = next(i for i, e in enumerate(ev1) if e[0] == "freeze")
before = ev1[:i_freeze]
check("no validation label parsed before the configuration is frozen", not any(e[0] == "parse" and e[2] == "final" for e in before) and not any(e[0] == "marker_parsed" for e in before),
      f"{len(before)} train parses before the freeze")
mk = [e for e in ev1 if e[0] == "marker_parsed"]
check("the planted marker is parsed once, only after the freeze and only after the look record was written", len(mk) == 1 and ev1.index(mk[0]) > i_freeze and mk[0][2] == "looks_lines=1", str(mk))
cfg = json.loads((out1 / "frozen_config_syn.json").read_text())
check("the frozen configuration holds the train-only choice (C, CV table, vocabulary) and no validation number", {"chosen_C", "cv_macro_ap_by_C", "atoms"} <= set(cfg) and "validation_per_atom" not in cfg, f"C={cfg['chosen_C']}")
# (d) selection independent of the validation labels
out2, ev2 = tmp / "o2", []
try:
    spied_run(out2, ev2, poison=True)
    check("a poisoned validation label makes the run fail", False)
except ValueError:
    cfg2 = json.loads((out2 / "frozen_config_syn.json").read_text())
    same = cfg2["chosen_C"] == cfg["chosen_C"] and cfg2["cv_macro_ap_by_C"] == cfg["cv_macro_ap_by_C"] and cfg2["config_hash"] == cfg["config_hash"]
    check("a poisoned validation label fails only at the final step; C, CV table and config hash are identical to the clean run (selection never saw validation labels)", same, f"C={cfg2['chosen_C']}, hash {cfg2['config_hash']}")
# (e) overwrite guard and look log
ev3 = []
try:
    spied_run(out1, ev3)
    check("a second run is refused while a final result exists", False)
except B.FinalResultExists:
    lines = (out1 / "looks.jsonl").read_text().splitlines()
    check("a second run is refused while a final result exists, BEFORE any validation label is read and without logging a look", not any(e[0] == "marker_parsed" or (e[0] == "parse" and e[2] == "final") for e in ev3) and len(lines) == 1, f"{len(lines)} look line(s)")
ev4 = []
spied_run(out1, ev4, allow_overwrite_final=True)
lines = [json.loads(x) for x in (out1 / "looks.jsonl").read_text().splitlines()]
check("with the explicit flag the overwrite is logged", len(lines) == 2 and lines[1]["overwrite_of_prior_final"] is True and lines[0]["overwrite_of_prior_final"] is False, f"look record keys: {sorted(lines[1])}")
check("each look line carries timestamp, code SHA, feature set, config hash and the running look count", all({"timestamp", "code_sha", "feature_set", "config_hash", "n_validation_looks_so_far"} <= set(x) for x in lines)
      and lines[1]["n_validation_looks_so_far"] == lines[0]["n_validation_looks_so_far"] + 1, f"counts {[x['n_validation_looks_so_far'] for x in lines]}")
check("the synthetic problem is learnable (the harness exercises the real fit)", res["validation_macro"]["f1"] > 0.5, f"macro F1 {res['validation_macro']['f1']}")
(ROOT / "results/phase4/label_isolation_selfcheck.md").write_text("# Phase 4 validation-label isolation self-check (synthetic, CPU)\n\n| check | result | detail |\n|---|---|---|\n" + "\n".join(
    f"| {n} | **{'PASS' if ok else 'FAIL'}** | {d} |" for n, ok, d in rows) + f"\n\nAll {len(rows)} checks passed: {all(r[1] for r in rows)}.\n")
raise SystemExit(0 if all(r[1] for r in rows) else 1)
