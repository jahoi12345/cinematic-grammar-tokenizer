"""Accuracy Step A, movement track (docs/acc_step_a_move_plan.md; the recorded rule): decoders D0-D3 (+ D4 in the secondary design) of the FROZEN geometry model on the 140-question
source-held-out ShotQA movement validation set, from committed probabilities/logits only (no model call, no GPU). ONE validation look is logged BEFORE any validation label is read.
NOTLD probabilities are read-only for a descriptive contradiction/atom table. Never opens ShotBench or CameraBench.
Usage: nice -n 19 uv run python scripts/p4_decode_move_val.py [--main-checkout DIR]"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.p4_camera_move.baseline import code_sha, record_look, wilson
from cgt.p4_camera_move.labels import MovementLabelStore, source_id
from cgt.p4_decode import move as M
from cgt.worktrees import main_checkout

ROOT = Path(__file__).resolve().parents[1]
P4 = ROOT / "results/phase4"
OUT = ROOT / "results/p4_stepa"
MIN_FREE_PERCENT = 25
CANDS = M.CANDIDATES


def free_percent() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.split(":")[1].strip().rstrip("%"))
    return 0


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def acc_entry(h: np.ndarray) -> dict:
    lo, hi = wilson(int(h.sum()), len(h))
    return {"acc": round(float(h.mean()), 4), "correct": int(h.sum()), "n": len(h), "wilson95_pct": [round(lo, 1), round(hi, 1)]}


def evaluate(questions, P, atoms, th, video, sources, ws, wg, gold_sets, extra_combined=None) -> dict:
    """Geometry and combined option accuracy, set metrics and source-bootstrap vs D0 for D0-D3 on one question set."""
    res = {"geometry": {}, "combined": {}, "sets": {}, "boot_vs_d0_geometry": {}, "boot_vs_d0_combined": {}}
    Hg, Hc = {}, {}
    for c in CANDS:
        Hg[c] = M.option_hits(questions, P, atoms, c, th)
        Hc[c] = M.option_hits(questions, P, atoms, c, th, video_logits=video, w_s=ws, w_g=wg)
        res["geometry"][c] = acc_entry(Hg[c])
        res["combined"][c] = acc_entry(Hc[c])
        res["sets"][c] = M.set_metrics(M.decode(P, atoms, c, th), gold_sets, atoms)
    if extra_combined:
        for name, (w_s2, w_g2) in extra_combined.items():
            Hc[name] = M.option_hits(questions, P, atoms, "D0", None, video_logits=video, w_s=w_s2, w_g=w_g2)
            res["combined"][name] = acc_entry(Hc[name])
    for c in [x for x in CANDS if x != "D0"]:
        res["boot_vs_d0_geometry"][c] = M.paired_bootstrap(Hg[c], Hg["D0"], sources)
    for c in [x for x in Hc if x != "D0"]:
        res["boot_vs_d0_combined"][c] = M.paired_bootstrap(Hc[c], Hc["D0"], sources)
    return res


def look_and_load(store, man, vp, vl, sha):
    """THE LOOK: logged BEFORE any validation label is read; then the 140 scored validation questions aligned with the stored probabilities and logits."""
    look = record_look(P4 / "validation_looks.jsonl", kind="acc_step_a_move", tag="acc_step_a_move_D0-D4", feature_sets=["flow", "depth", "video_lora_e3_logits"],
                       config_hash="thr_oof_f1grid+mask+split_seed0", code_sha_=sha)
    val_rows = store.rows(man["clips"]["val"], final_evaluation=True)
    rows_by = {r["clip"]: r for r in val_rows if r["atoms"]}
    if vl["clips"] != [c for c in vp["clips"] if c in rows_by] or set(vl["clips"]) != set(rows_by):
        raise SystemExit("clip order of probabilities and logits does not match the scored validation clips")
    clips = vl["clips"]
    pi = {c: i for i, c in enumerate(vp["clips"])}
    P = np.array([vp["probs"][pi[c]] for c in clips])
    Q = [rows_by[c] for c in clips]
    if [q["gold_letter"] for q in Q] != vl["gold"]:
        raise SystemExit("gold letters differ from the stored validation logits")
    V = [np.array(z) for z in vl["logits"]["normal"]]
    return look, Q, P, V, np.array([q["source"] for q in Q]), [q["atoms"] for q in Q]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-checkout", default=str(main_checkout()))
    ap.add_argument("--test-aid", action="store_true", help="mechanics check on TRAIN clips: no look, no validation label; needs --out-dir")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.test_aid and not a.out_dir:
        raise SystemExit("--test-aid needs --out-dir")
    main_root = Path(a.main_checkout)
    grpo = main_root / "data/external/shotqa/grpo.json"
    notld_path = main_root / "data/processed/night_of_the_living_dead_move.parquet"
    free = free_percent()
    if free < MIN_FREE_PERCENT:
        raise SystemExit(f"memory guard: {free}% free < {MIN_FREE_PERCENT}%; nothing started")
    sha = code_sha()
    man = json.loads((P4 / "split_manifest.json").read_text())
    store = MovementLabelStore(man["clips"]["val"], grpo_path=grpo)

    # ---------------- calibration (PRIMARY): OOF geometry probabilities of the TRAIN clips, train labels only
    oof = json.loads((P4 / "runs/video_v1/geom_oof.json").read_text())
    atoms = oof["atoms"]
    tr_clips = sorted(oof["probs"])
    tr_rows = store.rows(tr_clips)  # raises if any validation clip slipped in
    P_cal = np.array([oof["probs"][c] for c in tr_clips])
    Y_cal = np.array([[x in r["atoms"] for x in atoms] for r in tr_rows], dtype=bool)
    th = M.fit_thresholds(P_cal, Y_cal)
    cal_desc = {"n": len(tr_clips), "thresholds": dict(zip(atoms, th.tolist(), strict=True)), "positives": dict(zip(atoms, Y_cal.sum(0).tolist(), strict=True)),
                "gold_contradictory_share": round(float(np.mean([M.is_contradictory(r["atoms"])["contradictory"] for r in tr_rows])), 4)}
    cal_desc["oof_geometry_option_acc_in_sample_for_D2_D3"] = {c: acc_entry(M.option_hits(tr_rows, P_cal, atoms, c, th)) for c in CANDS}

    # ---------------- NOTLD (descriptive; no label exists; read-only)
    nd = pd.read_parquet(notld_path)
    Pn = nd[[f"p_{x}" for x in atoms]].to_numpy(dtype=np.float64)
    if np.isnan(Pn).any():
        raise SystemExit("NOTLD probabilities contain NaN")
    d0n = M.decode(Pn, atoms, "D0")
    if [" + ".join(s) for s in d0n] != nd["move_atoms"].tolist():
        raise SystemExit("D0 does not reproduce the stored NOTLD move_atoms")
    notld = {}
    for c in CANDS:
        sets = M.decode(Pn, atoms, c, th)
        cc = [M.is_contradictory(s) for s in sets]
        sizes = pd.Series([len(s) for s in sets]).value_counts().sort_index()
        notld[c] = {"n_shots": len(sets), "contradictory": sum(x["contradictory"] for x in cc), "contradictory_share": round(sum(x["contradictory"] for x in cc) / len(sets), 4),
                    "opposite_pair": sum(x["opposite_pair"] for x in cc), "static_with_other": sum(x["static_with_other"] for x in cc),
                    "static_alone_share": round(sum(s == ["static"] for s in sets) / len(sets), 4), "mean_set_size": round(float(np.mean([len(s) for s in sets])), 3),
                    "set_sizes": {int(k): int(v) for k, v in sizes.items()}, "atom_share": {x: round(sum(x in s for s in sets) / len(sets), 4) for x in atoms}}
    if notld["D0"]["contradictory"] != 462:
        raise SystemExit("D0 does not reproduce 462 contradictory NOTLD shots")

    # ---------------- validation probabilities / logits (no labels yet)
    vp = json.loads((P4 / "val_probs_flow_depth.json").read_text())
    vl = json.loads((P4 / "runs/video_v1/validation_logits.json").read_text())
    assert vp["atoms"] == atoms
    comb = json.loads((P4 / "runs/video_v1/combine.json").read_text())["weights"]
    ws, wg = comb["w_s"], comb["w_g"]

    if a.test_aid:  # mechanics only: TRAIN clips stand in for validation, no look, no validation label, scratch output
        look = {"n_validation_looks_so_far": 0, "n_logged_looks_so_far": 0, "test_aid": True}
        Q = tr_rows[:140]
        P = P_cal[:140]
        V = [np.zeros(4) for _ in Q]
        src = np.array([q["source"] for q in Q])
        gold_sets = [q["atoms"] for q in Q]
        all_src = sorted(set(src))
    else:
        look, Q, P, V, src, gold_sets = look_and_load(store, man, vp, vl, sha)
        all_src = sorted({source_id(c) for c in man["clips"]["val"]})
    prim = evaluate(Q, P, atoms, th, V, src, ws, wg, gold_sets)
    if not a.test_aid and (prim["geometry"]["D0"]["correct"] != 102 or prim["combined"]["D0"]["correct"] != 116):
        raise SystemExit(f"D0 does not reproduce 102/140 and 116/140: {prim['geometry']['D0']} {prim['combined']['D0']}")
    decision = M.decide({c: prim["geometry"][c]["acc"] for c in CANDS}, prim["boot_vs_d0_geometry"], {c: prim["sets"][c]["jaccard"] for c in CANDS})
    val_gold_contra = {"n": len(Q), "gold_set_contradictory": int(sum(M.is_contradictory(g)["contradictory"] for g in gold_sets)),
                       "questions_with_some_contradictory_option": int(sum(any(M.is_contradictory(q["option_atoms"][L])["contradictory"] for L in M.LETTERS) for q in Q))}

    # ---------------- SECONDARY (0022 literal): validation sources split by seed 0, label-free
    perm = [all_src[i] for i in np.random.default_rng(0).permutation(len(all_src))]
    cal_src = set(perm[: (len(perm) + 1) // 2])
    ic = np.array([s in cal_src for s in src])
    isel = ~ic
    th2 = M.fit_thresholds(P[ic], np.array([[x in g for x in atoms] for g, k in zip(gold_sets, ic, strict=True) if k], dtype=bool))
    geo_c = [M.option_geometry_scores(q, p, atoms) for q, p, k in zip(Q, P, ic, strict=True) if k]
    sem_c = [M.lsm(v) for v, k in zip(V, ic, strict=True) if k]
    gold_c = [M.LETTERS.index(q["gold_letter"]) for q, k in zip(Q, ic, strict=True) if k]
    w_d4 = M.fit_fusion_weights(sem_c, geo_c, gold_c)
    sel = lambda xs: [x for x, k in zip(xs, isel, strict=True) if k]
    sec = evaluate(sel(Q), P[isel], atoms, th2, sel(V), src[isel], ws, wg, sel(gold_sets), extra_combined={"D4": w_d4})
    sec_meta = {"calibration_sources": sorted(cal_src), "n_calibration_questions": int(ic.sum()), "n_selection_questions": int(isel.sum()), "n_selection_sources": len(set(src[isel])),
                "thresholds_calibration_half": dict(zip(atoms, th2.tolist(), strict=True)), "D4_weights": {"w_s": w_d4[0], "w_g": w_d4[1]},
                "note": "descriptive only; never used for choosing (plan section 1)"}
    sec_rank = M.decide({c: sec["geometry"][c]["acc"] for c in CANDS}, sec["boot_vs_d0_geometry"], {c: sec["sets"][c]["jaccard"] for c in CANDS})

    res = {"plan": "docs/acc_step_a_move_plan.md", "code_sha": sha, "look": look, "inputs_sha256_16": {p: sha16(ROOT / p) for p in (
        "results/phase4/val_probs_flow_depth.json", "results/phase4/runs/video_v1/geom_oof.json", "results/phase4/runs/video_v1/validation_logits.json")} | {"notld_move.parquet": sha16(notld_path)},
        "atoms": atoms, "frozen_fusion_weights": {"w_s": ws, "w_g": wg}, "calibration_primary_oof": cal_desc, "primary_validation": prim, "validation_gold_structure": val_gold_contra,
        "decision": decision, "secondary_0022_split": {**sec_meta, "results": sec, "would_rank": sec_rank}, "notld_descriptive": notld,
        "not_touched": ["ShotBench", "CameraBench", "NOTLD gold", "owned film", "user clip", "film_move defaults", "token rows", "accepted tables"]}
    out = Path(a.out_dir) if a.test_aid else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "move_val.json").write_text(json.dumps(res, indent=1))
    (out / "move_val.md").write_text(render(res))
    print(render(res))


def render(r: dict) -> str:
    p, s = r["primary_validation"], r["secondary_0022_split"]
    d = r["decision"]
    L = ["# Accuracy Step A, movement track: decoders of the frozen geometry model on ShotQA validation", "",
         (f"Plan: `{r['plan']}` (committed before this run). Code `{r['code_sha']}`. Validation look logged first: look {r['look']['n_validation_looks_so_far']} overall "
         f"({r['look']['n_logged_looks_so_far']} logged). Inputs are committed probabilities/logits only; no model call, no GPU; ShotBench and CameraBench were not opened."), "",
         "## Primary design: thresholds fitted on the 903 OOF train clips, selection on the 140 validation questions (19 sources)", "",
         "| candidate | geometry 4-option acc | Wilson 95% | diff vs D0 [source bootstrap 95%] | combined (0.3/0.7) acc | set exact | set Jaccard | macro F1 (atoms w/ support) | contradictory sets |",
         "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for c in CANDS:
        g, cb, st = p["geometry"][c], p["combined"][c], p["sets"][c]
        b = p["boot_vs_d0_geometry"].get(c)
        bs = f"{b['diff']:+.4f} [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]" if b else "-"
        L.append(f"| {c} | {g['acc']:.4f} ({g['correct']}/{g['n']}) | {g['wilson95_pct']} | {bs} | {cb['acc']:.4f} ({cb['correct']}/{cb['n']}) | {st['exact_match']:.4f} | {st['jaccard']:.4f} | "
                 f"{st['macro_f1_supported']:.4f} ({st['n_atoms_supported']}) | {st['contradictory_share']:.4f} |")
    L += ["", "Combined minus combined-D0 (descriptive): " + "; ".join(f"{c} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for c, v in p["boot_vs_d0_combined"].items()), "",
          (f"Validation gold structure: {r['validation_gold_structure']['gold_set_contradictory']} of {r['validation_gold_structure']['n']} gold sets are contradictory (unreachable for D1/D3); "
          f"{r['validation_gold_structure']['questions_with_some_contradictory_option']} questions offer at least one contradictory option."), "",
          f"**Decision (pre-registered rule):** top-ranked candidate {d['top_ranked']}; interval above 0: {d['ci_excludes_zero_above']}; Jaccard within 1 point: {d['jaccard_ok']}. "
          f"**Winner: {d['winner']}**" + (" (keep the current rule)." if d["winner"] == "D0" else "."), "",
          "D4 (fusion weights re-fitted) is not a primary candidate: in the primary design its calibration data are the OOF questions the frozen 0.3/0.7 were fitted on (identical by construction).", "",
          "OOF thresholds (calibration): " + ", ".join(f"{a} {t:.2f}" for a, t in r["calibration_primary_oof"]["thresholds"].items()), "",
          "## Per-atom F1 on validation (primary)", "", "| atom | support | " + " | ".join(CANDS) + " |", "|---|---:|" + "---:|" * len(CANDS)]
    for i, a in enumerate(r["atoms"]):
        L.append(f"| {a} | {p['sets']['D0']['per_atom'][i]['support']} | " + " | ".join(f"{p['sets'][c]['per_atom'][i]['f1']:.3f}" for c in CANDS) + " |")
    sr = s["results"]
    L += ["", "## Secondary design", "",
          (f"Validation sources split by seed 0: calibration half {s['n_calibration_questions']} questions ({len(s['calibration_sources'])} sources), selection half {s['n_selection_questions']} questions "
          f"({s['n_selection_sources']} sources). D4 weights fitted on the calibration half: w_s = {s['D4_weights']['w_s']}, w_g = {s['D4_weights']['w_g']}."), "",
          "| candidate | geometry acc | diff vs D0 [95%] | combined acc | combined diff vs D0 [95%] | set Jaccard |", "|---|---:|---|---:|---|---:|"]
    for c in [*CANDS, "D4"]:
        g = sr["geometry"].get(c)
        cb = sr["combined"][c]
        bg = sr["boot_vs_d0_geometry"].get(c)
        bc = sr["boot_vs_d0_combined"].get(c)
        L.append(f"| {c} | {f'{g['acc']:.4f} ({g['correct']}/{g['n']})' if g else '-'} | {f'{bg['diff']:+.4f} [{bg['ci95'][0]:+.4f}, {bg['ci95'][1]:+.4f}]' if bg else '-'} | "
                 f"{cb['acc']:.4f} ({cb['correct']}/{cb['n']}) | {f'{bc['diff']:+.4f} [{bc['ci95'][0]:+.4f}, {bc['ci95'][1]:+.4f}]' if bc else '-'} | "
                 f"{f'{sr['sets'][c]['jaccard']:.4f}' if c in sr['sets'] else '-'} |")
    L += ["", f"The rule applied to this half would pick: {s['would_rank']['winner']} (top-ranked {s['would_rank']['top_ranked']}); not used.", "",
          "## NOTLD (descriptive only; no label exists; accuracy unmeasured)", "",
          "| candidate | contradictory | share | opposite pair | static + other | static alone | mean set size |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c, v in r["notld_descriptive"].items():
        L.append(f"| {c} | {v['contradictory']}/{v['n_shots']} | {v['contradictory_share']:.1%} | {v['opposite_pair']} | {v['static_with_other']} | {v['static_alone_share']:.1%} | {v['mean_set_size']} |")
    L += ["", "Share of shots containing each atom:", "", "| atom | " + " | ".join(CANDS) + " |", "|---|" + "---:|" * len(CANDS)]
    for a in r["atoms"]:
        L.append(f"| {a} | " + " | ".join(f"{r['notld_descriptive'][c]['atom_share'][a]:.1%}" for c in CANDS) + " |")
    L += ["", ("Not changed: the frozen geometry model, `film_move.predict` (D0 stays the film runner's default), token rows, accepted tables. This is a recommendation only; "
          "no ShotBench run without the user's explicit go.")]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
