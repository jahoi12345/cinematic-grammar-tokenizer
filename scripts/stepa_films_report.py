"""Track F report (docs/plans/stepa_films.md section 4): token-slot changes of the Step A film mode vs the accepted NOTLD token table on the seeded
sample, the base-view parity check against the accepted Phase 3C scores, time per shot and the exploratory decomposition. CPU only; reads the
Track F cache of this repo, the accepted tables (read-only, --data-root) and the saved Step A parameters. A change is not an improvement.
Usage: python scripts/stepa_films_report.py --data-root <PM worktree> [--gpu-hours-note TEXT]"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from cgt.p3c_tokens.vocab import ATTRS
from cgt.p3s_stepa import film as F
from cgt.p3s_stepa import film_run as R
from cgt.tokens import v1

ROOT = Path(__file__).resolve().parents[1]
FILM = "night_of_the_living_dead"
TOKENS_SHA = "0e818c9d48546fc087609c539e9b9dd007b74100fd58b467c42f1350eaee9fb7"
ACCEPTED_RUN = "night_of_the_living_dead_d05369097f04"
SLOT_ATTRS = {"size": ["size"], "lens": ["lens"], "angle": ["angle"], "framing": ["framing"], "light": ["light_type", "light_cond"],
              "comp": ["composition"]}
VARIANTS = {  # exploratory decomposition: (views, calib, letter_bias, ensemble); none is a candidate
    "C0 (base view only: today's method D)": (["base"], False, True, False),
    "C1 (4 option orders)": (["base", "shift1", "shift2", "shift3"], False, True, False),
    "C1+C3 (+ hflip, crop80)": (list(F.FILM_VIEWS), False, True, False),
    "C1+C3+C4* (+ calibration)": (list(F.FILM_VIEWS), True, True, False),
    "C5 only (base + 3A blend)": (["base"], False, True, True),
    "winner without letter bias": (list(F.FILM_VIEWS), True, False, True),
    "WINNER C1+C3+C4*+C5": (list(F.FILM_VIEWS), True, True, True),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def per_kf_scores(rec: dict, sig: dict, params: dict, variant: tuple) -> dict[str, np.ndarray]:
    views, calib, lb, ens = variant
    out = {}
    for a in ATTRS:
        vl = {v: rec["views"][v][a] for v in rec["views"] if a in rec["views"][v]}
        out[a] = F.combine(a, vl, np.array(sig[a]) if ens else None, params, views=views, calib=calib, letter_bias=lb, ensemble=ens)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache", default=str(ROOT / f"data/cache/stepa_films/{FILM}_n200_seed20261017"))
    ap.add_argument("--out", default=str(ROOT / "results/stepa_films/notld_sample"))
    a = ap.parse_args()
    dr, cd = Path(a.data_root), Path(a.cache)
    tok_path = dr / f"data/processed/{FILM}_tokens_v1.parquet"
    if sha(tok_path) != TOKENS_SHA:
        raise SystemExit("accepted NOTLD token table hash mismatch")
    tok = pd.read_parquet(tok_path).set_index("shot_id")
    sample = json.loads((cd / "sample.json").read_text())
    recs = R.load_records(cd / "views.jsonl")
    sigj = json.loads((cd / "siglip3a_scores.json").read_text())
    sig = sigj["scores"]
    params = F.load_params()
    shots = [s for s in sample["shot_ids"] if all((s, kf) in recs for kf in R.KEYFRAMES)]
    kfname = {(s, kf): Path(tok.loc[s, kf]).name for s in shots for kf in R.KEYFRAMES}

    # parity: base view vs the accepted Phase 3C scores (method D = log_softmax of the n letter logits)
    acc = {kf: {r["shot_id"]: r for r in map(json.loads, (dr / f"data/cache/p3c/runs/{ACCEPTED_RUN}/scores_{kf}.jsonl").read_text().splitlines())}
           for kf in R.KEYFRAMES}
    diffs, same = [], 0
    for s in shots:
        for kf in R.KEYFRAMES:
            for att in ATTRS:
                mine = log_softmax(np.array(recs[(s, kf)]["views"]["base"][att]))
                theirs = np.array(acc[kf][s][att])
                diffs.append(float(np.abs(mine - theirs).max()))
                same += int(np.argmax(mine) == np.argmax(theirs))
    parity = {"keyframe_attr_pairs": len(diffs), "max_abs_diff": max(diffs), "median_abs_diff": float(np.median(diffs)), "same_argmax": same}

    # primaries per variant
    res = {}
    rows_out = []
    for name, var in VARIANTS.items():
        kfsc = {s: [per_kf_scores(recs[(s, kf)], sig[kfname[(s, kf)]], params, var) for kf in R.KEYFRAMES] for s in shots}
        prim = {s: R.primaries(kfsc[s]) for s in shots}
        attr_change = {att: sum(prim[s][att] != tok.loc[s, att] for s in shots) for att in ATTRS}
        slot_change = {sl: sum(any(prim[s][x] != tok.loc[s, x] for x in xs) for s in shots) for sl, xs in SLOT_ATTRS.items()}
        any_change = sum(any(prim[s][x] != tok.loc[s, x] for x in ATTRS) for s in shots)
        res[name] = {"attr_changes": attr_change, "slot_changes": slot_change, "shots_any_static_change": any_change,
                     "transitions": {att: {f"{o}->{n}": c for (o, n), c in sorted(Counter((tok.loc[s, att], prim[s][att]) for s in shots
                                                                                         if prim[s][att] != tok.loc[s, att]).items(), key=lambda t: -t[1])}
                                     for att in ATTRS}}
        if name.startswith("WINNER"):
            for s in shots:
                rec = tok.loc[s].to_dict()
                old = rec["token_row"]
                for att in ATTRS:
                    rec[att] = prim[s][att]
                    rec[f"{att}_consistent"] = len({int(np.argmax(k[att])) for k in kfsc[s]}) == 1
                new = v1.format_row(rec)
                rows_out.append({"shot_id": int(s), "old": old, "new": new})
            marker_or_label = sum(r["old"] != r["new"] for r in rows_out)
            res[name]["rows_differing_incl_star_marker"] = marker_or_label

    # time
    tdef = [sum(recs[(s, kf)]["default_s"] for kf in R.KEYFRAMES) for s in shots]
    ttot = [sum(recs[(s, kf)]["total_s"] for kf in R.KEYFRAMES) for s in shots]
    inv = json.loads((cd / "invocations.json").read_text()) if (cd / "invocations.json").exists() else []
    gpu_s = sum(i["seconds"] + i["model_load_s"] for i in inv)
    n = len(shots)
    out = {"film": FILM, "sample": {"n_requested": sample["n"], "seed": sample["seed"], "n_complete": n}, "accepted_token_table_sha256": TOKENS_SHA,
           "accepted_p3c_run": ACCEPTED_RUN, "params_source": "results/p3s_stepa/static_val.json (C1+C3+C4*+C5 fit, not refitted)",
           "parity_base_view_vs_accepted_scores": parity, "variants": res,
           "time": {"per_shot_default_s_mean": float(np.mean(tdef)), "per_shot_stepa_s_mean": float(np.mean(ttot)),
                    "per_shot_default_s_median": float(np.median(tdef)), "per_shot_stepa_s_median": float(np.median(ttot)),
                    "ratio_mean": float(np.mean(ttot) / np.mean(tdef)), "siglip_cpu_s_total": sigj["cpu_seconds"],
                    "gpu_hours": gpu_s / 3600, "invocations": inv}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    Path(a.out + "_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    w = res["WINNER C1+C3+C4*+C5"]
    print(json.dumps({"n": n, "parity": parity, "winner_slots": {k: f"{v}/{n} ({100 * v / n:.1f}%)" for k, v in w["slot_changes"].items()},
                      "any": w["shots_any_static_change"], "time": {k: v for k, v in out["time"].items() if k != "invocations"}}, indent=1))
    for name, r in res.items():
        print(name, {k: v for k, v in r["slot_changes"].items()}, "any", r["shots_any_static_change"])


if __name__ == "__main__":
    main()
