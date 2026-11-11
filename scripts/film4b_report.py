"""Report of the opt-in film mode 4B + C5 (docs/plans/film_4b_c5.md section 4): token-slot changes of 4B alone and 4B+C5 vs the accepted NOTLD
token table on the seeded sample, 4B+C5 vs 4B alone, static-slot agreement with the Track F winner rows, transitions, time per shot. CPU only;
reads this repo's film4b cache, the accepted token table (read-only, --data-root, SHA-256 checked) and the committed Track F rows.
A change is not an improvement.
Usage: python scripts/film4b_report.py --data-root <PM worktree> --cache <data/cache/film4b/...> --out results/film4b/notld_sample
       [--pipecheck-summary results/film4b/pipecheck/summary.json --projection-shots 200]"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.film4b import mode as M
from cgt.p3c_tokens.vocab import ATTRS
from cgt.tokens import v1

ROOT = Path(__file__).resolve().parents[1]
FILM = "night_of_the_living_dead"
TOKENS_SHA = "0e818c9d48546fc087609c539e9b9dd007b74100fd58b467c42f1350eaee9fb7"
TRACKF_ROWS = ROOT / "results/stepa_films/notld_sample_rows.jsonl"
TRACKF_DEFAULT_S_PER_SHOT = 2.3269529999999996  # results/stepa_films/notld_sample.json time.per_shot_default_s_mean (2B, a different run)
SLOT_ATTRS = {"size": ["size"], "lens": ["lens"], "angle": ["angle"], "framing": ["framing"], "light": ["light_type", "light_cond"],
              "comp": ["composition"]}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def static_slots(row: str) -> dict[str, str]:
    """Static slot values of a formatted v1 row, the display-only '*' removed."""
    kv = dict(x.split("=", 1) for x in row.strip("[]").split(" | "))
    return {k: kv[k].rstrip("*") for k in SLOT_ATTRS}


def changes(prim: dict, ref: dict, shots: list[int]) -> dict:
    """prim[s][attr] vs ref[s][attr]: per-attribute and per-slot counts, any static slot, transitions (ref -> prim)."""
    return {"attr_changes": {a: sum(prim[s][a] != ref[s][a] for s in shots) for a in ATTRS},
            "slot_changes": {sl: sum(any(prim[s][x] != ref[s][x] for x in xs) for s in shots) for sl, xs in SLOT_ATTRS.items()},
            "shots_any_static_change": sum(any(prim[s][a] != ref[s][a] for a in ATTRS) for s in shots),
            "transitions": {a: {f"{o}->{n}": c for (o, n), c in sorted(Counter((ref[s][a], prim[s][a]) for s in shots
                                                                                if prim[s][a] != ref[s][a]).items(), key=lambda t: (-t[1], t[0]))}
                            for a in ATTRS}}


def markdown(out: dict, n: int) -> str:
    """Generated summary tables (no interpretation beyond the fixed caveat)."""
    t = out["time"]
    lines = [f"# Film mode 4B + C5 on NOTLD: {n} shots (produced by scripts/film4b_report.py)",
             "**A change is not an improvement**: no film labels exist; these are change rates vs the accepted NOTLD table only.",
             (f"Sample: seed {out['sample']['seed']}, requested {out['sample']['n_requested']}, first {out['sample']['first']}, complete {n}. "
              f"Model {out['model']}. C5 weights {out['blend']['ensemble_w'] if out['blend'] else 'n/a'} (light_type 0: C5 cannot move it)."), "",
             "## Shots changed per slot", "", "| variant | " + " | ".join(SLOT_ATTRS) + " | any static |", "|---" * (len(SLOT_ATTRS) + 2) + "|"]
    for name, r in out["variants"].items():
        cells = [f"{r['slot_changes'][s]} ({100 * r['slot_changes'][s] / max(n, 1):.1f}%)" for s in SLOT_ATTRS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['shots_any_static_change']} ({100 * r['shots_any_static_change'] / max(n, 1):.1f}%) |")
    lines += ["", "## Per-attribute changes and transitions (old -> new)", ""]
    for name, r in out["variants"].items():
        lines.append(f"**{name}:** " + ", ".join(f"{a} {c}" for a, c in r["attr_changes"].items()))
        for a, tr in r["transitions"].items():
            if tr:
                lines.append(f"- {a}: " + ", ".join(f"{k} {v}" for k, v in tr.items()))
        lines.append("")
    lines += ["## Static-slot agreement with the Track F winner rows (2B C1+C3+C4*+C5; descriptive)", ""]
    for name, g in out["trackf_static_slot_agreement"].items():
        lines.append(f"- {name} ({g['shots']} shots in both): " + ", ".join(f"{k} {v}" for k, v in g["per_slot_same_as_trackf_winner"].items()))
    lines += ["", "## Time", "", (f"- 4B method D per shot (3 keyframes): mean {t['per_shot_4b_s_mean']:.2f} s, median {t['per_shot_4b_s_median']:.2f} s, "
               f"max {t['per_shot_4b_s_max']:.2f} s; 2B default {t['per_shot_default_2b_s_mean_trackf']:.2f} s/shot (Track F, a different run)."),
              f"- SigLIP CPU total {t['siglip_cpu_s_total']} s; model loads {t['model_load_s']} s; child seconds total {t['child_seconds_total']:.0f}.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache", default=str(M.out_dir(FILM, 200, 20261017)))
    ap.add_argument("--out", default=str(ROOT / "results/film4b/notld_sample"))
    ap.add_argument("--pipecheck-summary", default=None)
    ap.add_argument("--projection-shots", type=int, default=200)
    a = ap.parse_args()
    dr, cd = Path(a.data_root), Path(a.cache)
    tok_path = dr / f"data/processed/{FILM}_tokens_v1.parquet"
    if sha(tok_path) != TOKENS_SHA:
        raise SystemExit("accepted NOTLD token table hash mismatch")
    tok = pd.read_parquet(tok_path).set_index("shot_id")
    sample = json.loads((cd / "sample.json").read_text())
    recs = M.load_records(cd / "logits.jsonl")
    shots = [s for s in sample["shot_ids"] if all((s, kf) in recs for kf in M.KEYFRAMES)]
    n = len(shots)
    kfname = {(s, kf): Path(tok.loc[s, kf]).name for s in shots for kf in M.KEYFRAMES}
    ref = {s: {x: tok.loc[s, x] for x in ATTRS} for s in shots}

    kf4 = {s: [M.keyframe_vectors(recs[(s, kf)]) for kf in M.KEYFRAMES] for s in shots}
    prim = {"4B": {s: M.primaries(kf4[s]) for s in shots}}
    blend_info = None
    if (cd / "siglip3a_scores.json").exists():
        sigj = json.loads((cd / "siglip3a_scores.json").read_text())
        blend_info = json.loads((cd / "blend.json").read_text())
        if blend_info["ensemble_w"] != M.EXPECTED_W:
            raise SystemExit("cached blend weights differ from the committed Step A-4B C5 fit")
        w = blend_info["ensemble_w"]
        kfc = {s: [M.keyframe_vectors(recs[(s, kf)], sigj["scores"][kfname[(s, kf)]], w) for kf in M.KEYFRAMES] for s in shots}
        prim["4B+C5"] = {s: M.primaries(kfc[s]) for s in shots}
    else:
        kfc, sigj = None, None

    res = {name: changes(p, ref, shots) for name, p in prim.items()}
    if "4B+C5" in prim:
        res["4B+C5 vs 4B alone"] = changes(prim["4B+C5"], prim["4B"], shots)

    rows = []
    for s in shots:
        out = {"shot_id": int(s), "old": tok.loc[s, "token_row"]}
        for name, kfs in (("4B", kf4), ("4B+C5", kfc)):
            if kfs is None:
                continue
            rec = tok.loc[s].to_dict()
            for att in ATTRS:
                rec[att] = prim[name][s][att]
                rec[f"{att}_consistent"] = len({int(np.argmax(k[att])) for k in kfs[s]}) == 1
            out[name] = v1.format_row(rec)
        rows.append(out)

    # descriptive: static-slot agreement with the Track F winner rows (2B, C1+C3+C4*+C5) on shots in both samples
    tf = {r["shot_id"]: r for r in map(json.loads, TRACKF_ROWS.read_text().splitlines())}
    agree = {}
    for name in [k for k in ("4B", "4B+C5") if k in prim]:
        both = [r for r in rows if r["shot_id"] in tf]
        agree[name] = {"shots": len(both), "per_slot_same_as_trackf_winner": {
            sl: sum(static_slots(r[name])[sl] == static_slots(tf[r["shot_id"]]["new"])[sl] for r in both) for sl in SLOT_ATTRS}}

    tim = [sum(recs[(s, kf)]["total_s"] for kf in M.KEYFRAMES) for s in shots]
    img = [sum(recs[(s, kf)]["image_s"] for kf in M.KEYFRAMES) for s in shots]
    inv = json.loads((cd / "invocations.json").read_text()) if (cd / "invocations.json").exists() else []
    gpu_s = sum(i["seconds"] + i["model_load_s"] for i in inv)
    time_ = {"per_shot_4b_s_mean": float(np.mean(tim)) if n else None, "per_shot_4b_s_median": float(np.median(tim)) if n else None,
             "per_shot_4b_s_max": float(np.max(tim)) if n else None, "per_shot_image_state_s_mean": float(np.mean(img)) if n else None,
             "per_shot_default_2b_s_mean_trackf": TRACKF_DEFAULT_S_PER_SHOT,
             "note_2b": "2B default time quoted from Track F (results/stepa_films/notld_sample.json), a different run on the same machine",
             "siglip_cpu_s_total": sigj["cpu_seconds"] if sigj else None, "model_load_s": [i["model_load_s"] for i in inv],
             "child_seconds_total": gpu_s, "invocations": inv}
    out = {"film": FILM, "sample": {"n_requested": sample["n"], "seed": sample["seed"], "first": sample.get("first"), "n_complete": n},
           "accepted_token_table_sha256": TOKENS_SHA, "model": f"{M.REPO}@{M.REVISION} + LoRA {M.ADAPTER_SHA}",
           "blend": blend_info, "variants": res, "trackf_static_slot_agreement": agree, "time": time_,
           "note": "A change is not an improvement: no film labels exist; these are change rates only."}
    if a.pipecheck_summary:
        eq = [i.get("prefix_kv_vs_stock") for i in inv if i.get("prefix_kv_vs_stock")]
        worst = max(tim) if n else None
        out["projection"] = {"shots": a.projection_shots, "sec_per_shot_projection": worst, "basis": "slowest measured shot",
                             "gpu_hours_projected": (a.projection_shots * worst + max(time_["model_load_s"] or [0])) / 3600 if worst else None}
        Path(a.pipecheck_summary).write_text(json.dumps({"ok": n == len(sample["shot_ids"]) and "4B+C5" in prim, "n_shots": n,
                                                         "prefix_kv_vs_stock": eq, "sec_per_shot_projection": worst,
                                                         "projection": out["projection"], "time": {k: v for k, v in time_.items() if k != "invocations"}},
                                                        indent=1) + "\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1) + "\n")
    Path(a.out + "_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    Path(a.out + ".md").write_text(markdown(out, n))
    for name, r in res.items():
        print(name, "slots", r["slot_changes"], "any", r["shots_any_static_change"], "of", n)
    print(json.dumps({k: v for k, v in time_.items() if k != "invocations"}, indent=1))


if __name__ == "__main__":
    main()
