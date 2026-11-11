"""Track F step 1: check that the SAVED Step A winner parameters (results/p3s_stepa/static_val.json) reproduce the reported ShotQA validation numbers
when applied without any fit to the cached Step A view logits and Phase 3A scores (read-only, from the Step A worktree). CPU only. Reads ShotQA
validation only (val.jsonl labels + meta.jsonl titles for the film split); never gold, dev, ShotBench or CameraBench.
Usage: python scripts/stepa_films_params_check.py --stepa-root <a sibling working copy worktree> --out results/stepa_films/params_check.json"""
import argparse
import json
from pathlib import Path

from cgt.p3_static_attrs.data import image_films
from cgt.p3s_stepa import combine as C
from cgt.p3s_stepa import film
from cgt.p3s_stepa.split import load_split

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepa-root", required=True)
    ap.add_argument("--shotqa-meta", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/stepa_films/params_check.json"))
    a = ap.parse_args()
    sr = Path(a.stepa_root)
    sp = load_split(sr / "data/cache/p3b/val.jsonl", lambda ids: image_films(Path(a.shotqa_meta), ids)[0])
    qs = sp["questions"]
    views = {}
    for v in film.FILM_VIEWS:
        views[v] = {r["key"]: r["logits"] for r in map(json.loads, (sr / f"data/cache/p3s_stepa/views/{v}.jsonl").read_text().splitlines())}
    sig = {r["key"]: r["scores"] for r in map(json.loads, (sr / "data/cache/p3s_stepa/siglip3a.jsonl").read_text().splitlines())}
    params = film.load_params()
    sc = film.apply_saved_shotqa(qs, views, sig, params)
    rep = json.loads(film.STATIC_VAL.read_text())["results"][film.WINNER]
    out = {}
    for part, sub in (("selection", [q for q in qs if q["half"] == "sel"]), ("full", qs)):
        acc = C.accuracy(sub, sc)
        out[part] = {"reproduced_mean": acc["mean"], "reported_mean": rep[part]["mean"],
                     "max_abs_attr_diff": max(abs(acc["per_attribute"][k] - rep[part]["per_attribute"][k]) for k in acc["per_attribute"]),
                     "n": len(sub)}
    out["split"] = {k: sp["info"][k] for k in ("distinct_ids", "films", "seed")}
    out["params"] = params
    out["ok"] = all(out[p]["max_abs_attr_diff"] < 1e-12 for p in ("selection", "full"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("selection", "full", "ok")}, indent=1))


if __name__ == "__main__":
    main()
