"""EXPLORATORY, post-hoc breakdown of the published predictions of the six final evaluations (docs/phase4_video_plan.md Amendment 9; nothing here is pre-registered, nothing is retuned, nothing is re-evaluated).
It reads only `results/phase4/final_eval/raw/<dataset>__<tag>.predictions.jsonl` (options, gold, chosen) and the option wording; it opens no label file and touches no model. Any explanation suggested by
these tables is a HYPOTHESIS. Writes results/phase4/final_eval_exploratory.md.
Usage: uv run python scripts/p4_final_exploratory.py"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move.labels import parse_move_label

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/phase4/final_eval/raw"


def load(dataset: str, model: str, variant: str = "main") -> list[dict]:
    p = RAW / f"{FE.key_of(dataset, FE.MODEL_TAGS[model])}.predictions.jsonl"
    return [r for r in (json.loads(x) for x in p.read_text().splitlines()) if r["variant"] == variant]


def n_atoms(text: str) -> int:
    return len(parse_move_label(text).atoms)


def breakdown(dataset: str) -> list[str]:
    L = [f"## {dataset} (variant `main`)", ""]
    rows = {m: load(dataset, m) for m in FE.MODELS}
    base = rows["geometry"]
    offered = np.mean([n_atoms(q["options"][c]) > 1 for q in base for c in "ABCD"])
    gold = np.mean([len(q["gold_atoms"]) > 1 for q in base])
    L += [f"n = {len(base)} questions. Share of OFFERED options with more than one atom: {offered:.1%}; share of questions whose GOLD set has more than one atom: {gold:.1%}.", "",
          "| model | accuracy | predictions that are a compound option | accuracy when gold is single | accuracy when gold is compound | mean atoms predicted | mean atoms gold |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m, rs in rows.items():
        comp = np.array([n_atoms(r["options"][r["chosen"]]) > 1 for r in rs])
        hit = np.array([r["hit"] for r in rs])
        g = np.array([len(r["gold_atoms"]) > 1 for r in rs])
        L.append(f"| {m} | {hit.mean():.3f} | {comp.mean():.1%} | {hit[~g].mean():.3f} (n={int((~g).sum())}) | {hit[g].mean():.3f} (n={int(g.sum())}) | {np.mean([n_atoms(r['options'][r['chosen']]) for r in rs]):.2f} | {np.mean([len(r['gold_atoms']) for r in rs]):.2f} |")
    atoms = sorted({a for r in base for a in r["gold_atoms"]}, key=lambda a: -sum(a in r["gold_atoms"] for r in base))
    L += ["", "Hit rate among the questions whose gold set contains the atom (per-atom, with support):", "", "| atom | support | geometry | video | combined |", "|---|---:|---:|---:|---:|"]
    for a in atoms:
        idx = [i for i, r in enumerate(base) if a in r["gold_atoms"]]
        L.append(f"| {a} | {len(idx)} | " + " | ".join(f"{np.mean([rows[m][i]['hit'] for i in idx]):.2f}" for m in ("geometry", "video", "combined")) + " |")
    pick = Counter(rows["video"][i]["options"][rows["video"][i]["chosen"]] for i in range(len(base)))
    L += ["", "Most frequent option texts chosen by the video model: " + "; ".join(f"{t} ({c})" for t, c in pick.most_common(5)) + ".", ""]
    return L


def main() -> None:
    L = ["# EXPLORATORY breakdown of the final-evaluation predictions (post hoc; nothing pre-registered, nothing retuned; any explanation is a hypothesis)", "",
         "Computed only from the published prediction files. Differences between rows are not tested; the pre-registered comparison is `*_paired.md`.", ""]
    for d in FE.DATASETS:
        L += breakdown(d)
    Path(ROOT / "results/phase4/final_eval_exploratory.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
