"""Geometry (flow + depth) out-of-fold atom probabilities on the SAME 3 source-grouped folds as the video cross-fit (Amendment 2 items 2 and 6), TRAIN sources only: per clip, the
probabilities of the 15 atoms from a model fitted on the other two folds (per-atom logistic regression, balanced, C = the value chosen by source-grouped CV for flow+depth, read from
results/phase4/frozen_config_flow_depth.json). Reads TRAIN labels only through the locked store. CPU. Writes results/phase4/runs/video_<tag>/geom_oof.json.
Usage: uv run python scripts/p4_video_geom_oof.py --tag v1"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.p4_camera_move import crossfit as X
from cgt.p4_camera_move.baseline import fit_scores
from cgt.p4_camera_move.labels import MovementLabelStore, schema_moves

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()
    man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    store = MovementLabelStore(man["clips"]["val"])
    rows = [r for r in store.rows(man["clips"]["train"]) if r["atoms"]]
    atoms = [x for x in schema_moves() if any(x in r["atoms"] for r in rows)]
    feats = pd.concat([pd.read_parquet(ROOT / "data/cache/p4" / f).set_index("clip") for f in ("features_shotqa.parquet", "features_depth_shotqa.parquet")], axis=1)
    c_val = json.loads((ROOT / "results/phase4/frozen_config_flow_depth.json").read_text())["chosen_C"]
    X_all = feats.loc[[r["clip"] for r in rows]].to_numpy(dtype=np.float64)
    Y_all = np.array([[x in r["atoms"] for x in atoms] for r in rows], dtype=int)
    idx = {r["clip"]: i for i, r in enumerate(rows)}
    oof = {}
    for k in range(X.N_FOLDS):
        tr, ho = X.stage_split(rows, f"fold{k}")
        ti, hi = [idx[r["clip"]] for r in tr], [idx[r["clip"]] for r in ho]
        P = fit_scores(X_all[ti], Y_all[ti], X_all[hi], c_val)
        for r, p in zip(ho, P, strict=True):
            oof[r["clip"]] = [round(float(v), 6) for v in p]
    assert len(oof) == len(rows)
    out = ROOT / "results/phase4/runs" / f"video_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "geom_oof.json").write_text(json.dumps({"atoms": atoms, "C": c_val, "feature_sets": ["flow", "depth"], "n": len(oof), "probs": oof}))
    print(f"wrote geom_oof.json: {len(oof)} clips, C={c_val}, {len(atoms)} atoms")


if __name__ == "__main__":
    main()
