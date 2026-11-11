"""Baseline multi-label ATOM classifier on Phase 4 features with strict validation-label isolation (logic in cgt.p4_camera_move.baseline, docs/phase4_interface.md section 4).
Feature sets (--features, comma separated): flow (classical DIS geometry), depth (depth-parallax module), siglip (frozen SigLIP frame embeddings); --tag names the outputs
`results/phase4/{frozen_config,baseline_atoms,val_probs}_<tag>.*`. Every invocation is ONE validation look and appends a line to results/phase4/validation_looks.jsonl;
an existing final result for the tag is only replaced with --allow-overwrite-final (which is logged). Never reads ShotBench movement or CameraBench.
Usage: uv run python scripts/p4_baseline_atoms.py --features flow --tag flow"""
import argparse
import json
from pathlib import Path

import pandas as pd

from cgt.p4_camera_move.baseline import code_sha, run
from cgt.p4_camera_move.labels import MovementLabelStore

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILES = {"flow": "features_shotqa.parquet", "depth": "features_depth_shotqa.parquet", "siglip": "features_siglip_shotqa.parquet"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="flow")
    parser.add_argument("--tag", default="flow")
    parser.add_argument("--allow-overwrite-final", action="store_true")
    args = parser.parse_args()
    names = args.features.split(",")
    manifest = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    feats = pd.concat([pd.read_parquet(ROOT / "data/cache/p4" / FEATURE_FILES[n]).set_index("clip") for n in names], axis=1)
    store = MovementLabelStore(manifest["clips"]["val"])
    res = run(feats=feats, manifest=manifest, store=store, feature_sets=names, tag=args.tag, out_dir=ROOT / "results/phase4", looks_path=ROOT / "results/phase4/validation_looks.jsonl",
              code_sha=code_sha(), allow_overwrite_final=args.allow_overwrite_final)
    print((ROOT / f"results/phase4/baseline_atoms_{args.tag}.md").read_text())
    print("validation looks so far:", res["n_validation_looks_so_far"])


if __name__ == "__main__":
    main()
