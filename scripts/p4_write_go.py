"""Writes the author's go-ahead file(s) for the sealed Phase 4 final evaluations: `results/phase4/final_eval/GO_<dataset>__<model tag>`, one per evaluation, each bound to the dataset, the
model tag, the adapter hash, the configuration hash and the SHA-256 of the freeze record AS COMMITTED (`final_eval.go_text`). `scripts/p4_final_eval.py` refuses any go file with other
content. The hashes are recomputed from the committed freeze record, the adapter file and the frozen geometry configuration (`final_run.make_spec`), never typed in.
RUN BY THE AUTHOR ONLY, after the freeze was independently checked and the author decided. The implementer never runs it for the real directory.
Usage: uv run python scripts/p4_write_go.py --dataset shotbench_movement --model combined [--out-dir DIR]
       uv run python scripts/p4_write_go.py --all [--out-dir DIR]        (all six evaluations)"""
import argparse
from pathlib import Path

from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move import final_run as FR


def write_go(dataset: str, model: str, out_dir: Path) -> Path:
    spec, _registry, _adapter = FR.make_spec(dataset, model)
    out = Path(out_dir) / f"GO_{spec.key}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.go())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=FE.DATASETS)
    ap.add_argument("--model", choices=FE.MODELS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out-dir", default=str(FR.OUT_DIR))
    a = ap.parse_args()
    pairs = [(d, m) for d in FE.DATASETS for m in FE.MODELS] if a.all else [(a.dataset, a.model)]
    if not a.all and not (a.dataset and a.model):
        ap.error("--dataset and --model, or --all")
    for d, m in pairs:
        print("wrote", write_go(d, m, Path(a.out_dir)))
