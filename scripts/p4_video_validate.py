"""Descriptive validation of the FROZEN video cross-fit result (docs/phase4_video_plan.md Amendment 2 item 7): the selected-epoch adapter of the final fit, the geometry model (flow +
depth, fitted on all train clips) and their combination with the weights frozen from the OOF questions, on the 140 validation clips, plus the controls on the video model (all frames
replaced by the middle frame, shuffled frame order, grey frames). This invocation is ONE validation look: it is recorded in results/phase4/validation_looks.jsonl BEFORE the validation labels
are read. Numbers are descriptive; nothing is selected on them. The adapter SHA-256 is checked against the training manifest. No ShotBench movement, no CameraBench.
Usage: uv run python scripts/p4_video_validate.py --tag v1      (test aid for the mechanics: --test-aid --limit 6 [--cpu], uses TRAIN clips and records no look)"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "8")

import mlx.core as mx
import numpy as np
import pandas as pd
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load
from PIL import Image

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.score import answer_token_ids
from cgt.p4_camera_move import video_vlm as V
from cgt.p4_camera_move.baseline import (
    code_sha,
    fit_scores,
    option_scores,
    record_look,
    wilson,
)
from cgt.p4_camera_move.labels import MovementLabelStore, schema_moves
from cgt.p4_camera_move.video_score import ListView, score_samples

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
FRAMES = ROOT / "data/cache/p4/video_frames"
LETTERS = "ABCD"


def lsm(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def grey_like(frames: list[Path]) -> list[Path]:
    out = []
    for p in frames:
        w, h = Image.open(p).size
        g = FRAMES / "grey" / f"grey_{w}x{h}.jpg"
        if not g.exists():
            g.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (w, h), (128, 128, 128)).save(g, quality=92)
        out.append(g)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--test-aid", action="store_true", help="mechanics check on TRAIN clips with the base model: no look, no validation label, writes to --out-dir only")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
    base = ROOT / "results/phase4/runs" / f"video_{a.tag}"
    out = Path(a.out_dir) if a.out_dir else base
    out.mkdir(parents=True, exist_ok=True)
    comb = json.loads((base / "combine.json").read_text()) if (base / "combine.json").exists() else None
    if comb is None and not a.test_aid:
        raise SystemExit("combine.json is missing: the OOF analysis must run first")
    E = comb["selected_epoch_E"] if comb else 1
    ws, wg = (comb["weights"]["w_s"], comb["weights"]["w_g"]) if comb else (1.0, 1.0)
    ck = ROOT / "data/cache/p4/video_runs" / a.tag / "final" / f"ckpt_e{E}.safetensors"
    adapter_sha = None
    if not a.test_aid:
        adapter_sha = guards.sha256_file(ck)
        recorded = json.loads((base / "final/manifest.json").read_text())["checkpoints"][str(E)]["sha256"]
        if adapter_sha != recorded:
            raise guards.AdapterHashError("the selected adapter does not match the training manifest")
    man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    store = MovementLabelStore(man["clips"]["val"])
    train_rows = [r for r in store.rows(man["clips"]["train"]) if r["atoms"]]  # TRAIN labels
    atoms = [x for x in schema_moves() if any(x in r["atoms"] for r in train_rows)]
    feats = pd.concat([pd.read_parquet(ROOT / "data/cache/p4" / f).set_index("clip") for f in ("features_shotqa.parquet", "features_depth_shotqa.parquet")], axis=1)
    c_val = json.loads((ROOT / "results/phase4/frozen_config_flow_depth.json").read_text())["chosen_C"]
    Xtr = feats.loc[[r["clip"] for r in train_rows]].to_numpy(dtype=np.float64)
    Ytr = np.array([[x in r["atoms"] for x in atoms] for r in train_rows], dtype=int)
    eval_clips = man["clips"]["train"][: a.limit] if a.test_aid else list(man["clips"]["val"])
    P_all = fit_scores(Xtr, Ytr, feats.loc[eval_clips].to_numpy(dtype=np.float64), c_val)  # geometry model fitted on all train clips; features only
    # ---- the look is recorded BEFORE any validation label is read
    if not a.test_aid:
        record_look(ROOT / "results/phase4/validation_looks.jsonl", kind="video_validation", tag=f"video_{a.tag}", feature_sets=[f"video_lora_e{E}", "flow", "depth"],
                    config_hash=f"E{E}_ws{ws}_wg{wg}_adapter{adapter_sha[:16]}")
        rows_all = store.rows(man["clips"]["val"], final_evaluation=True)
    else:
        rows_all = store.rows(eval_clips)  # TRAIN clips only
    keep = [i for i, r in enumerate(rows_all) if r["atoms"]]
    rows = [rows_all[i] for i in keep]
    P = P_all[keep]
    samples = [V.build_sample(r, V.extract_frames(QA / r["clip"], a.frames, a.tokens, FRAMES)) for r in rows]
    model, processor = load(pinned_model_path())
    V.configure_processor(processor)
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
    if not a.test_aid:
        model.load_weights(list(mx.load(str(ck)).items()), strict=False)
    model.eval()
    ds = VisionDataset(ListView([]), model.config.__dict__, processor, None, True)
    golds = [r["gold_letter"] for r in rows]

    def variant(kind: str) -> list[dict]:
        out_s = []
        for i, s in enumerate(samples):
            fr = [Path(p) for p in s["images"]]
            if kind == "static_middle":
                fr = [fr[len(fr) // 2]] * len(fr)
            elif kind == "shuffled":
                fr = [fr[j] for j in np.random.default_rng([0, i]).permutation(len(fr))]
            elif kind == "grey":
                fr = grey_like(fr)
            out_s.append({**s, "images": [str(p) for p in fr]})
        return out_s
    conds = ("normal", "static_middle", "shuffled", "grey")
    logits = {c: score_samples(model, ds, variant(c), golds, ids4) for c in conds}
    geo = [np.array([option_scores(r, p, atoms)[c] for c in LETTERS]) for r, p in zip(rows, P, strict=True)]
    gold_i = np.array([LETTERS.index(g) for g in golds])
    src = np.array([r["source"] for r in rows])
    compound = np.array([r["kind"] != "single" for r in rows])

    def hit(sc):
        return np.array([int(np.argmax(s)) == g for s, g in zip(sc, gold_i, strict=True)])
    H = {"geometry_only": hit(geo)}
    for c in conds:
        H[f"video_only/{c}"] = hit([lsm(z) for z in logits[c]])
        H[f"combined/{c}"] = hit([ws * lsm(z) + wg * g for z, g in zip(logits[c], geo, strict=True)])
    usrc = sorted(set(src))
    rng = np.random.default_rng(0)
    boots = [np.concatenate([np.where(src == s)[0] for s in rng.choice(usrc, len(usrc), replace=True)]) for _ in range(1000)]

    def paired(x, y):
        d = [x[ix].mean() - y[ix].mean() for ix in boots]
        return {"diff": round(float(x.mean() - y.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    res = {"tag": a.tag, "test_aid": a.test_aid, "selected_epoch_E": E, "weights": {"w_s": ws, "w_g": wg}, "adapter_sha256": adapter_sha, "code_sha": code_sha(), "n_val_questions": len(rows), "n_sources": len(usrc),
           "option_accuracy": {k: {"acc": round(float(v.mean()), 4), "correct": int(v.sum()), "ci95": [round(x, 1) for x in wilson(int(v.sum()), len(v))], "single_label_acc": round(float(v[~compound].mean()), 4),
                                   "n_single": int((~compound).sum()), "compound_acc": round(float(v[compound].mean()), 4), "n_compound": int(compound.sum())} for k, v in H.items()},
           "paired_bootstrap_by_source_1000": {"combined_minus_video_only": paired(H["combined/normal"], H["video_only/normal"]), "combined_minus_geometry": paired(H["combined/normal"], H["geometry_only"]),
                                               "video_only_minus_geometry": paired(H["video_only/normal"], H["geometry_only"]), "video_normal_minus_static_middle": paired(H["video_only/normal"], H["video_only/static_middle"]),
                                               "video_normal_minus_shuffled": paired(H["video_only/normal"], H["video_only/shuffled"]), "video_normal_minus_grey": paired(H["video_only/normal"], H["video_only/grey"])},
           "chance": 0.25, "note": "descriptive; selected nothing; every pass of this script is a logged validation look" if not a.test_aid else "TEST AID on TRAIN clips with the base model: mechanics only"}
    (out / "validation.json").write_text(json.dumps(res, indent=1))
    (out / "validation_logits.json").write_text(json.dumps({"clips": [r["clip"] for r in rows], "gold": golds, "logits": {c: [np.round(z, 4).tolist() for z in logits[c]] for c in conds}}))
    L = [f"# Video model validation `{a.tag}` ({'TEST AID, TRAIN clips, base model' if a.test_aid else 'descriptive; 140-clip validation sources'}; E = {E}; w_s = {ws}, w_g = {wg})", "",
         f"{len(rows)} questions, {len(usrc)} sources; chance 25%. Adapter SHA-256 `{adapter_sha}`.", "", "| scorer | option accuracy | correct | 95% Wilson | single-label | compound |", "|---|---:|---:|---|---:|---:|"]
    L += [f"| {k} | {v['acc']:.4f} | {v['correct']}/{len(rows)} | {v['ci95']} | {v['single_label_acc']:.3f} (n={v['n_single']}) | {v['compound_acc']:.3f} (n={v['n_compound']}) |" for k, v in res["option_accuracy"].items()]
    L += ["", "Paired bootstrap by source (1,000 resamples): accuracy difference [95% interval]", ""]
    L += [f"- {k}: {v['diff']:+.4f} {v['ci95']}" for k, v in res["paired_bootstrap_by_source_1000"].items()]
    (out / "validation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
