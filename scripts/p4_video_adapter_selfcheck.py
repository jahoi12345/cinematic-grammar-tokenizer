"""CPU-only self-check of the 8-frames-as-images adapter (MLX forced onto the CPU; no GPU). Uses ShotQA TRAIN clips only (labels through the locked store: train rows only), the base
Qwen3-VL-2B (pinned revision, no Phase 3B adapter), the Phase 3B causal-safe loss and leak probe. Writes results/phase4/video_adapter_selfcheck.md.
Usage: uv run python scripts/p4_video_adapter_selfcheck.py [--frames 8 --tokens 64]"""
import argparse
import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)  # before any model is created: this check never touches the GPU

from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.utils import load

from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p4_camera_move import video_vlm as V
from cgt.p4_camera_move.labels import MovementLabelStore

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/external/shotqa"
rows = []


def check(name, ok, detail=""):
    rows.append((name, ok, detail))
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    check("frame indices: k centres of k equal bins, strictly increasing", V.frame_indices(96, 8) == [6, 18, 30, 42, 54, 66, 78, 90] and V.frame_indices(8, 8) == list(range(8)), str(V.frame_indices(96, 8)))
    man = json.loads((ROOT / "results/phase4/split_manifest.json").read_text())
    store = MovementLabelStore(man["clips"]["val"])
    row = next(r for r in store.rows(man["clips"]["train"][:40]) if r["atoms"])  # TRAIN labels only
    cache = Path(tempfile.mkdtemp())
    fr = V.extract_frames(QA / row["clip"], a.frames, a.tokens, cache)
    from PIL import Image
    sizes = [Image.open(p).size for p in fr]
    check(f"{a.frames} frames extracted, sides multiples of 32, area within the {a.tokens}-token budget", len(fr) == a.frames and all(w % 32 == 0 and h % 32 == 0 and w * h <= V.budget_px(a.tokens) for w, h in sizes), f"size {sizes[0]}")
    mt = [p.stat().st_mtime_ns for p in fr]
    check("frame extraction is idempotent (second call reuses the files)", V.extract_frames(QA / row["clip"], a.frames, a.tokens, cache) == fr and [p.stat().st_mtime_ns for p in fr] == mt)
    sample = V.build_sample(row, fr)
    check("sample prompt keeps the options and starts with the frame line; the answer turn is the gold letter", V.check_prompt_format(sample["messages"][0]["content"]) and sample["messages"][1]["content"] == row["gold_letter"], sample["messages"][0]["content"][:90].replace("\n", " | "))
    path = pinned_model_path()
    model, processor = load(path)
    V.configure_processor(processor)
    ds = VisionDataset(ListView([sample]), model.config.__dict__, processor, None, True)
    item = ds.process(sample)
    ids = np.array(item["input_ids"]).reshape(-1)
    grid = np.array(item["image_grid_thw"]).reshape(-1, 3)
    img_id = model.config.__dict__.get("image_token_id") or model.config.__dict__.get("image_token_index")
    n_img_tok = int((ids == img_id).sum())
    check("the processor makes one image_grid_thw row per frame", grid.shape[0] == a.frames, f"grid {grid.tolist()[:2]}...")
    check("image tokens in input_ids equal the grid-derived count and stay within K x budget", n_img_tok == V.image_token_count(grid) and n_img_tok <= a.frames * a.tokens, f"{n_img_tok} visual tokens, sequence {ids.size}")
    cm = np.array(item["completion_mask"]).reshape(-1)
    sup = ids[cm == 1]
    tok = processor.tokenizer
    check("the supervised tokens are exactly the answer letter and the end-of-turn tokens (no prompt token, no image token)", tok.decode([int(sup[0])]).strip() == row["gold_letter"] and not (np.isin(sup, [img_id])).any() and len(sup) <= 3, f"supervised {len(sup)} tokens: {tok.decode(sup.tolist())!r}")
    leak = leakage_probe(model, item)
    check("leak probe on the multi-image sample: without_mask is exactly 0.0 (causal-safe loss)", leak["without_mask"] == 0.0, str(leak))
    batch = {k: v for k, v in item.items()}
    loss = float(causal_loss_fn(model, batch, train_on_completions=True))
    check("causal-safe loss on the multi-image sample is finite and positive (CPU forward)", np.isfinite(loss) and loss > 0, f"loss {loss:.3f}")
    (ROOT / "results/phase4/video_adapter_selfcheck.md").write_text(f"# Video adapter self-check (CPU, base Qwen3-VL-2B pinned revision, {a.frames} frames x <= {a.tokens} tokens)\n\n| check | result | detail |\n|---|---|---|\n" + "\n".join(
        f"| {n} | **{'PASS' if ok else 'FAIL'}** | {d.replace('|', '/')} |" for n, ok, d in rows) + f"\n\nAll {len(rows)} checks passed: {all(r[1] for r in rows)}.\n")
    raise SystemExit(0 if all(r[1] for r in rows) else 1)


if __name__ == "__main__":
    main()
