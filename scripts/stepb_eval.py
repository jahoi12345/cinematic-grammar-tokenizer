"""Step B validation of one adapter (inference only): the deduplicated ShotQA film-held-out validation set (5,762 questions,
primary) and/or the CineScale2 split-v3 val questions (1,178, secondary). Letter logits, Phase 3B scorer. The adapter must be the
frozen one (sha 4c030967...) or registered in its run manifest. Never reads ShotBench, CameraBench, gold or dev.
  nice -n 19 <venv>/bin/python scripts/stepb_eval.py --adapter <file> --manifest results/stepb/runs/<tag>/manifest.json \
      --out results/stepb/runs/<tag>/val_ckpt050 [--sets shotqa,cs2]"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
from mlx_vlm.utils import load

from cgt.p3b_vlm import guards
from cgt.p3b_vlm.model_path import pinned_model_path
from cgt.p3b_vlm.score import answer_token_ids, evaluate_samples, summarize
from cgt.stepb.evaluate import evaluate_cs2, shotqa_points, summarize_cs2

ROOT = Path(__file__).resolve().parents[1]
REPO, REVISION = "Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"
DATA = ROOT / "data/cache/stepb"
FROZEN_SHA = "4c0309679c9cdf6f60e5bde3cf12fd79c409b0fb49d0bbfd84c8b36499299d1f"


class ListView:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def load_set(name: str) -> list[dict]:
    summ = json.loads((ROOT / "results/stepb/data_summary.json").read_text())["files"][name]
    p = DATA / f"{name}.jsonl"
    if guards.sha256_file(p) != summ["sha256"]:
        raise guards.DataRootViolation(f"{p} differs from the committed data summary")
    return [json.loads(x) for x in p.read_text().splitlines()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="shotqa,cs2")
    a = ap.parse_args()
    try:
        run(a)
    except guards.FailClosed as e:
        print(f"FAIL CLOSED ({type(e).__name__}, exit {e.exit_code}): {e}", flush=True)
        raise SystemExit(e.exit_code) from e


def run(a) -> None:
    adapter = Path(a.adapter)
    sha = guards.sha256_file(adapter)
    if sha != FROZEN_SHA:
        if not a.manifest:
            raise guards.AdapterHashError("a non-frozen adapter needs --manifest")
        guards.require_registered_hash(Path(a.manifest), adapter)
    sets = a.sets.split(",")
    model, processor = load(pinned_model_path(REPO, REVISION))
    cfg = model.config.__dict__
    ids4 = answer_token_ids(processor)
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=16, alpha=32.0, dropout=0.05, verbose=False)
    model.load_weights(list(mx.load(str(adapter)).items()), strict=False)
    model.eval()
    ds = VisionDataset(ListView([]), cfg, processor, None, True)
    out = {"adapter": str(adapter), "adapter_sha256": sha, "answer_token_ids": ids4}
    pred_dir = DATA / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.out).name + "_" + sha[:12]
    if "shotqa" in sets:
        val = load_set("shotqa_val_dedup")
        t0 = time.perf_counter()
        res = evaluate_samples(model, ds, [{**s, "images": [s["shrunk"]]} for s in val], ids4)
        out["shotqa"] = {**shotqa_points(summarize(res)), "seconds": round(time.perf_counter() - t0, 1)}
        (pred_dir / f"{stem}.shotqa.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
    if "cs2" in sets:
        val = load_set("cs2_val")
        t0 = time.perf_counter()
        res = evaluate_cs2(model, ds, val, ids4)
        out["cs2"] = {**summarize_cs2(res), "seconds": round(time.perf_counter() - t0, 1)}
        (pred_dir / f"{stem}.cs2.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
    if guards.sha256_file(adapter) != sha:
        raise guards.AdapterHashError("adapter changed during evaluation")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(str(a.out) + ".summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k in ("shotqa", "cs2")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
