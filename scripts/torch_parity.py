"""an earlier change torch parity (docs/plans/torch_parity.md): score the frozen Phase 3B adapter (MLX ckpt_100, sha 4c030967...) ported to transformers + peft
on the seeded ShotQA film-split VALIDATION subset (the first N of `p3b_eval_val.py --n`) and compare with the frozen MLX predictions (sha cebea813...).
Inference only. Never reads gold, dev, ShotBench or CameraBench. Per-question torch results stay under data/cache (gitignored); the committed
report holds aggregates only.
  nice -n 19 python scripts/torch_parity.py --src-root <worktree holding data/cache/p3b> --n 200 --dtype bfloat16 --out results/torch_parity/report.json
  nice -n 19 python scripts/torch_parity.py --diag-inputs ...   (CPU only: compare torch prompt ids / pixel values with mlx-vlm's VisionDataset)"""
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from cgt.eval.metrics import pick_option
from cgt.p3b_vlm import torch_port as tp
from cgt.p3b_vlm.model_path import pinned_model_path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_SHA = "4c0309679c9cdf6f60e5bde3cf12fd79c409b0fb49d0bbfd84c8b36499299d1f"
PRED_SHA = "cebea813002eb24971d6ae652b38a90a0f7fc7105ceed92d9a0da3c64770ece6"
RUN = "data/cache/p3b/runs/qwen3vl2b_lora_filmsplit_run1"
LETTERS = ("A", "B", "C", "D")
ATTRS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
RANK, SCALE = 16, 2.0
GPU_CAP_S = 75 * 60
MIN_FREE_PCT = 25


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"free percentage:\s*(\d+)%", out).group(1))


def letter_token_ids(processor) -> dict[str, int]:
    """Portable copy of cgt.p3b_vlm.score.answer_token_ids (that module imports mlx): the first assistant-answer token per letter."""
    tok = processor.tokenizer
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q?"}]}
    prompt = tok.encode(processor.apply_chat_template([user], add_generation_prompt=True), add_special_tokens=False)
    out = {}
    for letter in LETTERS:
        full = processor.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": letter}]}], add_generation_prompt=False)
        ids = tok.encode(full, add_special_tokens=False)
        rest = ids[len(prompt):]
        if ids[: len(prompt)] != prompt or len(rest) < 2 or tok.decode([rest[0]]).strip() != letter:
            raise ValueError(f"letter {letter}: assistant answer is not a single first token after the generation prompt")
        out[letter] = int(rest[0])
    if len(set(out.values())) != 4:
        raise ValueError(f"letter tokens not distinct: {out}")
    return out


def prompt_inputs(processor, sample: dict, image):
    """Tokens before the answer letter: chat template of the user turn (image, then the question text) + generation prompt."""
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["messages"][0]["content"]}]}
    text = processor.apply_chat_template([user], add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt")


def load_subset(src: Path, n: int) -> tuple[list[dict], list[dict]]:
    val = [json.loads(x) for x in (src / "data/cache/p3b/val.jsonl").read_text().splitlines()]
    pf = src / RUN / "val_predictions/ckpt_100.predictions.jsonl"
    if sha256(pf) != PRED_SHA:
        raise SystemExit(f"reference predictions sha mismatch: {pf}")
    ref = [json.loads(x) for x in pf.read_text().splitlines()]
    if len(ref) != len(val) or any(a["id"] != b["id"] for a, b in zip(val, ref)):
        raise SystemExit("reference predictions are not row-aligned with val.jsonl")
    idx = sorted(np.random.default_rng(0).permutation(len(val))[:n])
    cache = src / "data/cache/p3b/images_147456_32"
    sub = []
    for i in idx:
        s = dict(val[i])
        p = cache / (Path(s["images"][0]).stem + ".jpg")
        if not p.exists():
            raise SystemExit(f"shrunk image missing (not created here): {p.name}")
        s["images"] = [str(p)]
        sub.append(s)
    return sub, [ref[i] for i in idx]


def accuracy(rows: list[dict]) -> dict:
    per = {}
    for a in ATTRS:
        r = [x for x in rows if x["attribute"] == a]
        if r:
            per[a] = {"n": len(r), "acc": float(np.mean([x["pred"] == x["gold"] for x in r]))}
    return {"per_attribute": per, "mean_acc": float(np.mean([v["acc"] for v in per.values()]))}


MLX_INPUTS = ROOT / "data/cache/torch_parity/mlx_inputs"


def dump_mlx_inputs(sub) -> dict:
    """CPU only, no model weights, separate process (mlx-vlm registers its own processor classes, so it must not share a process with the torch
    scorer): save what mlx-vlm's VisionDataset.process fed the MLX evaluation (prompt ids before the answer, pixel_values, image_grid_thw)."""
    import mlx.core as mx
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.utils import load_config, load_image_processor, load_processor

    mx.set_default_device(mx.cpu)
    path = Path(pinned_model_path())
    mproc = load_processor(path, True)  # as mlx_vlm.utils.load
    mip = load_image_processor(path)
    if mip is not None:
        mproc.image_processor = mip
    ds = VisionDataset(type("L", (), {"__len__": lambda s: 0, "__getitem__": lambda s, i: None})(), load_config(path), mproc, None, True)
    MLX_INPUTS.mkdir(parents=True, exist_ok=True)
    for k, s in enumerate(sub):
        item = ds.process({"images": s["images"], "messages": s["messages"]})
        ids = np.array(item["input_ids"]).reshape(-1)
        p = int(np.flatnonzero(np.array(item["completion_mask"]).reshape(-1))[0])
        np.savez(MLX_INPUTS / f"{k:04d}.npz", ids=ids[:p], pix=np.array(item["pixel_values"].astype(mx.float32)), grid=np.array(item["image_grid_thw"]))
    return {"n": len(sub), "mlx_image_processor": type(mproc.image_processor).__name__}


def compare_inputs(proc, k: int, enc) -> tuple[bool, bool, float] | None:
    f = MLX_INPUTS / f"{k:04d}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    ids_ok = np.array_equal(z["ids"], enc["input_ids"][0].cpu().numpy())
    grid_ok = np.array_equal(z["grid"], enc["image_grid_thw"].cpu().numpy())
    pix = enc["pixel_values"].float().cpu().numpy()
    return ids_ok, grid_ok, float(np.abs(z["pix"] - pix).max()) if z["pix"].shape == pix.shape else float("inf")


def summarize_inputs(cmp: list) -> dict:
    c = [x for x in cmp if x is not None]
    return {"n_compared": len(c), "prompt_ids_identical": sum(x[0] for x in c), "image_grid_thw_identical": sum(x[1] for x in c),
            "pixel_values_max_absdiff": max((x[2] for x in c), default=None)}


def cpu_compare(sub) -> dict:
    """CPU only, torch side of the input check (no model weights)."""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(pinned_model_path())
    res = summarize_inputs([compare_inputs(proc, k, prompt_inputs(proc, s, Image.open(s["images"][0]).convert("RGB"))) for k, s in enumerate(sub)])
    res.update({"torch_image_processor": type(proc.image_processor).__name__, "letter_ids_torch": letter_token_ids(proc)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", required=True, help="worktree whose data/cache/p3b holds val.jsonl, the image cache and the frozen run")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-mlx-inputs", action="store_true", help="CPU, mlx-vlm process only")
    ap.add_argument("--compare-inputs", action="store_true", help="CPU, torch processor only")
    a = ap.parse_args()
    src = Path(a.src_root)
    sub, ref = load_subset(src, a.n)
    if a.dump_mlx_inputs or a.compare_inputs:
        res = dump_mlx_inputs(sub) if a.dump_mlx_inputs else cpu_compare(sub)
        print(json.dumps(res, indent=1))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=1))
        return
    run(a, src, sub, ref)


def run(a, src: Path, sub: list[dict], ref: list[dict]) -> None:
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from cgt.pipeline.gpulock import GpuLock, LockHeld

    adapter = src / RUN / "ckpt_100.safetensors"
    if sha256(adapter) != ADAPTER_SHA:
        raise SystemExit("adapter sha mismatch")
    peft_dir = ROOT / "data/cache/torch_parity/peft_ckpt100"
    conv = tp.write_peft_adapter(adapter, peft_dir, RANK, SCALE, src / RUN / "adapter_config.json")
    path = pinned_model_path()
    proc = AutoProcessor.from_pretrained(path)
    ids4 = letter_token_ids(proc)
    lock = GpuLock(f"torch-parity {a.dtype}")
    while True:
        try:
            lock.acquire()
            break
        except LockHeld as e:
            print(f"waiting: {e}", flush=True)
            time.sleep(30)
    try:
        fp = free_pct()
        if fp < MIN_FREE_PCT:
            raise SystemExit(f"memory guard: {fp}% free < {MIN_FREE_PCT}%")
        t_gpu = time.perf_counter()
        dtype = getattr(torch, a.dtype)
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to("mps")
        model = PeftModel.from_pretrained(model, str(peft_dir))
        model.eval()
        # loaded adapter == converted tensors (no silently dropped modules)
        n_lora = sum(1 for n, _ in model.named_parameters() if "lora_A" in n)
        if n_lora != conv["modules"]:
            raise SystemExit(f"peft loaded {n_lora} lora_A tensors, converter wrote {conv['modules']}")
        load_s = time.perf_counter() - t_gpu
        rows, secs, incmp = [], [], []
        stopped = None
        with torch.inference_mode():
            for k, s in enumerate(sub):
                if time.perf_counter() - t_gpu > GPU_CAP_S:
                    stopped = "gpu cap reached"
                    break
                enc = prompt_inputs(proc, s, Image.open(s["images"][0]).convert("RGB"))
                incmp.append(compare_inputs(proc, k, enc))
                enc = {k: (v.to("mps", dtype) if v.is_floating_point() else v.to("mps")) for k, v in enc.items()}
                torch.mps.synchronize()
                t = time.perf_counter()
                lg = model(**enc).logits[0, -1].float()
                four = np.array([float(lg[ids4[x]]) for x in LETTERS])
                torch.mps.synchronize()
                secs.append(time.perf_counter() - t)
                rows.append({"id": s["id"], "attribute": s["attribute"], "gold": s["answer"], "pred": pick_option(four, list(LETTERS)),
                             "logits": [round(float(x), 4) for x in four]})
        gpu_s = time.perf_counter() - t_gpu
    finally:
        lock.release()
    cache = ROOT / "data/cache/torch_parity"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"torch_{a.dtype}_n{len(rows)}.predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ref = ref[: len(rows)]
    cmp = tp.compare(ref, rows)
    acc_m, acc_t = accuracy(ref), accuracy(rows)
    d_pts = 100 * (acc_t["mean_acc"] - acc_m["mean_acc"])
    crit = {"P1_argmax_agreement_ge_0.98": cmp["argmax_agreement"] >= 0.98, "P2_mean_acc_diff_le_1pt": abs(d_pts) <= 1.0,
            "P3_logprob_absdiff_mean_le_0.15": cmp["logprob_absdiff_mean"] <= 0.15}
    rep = {"plan": "docs/plans/torch_parity.md", "device": "mps", "dtype": a.dtype, "n_planned": a.n, "n_scored": len(rows), "stopped": stopped,
           "subset": "val.jsonl rows sorted(np.random.default_rng(0).permutation(5826)[:N]) (= p3b_eval_val.py --n N)",
           "adapter_sha256": ADAPTER_SHA, "reference_predictions_sha256": PRED_SHA, "converted": conv, "lora_rank": RANK, "lora_scale": SCALE,
           "letter_token_ids": ids4, "inputs_vs_mlx_vlm": summarize_inputs(incmp), "parity": cmp, "accuracy_mlx_subset": acc_m, "accuracy_torch_subset": acc_t,
           "mean_acc_diff_points_torch_minus_mlx": d_pts, "criteria": crit, "pass": all(crit.values()),
           "torch_sec_per_question_median": float(np.median(secs)), "torch_sec_per_question_mean": float(np.mean(secs)),
           "torch_model_load_s": load_s, "gpu_minutes": gpu_s / 60, "gpu_cap_minutes": GPU_CAP_S / 60,
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__}}
    print(json.dumps({k: v for k, v in rep.items() if k not in ("accuracy_mlx_subset", "accuracy_torch_subset")}, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
