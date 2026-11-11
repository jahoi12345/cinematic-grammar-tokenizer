"""Torch (transformers + peft) pieces of the Step B LoRA recipe, mirroring scripts/stepb_train.py.

- `lr_at(update)`: the MLX schedule join_schedules([linear 0 -> peak over warmup], [cosine peak -> end over total - warmup]) evaluated at
  update - 1 (MLX's optimizer step counter starts at 0, so update k uses schedule(k - 1)).
- `build_item`: the torch-native training item: chat template of [user (image, text), assistant (letter)], completion mask = positions
  >= token length of [user] with the generation prompt (mlx-vlm VisionDataset._completion_prefix rule).
- `completion_loss`: cgt.p3b_vlm.loss.causal_loss_fn(train_on_completions=True) for an unpadded batch of one: fp32 logits at positions
  0..L-2, cross-entropy vs ids[1:], mean over positions whose target has completion mask 1.
- `grad_stats` / `loss_stats`: aggregate comparisons (numpy only).
torch is imported lazily so the numpy helpers stay importable without it."""
import math

import numpy as np

PEAK_LR, END_LR, WARMUP_FRAC, ACCUM, CLIP = 2e-5, 2e-6, 0.03, 16, 1.0
B0_UPDATES = 859


def schedule(step: float, total: int = B0_UPDATES, peak: float = PEAK_LR, end: float = END_LR, warmup_frac: float = WARMUP_FRAC) -> float:
    """MLX `join_schedules([linear_schedule(0, peak, max(w, 1)), cosine_decay(peak, max(total - w, 1), end)], [w])` at `step`."""
    w = round(warmup_frac * total)
    if step < w:
        return min(step, max(w, 1)) * (peak / max(w, 1))
    d = max(total - w, 1)
    s = min(step - w, d)
    return end + 0.5 * (1.0 + math.cos(math.pi / d * s)) * (peak - end)


def lr_at(update: int, total: int = B0_UPDATES) -> float:
    """Learning rate used by update `update` (1-based), as logged in updates.jsonl."""
    return schedule(update - 1, total)


def chat_messages(sample: dict) -> tuple[list[dict], list[dict]]:
    """(user-only conversation, full conversation) in the HF multimodal message format (image first, then the question text)."""
    msgs = sample["messages"]
    if [m["role"] for m in msgs] != ["user", "assistant"]:
        raise ValueError(f"{sample.get('id')}: expected one user and one assistant turn")
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": msgs[0]["content"]}]}
    asst = {"role": "assistant", "content": [{"type": "text", "text": msgs[1]["content"]}]}
    return [user], [user, asst]


def completion_mask_from_prefix(n_tokens: int, prefix_len: int) -> np.ndarray:
    """1 for positions >= prefix_len (the assistant answer and its end tokens), else 0."""
    if not 0 < prefix_len < n_tokens:
        raise ValueError(f"prefix length {prefix_len} must lie inside the sequence (length {n_tokens})")
    return (np.arange(n_tokens) >= prefix_len).astype(np.int32)


def build_item(processor, sample: dict, image) -> dict:
    """Torch training item: {input_ids (1, L), pixel_values, image_grid_thw, completion_mask (1, L)[, mm_token_type_ids]} as torch tensors."""
    import torch

    user, full = chat_messages(sample)
    text = processor.apply_chat_template(full, add_generation_prompt=False)
    prefix = processor.apply_chat_template(user, add_generation_prompt=True)
    enc = processor(text=[text], images=[image], return_tensors="pt")
    pids = processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"][0]
    plen, n = pids.shape[0], enc["input_ids"].shape[1]
    if plen >= n or not torch.equal(enc["input_ids"][0, :plen], pids):
        raise ValueError(f"{sample.get('id')}: the generation-prompt prefix is not a token prefix of the full conversation")
    cm = torch.from_numpy(completion_mask_from_prefix(n, plen)).unsqueeze(0)
    out = {"input_ids": enc["input_ids"], "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"], "completion_mask": cm}
    if "mm_token_type_ids" in enc:  # transformers 5.x Qwen3-VL needs it for M-RoPE
        out["mm_token_type_ids"] = enc["mm_token_type_ids"]
    return out


def completion_loss(logits, input_ids, completion_mask):
    """Mean fp32 cross-entropy over targets ids[:, 1:] whose completion mask is 1, from logits at positions 0..L-2.
    `logits` may cover L or L-1 positions (model fed the full or the shifted ids); batch of one, no padding."""
    import torch
    import torch.nn.functional as F

    seq = input_ids.shape[1] - 1
    lg = logits[:, :seq].float()
    labels = input_ids[:, 1:]
    mask = completion_mask[:, 1:].to(lg.dtype)
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels.reshape(-1), reduction="none").reshape(labels.shape)
    return (ce * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def mlx_key_of_torch_param(name: str) -> str:
    """peft parameter name `base_model.model.model.language_model.layers.L.x.y.lora_A.default.weight` -> MLX `language_model.model.layers.L.x.y.lora_a`."""
    parts = name.split(".")
    i = parts.index("layers")
    tail = parts[-3] if parts[-1] == "weight" and parts[-2] == "default" else parts[-2]
    ab = {"lora_A": "lora_a", "lora_B": "lora_b"}[tail]
    mod = ".".join(parts[i: i + 4])  # layers.L.<self_attn|mlp>.<proj>
    return f"language_model.model.{mod}.{ab}"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def grad_stats(ref: dict[str, np.ndarray], other: dict[str, np.ndarray]) -> dict:
    """Per-tensor cosine and relative norm difference of `other` vs `ref` (same keys, same shapes, MLX layout), plus global numbers."""
    if sorted(ref) != sorted(other):
        raise ValueError("gradient key sets differ")
    cos, rel, worst = [], [], []
    for k in sorted(ref):
        a, b = np.asarray(ref[k], np.float64).ravel(), np.asarray(other[k], np.float64).ravel()
        if a.shape != b.shape:
            raise ValueError(f"{k}: shape {a.shape} vs {b.shape}")
        c = _cos(a, b)
        cos.append(c)
        na = np.linalg.norm(a)
        rel.append(abs(np.linalg.norm(b) - na) / na if na > 0 else float("nan"))
        worst.append((c, k))
    ga = np.concatenate([np.asarray(ref[k], np.float64).ravel() for k in sorted(ref)])
    gb = np.concatenate([np.asarray(other[k], np.float64).ravel() for k in sorted(ref)])
    cos_a = np.array(cos)
    worst.sort()
    return {"n_tensors": len(cos), "cos_median": float(np.nanmedian(cos_a)), "cos_min": float(np.nanmin(cos_a)),
            "cos_p05": float(np.nanquantile(cos_a, 0.05)), "cos_mean": float(np.nanmean(cos_a)),
            "n_cos_below_0.99": int((cos_a < 0.99).sum()), "n_cos_below_0.95": int((cos_a < 0.95).sum()),
            "lowest_cos_tensors": [{"key": k, "cos": round(c, 5)} for c, k in worst[:5]],
            "rel_norm_diff_median": float(np.nanmedian(rel)), "rel_norm_diff_max": float(np.nanmax(rel)),
            "global_cos": _cos(ga, gb), "global_norm_ref": float(np.linalg.norm(ga)), "global_norm_other": float(np.linalg.norm(gb)),
            "global_rel_norm_diff": float(abs(np.linalg.norm(gb) - np.linalg.norm(ga)) / np.linalg.norm(ga))}


def loss_stats(ref: list[float], other: list[float]) -> dict:
    """Aggregate per-sample loss differences (other - ref)."""
    a, b = np.asarray(ref, np.float64), np.asarray(other, np.float64)
    if a.shape != b.shape:
        raise ValueError("loss lists differ in length")
    d = b - a
    return {"n": int(a.size), "mean_ref": float(a.mean()), "mean_other": float(b.mean()), "absdiff_mean": float(np.abs(d).mean()),
            "absdiff_max": float(np.abs(d).max()), "absdiff_median": float(np.median(np.abs(d))), "signed_diff_mean": float(d.mean()),
            "pearson": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else None}
