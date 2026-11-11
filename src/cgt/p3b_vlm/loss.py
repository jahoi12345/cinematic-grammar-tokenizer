"""Causal-safe training loss for mlx-vlm models and a leakage probe (docs/phase3b_interface.md section 4).

Why: mlx-vlm 0.7.4's `vision_language_loss_fn` passes the padding `attention_mask` to the model. For SmolVLM (Idefics3) that mask
is applied as a full (bidirectional) mask, so the answer token leaks into earlier positions and the loss collapses to ~0 within
about 100 steps (measured). With right padding no model mask is needed for a causal LM: real tokens never attend to the padding
that follows them, and padded positions are excluded from the loss. `causal_loss_fn` therefore calls the model with mask=None."""
import mlx.core as mx
import numpy as np
from mlx import nn

_SKIP = ("input_ids", "pixel_values", "attention_mask", "completion_mask")


def causal_loss_fn(model, batch, train_on_completions: bool = False, assistant_id: int = 77091):
    """Mean cross-entropy over supervised tokens (batch['completion_mask'] if train_on_completions), model called WITHOUT a mask."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    seq = input_ids.shape[1] - 1
    labels = batch["input_ids"][:, 1:]
    kwargs = {k: v for k, v in batch.items() if k not in _SKIP}
    logits = model(input_ids[:, :-1], batch["pixel_values"], None, **kwargs).logits.astype(mx.float32)
    if logits.shape[1] < labels.shape[1]:
        logits = mx.pad(logits, ((0, 0), (0, labels.shape[1] - logits.shape[1]), (0, 0)), mode="constant", constant_values=-100)
    elif logits.shape[1] > labels.shape[1]:
        logits = logits[:, -labels.shape[1]:, :]
    lengths = mx.minimum(mx.sum(attention_mask[:, :-1], axis=1), seq)
    loss_mask = mx.arange(seq)[None, :] < lengths[:, None]
    if train_on_completions:
        if "completion_mask" not in batch:
            raise ValueError("train_on_completions needs batch['completion_mask'] (VisionDataset builds it)")
        loss_mask = loss_mask * batch["completion_mask"][:, 1:]
    ce = nn.losses.cross_entropy(logits, labels)
    return (ce * loss_mask).sum() / mx.maximum(loss_mask.sum(), 1)


def leakage_probe(model, item: dict, first_supervised: int | None = None) -> dict:
    """Change ONE supervised token of a processed sample and measure how much logits at earlier positions move.
    Returns {"with_mask": x, "without_mask": y}: max |logit change| over positions before the changed token when the model is
    called with the padding mask / with mask=None. A causal-safe call has exactly 0.0. The probe runs the model in eval mode (dropout off) and restores the previous mode. `item` = VisionDataset.process(...) output."""
    was_training = bool(getattr(model, "training", False))
    model.eval()  # dropout off: with LoRA dropout active, two forward passes differ even for a causal model (false positive after resume)
    try:
        ids = np.array(item["input_ids"]).reshape(1, -1)
        am = np.array(item["attention_mask"]).reshape(1, -1)
        cm = np.array(item["completion_mask"]).reshape(-1)
        p = int(np.flatnonzero(cm)[0]) if first_supervised is None else first_supervised
        ids2 = ids.copy()
        ids2[0, p] = (int(ids[0, p]) + 1) % 1000 + 5
        kw = {k: mx.array(np.array(v)) for k, v in item.items() if k not in _SKIP}
        pv = mx.array(np.array(item["pixel_values"]))
        out = {}
        for name, mask in (("with_mask", mx.array(am[:, :-1])), ("without_mask", None)):
            a = np.array(model(mx.array(ids[:, :-1]), pv, mask, **kw).logits.astype(mx.float32))
            b = np.array(model(mx.array(ids2[:, :-1]), pv, mask, **kw).logits.astype(mx.float32))
            out[name] = float(np.abs(a - b).max(axis=-1)[0][: p - 1 + 1].max()) if p >= 1 else 0.0
        return out
    finally:
        if was_training:
            model.train()
