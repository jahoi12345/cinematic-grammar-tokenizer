"""Option-restricted accuracy from LETTER LOGITS (docs/phase3b_interface.md section 4, docs/phase3b_plan.md section 6).
The prompt is tokenised exactly as in training (VisionDataset on the full chat sample); the logits at the position BEFORE the first
supervised (answer) token are read at the four letter-token ids of the model's chat template; the argmax over A-D is the prediction.
Causal, so the answer token that follows is never visible. No generation, no answer extraction."""
import mlx.core as mx
import numpy as np

from cgt.eval.metrics import pick_option
from cgt.p3_static_attrs.labels import parse_label_set

LETTERS = ("A", "B", "C", "D")


def is_label_set(attribute: str, answer_text: str) -> bool:
    """True iff the correct option holds more than one DISTINCT schema label (the Phase 3A definition, `len(label set) > 1`). A repeated label such
    as "Long Lens, Long Lens" is ONE label, so a comma in the text must never be used to define a label set (erratum: results/phase3b/ERRATUM_strata.md).
    This only classifies answers into reporting strata; it has no effect on scoring."""
    return len(parse_label_set(attribute, answer_text)) > 1


_SKIP = ("input_ids", "pixel_values", "attention_mask", "completion_mask")


def answer_token_ids(processor) -> dict[str, int]:
    """Letter -> the single token the chat template puts first in an assistant answer (read from the template, not assumed).
    Raises if a letter does not map to exactly one token, the tokens are not distinct, or the answer is not followed by the end of turn."""
    tok = processor.tokenizer
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q?"}]}
    prompt = tok.encode(processor.apply_chat_template([user], add_generation_prompt=True), add_special_tokens=False)
    out = {}
    for letter in LETTERS:
        full = processor.apply_chat_template([user, {"role": "assistant", "content": [{"type": "text", "text": letter}]}],
                                             add_generation_prompt=False)
        ids = tok.encode(full, add_special_tokens=False)
        if ids[: len(prompt)] != prompt:
            raise ValueError("assistant turn does not extend the generation prompt")
        rest = ids[len(prompt):]
        if len(rest) < 2 or tok.decode([rest[0]]).strip() != letter:
            raise ValueError(f"letter {letter} is not a single first token of the answer: {rest[:3]}")
        out[letter] = int(rest[0])
    if len(set(out.values())) != 4:
        raise ValueError(f"letter tokens not distinct: {out}")
    return out


def letter_logits(model, item: dict, ids4: dict[str, int]) -> tuple[np.ndarray, int]:
    """(logits of A-D (4,), token id of the gold answer in the sample). `item` = VisionDataset(...).process(sample) with
    train_on_completions=True. Only the prompt part (positions before the answer token) is fed to the model."""
    ids = np.array(item["input_ids"]).reshape(1, -1)
    cm = np.array(item["completion_mask"]).reshape(-1)
    p = int(np.flatnonzero(cm)[0])
    kw = {k: mx.array(np.array(v)) for k, v in item.items() if k not in _SKIP}
    logits = model(mx.array(ids[:, :p]), mx.array(np.array(item["pixel_values"])), None, **kw).logits[0, -1].astype(mx.float32)
    four = np.array([float(logits[ids4[letter]]) for letter in LETTERS])
    return four, int(ids[0, p])


def evaluate_samples(model, vision_ds, samples: list[dict], ids4: dict[str, int], progress=None) -> list[dict]:
    """Per-question results for `samples` (chat samples with answer letter). `vision_ds` is a VisionDataset built with
    train_on_completions=True (only its `process` is used). Asserts the tokenised gold answer is the template's letter token.
    `progress(done, total)` is called after every question with COUNTS ONLY (never a score, a prediction or a gold letter).
    Error messages carry only the sample id: they never contain the gold letter, a prediction or a logit."""
    out = []
    total = len(samples)
    for s in samples:
        item = vision_ds.process({"images": s["images"], "messages": s["messages"]})
        four, gold_tok = letter_logits(model, item, ids4)
        if gold_tok != ids4[s["answer"]]:
            raise ValueError(f"{s['id']}: the tokenised gold answer is not the chat template's token for its letter")
        pred = pick_option(four, list(LETTERS))
        out.append({"id": s["id"], "attribute": s["attribute"], "gold": s["answer"], "pred": pred,
                    "compound": is_label_set(s["attribute"], s["options"][s["answer"]]), "logits": [round(float(x), 4) for x in four]})
        if progress is not None:
            progress(len(out), total)
    return out


def summarize(results: list[dict]) -> dict:
    """Per-attribute and unweighted-mean option-restricted accuracy (+ single-label / label-set strata; `n_compound` counts answers with more than one
    DISTINCT label, see `is_label_set`)."""
    attrs = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
    per = {}
    for a in attrs:
        r = [x for x in results if x["attribute"] == a]
        if not r:
            continue
        hit = np.array([x["pred"] == x["gold"] for x in r])
        comp = np.array([x["compound"] for x in r])
        per[a] = {"n": len(r), "correct": int(hit.sum()), "acc": float(hit.mean()),
                  "n_compound": int(comp.sum()), "acc_single": float(hit[~comp].mean()) if (~comp).any() else None,
                  "acc_compound": float(hit[comp].mean()) if comp.any() else None}
    return {"per_attribute": per, "mean_acc": float(np.mean([v["acc"] for v in per.values()])), "n": len(results)}
