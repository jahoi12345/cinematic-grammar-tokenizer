"""Step B evaluation helpers. ShotQA questions use the Phase 3B scorer unchanged (cgt.p3b_vlm.score.evaluate_samples / summarize);
CineScale2 questions use the same letter-logit read-out (cgt.p3b_vlm.score.letter_logits) but their own summary, because their
option texts include "Eye level", which the schema v1 label parser does not know (it lands with the schema v2 edit)."""
from collections.abc import Sequence

import numpy as np

LETTERS = ("A", "B", "C", "D")


def as_model_item(s: dict) -> dict:
    """The sample as fed to the model: the budget-shrunk image (`shrunk`) and the chat messages."""
    return {"images": [s["shrunk"]], "messages": s["messages"]}


def evaluate_cs2(model, vision_ds, samples: Sequence[dict], ids4: dict[str, int]) -> list[dict]:
    from cgt.eval.metrics import pick_option
    from cgt.p3b_vlm.score import letter_logits
    out = []
    for s in samples:
        item = vision_ds.process(as_model_item(s))
        four, gold_tok = letter_logits(model, item, ids4)
        if gold_tok != ids4[s["answer"]]:
            raise ValueError(f"{s['id']}: the tokenised gold answer is not the chat template's token for its letter")
        pred = pick_option(four, list(LETTERS))
        out.append({"id": s["id"], "cs2_class": s["cs2_class"], "gold": s["answer"], "pred": pred,
                    "pred_class": _class_of(s, pred), "logits": [round(float(x), 4) for x in four]})
    return out


def _class_of(s: dict, letter: str) -> str:
    from cgt.cinescale2.labels import SCHEMA_ANGLE_TEXT
    inv = {v: k for k, v in SCHEMA_ANGLE_TEXT.items()}
    return inv[s["options"][letter]]


def summarize_cs2(results: Sequence[dict]) -> dict:
    """Accuracy (points) over all CineScale2 questions, per-class recall (points) and the class-balanced mean recall."""
    if not results:
        return {"n": 0, "acc": None, "recall": {}, "balanced_recall": None}
    hit = np.array([r["pred"] == r["gold"] for r in results])
    classes = sorted({r["cs2_class"] for r in results})
    rec = {c: 100.0 * float(np.mean([r["pred"] == r["gold"] for r in results if r["cs2_class"] == c])) for c in classes}
    return {"n": len(results), "correct": int(hit.sum()), "acc": 100.0 * float(hit.mean()), "recall": rec,
            "balanced_recall": float(np.mean(list(rec.values())))}


def shotqa_points(summary: dict) -> dict:
    """Phase 3B summarize() output -> {'primary': mean in points, 'per_attribute': {attr: points}, 'n': ...}."""
    return {"primary": 100.0 * summary["mean_acc"], "n": summary["n"],
            "per_attribute": {a: 100.0 * v["acc"] for a, v in summary["per_attribute"].items()}}
