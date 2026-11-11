"""The LETTER-LOGIT + C5 ShotBench protocol (docs/plans/shotbench_4b_request.md section 2). NOT the matched generative protocol of
docs/p3m_matched_protocol.md: the official ShotBench prompt/decoding/parser produce a parsed LETTER, not option scores, so the Step A C5 score
blend cannot be applied to it. This protocol therefore has its own version string and its own hash.

Per question: the Step C 4B adapter's 4 letter logits at the answer position of the ShotQA TRAINING prompt (Phase 3B letter-logit scorer,
`cgt.p3b_vlm.score`, as the Step A "base" view and the 2B letter-logit run qwen3vl2b_lora_filmsplit), the frozen Phase 3A SigLIP head option
scores (as scripts/p3s_stepa_siglip.py), and the fixed C5 blend of Step A-4B:
    score = (1 - w_attr) * log_softmax(vlm letter logits) + w_attr * log_softmax(SigLIP option scores),
w_attr loaded from the committed Step A-4B fit (results/stepa_4b/val.json, results.C5.fit.ensemble_w); prediction = argmax, ties to the lowest
letter (cgt.eval.metrics.pick_option). Both score vectors are restricted to the option letters the question actually has (A-D; any other letter
is refused); for a 4-option question this is exactly `cgt.p3s_stepa.combine.mix` on the "base" view. Pure Python/numpy: no model, no data."""
import hashlib
import json

import numpy as np
from scipy.special import log_softmax

from cgt.eval.metrics import pick_option

PROTOCOL_VERSION = "p3m-letterlogit-c5-v1"
LETTERS = ("A", "B", "C", "D")
BLEND_CONFIG = "C5"
BLEND_SOURCE_REL = "results/stepa_4b/val.json"
BLEND_KEYS = ("results", "C5", "fit", "ensemble_w")
ATTRS = ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition")


def protocol_spec() -> dict:
    from cgt.p3b_vlm.images import BUDGET_PX, MULTIPLE
    from cgt.p3b_vlm.shotbench import PROMPT_TAIL
    return {"version": PROTOCOL_VERSION,
            "questions": "cgt.p3b_vlm.shotbench.load_shotbench_samples: the static image rows of the 7 categories, file order, 0 dropped required",
            "prompt": "ShotQA training user turn (cgt.p3b_vlm.shotbench.build_user_text): Question: {q}\\nOptions:\\n{k}. {v} (letter order)\\n" + PROMPT_TAIL,
            "system_prompt": None, "message_order": "user[image, text], assistant fixed 'A' (only positions BEFORE the answer token are run)",
            "vlm_scorer": "letter logits A-D at the answer position; answer tokens read from the chat template (cgt.p3b_vlm.score.letter_logits)",
            "image": {"budget_px": BUDGET_PX, "multiple": MULTIPLE, "cache": "cgt.p3b_vlm.images.prepare_eval_cache"},
            "views": ["base"], "calibration": None,
            "siglip_scorer": "Phase 3A chosen heads, mean head output over seeds 0,1,2 on l2-normalised SigLIP features of the ORIGINAL image; "
                             "multihot set log-likelihood or log mean class probability (cgt.eval.metrics)",
            "blend": "(1 - w_attr) * log_softmax(vlm) + w_attr * log_softmax(siglip) over the question's option letters; "
                     f"w from {BLEND_SOURCE_REL} {'.'.join(BLEND_KEYS)}",
            "decision": "argmax, ties to the lowest letter", "invalid": "impossible: every question gets a letter",
            "reported": {"primary": "C5 blend accuracy per attribute and unweighted 7-attribute mean",
                         "descriptive_only": "C0 = the same pass's VLM letter logits alone (never selectable)"}}


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_spec(), sort_keys=True).encode()).hexdigest()


def blend_sha256(ensemble_w: dict) -> str:
    """Hash of the blend parameters: the config name and the per-attribute weights (all 7 attributes, floats)."""
    if sorted(ensemble_w) != sorted(ATTRS):
        raise ValueError(f"blend weights must cover exactly the 7 attributes, got {sorted(ensemble_w)}")
    w = {a: float(ensemble_w[a]) for a in ATTRS}
    return hashlib.sha256(json.dumps({"config": BLEND_CONFIG, "ensemble_w": w}, sort_keys=True).encode()).hexdigest()


def load_blend(val_json_text: str) -> dict:
    """The C5 weights from the text of results/stepa_4b/val.json."""
    d = json.loads(val_json_text)
    for k in BLEND_KEYS:
        d = d[k]
    return {a: float(d[a]) for a in ATTRS}


def option_letters(options: dict) -> list[str]:
    letters = sorted(options)
    bad = [x for x in letters if x not in LETTERS]
    if bad or len(letters) < 2:
        raise ValueError("a question's option letters are not a subset of A-D with at least 2 options")
    return letters


def blend(vlm_logits4, siglip_scores: list[float], letters: list[str], w: float) -> np.ndarray:
    """C5 scores over `letters`: `vlm_logits4` = the 4 letter logits in A-D order; `siglip_scores` = one score per letter in `letters`."""
    idx = [LETTERS.index(x) for x in letters]
    v = np.asarray(vlm_logits4, dtype=float)[idx]
    s = np.asarray(siglip_scores, dtype=float)
    if s.shape != v.shape:
        raise ValueError("SigLIP scores do not match the option letters")
    return (1 - w) * log_softmax(v) + w * log_softmax(s)


def predict(vlm_logits4, siglip_scores: list[float], letters: list[str], w: float) -> tuple[str, str]:
    """(C5 letter, C0 letter): the blend's argmax and the VLM-only argmax over the question's letters (ties to the lowest letter)."""
    c5 = pick_option(blend(vlm_logits4, siglip_scores, letters, w), letters)
    v = np.asarray(vlm_logits4, dtype=float)[[LETTERS.index(x) for x in letters]]
    return c5, pick_option(v, letters)
