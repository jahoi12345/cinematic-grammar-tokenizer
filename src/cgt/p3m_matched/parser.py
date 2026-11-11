"""Answer extraction exactly as the official ShotBench scorer's LOCAL matchers (Vchitect/ShotBench evaluation/calculate_scores.py;
docs/p3m_matched_protocol.md section 1.4). The GPT-4o fallback and the unseeded random fallback are NOT replicated: an answer the local
matchers cannot map is INVALID (letter None), scored wrong by `cgt.p3m_matched.evaluate`."""
import copy as cp
import math
import re

PUNCT = ".()[],:;!*#{}"
_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def build_choices(options: dict) -> dict[str, str]:
    """Option keys as stored; empty and NaN values dropped."""
    return {k: v for k, v in options.items() if v and not (isinstance(v, float) and math.isnan(v))}


def extract_from_answer_tag(text: str) -> str:
    m = _TAG.search(text)
    return m.group(1).strip() if m else text.strip()


def can_infer_option(answer: str, choices: dict[str, str]):
    """Exactly one choice key as a whole whitespace token after replacing the punctuation by spaces; 'Z' if no key and a 'Z' token; else False."""
    answer_mod = cp.copy(extract_from_answer_tag(answer))
    for c in PUNCT:
        answer_mod = answer_mod.replace(c, " ")
    splits = [s.strip() for s in answer_mod.split()]
    hits = [k for k in choices if k in splits]
    if len(hits) == 1:
        return hits[0]
    if not hits and any(z in splits for z in ("Z", "")):
        return "Z"
    return False


def can_infer_text(answer: str, choices: dict[str, str]):
    """Exactly one choice whose lowercased option text is a substring of the lowercased extracted answer; else False."""
    raw = extract_from_answer_tag(answer).lower()
    hits = [k for k, v in choices.items() if str(v).lower() in raw]
    return hits[0] if len(hits) == 1 else False


def can_infer(answer: str, choices: dict[str, str]):
    return can_infer_option(answer, choices) or can_infer_text(answer, choices)


def parse_answer(answer: str, options: dict) -> tuple[str | None, str]:
    """(letter or None, method): method is 'option', 'text' or 'invalid'. 'Z' counts as a parsed letter (method 'option')."""
    choices = build_choices(options)
    r = can_infer_option(answer, choices)
    if r:
        return r, "option"
    r = can_infer_text(answer, choices)
    if r:
        return r, "text"
    return None, "invalid"
