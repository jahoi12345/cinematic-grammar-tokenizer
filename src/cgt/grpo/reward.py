"""GRPO pilot reward (docs/plans/grpo_pilot.md section 2.1): 1 if the parsed option letter of a sampled answer equals the gold letter,
else 0; no format reward. The answer is parsed with the official ShotBench LOCAL option matcher (cgt.p3m_matched.parser
.can_infer_option: exactly one option key as a whole whitespace token after punctuation removal). The text matcher is not used: in the
letter-only pilot an answer is one token, and an option-text match on a single word piece would reward a non-letter answer.

The trainer needs the reward inside the MLX graph, so `letter_table` decodes every vocabulary id ONCE into its option-letter index
(0-3 for A-D, -1 otherwise) with the same matcher; reward = (table[token] == gold index)."""
from collections.abc import Callable, Sequence

import numpy as np

from cgt.p3m_matched.parser import can_infer_option

LETTERS = ("A", "B", "C", "D")
_CHOICES = {k: k for k in LETTERS}   # keys are what the option matcher looks for; values unused by it


def parse_letter(text: str) -> str | None:
    r = can_infer_option(text, _CHOICES)
    return r if r in LETTERS else None


def reward(text: str, gold: str) -> float:
    return 1.0 if parse_letter(text) == gold else 0.0


def letter_table(decode: Callable[[list[int]], str], vocab_size: int) -> np.ndarray:
    """int32 (vocab_size,): option-letter index of each single-token answer, -1 if it does not parse to A-D."""
    out = np.full(vocab_size, -1, dtype=np.int32)
    for t in range(vocab_size):
        p = parse_letter(decode([t]))
        if p is not None:
            out[t] = LETTERS.index(p)
    return out


def rewards_from_tokens(table: np.ndarray, tokens: Sequence[int], gold: str) -> np.ndarray:
    return (np.asarray(table)[np.asarray(tokens)] == LETTERS.index(gold)).astype(np.float32)
