"""Token row v1: soft-vote primaries, fixed log length bins, global palette ids, no style slot. Pure python/numpy; the v0 formatter in
format.py is left untouched. Display only: the `*` marks an attribute whose three keyframes disagree on the primary label (it is not a token)."""
from pathlib import Path

import numpy as np
import yaml

from cgt.tokens import format as v0

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/tokens.yaml"
VERSION = "v1"
SLOTS = v0.SLOTS
STATIC = v0.STATIC


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def len_bin(sec: float, edges: list[float] | None = None) -> int:
    """Index of the log bin of a shot length: <1, 1-2, 2-4, 4-8, 8-16, 16-32, 32+ s (lower edge inclusive)."""
    edges = edges or load_config()["len_bins"]["edges"]
    return int(np.searchsorted(np.asarray(edges, dtype=float), float(sec), side="right"))


def len_token(sec, names: list[str] | None = None) -> str:
    if v0._none(sec):
        return v0.MISSING
    names = names or load_config()["len_bins"]["names"]
    return f"{names[len_bin(sec)]}s"


def softmax(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    e = np.exp(v - v.max())
    return e / e.sum()


def soft_vote(score_vectors: list) -> np.ndarray:
    """Mean of the per-keyframe probability vectors (each = softmax of that keyframe's scores)."""
    return np.mean([softmax(v) for v in score_vectors], axis=0)


def primary_and_margin(prob: np.ndarray) -> tuple[int, float, float]:
    """(argmax with ties to the lowest index, p1, p1 - p2)."""
    i = int(np.argmax(prob))
    s = np.sort(prob)[::-1]
    return i, float(s[0]), float(s[0] - s[1])


def palette_token(gid) -> str:
    return v0.MISSING if v0._none(gid) else f"g{int(gid):02d}"


def schema_move_atoms() -> list[str]:
    """The 16 direction-bearing move classes of configs/schema.yaml (read only): the atoms of the movement slot."""
    return list(yaml.safe_load((ROOT / "configs/schema.yaml").read_text())["attributes"]["move"]["values"])


def move_token(atoms, present: bool) -> str:
    """Movement slot: the predicted UNORDERED atom set in schema order joined by '+', `static` alone for a shot without movement. `n/a` only when the film has no
    movement table at all (`present` False); a shot without a prediction in a present table shows `?`."""
    if not present:
        return v0.NOT_AVAILABLE
    if atoms is None or (isinstance(atoms, float) and np.isnan(atoms)) or len(atoms) == 0:
        return v0.MISSING
    return "+".join(atoms)


def slot_values(rec: dict) -> dict[str, str]:
    v = v0.slot_values(rec)
    v["move"] = move_token(rec.get("move_atoms"), bool(rec.get("move_present", False)))
    v["len"] = len_token(rec.get("len_s"))
    v["palette"] = palette_token(rec.get("palette_global"))
    return v


def format_row(rec: dict) -> str:
    return "[" + " | ".join(f"{k}={x}" for k, x in slot_values(rec).items()) + "]"


def parse_row(text: str) -> dict[str, str]:
    return v0.parse_row(text)
