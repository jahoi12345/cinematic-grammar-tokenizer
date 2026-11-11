"""Step B selection rule (docs/plans/stepb_cinescale2.md section 5), fixed before training.

Inputs per checkpoint: `primary` = unweighted 7-attribute mean accuracy (in POINTS, 0-100) on the deduplicated ShotQA film-held-out
validation set; `per_attribute` points; `secondary` = CineScale2 val angle accuracy (points). Baseline = the frozen 4c030967 adapter
on the same sets.
Eligible: primary >= base + 0.5 AND every attribute >= base_attr - 2.0 AND angle >= base_angle - 1.0.
Winner: let P* = highest primary among eligible checkpoints; the tie set = eligible checkpoints with primary >= P* - 0.1; within the tie
set the highest secondary wins; equal secondary (compared rounded to 4 decimals) -> the earlier checkpoint (50 before 100) -> the
earlier candidate in run order (B0, B1, B2). No eligible checkpoint -> None (the frozen model stays)."""
from collections.abc import Sequence

MIN_GAIN, MAX_ATTR_DROP, MAX_ANGLE_DROP, TIE_BAND = 0.5, 2.0, 1.0, 0.1
ATTRS = ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition")


def eligibility(ck: dict, base: dict) -> dict:
    """{'eligible': bool, 'reasons': [...]} for one checkpoint against the baseline."""
    reasons = []
    if not ck["primary"] >= base["primary"] + MIN_GAIN:
        reasons.append(f"primary {ck['primary']:.2f} < baseline {base['primary']:.2f} + {MIN_GAIN}")
    for a in ATTRS:
        if not ck["per_attribute"][a] >= base["per_attribute"][a] - MAX_ATTR_DROP:
            reasons.append(f"{a} {ck['per_attribute'][a]:.2f} dropped more than {MAX_ATTR_DROP} below {base['per_attribute'][a]:.2f}")
    if not ck["per_attribute"]["angle"] >= base["per_attribute"]["angle"] - MAX_ANGLE_DROP:
        reasons.append(f"angle {ck['per_attribute']['angle']:.2f} more than {MAX_ANGLE_DROP} below {base['per_attribute']['angle']:.2f}")
    return {"eligible": not reasons, "reasons": reasons}


def select(checkpoints: Sequence[dict], base: dict, run_order: Sequence[str] = ("B0", "B1", "B2")) -> dict:
    """`checkpoints`: dicts with candidate, pct, primary, per_attribute, secondary. Returns {'winner': dict|None, 'table': [...]}"""
    table = []
    for ck in checkpoints:
        table.append({**ck, **eligibility(ck, base)})
    elig = [t for t in table if t["eligible"]]
    if not elig:
        return {"winner": None, "table": table, "p_star": None, "tie_set": []}
    p_star = max(t["primary"] for t in elig)
    ties = [t for t in elig if t["primary"] >= p_star - TIE_BAND]
    ties.sort(key=lambda t: (-round(t["secondary"], 4), t["pct"], list(run_order).index(t["candidate"])))
    return {"winner": ties[0], "table": table, "p_star": p_star,
            "tie_set": [(t["candidate"], t["pct"]) for t in ties]}
