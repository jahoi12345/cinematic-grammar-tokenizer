"""Step 3 selection rule and budget helpers (docs/plans/step3_gap.md sections 5-6, BINDING). CPU only.

Primary = unweighted 7-attribute mean (points) on dedup_v2 (5,822 q). Baseline = the plain Step C 4B (82.49).
Eligible iff primary >= base + 0.5 AND every attribute >= base_attr - 2.0 AND target mean (light_cond, composition, lens)
>= base target mean + 0.5. Winner: P* = highest eligible primary; tie set = eligible with primary >= P* - 0.1; within it the
highest target mean wins (compared rounded to 4 decimals), then the earlier candidate in run order (G0, G1, G2). G0 is eligible;
if it wins the gain is reported as "extra steps", not as the variant. No eligible candidate -> "no gain" (the plain 4B stays)."""
from collections.abc import Sequence

from cgt.step3 import config as C
from cgt.stepb.select import ATTRS
from cgt.stepc.select import (
    FAIL_CLOSED,
    TERMINATED,
    Budget,
    film_bootstrap,
    prior_crash_resumes,
    prior_gpu_hours,
)

__all__ = [
    "ATTRS",
    "FAIL_CLOSED",
    "MAX_ATTR_DROP",
    "MIN_GAIN",
    "MIN_TARGET_GAIN",
    "TERMINATED",
    "TIE_BAND",
    "Budget",
    "candidate_may_start",
    "eligibility",
    "film_bootstrap",
    "prior_crash_resumes",
    "prior_gpu_hours",
    "select",
    "target_mean",
    "train_proj_h",
]

MIN_GAIN, MAX_ATTR_DROP, MIN_TARGET_GAIN, TIE_BAND = 0.5, 2.0, 0.5, 0.1


def target_mean(per_attribute: dict, attrs: Sequence[str] = C.TARGET_ATTRS) -> float:
    return sum(per_attribute[a] for a in attrs) / len(attrs)


def eligibility(ck: dict, base: dict) -> dict:
    reasons = []
    if not ck["primary"] >= base["primary"] + MIN_GAIN:
        reasons.append(f"primary {ck['primary']:.2f} < baseline {base['primary']:.2f} + {MIN_GAIN}")
    for a in ATTRS:
        if not ck["per_attribute"][a] >= base["per_attribute"][a] - MAX_ATTR_DROP:
            reasons.append(f"{a} {ck['per_attribute'][a]:.2f} dropped more than {MAX_ATTR_DROP} below {base['per_attribute'][a]:.2f}")
    tm, tb = target_mean(ck["per_attribute"]), target_mean(base["per_attribute"])
    if not tm >= tb + MIN_TARGET_GAIN:
        reasons.append(f"target mean {tm:.2f} < baseline target mean {tb:.2f} + {MIN_TARGET_GAIN}")
    return {"eligible": not reasons, "reasons": reasons, "target_mean": tm}


def select(cands: Sequence[dict], base: dict, run_order: Sequence[str] = C.RUN_ORDER) -> dict:
    """`cands`: dicts with cand, primary, per_attribute. Returns {'winner', 'table', 'p_star', 'tie_set', 'extra_steps_only'}."""
    table = [{**c, **eligibility(c, base)} for c in sorted(cands, key=lambda c: run_order.index(c["cand"]))]
    elig = [t for t in table if t["eligible"]]
    if not elig:
        return {"winner": None, "table": table, "p_star": None, "tie_set": [], "extra_steps_only": False}
    p_star = max(t["primary"] for t in elig)
    ties = [t for t in elig if t["primary"] >= p_star - TIE_BAND]
    ties.sort(key=lambda t: (-round(t["target_mean"], 4), run_order.index(t["cand"])))
    w = ties[0]
    return {"winner": w, "table": table, "p_star": p_star, "tie_set": [t["cand"] for t in ties], "extra_steps_only": w["cand"] == "G0"}


def train_proj_h(cand: str, measured_s: dict | None = None, n_updates: int = C.N_UPDATES) -> float:
    """Projected training hours: (measured s/update of THIS candidate's variant if known, else of the previous finished candidate
    for the 1x variants, else the committed C.SEC_PER_UPDATE) x updates, +5% for load and saves."""
    measured_s = measured_s or {}
    s = measured_s.get(cand)
    if s is None and C.CANDIDATES[cand]["budget_px"] == C.BUDGET_1X:
        s = next((measured_s[k] for k in ("G0", "G1") if k in measured_s), None)
    s = s if s is not None else C.SEC_PER_UPDATE[cand]
    return 1.05 * s * n_updates / 3600


def candidate_may_start(budget: Budget, train_h: float, val_h: float) -> bool:
    """A candidate starts only if its whole training plus its validation fits the remaining cap and the hard stop (drop, never
    shorten). The proxy is not reserved: it runs afterwards only if it fits."""
    return budget.fits(train_h + val_h)
