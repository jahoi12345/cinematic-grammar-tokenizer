"""Step C selection rule, budget/deadline rule and the descriptive bootstrap (docs/plans/stepc_qwen3vl4b.md sections 5-6, BINDING).

Selection: primary = unweighted 7-attribute mean accuracy in POINTS
on the dedup_v2 ShotQA film-held-out validation set (5,822 questions); baseline = the frozen 2B 4c030967 on the same set.
Eligible: primary >= base + 0.5 AND every attribute >= base_attr - 2.0 AND angle >= base_angle - 1.0.
Winner: P* = highest primary among eligible checkpoints; tie set = eligible with primary >= P* - 0.1; the EARLIER checkpoint (50
before 100) of the tie set wins. No eligible checkpoint -> None ("no gain": the frozen 2B stays).
CPU only, no model is loaded here."""
import datetime as dt
from collections.abc import Callable, Sequence

import numpy as np

from cgt.stepb.select import ATTRS, MAX_ANGLE_DROP, MAX_ATTR_DROP, MIN_GAIN, TIE_BAND, eligibility

__all__ = ["ATTRS", "MAX_ANGLE_DROP", "MAX_ATTR_DROP", "MIN_GAIN", "TIE_BAND", "Budget", "eligibility", "film_bootstrap", "select"]


def select(checkpoints: Sequence[dict], base: dict) -> dict:
    """`checkpoints`: dicts with pct, primary, per_attribute (points). Returns {'winner': dict|None, 'table', 'p_star', 'tie_set'}."""
    table = [{**ck, **eligibility(ck, base)} for ck in sorted(checkpoints, key=lambda c: c["pct"])]
    elig = [t for t in table if t["eligible"]]
    if not elig:
        return {"winner": None, "table": table, "p_star": None, "tie_set": []}
    p_star = max(t["primary"] for t in elig)
    ties = sorted((t for t in elig if t["primary"] >= p_star - TIE_BAND), key=lambda t: t["pct"])
    return {"winner": ties[0], "table": table, "p_star": p_star, "tie_set": [t["pct"] for t in ties]}


class Budget:
    """GPU-hour cap + wall-clock hard stop. GPU hours = time spent inside GPU stages (lock held), summed by the runner.
    `fits(projection_h)` is the drop-and-record test made BEFORE a stage starts: the stage may start only if both
    used + projection <= cap and now + projection <= deadline. `stop_at(reserve_h)` is the latest wall-clock time a running
    stage may continue, keeping `reserve_h` for later stages (the training child is terminated then)."""

    def __init__(self, cap_h: float, deadline: dt.datetime, clock: Callable[[], dt.datetime] | None = None):
        self.cap_h, self.deadline = cap_h, deadline
        self.clock = clock or (lambda: dt.datetime.now().astimezone())
        self.used_h = 0.0

    def add(self, hours: float) -> None:
        self.used_h += hours

    def fits(self, projection_h: float) -> bool:
        return (self.used_h + projection_h <= self.cap_h
                and self.clock() + dt.timedelta(hours=projection_h) <= self.deadline)

    def stop_at(self, stage_start: dt.datetime, reserve_h: float = 0.0) -> dt.datetime:
        """Wall-clock time at which a stage started at `stage_start` must be terminated: the earlier of (deadline - reserve) and
        (the moment the GPU-hour cap minus reserve is used up)."""
        by_cap = stage_start + dt.timedelta(hours=max(self.cap_h - reserve_h - self.used_h, 0.0))
        return min(self.deadline - dt.timedelta(hours=reserve_h), by_cap)


def per_attr_mean(hit: np.ndarray, attr_idx: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Unweighted 7-attribute mean (points) of 0/1 `hit`, optionally with per-question multiplicities `weights`."""
    w = np.ones_like(hit, dtype=float) if weights is None else weights
    accs = []
    for k in range(len(ATTRS)):
        m = attr_idx == k
        accs.append(100.0 * float((hit[m] * w[m]).sum() / w[m].sum()))
    return float(np.mean(accs))


def film_bootstrap(hit_new: Sequence[int], hit_base: Sequence[int], attrs: Sequence[str], films: Sequence[str],
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """Descriptive film-clustered bootstrap of the gain (new - base, points, 7-attribute mean): resample the validation FILMS with
    replacement (seed 0, 2,000 draws), each drawn film contributing all its questions. Returns the point gain and the 2.5/97.5%
    percentiles. Paired: both models are scored on the same resampled questions."""
    hn, hb = np.asarray(hit_new, float), np.asarray(hit_base, float)
    ai = np.array([ATTRS.index(a) for a in attrs])
    uf, fi = np.unique(np.asarray(films), return_inverse=True)
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(uf), len(uf)), minlength=len(uf)).astype(float)
        w = cnt[fi]
        if any(w[ai == k].sum() == 0 for k in range(len(ATTRS))):
            continue
        gains.append(per_attr_mean(hn, ai, w) - per_attr_mean(hb, ai, w))
    point = per_attr_mean(hn, ai) - per_attr_mean(hb, ai)
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {"gain": point, "ci95": [float(lo), float(hi)], "n_boot": len(gains), "n_films": len(uf), "seed": seed}


VAL_ORDER = (100, 50)        # ckpt_100 first: the full pre-registered recipe (as the frozen 2B ckpt_100); ckpt_050 second
FAIL_CLOSED = {3, 4, 5, 6, 7, 8, 9, 10}
TERMINATED = -999


def training_may_start(budget: Budget, train_proj_h: float, val_proj_h: float) -> bool:
    """Training starts only if at least the 50% checkpoint and one validation fit (plan section 6)."""
    return budget.fits(0.5 * train_proj_h + val_proj_h)


def validatable(train_rc: int, manifest: dict) -> list[int]:
    """Checkpoints that may be validated, in VAL_ORDER. A fail-closed exit (stopping rule, leak, data, hash, length, dirty tree)
    makes the run invalid: nothing is validated. A normal finish -> both. A termination at the cap/hard stop (or an unresolved
    crash) -> only checkpoints already saved and registered (normally ckpt_050; its weights do not depend on later updates)."""
    if train_rc in FAIL_CLOSED:
        return []
    have = {int(k) for k in manifest.get("checkpoints", {})}
    return [p for p in VAL_ORDER if p in have]


def _jsonl(path) -> list[dict]:
    from pathlib import Path
    p = Path(path)
    return [__import__("json").loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


CRASH_RESUME_MSGS = ("training crashed: resuming ONCE", "training crashed: crash resume ")


def prior_crash_resumes(runner_log) -> int:
    """Crash resumes already made, across runner launches (amendment 2): the runner records every crash resume it starts as a
    runner.jsonl event ("training crashed: resuming ONCE ..." before amendment 2, "training crashed: crash resume k of N ..." after).
    A author restart is logged with a different message and is never counted. (restarts.jsonl, written by the trainer, lists
    every resume of either kind and is kept as the cross-check: its lines = crash resumes + author restarts.)"""
    crash_end = [e for e in _jsonl(runner_log) if e.get("msg") == "training did not finish"
                 and e.get("rc") not in (None, 0, TERMINATED, -998, *FAIL_CLOSED)]
    # amendment 3: a crash that ended a runner (the 22:06 crash under the pre-amendment-2 code) and was then resumed by a relaunch
    # counts as a crash resume too.
    return sum(str(e.get("msg", "")).startswith(CRASH_RESUME_MSGS) for e in _jsonl(runner_log)) + len(crash_end)


def prior_gpu_hours(runner_log, now: dt.datetime | None = None) -> float:
    """GPU hours already used by earlier launches of the runner (amendment 2), so a relaunch keeps the same GPU-hour cap: the sum
    of the per-stage `stage_h` events in runner.jsonl, plus, for a stage whose runner was killed before it could log `stage_h`
    (a stopped runner), the time from that stage's start to the next "runner started" event, or to `now` (conservative)."""
    now = now or dt.datetime.now().astimezone()
    total, open_start = 0.0, None
    for e in _jsonl(runner_log):
        t = dt.datetime.fromisoformat(e["time"])
        if e.get("msg") == "runner started" and open_start is not None:
            total += (t - open_start).total_seconds() / 3600
            open_start = None
        if e.get("msg") == "lock acquired, child started":
            open_start = t
        elif "stage_h" in e:
            total += float(e["stage_h"] or 0.0)
            open_start = None
    if open_start is not None:
        total += (now - open_start).total_seconds() / 3600
    return total
