"""Small pure helpers for the Qwen3-VL-4B feasibility measurement (docs/plans/qwen3vl4b_feasibility.md): timing summaries and
linear cost projections. No model code here."""
import math
import statistics


def n_updates(n_samples: int, accum: int) -> int:
    """Optimizer updates for one epoch of `n_samples` with accumulation `accum` (the last partial update counts)."""
    if n_samples < 0 or accum < 1:
        raise ValueError("n_samples must be >= 0 and accum >= 1")
    return math.ceil(n_samples / accum)


def timing_summary(seconds: list[float], warmup: int) -> dict:
    """Median/mean/min/max of `seconds` after dropping the first `warmup` entries. Raises if nothing is left."""
    kept = list(seconds)[warmup:]
    if not kept:
        raise ValueError(f"no measurements left after {warmup} warm-up entries")
    return {"n": len(kept), "median": statistics.median(kept), "mean": statistics.fmean(kept), "min": min(kept), "max": max(kept)}


def project_hours(sec_per_unit: float, n_units: int) -> float:
    """Linear projection in hours."""
    if sec_per_unit < 0 or n_units < 0:
        raise ValueError("negative input")
    return sec_per_unit * n_units / 3600.0
