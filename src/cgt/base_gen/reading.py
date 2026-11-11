"""Pre-stated reading rule of the base-generation experiment (docs/plans/base_gen_matched_budget.md section 6, BINDING).
d = 3B@f - 2B@f (points, unweighted 7-attribute mean, dedup_v2), paired film-clustered bootstrap 95% interval [lo, hi]."""

EQUIV_MARGIN = 1.0   # points


def read(d: float, lo: float, hi: float) -> str:
    if hi < 0:
        return "NEWER_GENERATION_MATTERS"        # the older-generation 3B is reliably below the newer 2B under our recipe
    if lo > 0:
        return "OLDER_3B_BETTER"                 # the bigger older base is reliably above: our 2B result is not a generation effect
    if abs(d) <= EQUIV_MARGIN:
        return "RECIPE_EXPLAINS"                 # interval contains 0 and |d| <= 1 point: base generation makes no detectable difference
    return "INCONCLUSIVE"                        # interval contains 0 but |d| > 1 point
