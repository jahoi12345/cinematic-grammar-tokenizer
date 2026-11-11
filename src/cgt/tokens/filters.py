"""Filter, sample and paging logic of the browse app (pure pandas; no streamlit)."""
import numpy as np
import pandas as pd

from cgt.tokens.format import PEOPLE_MAX, people_token

FILTER_COLUMNS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition", "key", "dof", "sound"]
KEYFRAME_MODES = ("any", "disagree", "agree")


def people_bucket(s: pd.Series) -> pd.Series:
    return s.map(lambda n: people_token(n))


def apply_filters(df: pd.DataFrame, selections: dict[str, list] | None = None, len_range: tuple[float, float] | None = None,
                  people: list[str] | None = None, keyframes: str = "any", move_atoms: list[str] | None = None) -> pd.DataFrame:
    """Rows matching ALL filters. An empty or missing selection for a column means 'no filter'. `people` holds display values ('0'..'4+').
    `keyframes`: 'any', 'disagree' (at least one of the 7 attributes' keyframes disagree), 'agree' (all 7 agree)."""
    if keyframes not in KEYFRAME_MODES:
        raise ValueError(f"keyframes must be one of {KEYFRAME_MODES}")
    mask = pd.Series(True, index=df.index)
    for col, vals in (selections or {}).items():
        if vals:
            mask &= df[col].isin(vals)
    if len_range is not None:
        mask &= df["len_s"].between(len_range[0], len_range[1])
    if people:
        mask &= people_bucket(df["people"]).isin(people)
    if move_atoms and "move_atoms" in df.columns:  # any of the selected atoms is in the shot's predicted (unordered) set
        wanted = set(move_atoms)
        mask &= df["move_atoms"].map(lambda a: a is not None and len(wanted & set(a)) > 0)
    if keyframes == "disagree":
        mask &= df["n_kf_disagree"] > 0
    elif keyframes == "agree":
        mask &= df["n_kf_disagree"] == 0
    return df[mask]


def random_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.iloc[np.sort(np.random.default_rng(seed).permutation(len(df))[:n])]


def page_count(n_rows: int, size: int) -> int:
    return max(1, -(-n_rows // size))


def get_page(df: pd.DataFrame, page: int, size: int) -> pd.DataFrame:
    """1-based page, clamped to the valid range."""
    page = min(max(1, page), page_count(len(df), size))
    return df.iloc[(page - 1) * size: page * size]


__all__ = ["PEOPLE_MAX", "apply_filters", "get_page", "page_count", "people_bucket", "random_sample"]
