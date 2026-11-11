"""CPU tests of the browse-app filter, sample and paging logic."""
import pandas as pd
import pytest

from cgt.tokens import filters


@pytest.fixture
def df():
    n = 30
    return pd.DataFrame({"shot_id": range(n), "size": ["CU", "MS", "WS"] * 10, "lens": ["wide"] * n, "angle": ["low"] * n, "framing": ["single"] * n,
                         "light_type": ["daylight"] * n, "light_cond": ["hard", "soft"] * 15, "composition": ["center"] * n,
                         "key": ["low-key", "mid-key", "high-key"] * 10, "dof": ["deep"] * n, "sound": ["score"] * n,
                         "len_s": [float(i) for i in range(n)], "people": [i % 7 for i in range(n)],
                         "n_kf_disagree": [i % 3 for i in range(n)]})


def test_empty_selection_is_no_filter(df):
    assert len(filters.apply_filters(df)) == 30
    assert len(filters.apply_filters(df, {"size": [], "key": []})) == 30


def test_filters_combine_with_and_and_multiselect_is_or(df):
    out = filters.apply_filters(df, {"size": ["CU", "MS"], "key": ["low-key"]})
    assert set(out["size"]) == {"CU"} and set(out["key"]) == {"low-key"}
    assert len(filters.apply_filters(df, {"size": ["CU", "MS"]})) == 20


def test_length_range_is_inclusive(df):
    out = filters.apply_filters(df, len_range=(5.0, 9.0))
    assert out.shot_id.tolist() == [5, 6, 7, 8, 9]


def test_people_uses_display_buckets(df):
    assert set(filters.apply_filters(df, people=["4+"]).people) == {4, 5, 6}
    assert set(filters.apply_filters(df, people=["0", "1"]).people) == {0, 1}


def test_keyframe_modes(df):
    assert (filters.apply_filters(df, keyframes="disagree").n_kf_disagree > 0).all()
    assert (filters.apply_filters(df, keyframes="agree").n_kf_disagree == 0).all()
    assert len(filters.apply_filters(df, keyframes="any")) == 30
    with pytest.raises(ValueError):
        filters.apply_filters(df, keyframes="sometimes")


def test_random_sample_is_seeded_sorted_and_bounded(df):
    a, b = filters.random_sample(df, 5, 1), filters.random_sample(df, 5, 1)
    assert a.shot_id.tolist() == b.shot_id.tolist() and len(a) == 5 and a.shot_id.is_monotonic_increasing
    assert filters.random_sample(df, 5, 2).shot_id.tolist() != a.shot_id.tolist()
    assert len(filters.random_sample(df.iloc[:3], 5, 1)) == 3


def test_paging(df):
    assert filters.page_count(30, 8) == 4 and filters.page_count(0, 8) == 1
    assert filters.get_page(df, 1, 8).shot_id.tolist() == list(range(8))
    assert filters.get_page(df, 4, 8).shot_id.tolist() == [24, 25, 26, 27, 28, 29]
    assert filters.get_page(df, 99, 8).shot_id.tolist() == [24, 25, 26, 27, 28, 29]  # clamped
    assert filters.get_page(df, 0, 8).shot_id.tolist() == list(range(8))


def test_move_atom_filter_matches_any_selected_atom(df):
    d = df.assign(move_atoms=[["static"] if i % 3 == 0 else (["pan-left", "push-in"] if i % 3 == 1 else None) for i in range(len(df))])
    assert set(filters.apply_filters(d, move_atoms=["push-in"]).shot_id) == {i for i in range(30) if i % 3 == 1}
    assert set(filters.apply_filters(d, move_atoms=["static", "pan-left"]).shot_id) == {i for i in range(30) if i % 3 in (0, 1)}
    assert len(filters.apply_filters(d, move_atoms=[])) == 30 and len(filters.apply_filters(df, move_atoms=["static"])) == 30  # no column: no filter
