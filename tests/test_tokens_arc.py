"""CPU tests of the visual-arc series (synthetic token table)."""
import numpy as np
import pandas as pd
import pytest

from cgt.tokens import arc


@pytest.fixture
def df():
    n = 40
    return pd.DataFrame({"start_min": np.linspace(0, 19.9, n), "len_s": np.linspace(1, 10, n), "size": ["CU", "MS", "WS", "ECU"] * 10, "key": ["low-key", "mid-key"] * 20,
                         "people": [0, 1, 2, 9] * 10, "sound": ["dialogue", "score"] * 20, "light_cond": ["high-contrast", "soft"] * 20, "light_type": "daylight",
                         "angle": "low", "kf_all_agree": [True, False] * 20})


def test_windows_are_by_start_time(df):
    assert arc.windows(df, 5.0).tolist() == [int(x // 5) for x in df.start_min]


def test_series_values_by_hand(df):
    t = arc.series_table(df, 5.0, n_boot=50)
    assert t.window.nunique() == 4 and t.n_shots.eq(10).all()
    win0 = t[t.window == 0].set_index("series")
    assert win0.loc["cuts_per_min", "value"] == 2.0
    d0 = df[arc.windows(df, 5.0) == 0]
    assert win0.loc["close_share", "value"] == pytest.approx(d0["size"].isin(arc.CLOSE).mean())
    assert win0.loc["people_mean", "value"] == pytest.approx(d0.people.clip(upper=4).mean())
    assert win0.loc["mean_size_ordinal", "value"] == pytest.approx(d0["size"].map(arc.SIZE_ORDER.index).mean())
    assert win0.loc["median_len_s", "value"] == pytest.approx(d0.len_s.median())
    assert set(t.kind) == {"heuristic", "prediction", "prediction-stability"}


def test_bootstrap_is_seeded_and_contains_the_mean_for_a_constant_and_a_mix():
    assert arc.bootstrap_ci(np.ones(20)) == (1.0, 1.0)
    lo, hi = arc.bootstrap_ci(np.array([0.0] * 10 + [1.0] * 10), seed=3)
    assert lo < 0.5 < hi and (lo, hi) == arc.bootstrap_ci(np.array([0.0] * 10 + [1.0] * 10), seed=3)
    assert np.isnan(arc.bootstrap_ci(np.array([]))[0])


def test_spearman_detects_a_monotone_series(df):
    t = arc.series_table(df, 5.0, n_boot=10)
    s = arc.spearman_with_time(t).set_index("series")
    assert s.loc["median_len_s", "spearman_with_time"] == pytest.approx(1.0)


def test_last_partial_window_rate_uses_its_real_duration(df):
    d = df.assign(end_s=df.start_min * 60 + 5)
    d = d[d.start_min < 17.5].assign(end_s=lambda x: np.where(x.start_min == x.start_min.max(), 17.5 * 60, x.start_min * 60 + 5))
    t = arc.series_table(d, 5.0, n_boot=5)
    last = t[(t.series == "cuts_per_min") & (t.window == 3)].iloc[0]
    assert last.n_shots / 2.5 == pytest.approx(last.value)


def test_movement_series_by_hand_and_requires_a_movement_table(df):
    with pytest.raises(ValueError):
        arc.movement_series_table(df)
    atoms = [["static"] if i % 4 == 0 else (["static", "pan-left"] if i % 4 == 1 else (["push-in", "tilt-up", "move-left"] if i % 4 == 2 else ["zoom-in"])) for i in range(len(df))]
    t = arc.movement_series_table(df.assign(move_atoms=atoms), 5.0, n_boot=20)
    win0 = t[t.window == 0].set_index("series")
    d0 = [a for a, ww in zip(atoms, arc.windows(df, 5.0), strict=True) if ww == 0]
    assert win0.loc["move_static_share", "value"] == pytest.approx(sum("static" in a for a in d0) / len(d0))
    assert win0.loc["move_static_alone_share", "value"] == pytest.approx(sum(a == ["static"] for a in d0) / len(d0))
    assert win0.loc["move_mean_set_size", "value"] == pytest.approx(sum(map(len, d0)) / len(d0))
    assert win0.loc["move_pan_share", "value"] == pytest.approx(sum("pan-left" in a for a in d0) / len(d0))
    assert set(t.kind) == {"prediction (movement)"} and t.series.nunique() == 9


def test_contradictory_share_uses_the_declared_definition(df):
    sets = [["static"], ["static", "pan-left"], ["pan-left", "pan-right"], ["push-in", "tilt-up"], ["zoom-in", "zoom-out", "arc"], ["boom-up"]]
    s = pd.Series(sets)
    assert arc.contradictory_share(s) == pytest.approx(3 / 6)  # static+other, opposite pair, opposite pair (+arc)
    d = df.iloc[:6].assign(move_atoms=sets, start_min=[0, 1, 2, 5, 6, 7])
    t = arc.contradictory_series(d, 5.0, n_boot=10)
    assert t.set_index("window").value.to_dict() == {0: pytest.approx(2 / 3), 1: pytest.approx(1 / 3)} and set(t.series) == {"move_contradictory_share"}
