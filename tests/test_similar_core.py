"""Similar shots (docs/similar_shots_spec.md): slot rules, exclusions, ties, symmetry, determinism, analysis helpers. Synthetic tables only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cgt.tokens import similar

REPO = Path(__file__).resolve().parents[1]
ATTRS = ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition")
BASE = {"size": "MS", "lens": "medium", "angle": "low", "framing": "single", "light_type": "daylight", "light_cond": "soft", "composition": "center",
        "key": "mid-key", "dof": "deep", "people": 1, "sound": "dialogue", "len_s": 3.0, "palette_global": 5, "move_atoms": ["static"]}


def table(rows: list[dict], move_present: bool = True) -> pd.DataFrame:
    recs = []
    for i, r in enumerate(rows):
        d = {**BASE, **r, "shot_id": r.get("shot_id", i), "start_s": r.get("start_s", 60.0 * i)}
        for a in ATTRS:
            d.setdefault(f"{a}_consistent", True)
        d["move_present"] = move_present
        d["token_row"] = f"row{d['shot_id']}"
        recs.append(d)
    return pd.DataFrame(recs)


def pair(a: dict, b: dict, centroids=None, move_present=True) -> dict:
    """Slot scores of shot 0 (= a) against shot 1 (= b)."""
    f = similar.features(table([a, b], move_present), centroids)
    return {k: float(v[1]) for k, v in similar.slot_scores_row(f, 0).items()}, f


def score(a: dict, b: dict, centroids=None, move_present=True) -> float:
    f = similar.features(table([a, b], move_present), centroids)
    return float(similar.similarity_row(f, 0)[1])


def random_table(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pools = {"size": ["ECU", "CU", "MCU", "MS", "MWS", "WS", "EWS", None], "lens": ["long", "medium", "wide", "ultra-wide"], "angle": ["low", "high", "dutch"],
             "framing": ["single", "insert", "2-shot"], "light_type": ["daylight", "artificial"], "light_cond": ["soft", "hard", "high-contrast"],
             "composition": ["center", "left-heavy"], "key": ["low-key", "mid-key", "high-key"], "dof": ["shallow", "medium", "deep"], "sound": ["score", "dialogue", "ambient"]}
    atoms = [["static"], ["static", "pan-left"], ["pan-left", "pan-right"], ["push-in"], ["tilt-up", "boom-up"], None]
    rows = []
    for i in range(n):
        r = {k: v[rng.integers(len(v))] for k, v in pools.items()}
        r.update({"people": int(rng.integers(0, 7)), "len_s": float(rng.uniform(0.3, 40)), "palette_global": int(rng.integers(0, 6)),
                  "move_atoms": atoms[rng.integers(len(atoms))], "start_s": 10.0 * i})
        for a in ATTRS:
            r[f"{a}_consistent"] = bool(rng.random() > 0.3)
        rows.append(r)
    return table(rows)


CENTS = np.array([[float(i)] * 15 for i in range(6)])  # centroid i at distance proportional to i: D[a, b] = |a - b| / 5


# ---- slot rules (S3-S6, S8-S10) ------------------------------------------------------------------------------------------------------------------------

def test_identical_rows_score_one():
    assert score({}, {}) == 1.0
    assert score({}, {}, CENTS) == 1.0


def test_ordinal_slots_graded():  # S3
    s, _ = pair({"size": "MCU", "lens": "long", "key": "low-key", "dof": "shallow", "people": 0, "len_s": 0.5},
                {"size": "MS", "lens": "ultra-wide", "key": "mid-key", "dof": "deep", "people": 7, "len_s": 3.0})
    assert s["size"] == pytest.approx(5 / 6) and s["lens"] == 0.0 and s["key"] == 0.5 and s["DoF"] == 0.0
    assert s["people"] == 0.0 and s["len"] == pytest.approx(1 - 2 / 6)  # bins <1 (0) vs 2-4 (2)
    s, _ = pair({"people": 4}, {"people": 9})
    assert s["people"] == 1.0  # 4+ caps
    s, _ = pair({"people": 2}, {"people": 3})
    assert s["people"] == 0.75


def test_categorical_slots_exact():
    s, _ = pair({"angle": "low", "framing": "single", "composition": "center", "sound": "score"}, {"angle": "high", "framing": "single", "composition": "left-heavy", "sound": "score"})
    assert (s["angle"], s["framing"], s["comp"], s["sound"]) == (0.0, 1.0, 0.0, 1.0)


def test_light_halves():  # S10
    s, _ = pair({"light_type": "daylight", "light_cond": "soft"}, {"light_type": "daylight", "light_cond": "hard"})
    assert s["light"] == 0.5
    s, _ = pair({"light_type": "daylight", "light_cond": "soft"}, {"light_type": "moonlight", "light_cond": "soft"})
    assert s["light"] == 0.5
    s, _ = pair({"light_cond_consistent": False}, {})
    assert s["light"] == 0.75


def test_star_counts_half():  # S4
    s, _ = pair({"size_consistent": False}, {})
    assert s["size"] == 0.5
    s, _ = pair({"size_consistent": False}, {"size_consistent": False})
    assert s["size"] == 0.25
    s, _ = pair({"angle_consistent": False}, {"angle": "high"})
    assert s["angle"] == 0.0
    assert score({"composition_consistent": False}, {}) == pytest.approx(1 - 0.5 / 13)


def test_move_jaccard():  # S5
    s, _ = pair({"move_atoms": ["static", "pan-left"]}, {"move_atoms": ["static"]})
    assert s["move"] == 0.5
    s, _ = pair({"move_atoms": ["pan-left", "tilt-up", "push-in"]}, {"move_atoms": ["tilt-up", "push-in", "zoom-in"]})
    assert s["move"] == 0.5
    s, _ = pair({"move_atoms": ["pan-left"]}, {"move_atoms": ["pan-right"]})
    assert s["move"] == 0.0


def test_palette_centroid_distance_and_fallback():  # S6
    s, f = pair({"palette_global": 1}, {"palette_global": 2}, CENTS)
    assert f.palette_mode == "centroid" and s["palette"] == pytest.approx(1 - 1 / 5)
    s, _ = pair({"palette_global": 0}, {"palette_global": 5}, CENTS)
    assert s["palette"] == pytest.approx(0.0)
    s, f = pair({"palette_global": 1}, {"palette_global": 2})
    assert f.palette_mode == "exact_id" and s["palette"] == 0.0
    # palette_tok fallback when palette_global is absent
    df = table([{"palette_global": 3}, {"palette_global": 3}]).drop(columns="palette_global").assign(palette_tok=["g03", "g03"])
    assert similar.slot_scores_row(similar.features(df), 0)["palette"][1] == 1.0


def test_move_absent_drops_slot():  # S8
    s, f = pair({}, {}, move_present=False)
    assert "move" not in s and f.n_slots == 12
    assert score({}, {}, move_present=False) == 1.0
    assert score({"sound": "score"}, {}, move_present=False) == pytest.approx(11 / 12)


def test_missing_counts_zero():  # S9
    for k, v in (("size", None), ("size", "?"), ("sound", None), ("palette_global", None), ("move_atoms", None), ("people", None), ("dof", float("nan"))):
        s, _ = pair({k: v}, {k: v})
        slot = {"palette_global": "palette", "move_atoms": "move", "dof": "DoF"}.get(k, k)
        assert s[slot] == 0.0, k
    assert score({"size": None}, {"size": None}) == pytest.approx(12 / 13)


# ---- exclusions, ranking, ties (S1, S2, S7) -----------------------------------------------------------------------------------------------------------

def test_self_and_neighbours_excluded():  # S1, S2
    rows = [{"sound": "score"}] * 5 + [{"sound": "ambient"}] * 3 + [{"sound": "score"}]  # shot 2 equals its neighbours, the only far equal is 8
    df = table(rows)
    top = similar.top_k_table(df, k=3)
    t2 = top[top.shot_id == 2]
    assert t2.iloc[0].other_shot_id == 8 and t2.iloc[0].score == 1.0
    for r in top.itertuples():
        assert abs(r.other_shot_id - r.shot_id) > 2
    assert list(similar.candidates(9, 2)) == [5, 6, 7, 8]
    assert list(similar.candidates(9, 0)) == [3, 4, 5, 6, 7, 8]


def test_neighbour_exclusion_uses_order_not_ids():
    df = table([{"shot_id": 10 * i, "sound": "score" if i in (0, 2) else "ambient"} for i in range(8)])
    t = similar.similar_to(df, 0, k=8)
    assert 20 not in set(t.other_shot_id) and set(t.other_shot_id) == {30, 40, 50, 60, 70}


def test_ties_to_smaller_shot_id():  # S7
    df = table([{"sound": "score"}] + [{"sound": "ambient"}] * 3 + [{"sound": "score"}] * 4)
    t = similar.similar_to(df, 0, k=5)
    assert list(t.other_shot_id) == [4, 5, 6, 7, 3]
    top = similar.top_k_table(df, k=2)
    assert bool(top[(top.shot_id == 0) & (top["rank"] == 1)].tie_next.iloc[0])


def test_top_k_count_and_order():
    top = similar.top_k_table(random_table(), k=5)
    assert (top.groupby("shot_id").size() == 5).all()
    for _, d in top.groupby("shot_id"):
        assert list(d["rank"]) == [1, 2, 3, 4, 5] and (np.diff(d.score.to_numpy()) <= 0).all()


# ---- symmetry and determinism --------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_score_symmetric_exactly(seed):
    f = similar.features(random_table(seed=seed), CENTS)
    m = similar.similarity_matrix(f)
    assert np.array_equal(m, m.T) and (m >= 0).all() and (m <= 1 + 1e-12).all()


def test_deterministic_and_row_order_invariant():
    df = random_table(seed=3)
    a = similar.top_k_table(df, centroids=CENTS)
    b = similar.top_k_table(df.sample(frac=1.0, random_state=7), centroids=CENTS)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a, similar.top_k_table(df, centroids=CENTS))


def test_matrix_matches_hand_sum():
    df = random_table(seed=4)
    f = similar.features(df, CENTS)
    for i in (0, 7, 21):
        parts = similar.slot_scores_row(f, i)
        assert list(parts) == similar.SLOTS
        assert np.allclose(similar.similarity_row(f, i), sum(parts.values()) / 13)


# ---- analysis helpers ----------------------------------------------------------------------------------------------------------------------------------

def test_gap_stats_and_reference():
    st = similar.gap_stats([0.5, 2, 6, 12, 40, 90])
    assert st["hist"] == {"0-1": 1, "1-5": 1, "5-10": 1, "10-30": 1, "30-60": 1, "60+": 1}
    assert st["share_gt_10"] == 0.5 and st["n"] == 6
    g = similar.all_candidate_gaps(np.arange(6) * 60.0)
    assert len(g) == 2 * 6 and sorted(set(g)) == [3.0, 4.0, 5.0]  # ordered pairs with |dp| > 2


def test_long_range_pairs_unordered_once():
    top = pd.DataFrame({"shot_id": [0, 30, 5, 9], "rank": 1, "other_shot_id": [30, 0, 40, 10], "score": [0.9, 0.9, 0.95, 0.99],
                        "gap_min": [15.0, 15.0, 20.0, 0.5]})
    p = similar.long_range_pairs(top, 10.0, 10)
    assert list(zip(p.a, p.b)) == [(5, 40), (0, 30)] and list(p.mutual) == [False, True]
    assert similar.mutual_top1_share(top) == 0.5


def _script():
    spec = importlib.util.spec_from_file_location("similar_shots_script", REPO / "scripts/similar_shots.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_script_refuses_owned_films():
    m = _script()
    assert m.public_film("night_of_the_living_dead")
    assert not m.public_film("not_a_film")
    owned = [f["id"] for f in yaml.safe_load((REPO / "configs/films.yaml").read_text())["films"] if f.get("public_domain") is not True]  # the real owned film, read at runtime
    if not owned:
        pytest.skip("no owned film in this registry (trimmed release tree)")
    assert not m.public_film(owned[0])


def test_equal_rational_scores_tie_to_smaller_id():  # 12/13 reached by two slot paths differs in the last float bit before rounding
    y = {"composition": "left-heavy"}                           # loses 1 (comp) -> 12/13
    x = {"size": "CU", "len_s": 1.5, "people": 3}               # loses 2/6 (size) + 1/6 (len) + 2/4 (people) = 1 -> 12/13
    far = {"size": "EWS", "lens": "ultra-wide", "angle": "aerial", "sound": "score", "palette_global": 0}
    df = table([{}, far, far, y, x])
    f = similar.features(df)
    raw = sum(similar.slot_scores_row(f, 0).values()) / 13
    assert raw[3] != raw[4] and raw[4] > raw[3]  # the float sums differ (the test is not vacuous)
    s = similar.similarity_row(f, 0)
    assert s[3] == s[4]
    assert list(similar.similar_to(df, 0, k=2).other_shot_id) == [3, 4]
    top = similar.top_k_table(df, k=1)
    r = top[top.shot_id == 0].iloc[0]
    assert r.other_shot_id == 3 and bool(r.tie_next)


def test_scores_rounded_to_twelve_decimals():  # an earlier change, similar-shots advisory 2 (M11): pin the precision of the rounding (spec Amendment 1: 12 decimals)
    y = {"composition": "left-heavy"}  # 12/13 = 0.923076923076923... has more than 12 decimals
    far = {"size": "EWS", "lens": "ultra-wide", "angle": "aerial", "sound": "score", "palette_global": 0}
    df = table([{}, far, y])
    s = similar.similarity_row(similar.features(df), 0)
    assert s[2] == 0.923076923077  # 12/13 rounded to 12 decimals; a 17-decimal rounding gives 0.9230769230769231
    assert np.array_equal(s, np.round(s, 12))
