"""The seven headline charts. Ported checks: every chart builds from the
real Night of the Living Dead table and shows the table's numbers (barcode widths sum to the running time, transition rows and slot distributions sum to
1, page pieces cover the film once, every shot start is counted); the same checks on the synthetic tables of test_demo_guard, so they run without data;
keyframes for a LOCAL ONLY film are refused in public mode. Skips cleanly where altair is missing."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("altair")
REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "app/demo"
if not (DEMO / "viz.py").exists():
    pytest.skip("app/demo/viz.py not on this branch", allow_module_level=True)
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import viz as V
from test_demo_guard import CLIP, NOTLD, OWNED, core, make_root
from test_demo_v3_revamp import emoji_in

FILM = "night_of_the_living_dead"
TABLE = f"data/processed/{FILM}_tokens_v1.parquet"


def _real_root() -> Path | None:
    for r in (os.environ.get("CGT_DEMO_ROOT"), REPO, REPO.parents[2] if len(REPO.parents) > 2 else None):
        if r and (Path(r) / TABLE).is_file():
            return Path(r)
    return None


ROOT = _real_root()
real = pytest.mark.skipif(ROOT is None, reason="the public film's token table is not present")


@pytest.fixture(scope="module")
def real_df():
    return V.prepare(core.load_table(ROOT, FILM, True)[0])


@pytest.fixture
def troot(tmp_path):
    return make_root(tmp_path, frames=False)


@pytest.fixture
def syn_df(troot):
    return V.prepare(core.load_table(troot, NOTLD, True)[0])


def _check_all(df, n_expected=None):
    bc = V.barcode(df)
    assert len(bc) == len(df) and (n_expected is None or len(df) == n_expected)
    assert np.isclose(((bc["end_min"] - bc["start_min"]) * 60).sum(), df["end_s"].max() - df["start_s"].min(), atol=1e-6)
    assert np.allclose(bc["start_min"].to_numpy()[1:], bc["end_min"].to_numpy()[:-1])  # contiguous: no gap, no overlap
    for lane, col, _, cons in V.LANES:
        assert (bc[lane] == df[col].astype(str)).all()
        if cons:
            assert (bc[f"low_{lane}"] == ~df[cons].astype(bool)).all()
    assert (bc["token_row"] == df["token_row"]).all()
    t = V.transitions(df)
    rows = t.groupby("from")["share"].sum()
    assert np.allclose(rows[t.groupby("from")["row_n"].first() > 0], 1.0)  # every size with an outgoing cut: its row sums to 1
    pairs = list(zip(df["size"].astype(str).iloc[:-1], df["size"].astype(str).iloc[1:], strict=True))
    assert t["n"].sum() == sum(1 for a, _ in pairs if a in set(t["from"]))
    for r in t.itertuples():
        assert r.n == sum(1 for a, b in pairs if a == r._1 and b == r.to)
    assert (t["diag"] == (t["from"] == t["to"])).all()
    fp = V.fingerprint(df)
    assert np.allclose(fp.groupby("slot")["share"].sum(), 1.0)
    assert (fp.groupby("slot")["n"].sum() == len(df)).all() and (fp["share_low"] <= fp["share"] + 1e-12).all()
    for label, col, src, cons in V.FINGERPRINT:
        if col not in df.columns:
            continue
        g = fp[fp["slot"] == label].set_index("value")
        assert (g["source"] == src).all()
        if col != "palette_tok":
            assert g["n"].to_dict() == df[col].astype(str).value_counts().to_dict()
        if cons and cons in df.columns:
            low = df[~df[cons].astype(bool)][col].astype(str).value_counts() / len(df)
            assert np.allclose(g["share_low"].reindex(low.index), low)
    rate = V.cut_rate(df)
    assert rate["starts"].sum() == len(df)
    fast = V.fastest_minutes(rate)
    assert fast["starts"].max() == rate["starts"].max()
    assert all(abs(a - b) > 1 for i, a in enumerate(fast["minute"]) for b in list(fast["minute"])[i + 1:])
    p = V.page_pieces(df)
    assert np.isclose((p["x1"] - p["x0"]).sum(), df["end_s"].max() - df["start_s"].min(), atol=1e-6)
    assert (p["x0"] >= 0).all() and (p["x1"] <= V.LINE_MIN * 60 + 1e-9).all() and (p["x1"] > p["x0"]).all()
    assert set(p["shot_id"]) == set(df["shot_id"])
    per_shot = p.assign(w=p["x1"] - p["x0"]).groupby("shot_id")["w"].sum()
    assert np.allclose(per_shot.sort_index().to_numpy(), df.set_index("shot_id")["len_s"].sort_index().to_numpy(), atol=1e-6)  # each shot once
    tr = V.size_trend(df)
    assert tr["closeness"].dropna().between(0, 6).all()
    sw = V.swings(tr)
    assert len(sw) <= 3 and (sw["end_min"] > sw["start_min"]).all() and np.allclose(sw["delta"], sw["to"] - sw["from"])
    e = V.strip_edges(df, 2400)
    assert e[0] == 0 and e[-1] == 2400 and np.diff(e).sum() == 2400 and (np.diff(e) >= 0).all()


def test_synthetic_tables_show_their_numbers(syn_df):
    _check_all(syn_df, 60)


@real
def test_real_table_shows_its_numbers(real_df):
    _check_all(real_df, 1077)
    assert len(V.fastest_minutes(V.cut_rate(real_df))) == 3 and 1 <= len(V.swings(V.size_trend(real_df))) <= 3


@real
def test_move_classes_follow_the_atoms(real_df):
    for atoms, cls in zip(real_df["move_atoms"], real_df["move_class"], strict=True):
        s = set(atoms)
        assert cls == ("static alone" if s == {"static"} else "static + other atoms" if "static" in s else "movement only")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_chart_builds(syn_df, theme):
    for spec in (V.barcode_chart(syn_df, theme, mark_shot=3), V.barcode_chart(syn_df, theme), V.rhythm_chart(syn_df, theme), V.size_lane_chart(syn_df, theme),
                 V.transition_chart(syn_df, theme), *V.pace_charts(syn_df, theme), V.page_chart(syn_df, theme)):
        d = spec.to_dict()
        assert d.get("vconcat") or d.get("layer") or d.get("mark")
        assert d["usermeta"] == {"cgt_viz": "v3.2"}
    html = V.fingerprint_html(syn_df, theme)
    assert html.count('class="cgt-vfpanel"') == len(V.FINGERPRINT)
    assert V.legend_html(syn_df, V.LANES, theme).count("cgt-vlrow") == len(V.LANES) + 1


@real
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_real_contact_strip_is_time_proportional(real_df, theme):
    img = V.contact_strip(ROOT, FILM, True, real_df.head(12), width=240, height=20)
    assert img.size == (240, 20)
    have, missing, thin = V.strip_counts(ROOT, FILM, True, real_df)
    assert have + missing == len(real_df) and have > 1000 and thin >= 0


@pytest.mark.parametrize("film", [OWNED, CLIP])
def test_keyframes_of_local_only_films_refused_in_public_mode(troot, syn_df, film):
    with pytest.raises(PermissionError):
        V.contact_strip(troot, film, True, syn_df.head(4), width=40, height=4)
    with pytest.raises(PermissionError):
        V.strip_counts(troot, film, True, syn_df)
    with pytest.raises(PermissionError):
        V.overview_html(troot, film, True, syn_df, "light")


def test_no_emoji_and_no_streamlit_in_viz():
    src = (DEMO / "viz.py").read_text(encoding="utf-8")
    assert not emoji_in(src) and "import streamlit" not in src
    assert not (REPO / "app/viz_lab").exists()  # removed: app/demo/viz.py is the one home of the charts
