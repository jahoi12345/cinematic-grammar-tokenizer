"""Demo app v1.0: About page, local-only Add-clip page, CSV export, full token rows,
readable chart labels, titles, toolbar. Synthetic tmp roots only; never touches film data."""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if not (REPO / "app/demo/core.py").exists():
    pytest.skip("app/demo not on this branch", allow_module_level=True)

try:
    from _private_values import owned_contra_ratio
except ImportError:  # the private loader is not part of an exported copy: nothing private to guard against
    def owned_contra_ratio() -> str:
        return "<no private value>"
from test_demo_guard import CLIP, NOTLD, OWNED, SECRET, _table, _texts, core, make_root, run_app
from test_deown_checks import _owned

TITLE = "Night of the Living Dead (1968)"
RAW_SERIES = list(core.SERIES_LABELS)


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path, [{"id": NOTLD, "public_domain": True, "title": "Night of the Living Dead", "year": 1968},
                                {"id": OWNED, "public_domain": False, "title": "Owned Film X", "year": 2000}])


@pytest.fixture
def real_root(tmp_path):
    """The REAL owned id and title in a tmp registry (read at runtime; this file never names them)."""
    fid, _ = _owned()
    import yaml
    real = next(f for f in yaml.safe_load((REPO / "configs/films.yaml").read_text())["films"] if f["id"] == fid)
    r = make_root(tmp_path / "real", [{"id": NOTLD, "public_domain": True, "title": "Night of the Living Dead", "year": 1968},
                             {"id": fid, "public_domain": False, "title": real.get("title", fid), "year": real.get("year")}], [], frames=False)
    _table(fid, seed=7).to_parquet(r / f"data/processed/{fid}_tokens_v1.parquet")
    return r


# ---- B1 About ------------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("public", [True, False])
def test_about_is_first_and_explains_the_row(monkeypatch, root, public):
    at = run_app(monkeypatch, root, public, "About")
    assert at.sidebar.radio(key="page").options[0] == "Home" and "About" in at.sidebar.radio(key="page").options  # v2.0 C1: Home is first
    text = _texts(at).replace("`", "")
    for slot, _, _ in core.SLOT_HELP:
        assert f"| {slot} |" in text
    assert len(core.SLOT_HELP) == 13
    assert "| * | low confidence" in text and "| ? | missing" in text
    for cap in (core.CAPTION_PREDICTIONS, core.CAPTION_SHAPES, core.CAPTION_INTERVALS, core.CAPTION_MOVEMENT_GENERIC, core.CAPTION_SIMILAR):
        assert cap.replace("`", "") in text
    assert "INFLATED" in text and "Within-film shapes only" in text and "UNMEASURED" in text
    assert TITLE in text


def test_about_slots_match_tokens_config():
    import yaml
    assert [s for s, _, _ in core.SLOT_HELP] == yaml.safe_load((REPO / "configs/tokens.yaml").read_text())["slots"]


def test_about_default_page_without_selection(monkeypatch, root):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    monkeypatch.setenv("CGT_DEMO_PUBLIC", "1")
    at = testing.AppTest.from_file(str(REPO / "app/demo/app.py"), default_timeout=120)
    at.run()
    assert not at.exception and at.sidebar.radio(key="page").value == "Home"


def test_about_public_never_names_owned_film(monkeypatch, root, real_root):
    low = _texts(run_app(monkeypatch, root, True, "About")).lower()
    assert [s for s in (*SECRET, "owned film x") if s in low] == []
    _, terms = _owned()
    low = _texts(run_app(monkeypatch, real_root, True, "About")).lower()
    assert [t for t in terms if t in low] == [] and owned_contra_ratio() not in low
    assert "local only" not in low  # the local-mode banner note is not shown in public mode


def test_about_without_films(monkeypatch, root):
    (root / "configs/films.yaml").unlink()  # public mode fails closed: no film, About still renders
    text = _texts(run_app(monkeypatch, root, True, "About"))
    assert "No film is available in this mode" in text


# ---- B2 Add your own clip ------------------------------------------------------------------------------------------------------------------------------

def test_add_clip_absent_in_public_mode(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "About")
    assert "Add your own clip" not in at.sidebar.radio(key="page").options
    with pytest.raises(ValueError):
        at.sidebar.radio(key="page").set_value("Add your own clip").run()
    assert "tokenize_video.py" not in _texts(at)


def test_add_clip_local_shows_command_only(monkeypatch, root):
    at = run_app(monkeypatch, root, False, "Add your own clip")
    assert at.sidebar.radio(key="page").options[-1] == "Add your own clip"
    text = _texts(at)
    assert core.ADD_CLIP_COMMAND in text and "UNCHECKED" in text and "--cut-threshold N" in text and "stays local" in text
    assert len(at.get("file_uploader")) == 0 and len(at.get("button")) == 0 and len(at.get("download_button")) == 0
    src = (REPO / "app/demo/app.py").read_text() + (REPO / "app/demo/core.py").read_text()
    assert "subprocess" not in src and "file_uploader" not in src and "os.system" not in src


# ---- B3 CSV --------------------------------------------------------------------------------------------------------------------------------------------

def test_csv_columns_and_guard(root):
    df, _ = core.load_table(root, NOTLD, True)
    view = df.iloc[5:17]
    out = pd.read_csv(io.BytesIO(core.csv_bytes(root, NOTLD, True, view)))
    assert list(out.columns) == [c for c in df.columns if c not in ("kf_start", "kf_mid", "kf_end")]
    assert len(out) == 12 and set(out["film_id"]) == {NOTLD} and list(out["shot_id"]) == list(range(5, 17))
    raw = core.csv_bytes(root, NOTLD, True, view).decode()
    assert "data/interim" not in raw and "keyframes" not in raw and ".jpg" not in raw
    assert "static;pan-left" in raw  # list cells joined, not numpy reprs
    odf, _ = core.load_table(root, OWNED, False)
    for f in (OWNED, CLIP):
        with pytest.raises(PermissionError):
            core.csv_bytes(root, f, True, odf)
    with pytest.raises(ValueError):
        core.csv_bytes(root, NOTLD, True, odf)  # rows of another film under the public film's name
    assert core.csv_bytes(root, OWNED, False, odf)  # local mode exports the owned film locally


def _capture_csv(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.runtime.media_file_manager import MediaFileManager
    got = []
    orig = MediaFileManager.add

    def add(self, data, mimetype, *a, **k):
        if mimetype == "text/csv":
            got.append((k.get("file_name"), data))
        return orig(self, data, mimetype, *a, **k)
    monkeypatch.setattr(MediaFileManager, "add", add)
    return got


def test_csv_download_in_app_public_only_public_film(monkeypatch, root):
    got = _capture_csv(monkeypatch)
    at = run_app(monkeypatch, root, True, "Shot explorer")
    assert len(at.get("download_button")) == 1
    assert at.get("download_button")[0].proto.label == "CSV (60 rows)"  # short label, not cut at 800 px
    _, data = got[-1]
    out = pd.read_csv(io.BytesIO(data))
    assert set(out["film_id"]) == {NOTLD} and len(out) == 60
    assert not {"kf_start", "kf_mid", "kf_end"} & set(out.columns)
    low = data.decode().lower()
    assert [s for s in SECRET if s in low] == [] and "data/interim" not in low


def test_csv_download_follows_filters(monkeypatch, root):
    got = _capture_csv(monkeypatch)
    at = run_app(monkeypatch, root, True, "Shot explorer")
    at.sidebar.multiselect(key="ex_size").set_value(["CU"]).run()
    assert not at.exception
    out = pd.read_csv(io.BytesIO(got[-1][1]))
    df, _ = core.load_table(root, NOTLD, True)
    assert len(out) == int((df["size"] == "CU").sum()) > 0 and set(out["size"]) == {"CU"}


# ---- B4 token rows -------------------------------------------------------------------------------------------------------------------------------------

def test_token_rows_rendered_complete_and_wrapped(monkeypatch, root):
    p = root / f"data/processed/{NOTLD}_tokens_v1.parquet"
    df = pd.read_parquet(p)
    df["token_row"] = [f"[size=EWS* | lens=ultra-wide* | angle=low | framing=establishing* | light=artificial/high-contrast* | comp=right-heavy* | key=low-key | "
                       f"DoF=shallow | people=4+ | sound=ambient | len=16-32s | palette=g21 | move=push-in+boom-up+tilt-down+move-left+pan-right | shot={i:05d}END]" for i in range(len(df))]
    df.to_parquet(p)
    rows = set(df["token_row"])
    at = run_app(monkeypatch, root, True, "Shot explorer")
    codes = [c for c in at.get("code") if c.proto.code_text.startswith("[")]
    assert len(codes) == 8 + 8 * 5  # 8 shots on the page + 5 similar shots each
    for c in codes:
        assert c.proto.code_text in rows and c.proto.code_text.endswith("END]") and len(c.proto.code_text) > 200
        assert c.proto.wrap_lines is True


# ---- B5 labels, titles, toolbar ------------------------------------------------------------------------------------------------------------------------

def _chart_specs(at) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("vega_lite_chart")]


def _encodings(spec):
    if isinstance(spec, dict):
        if "encoding" in spec:
            yield spec["encoding"]
        for v in spec.values():
            yield from _encodings(v)
    elif isinstance(spec, list):
        for v in spec:
            yield from _encodings(v)


def test_chart_legends_and_tooltips_are_readable(monkeypatch, root):
    at = run_app(monkeypatch, root, True, "Film overview")
    specs = _chart_specs(at)
    assert len(specs) >= 7
    seen_labels = set()
    for spec in specs:
        for enc in _encodings(spec):
            col = enc.get("color", {})
            assert col.get("field") != "series"
            for t in enc.get("tooltip", []) if isinstance(enc.get("tooltip"), list) else []:
                assert t.get("title") or t.get("field") not in ("series", "ci_lo", "ci_hi", "n_shots", "flag_sparse")
    from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
    for c in at.get("vega_lite_chart"):
        for ds in c.proto.datasets:
            d = convert_arrow_bytes_to_pandas_df(ds.data.data)
            if "label" in d.columns:
                seen_labels |= set(d["label"])
    assert {"shot starts per minute", "median shot length (s)", "share of dialogue shots", "share of contradictory sets"} <= seen_labels
    assert not set(RAW_SERIES) & seen_labels


def test_film_titles_and_toolbar(monkeypatch, root):
    st = pytest.importorskip("streamlit")
    st.set_option("client.toolbarMode", "auto")  # process-global: reset so that the app itself must set it
    at = run_app(monkeypatch, root, True, "Film overview")
    assert at.sidebar.selectbox(key="ov_film").options == [TITLE]
    assert at.sidebar.selectbox(key="ov_film").value == NOTLD
    assert f"Film overview: {TITLE}" in _texts(at)
    assert st.get_option("client.toolbarMode") == "minimal"
    at = run_app(monkeypatch, root, False, "Shot explorer", ex_film=OWNED)
    assert "Shot explorer: Owned Film X (2000)" in _texts(at) and core.BANNER_LOCAL_ONLY in _texts(at)


def test_side_by_side_public_explains_single_film(monkeypatch, root):
    text = _texts(run_app(monkeypatch, root, True, "Side by side"))
    assert "needs two listed films" in text and "only 1 such film is available here" in text and "public domain" in text and TITLE in text


# ---- an earlier review follow-up (A5): the Add-clip page's own public check, independent of the page list ------------------------------------------------------------

def _render_add_clip(root_s: str, app_s: str, public: bool):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("demo_app_unit", app_s)  # not __main__: importing renders nothing
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.page_add_clip(Path(root_s), public, [])


@pytest.mark.parametrize("public", [True, False])
def test_page_add_clip_renders_nothing_in_public_mode(monkeypatch, root, public):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CGT_DEMO_ROOT", str(root))
    at = testing.AppTest.from_function(_render_add_clip, args=(str(root), str(REPO / "app/demo/app.py"), public), default_timeout=60)
    at.run()
    assert not at.exception
    text = _texts(at)
    if public:
        assert "tokenize_video.py" not in text and len(at.header) == 0 and len(at.markdown) == 0 and len(at.code) == 0
        assert len(at.warning) == 0 and len(at.info) == 0 and len(at.title) == 0  # importing the app did not run main()
    else:  # positive control: the same call in local mode renders the page
        assert core.ADD_CLIP_COMMAND in text and "--client.toolbarMode minimal" in text


def test_run_command_documented_with_toolbar_flag(monkeypatch, root):
    if (REPO / "docs/demo_spec.md").exists():  # the spec is not part of a release tree
        assert "--client.toolbarMode minimal" in (REPO / "docs/demo_spec.md").read_text()
    assert "CGT_DEMO_PUBLIC=1 uv run streamlit run app/demo/app.py --client.toolbarMode minimal" in _texts(run_app(monkeypatch, root, True, "About"))
    assert "near-tie" not in _texts(run_app(monkeypatch, root, True, "About"))


def _style():
    import importlib.util
    spec = importlib.util.spec_from_file_location("demo_style_v1_colours", REPO / "app/demo/style.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GROUNDS = ("#F7F7F5", "#0F1113")  # page backgrounds of [theme.light] / [theme.dark] in .streamlit/config.toml


def _lum(h: str) -> float:
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _categorical(colr: dict) -> bool:
    """The categorical colour scale: exactly the explicit range style.CHART_COLORS (one palette for both themes; a named scheme no
    longer passes), and every colour of it at least 3:1 against both page backgrounds (WCAG non-text contrast)."""
    colours = _style().CHART_COLORS
    rng = (colr.get("scale") or {}).get("range")
    return rng == colours and len(set(colours)) >= 6 and all(_contrast(c, g) >= 3.0 for c in colours for g in GROUNDS)


def test_chart_palette_matches_theme_and_contrast():
    import tomllib
    colours = _style().CHART_COLORS
    theme = tomllib.loads((REPO / ".streamlit/config.toml").read_text())["theme"]
    assert theme["chartCategoricalColors"] == colours
    assert theme["light"]["backgroundColor"] == GROUNDS[0] and theme["dark"]["backgroundColor"] == GROUNDS[1]
    assert all(_contrast(c, g) >= 3.0 for c in colours for g in GROUNDS)
    assert not _categorical({"scale": {"scheme": "tableau10"}}) and not _categorical({"scale": {"range": colours[::-1]}})
    assert _contrast("#E0E0E0", GROUNDS[0]) < 3.0  # the contrast check can fail: a pale grey is below 3:1 on the light ground


def test_chart_legends_wrap_with_opaque_swatches(monkeypatch, root):
    """every colour legend wraps (two columns, one for more than three series) with full-opacity swatches and a categorical scheme (the band layer shares the line's legend)."""
    at = run_app(monkeypatch, root, True, "Film overview")
    n = 0
    viz_charts = 0
    for c in at.get("vega_lite_chart"):
        spec = json.loads(c.proto.spec)
        if spec.get("usermeta") == {"cgt_viz": "v3.2"}:  # v3.2 charts: HTML legends, only the pinned palettes (tests/test_demo_v32_viz.py)
            from test_demo_v32_viz import viz_spec_ok
            assert viz_spec_ok(spec)
            viz_charts += 1
            continue
        for enc in _encodings(spec):
            colr = enc.get("color")
            if not colr or "field" not in colr:
                continue
            lg = colr.get("legend")
            assert lg is not None and lg.get("columns") in (1, 2) and lg.get("symbolOpacity") == 1 and _categorical(colr)
            n += 1
    assert n >= 14  # line + band of the 7+ series charts
    assert viz_charts == 6  # page, size lane, pace (two), rhythm, cutting grammar
    assert any('"labelAngle": 0' in c.proto.spec for c in at.get("vega_lite_chart"))  # horizontal length-bin labels
    assert sum(1 for c in at.get('vega_lite_chart') for e in _encodings(json.loads(c.proto.spec)) if ((e.get('color') or {}).get('legend') or {}).get('columns') == 1) >= 2  # v3.2: viz charts carry legend None
