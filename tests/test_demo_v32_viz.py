"""Demo v3.2: the seven headline charts in the demo (app/demo/viz.py; docs/demo_spec.md v3.2).

Pins: the VIZ_* palettes of style.py (exact values, every colour >= 3:1 against its theme ground, the shared ones against both grounds); every chart of
Home and Film overview uses only pinned colours; the Home hero (barcode) shows the example's film, never a LOCAL ONLY film in either mode, and no hero
when no public film qualifies; Film overview shows the six other charts in both modes (LOCAL ONLY films only in local mode, with their banner);
no Streamlit widget is added (the page chart's slot switch is a Vega-Lite binding); the tooltip shows the full token row (no line clamp).
`viz_snippet_problem` is the exact-match classifier of the new '<div class="cgt-viz' snippets used by an earlier review, an earlier review and an earlier review harnesses: a snippet passes
only if it equals viz.home_html / viz.overview_html for the film this page may show (Home: the hero caption's film, which must be public; Film overview:
the sidebar selection, a film listed in this mode), in the light or dark theme. Synthetic tmp roots (test_demo_guard fixture); no gold/dev labels."""
import functools
import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "app/demo"
if not (DEMO / "viz.py").exists():
    pytest.skip("app/demo/viz.py not on this branch", allow_module_level=True)

from test_demo_guard import CLIP, NOTLD, OWNED, core, run_app
from test_demo_v2_frontend import root  # noqa: F401 - pytest fixture

GROUNDS = {"light": "#F7F7F5", "dark": "#0F1113"}
HERO_CAP = re.compile(r'^<div class="cgt-hero-cap">(.+), shot (\d+) at (\d+\.\d) min, mid keyframe</div>$')


# ---- the shared classifier --------------------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def load_viz(app_dir: str):
    """core, style and viz of the app under test (by file path), with `core`/`style` bound to that app's own modules while viz is executed."""
    pytest.importorskip("altair")  # viz needs altair (absent from the CPU venv: those tests skip there)
    d = Path(app_dir)
    mods = {}
    saved = {k: sys.modules.get(k) for k in ("core", "style")}
    sys.path.insert(0, str(d))
    try:
        for name in ("core", "style", "viz"):
            spec = importlib.util.spec_from_file_location(f"v32_{name}_{abs(hash(app_dir))}", d / f"{name}.py")
            m = importlib.util.module_from_spec(spec)
            if name == "viz":
                sys.modules["core"], sys.modules["style"] = mods["core"], mods["style"]
            spec.loader.exec_module(m)
            mods[name] = m
    finally:
        sys.path.pop(0)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mods


def _texts(at) -> list[str]:
    out = []

    def rec(n):
        for attr in ("value", "body"):
            try:
                v = getattr(n, attr, None)
            except Exception:  # noqa: BLE001
                v = None
            if isinstance(v, str):
                out.append(v)
                break  # one text per element (a markdown element exposes the same string as value and body)
        for c in getattr(n, "children", {}).values():
            rec(c)
    rec(at._tree)
    return out


@functools.lru_cache(maxsize=64)
def _snippets(app_dir: str, r: str, film: str, public: bool, kind: str, label: str, shot: int, stamp: int) -> frozenset:
    m = load_viz(app_dir)
    df = m["viz"].prepare(m["core"].load_table(Path(r), film, public)[0])
    out = set()
    for theme in ("light", "dark"):
        h = m["viz"].home_html(df, label, shot, theme) if kind == "home" else m["viz"].overview_html(Path(r), film, public, df, theme)
        out |= set(h.values())
    return frozenset(out)


def _stamp(m, r: Path, film: str) -> int:
    return m["core"].table_path(r, film)[0].stat().st_mtime_ns


def viz_snippet_problem(t: str, at, r: Path, public: bool, app_dir: Path = DEMO) -> str | None:
    """None if `t` is exactly one of the v3.2 snippets this page may show, else the reason (unknown snippets always fail)."""
    if not t.startswith('<div class="cgt-viz'):
        return "not a v3.2 viz snippet"
    m = load_viz(str(app_dir))
    c = m["core"]
    page = at.session_state["page"] if "page" in at.session_state else None  # noqa: SIM401 - AppTest state has no .get
    listed = c.listed_films(r, public)
    if page == "Home":
        caps = [HERO_CAP.match(x) for x in _texts(at) if x.startswith('<div class="cgt-hero-cap">')]
        caps = [x for x in caps if x]
        if len(caps) != 1:
            return "viz snippet on Home without exactly one hero caption"
        label, shot = caps[0].group(1), int(caps[0].group(2))
        films = [f for f in listed if not c.is_local_only(r, f) and c.film_label(r, f) == label]  # never a LOCAL ONLY film, in either mode
        for f in films:
            if t in _snippets(str(app_dir), str(r), f, public, "home", label, shot, _stamp(m, r, f)):
                return None
        return "Home viz snippet is not the hero film's"
    if page == "Film overview":
        film = at.session_state["ov_film"] if "ov_film" in at.session_state else (listed[0] if listed else None)
        if film not in listed:
            return "overview film not listed in this mode"
        return None if t in _snippets(str(app_dir), str(r), film, public, "overview", "", 0, _stamp(m, r, film)) else "overview viz snippet not of the selected film"
    return f"viz snippet on page {page!r}"


# ---- palettes ---------------------------------------------------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _style():
    """style.py alone (pure, no streamlit or altair import at module level): the palette and CSS pins run in every venv."""
    spec = importlib.util.spec_from_file_location("v32_style_only", DEMO / "style.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _lum(h: str) -> float:
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_viz_palettes_pinned_and_contrast():
    s = _style()
    assert s.VIZ_RAMP == {"light": ["#022D2B", "#073E3B", "#0A504D", "#07625F", "#137571", "#348682", "#4D9893"],
                          "dark": ["#93FEF7", "#80E5DE", "#6DCDC6", "#5AB5AF", "#479D98", "#358782", "#21716D"]}
    assert s.VIZ_GREYS == {"light": ["#27292B", "#56585A", "#8A8C8E"], "dark": ["#616365", "#A1A3A5", "#E6E8EA"]}
    assert s.VIZ_NOMINAL == ["#1F8F88", "#C27A1A", "#4F78C4", "#6F8A2E", "#8E5FA8"] and set(s.VIZ_NOMINAL) <= set(s.CHART_COLORS)
    assert s.VIZ_HIGHLIGHT == s.CHART_COLORS[2] and s.VIZ_NEUTRAL == "#808389"
    # v3.3: the quiet and text tokens pinned exactly as well
    assert s.VIZ_MISSING == {"light": "#D9DAD5", "dark": "#2C3035"} and s.VIZ_LINE == {"light": "#D9DAD5", "dark": "#2C3035"}
    assert s.VIZ_SURFACE == {"light": "#ECEDEA", "dark": "#191C1F"} and s.ACCENT == {"light": "#146964", "dark": "#3FB5AD"}
    assert s.VIZ_INK == {"light": "#16181B", "dark": "#ECEBE7"} and s.VIZ_MUTED == {"light": "#565A61", "dark": "#A2A6AD"}
    for t, g in GROUNDS.items():
        assert s.VIZ_GROUND[t] == g
        for c in s.VIZ_RAMP[t] + s.VIZ_GREYS[t]:
            assert contrast(c, g) >= 3.0, (t, c)
        lum = [_lum(c) for c in s.VIZ_RAMP[t]]
        assert lum == sorted(lum) if t == "light" else lum == sorted(lum, reverse=True)  # monotone: index 0 most intense on that ground
        assert contrast(s.VIZ_MUTED[t], g) >= 4.5 and contrast(s.VIZ_INK[t], g) >= 7
    for c in [*s.VIZ_NOMINAL, s.VIZ_HIGHLIGHT, s.VIZ_NEUTRAL]:
        assert all(contrast(c, g) >= 3.0 for g in GROUNDS.values()), c
    assert s.contrast("#000000", "#FFFFFF") == pytest.approx(21.0)


def test_palette_check_can_fail():
    assert contrast("#9A9DA3", GROUNDS["light"]) < 3.0  # the old lab neutral would fail the pin on the light ground


def _ranges(spec) -> list:
    return [json.loads(m) for m in re.findall(r'"range": (\[[^\]]*\])', json.dumps(spec))]


# v3.3: every colour value of a v3.2 chart is checked, not only `range` lists. Data colours (scale ranges, and mark / encoding colours
# by default) may only be the pinned data palettes; the theme tokens ink / muted / ground / surface / accent only in the uses listed here
# ((mark type, channel, mark property or encoding value) -> tokens); config colours only ink / muted / line; a named scheme or a colour that is not a pinned hex fails.
TOKEN_USES = {("rect", "stroke", "mark"): {"ground"},  # 1-px page-colour gaps (page chart, cutting grammar)
              ("rect", "stroke", "value"): {"ink"},  # hover / Home-shot outline (barcode, a conditional encoding value)
              ("point", "color", "mark"): {"ink"}, ("text", "color", "mark"): {"ink", "accent"},  # the Home-shot pin and its label; swing numbers
              ("rule", "color", "mark"): {"muted"}, ("line", "color", "mark"): {"muted", "accent"},  # rhythm shot steps; pace implied length, trends
              ("rect", "color", "mark"): {"accent"}, ("rect", "color", "value"): {"surface"},  # rhythm swing bands; grammar diagonal + empty cells
              ("circle", "color", "mark"): {"accent"}}  # pace dots
COLOUR_KEYS = ("color", "fill", "stroke")


def palette_tokens(s) -> dict[str, set[str]]:
    """Pinned colour sets of style.py by role (both themes), upper-case hex."""
    def both(d):
        return {d["light"].upper(), d["dark"].upper()}
    data = {c.upper() for c in [*s.CHART_COLORS, *s.VIZ_NOMINAL, s.VIZ_HIGHLIGHT, s.VIZ_NEUTRAL]}
    for t in ("light", "dark"):
        data |= {c.upper() for c in [*s.VIZ_RAMP[t], *s.VIZ_GREYS[t], s.VIZ_MISSING[t]]}
    return {"data": data, "ink": both(s.VIZ_INK), "muted": both(s.VIZ_MUTED), "ground": both(s.VIZ_GROUND), "surface": both(s.VIZ_SURFACE),
            "accent": both(s.ACCENT), "line": both(s.VIZ_LINE)}


def spec_colour_problems(spec: dict, s) -> list:
    """Every colour of a chart spec (datasets excluded) that breaks the rules above, as (where, value); [] = clean."""
    tok = palette_tokens(s)
    probs = []

    def ok(v, allowed: set) -> bool:
        return isinstance(v, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", v) is not None and v.upper() in allowed

    def walk(n, mark: str | None, where: str):
        if isinstance(n, list):
            for x in n:
                walk(x, mark, where)
            return
        if not isinstance(n, dict):
            return
        if "scheme" in n:
            probs.append((where, f"named scheme {n['scheme']!r}"))
        if "mark" in n:
            m = n["mark"]
            mark = m if isinstance(m, str) else m.get("type") if isinstance(m, dict) else None
            if isinstance(m, dict):
                for k in COLOUR_KEYS:
                    if k in m and m[k] is not None and not ok(m[k], tok["data"] | set().union(*(tok[x] for x in TOKEN_USES.get((mark, k, "mark"), ())))):
                        probs.append((f"{where}/mark.{k}", m[k]))
        for ch, enc in (n.get("encoding") or {}).items():
            if ch not in COLOUR_KEYS or not isinstance(enc, dict):
                continue
            listed = tok["data"] | set().union(*(tok[x] for x in TOKEN_USES.get((mark, ch, "value"), ())))
            conds = enc.get("condition", [])
            for c in (conds if isinstance(conds, list) else [conds]) + [enc]:
                if isinstance(c, dict) and c.get("value") is not None and not ok(c["value"], listed):
                    probs.append((f"{where}/encoding.{ch}.value", c["value"]))
                sc = c.get("scale") if isinstance(c, dict) else None
                if isinstance(sc, dict):
                    if "scheme" in sc:
                        probs.append((f"{where}/encoding.{ch}.scale", f"named scheme {sc['scheme']!r}"))
                    for v in sc.get("range", []) if isinstance(sc.get("range"), list) else [sc.get("range")] if "range" in sc else []:
                        if not ok(v, tok["data"]):  # a range is data colour: no theme token
                            probs.append((f"{where}/encoding.{ch}.scale.range", v))
        for k, v in n.items():
            if k in ("datasets", "data", "config", "encoding", "mark"):
                continue
            walk(v, mark, f"{where}/{k}")

    walk({k: v for k, v in spec.items() if k not in ("datasets", "config")}, None, "")
    cfg = json.dumps(spec.get("config", {}))
    for v in re.findall(r'"(?:\w*[cC]olor|fill|stroke|background)": "([^"]*)"', cfg):
        if v != "transparent" and not ok(v, tok["ink"] | tok["muted"] | tok["line"]):
            probs.append(("config", v))
    body = json.dumps({k: v for k, v in spec.items() if k != "datasets"})
    every = set().union(*tok.values())
    probs += [("hex literal", h) for h in re.findall(r"#[0-9A-Fa-f]{6}\b", body) if h.upper() not in every]
    return probs


def viz_spec_ok(spec: dict, app_dir: Path = DEMO) -> bool:
    """A v3.2 chart (usermeta cgt_viz): every colour value (mark, encoding value, condition, scale range, config) is a pinned colour in an allowed
    role, and no named scheme (spec_colour_problems); an earlier change's range check (viz.palette_sets()) is kept as well."""
    m = load_viz(str(app_dir))
    allowed = set(m["viz"].palette_sets())
    rs = _ranges(spec)
    return (spec.get("usermeta") == {"cgt_viz": "v3.2"} and all(set(r) <= allowed for r in rs)
            and spec_colour_problems(spec, m["style"]) == [])


def _specs(at) -> list[dict]:
    return [json.loads(c.proto.spec) for c in at.get("vega_lite_chart")]


# ---- Home hero --------------------------------------------------------------------------------------------------------------------------------------------

def _qualify(r: Path, film: str, ok: bool = True) -> None:
    p = r / f"data/processed/{film}_tokens_v1.parquet"
    df = pd.read_parquet(p)
    df["token_row"] = [f"[size={'CU' if ok else 'WS'} | lens=wide | people=1 | key=mid-key]" for _ in range(len(df))]
    df.to_parquet(p)


@pytest.mark.parametrize("public", [True, False])
def test_home_hero_is_the_example_film_barcode(monkeypatch, root, public):  # noqa: F811
    for f in (NOTLD, OWNED, CLIP):
        _qualify(root, f)
    at = run_app(monkeypatch, root, public, "Home")
    snips = [t for t in _texts(at) if t.startswith('<div class="cgt-viz')]
    assert len(snips) == 4 and all(viz_snippet_problem(t, at, root, public) is None for t in snips)
    specs = _specs(at)
    assert len(specs) == 1 and viz_spec_ok(specs[0])
    assert core.film_label(root, NOTLD) in snips[0] and OWNED not in json.dumps(specs) and CLIP not in json.dumps(specs)
    assert "shot_hover" in json.dumps(specs[0]) and "token row" in json.dumps(specs[0])  # hover lights the shot and shows its row
    cap = next(t for t in _texts(at) if t.startswith('<div class="cgt-hero-cap">'))
    shot = int(HERO_CAP.match(cap).group(2))
    assert f"datum.shot_id == {shot}" in json.dumps(specs[0])  # the example shot is outlined in the barcode


@pytest.mark.parametrize("public", [True, False])
def test_no_hero_when_only_local_films_qualify(monkeypatch, root, public):  # noqa: F811
    _qualify(root, NOTLD, False)
    for f in (OWNED, CLIP):
        _qualify(root, f)
    at = run_app(monkeypatch, root, public, "Home")
    assert not [t for t in _texts(at) if 'class="cgt-viz' in t] and not at.get("vega_lite_chart") and len(at.image) == 0


def test_classifier_rejects_foreign_snippets(monkeypatch, root):  # noqa: F811
    for f in (NOTLD, OWNED, CLIP):
        _qualify(root, f)
    m = load_viz(str(DEMO))
    home = run_app(monkeypatch, root, False, "Home")
    df = m["viz"].prepare(core.load_table(root, OWNED, False)[0])
    owned = m["viz"].home_html(df, core.film_label(root, OWNED), 1, "light")
    assert all(viz_snippet_problem(t, home, root, False) for t in owned.values() if core.film_label(root, OWNED) in t or "shot 1," in t)
    assert viz_snippet_problem('<div class="cgt-viz cgt-vnote">anything else</div>', home, root, False)
    real = [t for t in _texts(home) if t.startswith('<div class="cgt-viz')]
    assert viz_snippet_problem(real[0].replace("shots", "shots!"), home, root, False)
    ov = run_app(monkeypatch, root, True, "Film overview")
    dfo = m["viz"].prepare(core.load_table(root, OWNED, False)[0])
    other = m["viz"].overview_html(root, OWNED, False, dfo, "light")["fingerprint"]
    assert viz_snippet_problem(other, ov, root, True)  # a LOCAL ONLY film's fingerprint on the public overview
    assert viz_snippet_problem(real[0], ov, root, True)  # a Home snippet on the overview


# ---- Film overview ----------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("public,film", [(True, None), (False, None), (False, OWNED), (False, CLIP)])
def test_overview_shows_six_charts_in_sections(monkeypatch, root, public, film):  # noqa: F811
    pytest.importorskip("streamlit.testing.v1")
    at = run_app(monkeypatch, root, public, "Film overview")
    if film:
        at.sidebar.selectbox(key="ov_film").set_value(film).run()
    assert not at.exception
    snips = [t for t in _texts(at) if t.startswith('<div class="cgt-viz')]
    assert snips and all(viz_snippet_problem(t, at, root, public) is None for t in snips), [viz_snippet_problem(t, at, root, public) for t in snips]
    for title in ("The film as a page", "Contact-sheet barcode", "Pace", "Shot-size rhythm", "Cutting grammar", "Film fingerprint"):
        assert sum(f"<h4>{title}</h4>" in t for t in snips) == 1, title
    assert sum(t.startswith('<div class="cgt-viz cgt-vnote">') for t in snips) == 6  # every chart repeats the pointer to the full note
    viz_specs = [s for s in _specs(at) if s.get("usermeta")]
    assert len(viz_specs) == 6 and all(viz_spec_ok(s) for s in viz_specs)
    assert any("colour_by" in json.dumps(s) and '"input": "radio"' in json.dumps(s) for s in viz_specs)  # the slot switch is a chart binding
    assert len(at.image) == 1  # the contact strip
    errs = [e.value for e in at.error]
    shown = at.session_state["ov_film"]
    if core.is_local_only(root, shown):
        assert any(core.BANNER_LOCAL_ONLY in e for e in errs)  # the LOCAL ONLY banner stays on top of the charts
    else:
        assert not errs


def test_overview_public_never_lists_local_films(monkeypatch, root):  # noqa: F811
    at = run_app(monkeypatch, root, True, "Film overview")
    assert list(at.sidebar.selectbox(key="ov_film").options) == [core.film_label(root, NOTLD)]
    blob = json.dumps(_specs(at)) + " ".join(_texts(at))
    assert OWNED not in blob and CLIP not in blob


def _widget_keys(n) -> set:
    k = getattr(n, "key", None)
    out = {k} if isinstance(k, str) and k and not k.startswith("$$") and type(n).__name__ not in ("Block", "Button", "DownloadButton") else set()
    for c in getattr(n, "children", {}).values():
        out |= _widget_keys(c)
    return out


def test_no_new_widgets_on_home_and_overview(monkeypatch, root):  # noqa: F811
    for page, keys in (("Home", {"page"}), ("Film overview", {"page", "ov_film", "ov_win"})):
        at = run_app(monkeypatch, root, True, page)
        assert _widget_keys(at._tree) == keys, page


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_tooltip_shows_the_full_row(theme):
    css = _style().css(theme)
    rule = re.search(r"#vg-tooltip-element table tr td\.value\{([^}]*)\}", css).group(1)
    assert "-webkit-line-clamp:none" in rule and "white-space:normal" in rule and "display:table-cell" in rule and "overflow-wrap:anywhere" in rule
