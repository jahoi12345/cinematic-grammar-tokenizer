"""Demo v3.3: an earlier review advisories on demo v3.2 (a verification record; docs/demo_spec.md v3.3).

Pins:
- A1 tooltip: below 1240 px the tooltip is at most half the viewport wide (vega-tooltip then always finds a side), on phones a full-width band 12 px
  from both edges; the full-row rule (no line clamp) stays.
- A2 live theme switch: every theme-paired chart colour (light, dark) is re-mapped with CSS light-dark() in the SVG (fill / stroke / stop-color) and
  in the HTML legend swatches; the cutting-grammar labels are drawn in layers named by their ink in both themes; the v3.2 CSS text colours (kickers,
  notes, pin) are light-dark() pairs; the cutting grammar uses only exact ramp colours whose label reads >= 4.5:1 in both themes.
- A3 palette guard: viz_spec_ok rejects an off-palette band colour, a named scheme, a theme token in a scale range, a CSS colour name.
- A4 contact strip: in public mode the app opens only image files that core.frame_path returned for the public film.
- A5 pace dots >= 3:1 on both grounds. A6 the two selection rules are in the legends. A7 the Home caveat renders above the hero.
Synthetic tmp roots (test_demo_guard / test_demo_v2_frontend fixtures); no real film, no gold/dev labels."""
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "app/demo"
if not (DEMO / "viz.py").exists():
    pytest.skip("app/demo/viz.py not on this branch", allow_module_level=True)

from test_demo_guard import CLIP, NOTLD, OWNED, core, run_app
from test_demo_v2_frontend import root  # noqa: F401 - pytest fixture
from test_demo_v32_viz import (
    GROUNDS,
    _qualify,
    _specs,
    _style,
    _texts,
    contrast,
    load_viz,
    spec_colour_problems,
    viz_spec_ok,
)


def _mix(c: str, a: float, g: str) -> str:
    x = [round(int(c[i:i + 2], 16) * a + int(g[i:i + 2], 16) * (1 - a)) for i in (1, 3, 5)]
    return "#" + "".join(f"{v:02X}" for v in x)


def _rules(css: str) -> list[tuple[str, str]]:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    return [(m.group(1).strip(), m.group(2)) for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css)]


def _media_block(css: str, query: str) -> str:
    i = css.index(f"@media ({query}){{")
    depth, j = 0, css.index("{", i)
    for k in range(j, len(css)):
        depth += {"{": 1, "}": -1}.get(css[k], 0)
        if depth == 0:
            return css[j + 1:k]
    raise AssertionError(query)


# ---- A1 tooltip ---------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a1_tooltip_stays_inside_the_viewport(theme):
    css = _style().css(theme)
    mid = _media_block(css, "max-width:1240px")
    assert "#vg-tooltip-element{max-width:calc(50vw - 40px) !important}" in mid  # always fits on one side of the pointer (offset 10 px + padding)
    band = [d for s, d in _rules(css) if s.endswith("#vg-tooltip-element") and "left:12px" in d]
    assert band and all(x in band[0] for x in ("left:12px !important", "right:12px !important", "max-width:none !important", "width:auto !important"))
    assert css.index("left:12px !important") > css.index("@media (max-width:640px){\n  #vg-tooltip-element")  # inside the phone query
    rule = re.search(r"#vg-tooltip-element table tr td\.value\{([^}]*)\}", css).group(1)  # the full row stays unclamped
    assert "-webkit-line-clamp:none" in rule and "white-space:normal" in rule and "overflow-wrap:anywhere" in rule
    for s, d in _rules(css):
        if "vg-tooltip" in s:
            assert "line-clamp:" not in d.replace("line-clamp:none", "") and "max-height" not in d and "ellipsis" not in d, s


# ---- A2 live theme switch -------------------------------------------------------------------------------------------------------------------------------

def test_a2_theme_pairs_cover_every_theme_dependent_colour():
    s = _style()
    pairs = s.theme_pairs()
    seen = {}
    for lt, dk in pairs:
        for v in (lt, dk):
            assert seen.setdefault(v.upper(), (lt, dk)) == (lt, dk), f"{v} belongs to two pairs"
    want = list(zip(s.VIZ_RAMP["light"], s.VIZ_RAMP["dark"], strict=True)) + list(zip(s.VIZ_GREYS["light"], s.VIZ_GREYS["dark"], strict=True))
    want += [(d["light"], d["dark"]) for d in (s.VIZ_MISSING, s.VIZ_GROUND, s.VIZ_SURFACE, s.VIZ_LINE, s.VIZ_INK, s.VIZ_MUTED, s.ACCENT)]
    assert set(want) == set(pairs)
    shared = {c.upper() for c in [*s.CHART_COLORS, *s.VIZ_NOMINAL, s.VIZ_HIGHLIGHT, s.VIZ_NEUTRAL, "#FFFFFF"]}
    assert not shared & set(seen)  # theme-independent colours are never re-mapped
    for c in shared - {"#FFFFFF"}:
        assert all(contrast(c, g) >= 3.0 for g in GROUNDS.values()), c  # and they read on both grounds


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a2_css_remaps_every_pair_in_both_directions(theme):
    s = _style()
    css = s.css(theme)
    ch = '[data-testid="stVegaLiteChart"]'
    for lt, dk in s.theme_pairs():
        ld = f"light-dark({lt},{dk})"
        for attr in ("fill", "stroke", "stop-color"):
            assert f'{ch} [{attr}="{lt}" i],{ch} [{attr}="{dk}" i]{{{attr}:{ld} !important}}' in css, (attr, lt)
        assert f".cgt-vsw.cgt-c-{lt[1:].lower()},.cgt-vsw.cgt-c-{dk[1:].lower()}{{background:{ld} !important}}" in css
        assert contrast(lt, GROUNDS["light"]) != contrast(dk, GROUNDS["light"]) or lt == dk
    for a, ca in (("k", "#16181B"), ("w", "#FFFFFF")):
        for b, cb in (("k", "#16181B"), ("w", "#FFFFFF")):
            assert f"{ch} .cgt_lab_{a}{b}_marks text{{fill:light-dark({ca},{cb}) !important}}" in css
    # the v3.2 HTML text colours follow the theme too (kicker, notes, pin), and stay readable on both grounds
    rules = dict(_rules(css))
    assert f"color:light-dark({s.ACCENT['light']},{s.ACCENT['dark']})" in rules[".cgt-vkick"]
    assert f"color:light-dark({s.VIZ_MUTED['light']},{s.VIZ_MUTED['dark']})" in rules[".cgt-vnote,.cgt-vcap"]
    assert f"color:light-dark({s.VIZ_INK['light']},{s.VIZ_INK['dark']})" in rules[".cgt-vpin"]
    for t in ("light", "dark"):
        assert contrast(s.ACCENT[t], GROUNDS[t]) >= 4.5 and contrast(s.VIZ_MUTED[t], GROUNDS[t]) >= 4.5
    # A8: the Streamlit-themed charts' own grey axis text (3.46:1 on the light ground) is drawn in the muted pair
    assert contrast(s.STREAMLIT_CHART_GREY, GROUNDS["light"]) < 4.5
    assert f'{ch} .role-axis-label text[fill="{s.STREAMLIT_CHART_GREY}" i]' in css


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a2_grammar_cells_and_labels_in_both_themes(root, theme):  # noqa: F811
    m = load_viz(str(DEMO))
    viz, s = m["viz"], m["style"]
    df = viz.prepare(m["core"].load_table(root, NOTLD, True)[0])
    spec = viz.transition_chart(df, theme).to_dict()
    sc = next(lay["encoding"]["color"]["scale"] for lay in spec["layer"] if lay["encoding"]["color"].get("scale"))
    assert sc["type"] == "threshold" and sc["range"] == viz.grammar_range(theme) and set(sc["range"]) <= set(s.VIZ_RAMP[theme])
    for t in ("light", "dark"):  # every cell colour: >= 3:1 on its ground, and its label >= 4.5:1, in the theme drawn and after a live switch
        for c in viz.grammar_range(t):
            assert contrast(c, GROUNDS[t]) >= 3.0 and contrast(c, viz.text_on(c)) >= 4.5, (t, c)
    rows = next(v for v in spec["datasets"].values() if v and "lab" in v[0])
    mx = max(r["share"] for r in rows if not r["diag"]) or 1.0
    names = {lay.get("name") for lay in spec["layer"]}
    for r in rows:
        if r["flat"]:
            assert r["lab"] == "mm" and r["ink"] == s.VIZ_MUTED[theme]
            continue
        want = "".join("k" if viz.text_on(viz.grammar_fill(min(r["share"], mx), mx, t)) == "#16181B" else "w" for t in ("light", "dark"))
        assert r["lab"] == want and r["ink"] == viz.text_on(viz.grammar_fill(min(r["share"], mx), mx, theme))
        assert f"cgt_lab_{want}" in names
    # the threshold scale Vega draws and grammar_fill agree on every edge (bisect-right on the same floats)
    edges = viz.grammar_bins(mx)
    assert sc["domain"] == edges and viz.grammar_fill(edges[0], mx, theme) == viz.grammar_range(theme)[1]


def test_a2_legend_swatches_carry_their_colour_class(root):  # noqa: F811
    m = load_viz(str(DEMO))
    viz = m["viz"]
    df = viz.prepare(m["core"].load_table(root, NOTLD, True)[0])
    for theme in ("light", "dark"):
        for h in viz.overview_html(root, NOTLD, True, df, theme).values():
            for cls, sty in re.findall(r'<span class="cgt-vsw ([^"]*)" style="([^"]*)"', h):
                assert cls == "cgt-c-" + re.match(r"background:#([0-9A-Fa-f]{6})", sty).group(1).lower()
            assert '<span class="cgt-vsw" ' not in h


# ---- A3 palette guard -----------------------------------------------------------------------------------------------------------------------------------

def _all_specs(viz, df, theme):
    top, bottom = viz.pace_charts(df, theme)
    return {"hero": viz.barcode_chart(df, theme, mark_shot=int(df["shot_id"].iloc[0])), "rhythm": viz.rhythm_chart(df, theme),
            "lane": viz.size_lane_chart(df, theme), "grammar": viz.transition_chart(df, theme), "pace_top": top, "pace_bottom": bottom,
            "page": viz.page_chart(df, theme)}


def test_a3_every_real_chart_passes_the_full_colour_check(root):  # noqa: F811
    m = load_viz(str(DEMO))
    df = m["viz"].prepare(m["core"].load_table(root, NOTLD, True)[0])
    for theme in ("light", "dark"):
        for name, ch in _all_specs(m["viz"], df, theme).items():
            spec = ch.to_dict()
            assert spec_colour_problems(spec, m["style"]) == [], (theme, name)
            assert viz_spec_ok(spec), (theme, name)


def test_a3_off_palette_colours_and_schemes_fail(root):  # noqa: F811
    m = load_viz(str(DEMO))
    viz, s = m["viz"], m["style"]
    df = viz.prepare(m["core"].load_table(root, NOTLD, True)[0])
    top = viz.pace_charts(df, "light")[0].to_dict()
    assert viz_spec_ok(top)
    band = json.loads(json.dumps(top).replace(f'"color": "{s.VIZ_HIGHLIGHT}"', '"color": "#FF00FF"'))  # the fastest-minute band
    assert band != top and not viz_spec_ok(band)
    named = json.loads(json.dumps(top).replace(f'"color": "{s.VIZ_HIGHLIGHT}"', '"color": "magenta"'))
    assert not viz_spec_ok(named)
    base = {"usermeta": {"cgt_viz": "v3.2"}, "mark": "rect"}
    assert not viz_spec_ok({**base, "encoding": {"color": {"field": "x", "type": "nominal", "scale": {"scheme": "rainbow"}}}})  # named scheme
    assert not viz_spec_ok({**base, "encoding": {"stroke": {"value": "#00FF00"}}})
    assert not viz_spec_ok({**base, "encoding": {"color": {"field": "x", "type": "nominal", "scale": {"range": [s.ACCENT["light"], s.VIZ_RAMP["light"][0]]}}}})
    assert not viz_spec_ok({**base, "encoding": {"color": {"field": "x", "type": "nominal", "scale": {"range": [s.VIZ_INK["dark"]]}}}})  # ink as data
    assert not viz_spec_ok({"usermeta": {"cgt_viz": "v3.2"}, "mark": {"type": "bar", "color": s.ACCENT["light"]}})  # accent bars: not a listed use
    assert not viz_spec_ok({"usermeta": {"cgt_viz": "v3.2"}, "mark": {"type": "rect", "color": s.VIZ_SURFACE["dark"]}})  # surface as a mark colour
    assert viz_spec_ok({**base, "encoding": {"color": {"condition": {"test": "datum.flat", "value": s.VIZ_SURFACE["dark"]}, "field": "share",
                                                      "type": "quantitative", "scale": {"range": s.VIZ_RAMP["dark"]}}}})  # the listed use
    assert not viz_spec_ok({**base, "config": {"axis": {"labelColor": "#FF00FF"}}})


# ---- A4 contact strip: frames only via core.frame_path, only the public film ------------------------------------------------------------------------------

def test_a4_contact_strip_opens_only_frame_path_files_of_the_public_film(monkeypatch, root):  # noqa: F811
    from PIL import Image
    opened, granted = [], set()
    orig_open = Image.open

    def spy_open(fp, *a, **k):
        if isinstance(fp, (str, os.PathLike)):  # Streamlit re-opens the encoded PNG bytes (BytesIO): only files on disk count
            opened.append(str(Path(fp).resolve()))
        return orig_open(fp, *a, **k)
    monkeypatch.setattr(Image, "open", spy_open)
    at = run_app(monkeypatch, root, True, "Film overview")
    assert len(at.image) == 1
    kdir = str((root / f"data/interim/{NOTLD}/keyframes").resolve()) + os.sep
    assert opened and all(p.startswith(kdir) for p in opened), [p for p in opened if not p.startswith(kdir)]
    # direct call: every file the strip opens is one core.frame_path returned for this film in this mode
    m = load_viz(str(DEMO))
    viz, c = m["viz"], m["core"]
    orig_fp = c.frame_path

    def spy_fp(r, film, rel, public):
        p = orig_fp(r, film, rel, public)
        if p is not None:
            granted.add(str(Path(p).resolve()))
        return p
    monkeypatch.setattr(c, "frame_path", spy_fp)
    opened.clear()
    df = viz.prepare(c.load_table(root, NOTLD, True)[0])
    viz.contact_strip(root, NOTLD, True, df)
    assert opened and set(opened) <= granted and all(p.startswith(kdir) for p in opened)


def test_a4_spy_catches_a_bypass(monkeypatch, root):  # noqa: F811 - the probe can fail: a strip reading another film's frames directly is seen
    from PIL import Image
    opened = []
    orig_open = Image.open
    monkeypatch.setattr(Image, "open", lambda fp, *a, **k: opened.append(str(Path(fp).resolve())) or orig_open(fp, *a, **k))
    m = load_viz(str(DEMO))
    df = m["viz"].prepare(m["core"].load_table(root, NOTLD, True)[0])
    p = root / str(df["kf_mid"].iloc[0]).replace(NOTLD, OWNED)
    assert p.is_file()
    Image.open(p).close()
    kdir = str((root / f"data/interim/{NOTLD}/keyframes").resolve()) + os.sep
    assert not all(x.startswith(kdir) for x in opened)


# ---- A5 / A6 --------------------------------------------------------------------------------------------------------------------------------------------

def test_a5_pace_dots_read_on_both_grounds(root):  # noqa: F811
    m = load_viz(str(DEMO))
    viz, s = m["viz"], m["style"]
    df = viz.prepare(m["core"].load_table(root, NOTLD, True)[0])
    for t in ("light", "dark"):
        top = viz.pace_charts(df, t)[0].to_dict()
        dots = [lay for lay in top["layer"] if (lay["mark"]["type"] if isinstance(lay["mark"], dict) else lay["mark"]) == "circle"]
        assert len(dots) == 1 and dots[0]["mark"]["opacity"] == viz.PACE_DOT_OPACITY and dots[0]["mark"]["color"] == s.ACCENT[t]
        assert contrast(_mix(s.ACCENT[t], viz.PACE_DOT_OPACITY, GROUNDS[t]), GROUNDS[t]) >= 3.0, t
    assert contrast(_mix(s.ACCENT["light"], 0.6, GROUNDS["light"]), GROUNDS["light"]) < 3.0  # the v3.2 value fails: the check can fail


def test_a6_selection_rules_in_the_legends(root):  # noqa: F811
    m = load_viz(str(DEMO))
    viz = m["viz"]
    df = viz.prepare(m["core"].load_table(root, NOTLD, True)[0])
    for t in ("light", "dark"):
        pace, rhythm = viz.pace_legend_html(df, t), viz.rhythm_legend_html(df, t)
        assert viz.PACE_RULE in pace and "no two adjacent" in viz.PACE_RULE and "most shot starts" in viz.PACE_RULE
        if viz.swings(viz.size_trend(df)).shape[0]:
            assert viz.SWING_RULE in rhythm and rhythm.index(viz.SWING_RULE) < rhythm.index('<ol class="cgt-vswings">')
        assert "turning points" in viz.SWING_RULE and "only rise or only fall" in viz.SWING_RULE and "largest change" in viz.SWING_RULE
        assert "sigma 100 s" in rhythm  # the smoother, as in v3.2


# ---- A7 Home caveat above the hero -------------------------------------------------------------------------------------------------------------------------

def _order(at) -> list[tuple[str, str]]:
    out = []

    def rec(n):
        kind = type(n).__name__
        txt = ""
        for attr in ("value", "body"):
            try:
                v = getattr(n, attr, None)
            except Exception:  # noqa: BLE001
                v = None
            if isinstance(v, str):
                txt = v
                break
        proto = getattr(n, "proto", None)
        if proto is not None and proto.DESCRIPTOR.name in ("ArrowVegaLiteChart", "VegaLiteChart"):
            kind = "Chart"
        out.append((kind, txt))
        ch = getattr(n, "children", {}) or {}
        for k in sorted(ch):
            rec(ch[k])
    rec(at._tree)
    return out


@pytest.mark.parametrize("public", [True, False])
def test_a7_home_caveat_renders_above_the_hero(monkeypatch, root, public):  # noqa: F811
    for f in (NOTLD, OWNED, CLIP):
        _qualify(root, f)
    at = run_app(monkeypatch, root, public, "Home")
    seq = _order(at)
    cav = [i for i, (k, t) in enumerate(seq) if k == "Warning" and t == core.CAPTION_PREDICTIONS]
    chart = [i for i, (k, _) in enumerate(seq) if k == "Chart"]
    snip = [i for i, (_, t) in enumerate(seq) if t.startswith('<div class="cgt-viz')]
    assert cav and chart and snip, (cav, chart, snip)
    assert cav[0] < chart[0] and cav[0] < snip[0]
    assert len(_specs(at)) == 1 and sum(t.startswith('<div class="cgt-viz') for t in _texts(at)) == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
