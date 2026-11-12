"""The static demo (app/static/, published on GitHub Pages): its generated data equals the guarded token table, the build refuses any film
that is not public domain, every caveat is the demo's own constant byte for byte, no emoji, chart colours only from the pinned palettes,
and no request leaves the site except the pinned chart library and the web fonts. Data checks need the public demo data (demo_data/ in the
published repository, data/release/demo_bundle/ locally, or CGT_DEMO_ROOT) and are skipped without it."""
import html
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "app" / "static"
DATA = STATIC / "data"
pytest.importorskip("altair")  # the chart code needs altair (absent from the CPU-only environment: this file runs with the demo's)
sys.path.insert(0, str(REPO / "app" / "demo"))
sys.path.insert(0, str(REPO / "src"))

import core
import style
import viz

from cgt.tokens import filters
from cgt.tokens.format import people_token


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load("build_static_app", "scripts/build_static_app.py")
assemble = _load("assemble_pages_site", "scripts/assemble_pages_site.py")

CDN = {
    "https://cdn.jsdelivr.net/npm/vega@6.2.0/build/vega.min.js",
    "https://cdn.jsdelivr.net/npm/vega-lite@6.4.1/build/vega-lite.min.js",
    "https://cdn.jsdelivr.net/npm/vega-embed@7.1.0/build/vega-embed.min.js",
}
FONTS = {"https://fonts.googleapis.com", "https://fonts.gstatic.com",
         ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400"
          "&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap")}
LINKS = {"https://github.com/jahoi12345/cinematic-grammar-tokenizer"}  # navigation links only (an <a href>, never fetched)


def data_root() -> Path | None:
    env = os.environ.get("CGT_DEMO_ROOT")
    for cand in ([Path(env)] if env else []) + [REPO / "demo_data", REPO / "data/release/demo_bundle"]:
        if (cand / "configs/films.yaml").is_file() and list((cand / "data/processed").glob("*_tokens_v1.parquet")):
            return cand
    return None


ROOT = data_root()
needs_data = pytest.mark.skipif(ROOT is None, reason="no public demo data (demo_data/ or data/release/demo_bundle/)")


def c_ok(fj: dict) -> bool:
    """No fingerprint title is left in the shipped HTML (the page adds them)."""
    return all('class="cgt-vfr" title=' not in fj["html"][t]["fingerprint"] for t in ("light", "dark"))


def site() -> dict:
    return json.loads((DATA / "site.json").read_text(encoding="utf-8"))


def films() -> list[str]:
    return [f["id"] for f in site()["films"]]


def jload(film: str, name: str) -> dict:
    return json.loads((DATA / film / name).read_text(encoding="utf-8"))


# ---- public only ------------------------------------------------------------------------------------------------------------------------------------

def test_site_is_a_public_build_of_public_domain_films_only():
    s = site()
    assert s["public"] is True and films()
    reg = __import__("yaml").safe_load((REPO / "configs/films.yaml").read_text())
    public = {f["id"] for f in reg["films"] if f.get("public_domain") is True}
    assert set(films()) <= public
    on_disk = {p.name for p in DATA.iterdir() if p.is_dir()}
    assert on_disk == set(films())  # no data folder of any other film


def _fake_root(tmp_path: Path, film: str, public_domain: bool) -> Path:
    root = tmp_path / "root"
    (root / "configs").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    (root / "configs/films.yaml").write_text(
        f"films:\n  - id: {film}\n    title: Synthetic\n    year: 2000\n    public_domain: {'true' if public_domain else 'false'}\n")
    pd.DataFrame({"film_id": [film], "shot_id": [0], "start_s": [0.0], "end_s": [2.0], "len_s": [2.0], "token_row": ["[size=CU]"]}).to_parquet(
        root / f"data/processed/{film}_tokens_v1.parquet")
    return root


@pytest.mark.parametrize("public_domain", [False, True])
def test_build_refuses_a_film_that_is_not_public_in_both_registries(tmp_path, public_domain):
    """Not public under the data root -> refused; public there but unknown to this checkout's registry -> refused too. Nothing is written."""
    root = _fake_root(tmp_path, "synthetic_private_film", public_domain)
    out = tmp_path / "out"
    with pytest.raises(build.Refused):
        build.build(root, out)
    assert not out.exists()
    assert build.main(["--root", str(root), "--out", str(out)]) == 3
    assert not out.exists()


@needs_data
def test_build_refuses_an_explicit_non_public_film(tmp_path):
    out = tmp_path / "out"
    assert build.main(["--root", str(ROOT), "--out", str(out), "--film", "not_a_public_film"]) == 3
    assert not out.exists()


def test_app_renders_film_pages_only_for_films_of_site_json():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'r.page === "overview" && ids.indexOf(film) >= 0' in js and 'r.page === "explorer" && ids.indexOf(film) >= 0' in js
    assert re.findall(r"fetch\(([^,)]+)", js) == ["DATA + path"]  # the only fetch: files under data/


# ---- the data equals the table ----------------------------------------------------------------------------------------------------------------------

@needs_data
def test_shots_json_equals_the_guarded_table():
    for film in films():
        df, _ = core.load_table(ROOT, film, True)
        sj = jload(film, "shots.json")
        c = sj["columns"]
        assert sj["n"] == len(df) == len(c["shot_id"])
        assert c["shot_id"] == df["shot_id"].astype(int).tolist()
        assert c["token_row"] == df["token_row"].tolist()
        assert c["len_s"] == df["len_s"].astype(float).tolist()
        assert c["start_min_txt"] == [f"{x:.1f}" for x in df["start_min"]] and c["len_txt"] == [f"{x:.1f}" for x in df["len_s"]]
        assert c["detail"] == [[[a, f"{r[f'{a}_p1']:.2f}", f"{r[f'{a}_margin']:.2f}"] for a in build.STATIC] for r in df.to_dict("records")]
        assert c["people_tok"] == [people_token(v) for v in df["people"]]
        assert c["n_kf_disagree"] == df["n_kf_disagree"].astype(int).tolist()
        for slot, col in core.slot_options(df).items():
            assert c[col] == [None if pd.isna(v) else str(v) for v in df[col]], slot
        assert c["move_atoms"] == [sorted(set(map(str, a))) for a in df["move_atoms"]]
        for k in ("kf_start", "kf_mid", "kf_end"):
            assert c[k] == [Path(r).name if core.frame_path(ROOT, film, r, True) else None for r in df[k]]
        o = sj["options"]
        assert [(s["label"], s["column"], s["values"]) for s in o["slots"]] == [
            (slot, col, sorted(df[col].dropna().unique().tolist())) for slot, col in core.slot_options(df).items()]
        assert o["people"] == ["0", "1", "2", "3", "4+"]
        assert o["move"] == sorted({a for xs in df["move_atoms"] if xs is not None for a in xs})
        assert o["len_range"] == [float(df.len_s.min()), float(df.len_s.max())]
        assert o["keyframe_modes"] == list(filters.KEYFRAME_MODES)
        assert sj["page_size"] == 8


@needs_data
def test_similar_shots_equal_core_similar_table():
    for film in films():
        df, _ = core.load_table(ROOT, film, True)
        top = core.similar_table(ROOT, film, True, df)
        sj = jload(film, "shots.json")
        ids = sj["columns"]["shot_id"]
        got = {(ids[i], r[0], r[1], r[2], r[3], r[4]) for i, v in enumerate(sj["similar"]) for r in v}
        assert got == {(int(s.shot_id), int(s.rank), int(s.other_shot_id), f"{s.other_start_s / 60:.1f}", f"{s.gap_min:.1f}", f"{s.score:.3f}")
                       for s in top.itertuples()}
        assert sj["similar_caption"].startswith(core.CAPTION_SIMILAR)


@needs_data
def test_film_json_stats_and_captions():
    for film in films():
        df, version = core.load_table(ROOT, film, True)
        fj = jload(film, "film.json")
        s = core.length_stats(df)
        assert fj["stats"]["shots"] == s["shots"] == len(df) and fj["version"] == version
        assert fj["tiles"][0] == ["shots", f"{len(df):,}"]
        assert fj["movement_caption"] == core.movement_caption(film, df, ROOT)
        assert fj["label"] == core.film_label(ROOT, film)
        vdf = viz.prepare(df)  # the fingerprint titles the page rebuilds from fingerprint_tips equal viz.fingerprint_html's
        titles = re.findall(r'class="cgt-vfr" title="([^"]*)"', viz.fingerprint_html(vdf, "light"))
        assert [html.escape(build.fingerprint_title(t, len(df))) for t in fj["fingerprint_tips"]] == titles
        js = (STATIC / "app.js").read_text(encoding="utf-8")
        assert '" = " + tip[1] + ": " + tip[2] + " of " + fj.stats.shots + " shots (" + tip[3] + "%)"' in js
        assert c_ok(fj)
        entry = next(f for f in site()["films"] if f["id"] == film)
        assert entry["shots"] == len(df) and entry["label"] == fj["label"]


@needs_data
def test_filters_in_the_page_match_apply_filters_on_a_sample():
    """The page's filter rule (all slots ANDed, values ORed within a slot, people buckets, any move atom, keyframe mode, inclusive length
    range) recomputed in Python from shots.json equals cgt.tokens.filters.apply_filters on the table."""
    for film in films():
        df, _ = core.load_table(ROOT, film, True)
        c = jload(film, "shots.json")["columns"]

        def page_rule(sel, rng, people, kf, move, c=c):
            out = []
            for i in range(len(c["shot_id"])):
                if any(v and c[col][i] not in v for col, v in sel.items()):
                    continue
                if rng and not (rng[0] <= c["len_s"][i] <= rng[1]):
                    continue
                if people and c["people_tok"][i] not in people:
                    continue
                if move and not set(c["move_atoms"][i] or []) & set(move):
                    continue
                if kf == "disagree" and not c["n_kf_disagree"][i] > 0 or kf == "agree" and c["n_kf_disagree"][i] != 0:
                    continue
                out.append(c["shot_id"][i])
            return out
        cases = [({"size": ["CU", "MCU"], "angle": ["low"]}, None, ["1"], "any", []), ({}, (2.0, 10.0), [], "agree", ["push-in", "static"]),
                 ({"sound": ["dialogue"], "key": ["low-key"]}, None, ["2", "4+"], "disagree", []), ({}, None, [], "any", [])]
        for sel, rng, people, kf, move in cases:
            ref = filters.apply_filters(df, sel, rng, people, kf, move)["shot_id"].astype(int).tolist()
            assert page_rule(sel, rng, people, kf, move) == ref


@needs_data
def test_rebuild_is_identical(tmp_path):
    out = tmp_path / "data"
    res = build.build(ROOT, out)
    committed = sorted(str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file())
    assert sorted(res["files"]) == committed
    for rel in committed:
        if rel.endswith(".png"):
            from PIL import Image
            with Image.open(out / rel) as a, Image.open(DATA / rel) as b:
                assert a.size == b.size and a.tobytes() == b.tobytes(), rel
        else:
            assert (out / rel).read_bytes() == (DATA / rel).read_bytes(), rel


# ---- caveats, emoji, palettes, requests --------------------------------------------------------------------------------------------------------------

CAVEAT_KEYS = {"predictions": core.CAPTION_PREDICTIONS, "model": core.CAPTION_MODEL, "intervals": core.CAPTION_INTERVALS, "shapes": core.CAPTION_SHAPES,
               "similar": core.CAPTION_SIMILAR, "movement_generic": core.CAPTION_MOVEMENT_GENERIC,
               "explorer": core.CAPTION_PREDICTIONS + " A * marks an attribute whose start/mid/end keyframes give different predictions."}


def test_caveats_are_the_demo_constants_byte_for_byte():
    text = site()["text"]
    for k, v in CAVEAT_KEYS.items():
        assert text[k] == v, k
    assert site()["slot_help"] == [list(x) for x in core.SLOT_HELP] and site()["marker_help"] == [list(x) for x in core.MARKER_HELP]


def test_every_caveat_is_drawn_visible_where_the_app_shows_it():
    """Each caveat goes through caveat() (a visible callout) or caption() (visible text); never inside a collapsed element."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for call in ("caveat(t.predictions)", "caveat(t.explorer)", "caveat(t.shapes)", "caveat(fj.movement_caption)", "caption(t.intervals)",
                 "caption(t.shapes)", "caption(t.predictions)", "caption(site.text.model)", 'h("p", { text: t.model })'):
        assert call in js, call
    for k in ("predictions", "shapes", "intervals", "movement_generic", "similar"):  # the About page's caveat list
        assert f"t.{k}]" in js, k
    body = js[js.index("function caveat("):js.index("function info(")]
    assert "details" not in body and "hidden" not in body and 'text: text' in body


def test_no_emoji_in_the_static_app():
    for p in sorted(STATIC.rglob("*")):
        if p.is_file() and p.suffix in (".html", ".css", ".js", ".json"):
            t = p.read_text(encoding="utf-8")
            t2 = re.sub(r"&#(x[0-9a-fA-F]+|\d+);", lambda m: chr(int(m.group(1)[1:], 16) if m.group(1)[0] == "x" else int(m.group(1))), t)
            assert not style.EMOJI.search(t2), p.name


def _colors(obj, out: set):
    if isinstance(obj, dict):
        for v in obj.values():
            _colors(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _colors(v, out)
    elif isinstance(obj, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", obj):
        out.add(obj.upper())


def test_chart_specs_are_valid_and_use_only_the_pinned_palettes():
    import altair as alt
    allowed = {c.upper() for c in viz.palette_sets()} | {c.upper() for c in style.LABEL_ON.values()}
    for film in films():
        for group in ("charts.json", "charts_home.json"):
            p = DATA / film / group
            if not p.exists():
                continue
            ch = json.loads(p.read_text(encoding="utf-8"))
            assert set(ch["specs"]) == {"light", "dark"} and ch["specs"]["light"].keys() == ch["specs"]["dark"].keys()
            for t in ("light", "dark"):
                for name, spec in ch["specs"][t].items():
                    assert spec["$schema"].endswith(f"/vega-lite/{alt.SCHEMA_VERSION}.json"), name
                    assert set(spec["usedDatasets"]) <= set(ch["datasets"]), name
                    assert '"url"' not in json.dumps(spec), name  # data inline only: no chart fetches anything
                    found: set = set()
                    _colors(spec, found)
                    assert found <= allowed, (name, found - allowed)
            found = set()
            _colors(ch["datasets"], found)
            assert found <= allowed, found - allowed
    home = jload(films()[0], "charts_home.json")
    assert list(home["specs"]["light"]) == ["barcode"]


def _urls(text: str) -> set[str]:
    return set(re.findall(r"https?://[^\s\"'<>)]+", text))


def test_no_external_request_except_the_pinned_chart_library_and_fonts():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    scripts = set(re.findall(r'<script[^>]*\ssrc="([^"]+)"', html))
    assert scripts == CDN | {"app.js"}
    for tag in re.findall(r"<script[^>]*\ssrc=\"https[^>]*>", html):  # pinned and integrity-checked
        assert 'integrity="sha256-' in tag and 'crossorigin="anonymous"' in tag
    links = set(re.findall(r'<link[^>]*\shref="([^"]+)"', html))
    assert links == (FONTS | {"app.css"})
    assert _urls(html) <= CDN | FONTS | LINKS
    assert _urls((STATIC / "app.js").read_text(encoding="utf-8")) <= LINKS
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert not _urls(css) and "url(" not in css and "@import" not in css
    for p in DATA.rglob("*.json"):
        bad = {u for u in _urls(p.read_text(encoding="utf-8")) if not u.startswith("https://vega.github.io/schema/vega-lite/")}
        assert not bad, (p.name, bad)


# ---- token chips: the page's port equals style.chips_html ---------------------------------------------------------------------------------------------

def _js_functions(names: list[str]) -> str:
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    out = [re.search(r"var MODEL_SLOTS = .*?;\n\s*var HEURISTIC_SLOTS = .*?;", js, re.DOTALL).group(0)]
    for n in names:
        start = js.index(f"function {n}(")
        depth, i = 0, js.index("{", start)
        while True:
            depth += {"{": 1, "}": -1}.get(js[i], 0)
            i += 1
            if depth == 0:
                break
        out.append(js[start:i])
    return "\n".join(out)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_js_chips_equal_python_chips(tmp_path):
    rows = ["[size=MCU | lens=medium* | angle=? | framing=single | light=artificial*/hard* | comp=n/a | key=mid-key | move=pan-left+static]",
            "[a=b]", "not a row", "[size=CU | broken]", "[size=<b>&'\"]", "[ size = CU ]", ""]
    if ROOT is not None:
        rows += core.load_table(ROOT, films()[0], True)[0]["token_row"].tolist()
    src = _js_functions(["esc", "parseRow", "chipsHtml"])
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    prog = src + "\nconst rows = JSON.parse(require('fs').readFileSync(process.argv[2], 'utf8'));\n" \
                 "process.stdout.write(JSON.stringify(rows.map(r => [chipsHtml(r, 'md'), chipsHtml(r, 'lg')])));\n"
    (tmp_path / "run.js").write_text(prog)
    res = subprocess.run(["node", str(tmp_path / "run.js"), str(tmp_path / "rows.json")], capture_output=True, text=True, timeout=60, check=True)
    got = json.loads(res.stdout)
    assert got == [[style.chips_html(r, "md"), style.chips_html(r, "lg")] for r in rows]


# ---- site assembly -----------------------------------------------------------------------------------------------------------------------------------

def test_assembly_rejects_unexpected_keyframe_names():
    for bad in ("../00001_mid.jpg", "x.jpg", "00001_mid.png", "/abs/00001_mid.jpg"):
        with pytest.raises(assemble.AssemblyError):
            assemble.keyframe_names({"columns": {"kf_mid": [bad]}})
    assert assemble.keyframe_names({"columns": {"kf_mid": ["00001_mid.jpg", None], "kf_end": ["00001_end.jpg"]}}) == ["00001_end.jpg", "00001_mid.jpg"]
    with pytest.raises(assemble.AssemblyError):
        assemble.paper_with_link("<html>no body</html>")
    assert 'href="app/" target="_top"' in assemble.paper_with_link("<html><body class='x'><p>paper</p></body></html>")


@needs_data
def test_assembly_builds_the_site(tmp_path):
    paper, pdf = tmp_path / "paper.html", tmp_path / "paper.pdf"
    paper.write_text("<!doctype html><html><head></head><body><h1>Paper</h1></body></html>")
    pdf.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "_site"
    res = assemble.assemble(out, paper, pdf, STATIC, ROOT)
    assert res["films"] == films()
    assert (out / "index.html").read_text().count('href="app/"') == 1 and (out / "paper_public.pdf").read_bytes() == b"%PDF-1.4\n"
    for name in ("index.html", "app.css", "app.js", "data/site.json"):
        assert (out / "app" / name).read_bytes() == (STATIC / name).read_bytes()
    for film in films():
        want = assemble.keyframe_names(jload(film, "shots.json"))
        assert sorted(p.name for p in (out / "app/keyframes" / film).iterdir()) == want == sorted(want)
        assert res["keyframes"][film] == len(want)
    assert sorted(p.name for p in (out / "app/keyframes").iterdir()) == sorted(films())


def test_pages_workflow_assembles_and_uploads_the_site():
    import yaml
    cands = [REPO / ".github/workflows/pages.yml", REPO / "scripts/release/public_overlay/.github/workflows/pages.yml"]
    wf = yaml.safe_load(next(p for p in cands if p.is_file()).read_text())
    steps = [s for j in wf["jobs"].values() for s in j["steps"]]
    runs = [s["run"] for s in steps if "run" in s]
    assert runs == [("python3 scripts/assemble_pages_site.py --out _site --paper site/index.html --pdf site/paper_public.pdf --app app/static "
                     "--data-root demo_data")]
    assert [s["with"]["path"] for s in steps if "upload-pages-artifact" in s.get("uses", "")] == ["_site"]
    assert any("deploy-pages" in s.get("uses", "") for s in steps)
    assert not any("pip" in r or "install" in r for r in runs)  # no package installed: the data files are committed
