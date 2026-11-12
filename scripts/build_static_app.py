"""Build the data of the static demo (app/static/: plain HTML, CSS and JavaScript, no server) from the guarded public demo data.

Usage: PYTHONPATH=src python scripts/build_static_app.py [--root DATA_ROOT] [--out app/static/data] [--film ID ...]

DATA_ROOT holds configs/films.yaml, data/processed/<film>_tokens_v1.parquet and data/interim/<film>/keyframes/ (the public repository's
demo_data/ folder; default: env CGT_DEMO_ROOT, else demo_data/ of this checkout, else data/release/demo_bundle/). Public mode only: every
film is read through app/demo/core.py with public=True, and a film is written only if it is public domain in the registry under DATA_ROOT
AND in this checkout's configs/films.yaml and is not local only. Anything else is refused (exit 3) before a file is written.

Writes (and only these): OUT/site.json, OUT/<film>/film.json, OUT/<film>/shots.json, OUT/<film>/charts.json, OUT/<film>/charts_home.json,
OUT/<film>/strip.png.
Keyframes are NOT copied here: the site assembly (scripts/assemble_pages_site.py) copies the ones shots.json names into the published site.
Chart specs are the Vega-Lite specs of app/demo/viz.py (and the visual-arc charts of the Streamlit app, rebuilt here without streamlit),
one per theme; their data sets are stored once per film and shared by both themes."""
import argparse
import hashlib
import html
import io
import json
import os
import re
import shutil
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "app" / "demo", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import core
import style
import viz

from cgt.tokens import filters
from cgt.tokens.format import PEOPLE_MAX, people_token

FORMAT_VERSION = 1
THEMES = ("light", "dark")
MODES = ("equal20", "min5")
STATIC = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]  # = app.STATIC
PANELS = [("Shot rhythm (heuristic)", ["cuts_per_min", "median_len_s"]),  # = app.PANELS (the visual-arc panels)
          ("Scale (model prediction)", ["close_share", "mean_size_ordinal"]),
          ("Tone and light (key: heuristic; light: model prediction)", ["low_key_share", "high_contrast_share", "daylight_share"]),
          ("People and sound (heuristics)", ["people_mean", "multi_person_share", "dialogue_share"]),
          ("Angle (model; no eye-level class) and keyframe stability", ["low_angle_share", "keyframes_all_agree_share"])]
MOVE_A = ["move_static_share", "move_static_alone_share", "move_contradictory_share"]
LEGEND_LONG_LABEL = 28
KF_COLS = ("kf_start", "kf_mid", "kf_end")
EXPLORER_CAVEAT_SUFFIX = " A * marks an attribute whose start/mid/end keyframes give different predictions."
COMPARE_INFO = ("The side-by-side view needs two listed films with a v1 token table. Public mode lists only films marked public domain in the committed film "
                "registry, and only {n} such film is available here ({have}), so there is nothing to compare it with. Its within-film arc is on the Film "
                "overview page.")
SIMILAR_NOTE = " Palette centroids are not available under this data root: the palette slot is an exact id match here."


class Refused(Exception):
    pass


def default_root() -> Path:
    env = os.environ.get("CGT_DEMO_ROOT")
    if env:
        return Path(env)
    for cand in (ROOT / "demo_data", ROOT / "data/release/demo_bundle"):
        if (cand / "configs/films.yaml").is_file():
            return cand
    return ROOT / "demo_data"


def public_films(root: Path, wanted: list[str] | None = None) -> list[str]:
    """Films to build: listed in public mode under ROOT, public domain in both registries, not local only. Fail closed."""
    allowed = core.public_ids(root) & core.public_ids(core.CODE_ROOT)
    listed = core.listed_films(root, public=True)
    films = [f for f in listed if f in allowed and not core.is_local_only(root, f)]
    if wanted:
        bad = [f for f in wanted if f not in films]
        if bad:
            raise Refused(f"not a public-domain film with a table under the data root: {', '.join(bad)}")
        films = [f for f in films if f in wanted]
    if not films:
        raise Refused("no public-domain film with a token table under the data root")
    return films


# ---- charts (Vega-Lite specs) ------------------------------------------------------------------------------------------------------------------------

def legend(columns: int) -> alt.Legend:  # = app.legend
    return alt.Legend(orient="bottom", columns=columns, labelLimit=360, symbolOpacity=1, symbolType="stroke", symbolStrokeWidth=3)


def legend_columns(labels: list[str]) -> int:  # = app.legend_columns
    return 2 if len(labels) <= 3 and max((len(x) for x in labels), default=0) <= LEGEND_LONG_LABEL else 1


def series_chart(tbl: pd.DataFrame, names: list[str], x: str, x_title: str, theme: str):
    """The visual-arc chart of the Streamlit app (app.series_chart), styled with the v3.2 chart look for one theme."""
    d = tbl[tbl.series.isin(names)].copy()
    d["label"] = d["series"].map(core.series_label)
    d = d[["label", x, "value", "ci_lo", "ci_hi", "n_shots", "flag_sparse"]]
    scale = alt.Scale(range=style.CHART_COLORS)
    cols = legend_columns([core.series_label(n) for n in names])
    base = alt.Chart(d).encode(x=alt.X(f"{x}:Q", title=x_title))
    line = base.mark_line(point=True).encode(y=alt.Y("value:Q", title=None), color=alt.Color("label:N", title=None, scale=scale, legend=legend(cols)), tooltip=[
        alt.Tooltip("label:N", title="series"), alt.Tooltip("value:Q", title="value", format=".3f"), alt.Tooltip("ci_lo:Q", title="interval low", format=".3f"),
        alt.Tooltip("ci_hi:Q", title="interval high", format=".3f"), alt.Tooltip("n_shots:Q", title="shots in window"),
        alt.Tooltip("flag_sparse:N", title="fewer than 15 shots")])
    band = base.mark_area(opacity=0.15).encode(y="ci_lo:Q", y2="ci_hi:Q", color=alt.Color("label:N", title=None, scale=scale, legend=legend(cols))).transform_filter(
        "isValid(datum.ci_lo)")
    sparse = alt.Chart(d[d.flag_sparse]).mark_point(shape="cross", size=60, color=style.VIZ_NEUTRAL).encode(x=f"{x}:Q", y="value:Q")
    rows = -(-len(names) // cols)
    layers = band + line + sparse if bool(d.flag_sparse.any()) else band + line  # no empty layer (Vega warns on an empty extent)
    return viz.themed(layers.properties(height=220 + 20 * rows, width="container"), theme)


def len_order(bins) -> list[str]:  # = the sort of app.length_bin_chart
    def lower(b: str) -> float:
        t = str(b).rstrip("s")
        try:
            return -1.0 if t.startswith("<") else float(t.split("-")[0].rstrip("+"))
        except ValueError:
            return float("inf")
    return sorted(bins, key=lower)


def length_bin_chart(df: pd.DataFrame, theme: str):  # = app.length_bin_chart (accent of the theme)
    d = df["len_tok"].value_counts(normalize=True).rename_axis("bin").reset_index(name="share")
    order = len_order(d["bin"])
    return viz.themed(alt.Chart(d).mark_bar(color=style.ACCENT[theme]).encode(
        x=alt.X("bin:N", sort=order, title="shot length bin (seconds)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("share:Q", title="share of shots", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("bin:N", title="length bin"), alt.Tooltip("share:Q", title="share of shots", format=".1%")]).properties(
        height=220, width="container"), theme)


def _clean(obj):
    """JSON-safe copy: NaN/inf become null (Vega reads null as missing)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def split_datasets(spec: dict, store: dict) -> dict:
    """Move a spec's top-level `datasets` into the shared store (same name = same content: Altair names a data set by its hash)."""
    ds = spec.pop("datasets", {}) or {}
    for name, values in ds.items():
        v = _clean(values)
        if name in store and store[name] != v:
            raise ValueError(f"data set {name} differs between charts")
        store[name] = v
    spec["usedDatasets"] = sorted(ds)
    return spec


def film_charts(film: str, df: pd.DataFrame, vdf: pd.DataFrame, hero_id: int | None, arcs: dict, moves: dict) -> dict:
    store: dict = {}
    out = {t: {} for t in THEMES}
    for t in THEMES:
        charts = {"length": length_bin_chart(df, t), "barcode": viz.barcode_chart(vdf, t, mark_shot=hero_id), "page": viz.page_chart(vdf, t),
                  "size_lane": viz.size_lane_chart(vdf, t), "rhythm": viz.rhythm_chart(vdf, t), "grammar": viz.transition_chart(vdf, t)}
        top, bottom = viz.pace_charts(vdf, t)
        charts["pace_top"], charts["pace_bottom"] = top, bottom
        for mode in MODES:
            for i, (_, names) in enumerate(PANELS):
                charts[f"arc_{mode}_{i}"] = series_chart(arcs[mode], names, "window_mid_min", "minutes into the film (window centre)", t)
            if moves:
                charts[f"move_{mode}_0"] = series_chart(moves[mode], MOVE_A, "window_mid_min", "minutes (window centre)", t)
                charts[f"move_{mode}_1"] = series_chart(moves[mode], [f"move_{g}_share" for g in core.arc.MOVE_GROUPS], "window_mid_min",
                                                        "minutes (window centre)", t)
        for name, ch in charts.items():
            out[t][name] = _clean(split_datasets(ch.to_dict(), store))
    return {"format": FORMAT_VERSION, "film": film, "datasets": store, "specs": out}


def split_groups(charts: dict) -> dict[str, dict]:
    """Two files per film: the Home hero (barcode only, loaded on Home) and every other chart (loaded on the Film overview)."""
    out = {}
    for group, keep in (("charts_home", lambda n: n in HOME_CHARTS), ("charts", lambda n: n not in HOME_CHARTS)):
        specs = {t: {n: s for n, s in charts["specs"][t].items() if keep(n)} for t in THEMES}
        used = sorted({d for t in THEMES for s in specs[t].values() for d in s["usedDatasets"]})
        out[group] = {"format": FORMAT_VERSION, "film": charts["film"], "datasets": {d: charts["datasets"][d] for d in used}, "specs": specs}
    return out


HOME_CHARTS = ("barcode",)


# ---- tables -----------------------------------------------------------------------------------------------------------------------------------------

def hero_row(root: Path, film: str, df: pd.DataFrame):
    """= app.hero_shot for one film: the first shot (by id) whose row parses, is CU/MCU of one person with no `*`, and has a mid keyframe."""
    if "token_row" not in df.columns or core.is_local_only(root, film):
        return None
    for r in df.sort_values("shot_id").to_dict("records"):
        slots = {t["slot"]: t for t in style.parse_row(r.get("token_row"))}
        if (slots.get("size", {}).get("value") in ("CU", "MCU") and slots.get("people", {}).get("value") == "1"
                and not any(t["low"] for t in slots.values()) and core.frame_path(root, film, r.get("kf_mid"), True) is not None):
            return r
    return None


def kf_name(root: Path, film: str, rel) -> str | None:
    p = core.frame_path(root, film, rel, True)
    return p.name if p is not None else None


def _plain(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def shots_json(root: Path, film: str, df: pd.DataFrame, top: pd.DataFrame) -> dict:
    """Columnar shot table: what the explorer filters on and shows. Display strings are formatted here (same format codes as the app)."""
    opts = core.slot_options(df)
    has_move = core.has_movement(df)
    cols: dict[str, list] = {"shot_id": [int(x) for x in df["shot_id"]], "start_s": [float(x) for x in df["start_s"]],
                             "len_s": [float(x) for x in df["len_s"]], "token_row": [str(x) if isinstance(x, str) else "" for x in df["token_row"]]}
    for c in opts.values():
        cols[c] = [None if pd.isna(v) else str(v) for v in df[c]]
    cols["people_tok"] = [people_token(v) for v in df["people"]]
    cols["move_atoms"] = [sorted(set(map(str, a))) if a is not None and not isinstance(a, float) else None for a in df["move_atoms"]] if has_move else None
    cols["n_kf_disagree"] = [int(x) for x in df["n_kf_disagree"]] if "n_kf_disagree" in df.columns else None
    for k in KF_COLS:
        cols[k] = [kf_name(root, film, r) for r in df[k]]
    # numbers are written as preformatted strings (the app's format codes) and the page adds the words: the data files carry no prose
    # sentence with a number in it (the release scanner reads such sentences as possible counts of other films)
    cols["start_min_txt"] = [f"{x:.1f}" for x in df["start_min"]]
    cols["len_txt"] = [f"{x:.1f}" for x in df["len_s"]]
    if all(f"{a}_p1" in df.columns for a in STATIC):
        cols["detail"] = [[[a, f"{r[f'{a}_p1']:.2f}", f"{r[f'{a}_margin']:.2f}"] for a in STATIC] for r in df.to_dict("records")]
    pos = {int(x): i for i, x in enumerate(df["shot_id"])}
    sim: list[list] = [[] for _ in range(len(df))]
    for s in top.sort_values(["shot_id", "rank"]).itertuples():
        sim[pos[int(s.shot_id)]].append([int(s.rank), int(s.other_shot_id), f"{s.other_start_s / 60:.1f}", f"{s.gap_min:.1f}", f"{s.score:.3f}"])
    mode = top.attrs.get("palette_mode", "centroid")
    lo, hi = float(df.len_s.min()), float(df.len_s.max())
    options = {"slots": [{"label": slot, "column": col, "values": sorted(df[col].dropna().unique().tolist())} for slot, col in opts.items()],
               "people": [str(i) for i in range(PEOPLE_MAX)] + [f"{PEOPLE_MAX}+"],
               "move": sorted({a for xs in df["move_atoms"] if xs is not None for a in xs}) if has_move else None,
               "len_range": [lo, hi], "keyframe_modes": list(filters.KEYFRAME_MODES) if "n_kf_disagree" in df.columns else None}
    return {"format": FORMAT_VERSION, "film": film, "n": len(df), "columns": {k: v for k, v in cols.items() if v is not None},
            "options": _clean(options), "similar": sim, "similar_caption": core.CAPTION_SIMILAR + ("" if mode == "centroid" else SIMILAR_NOTE),
            "page_size": 8}


def film_json(root: Path, film: str, label: str, df: pd.DataFrame, version: str, vdf: pd.DataFrame, hero, arcs: dict, moves: dict) -> dict:
    s = core.length_stats(df)
    out = {"format": FORMAT_VERSION, "film": film, "label": label, "version": version,
           "stats": {k: _plain(v) for k, v in s.items()},
           "tiles": [["shots", f"{s['shots']:,}"], ["runtime (min)", f"{s['running_min']:.1f}"], ["median shot (s)", f"{s['median_s']:.1f}"],
                     ["mean shot (s)", f"{s['mean_s']:.1f}"]],
           "length_caption": (f"Shot length (heuristic cut table): quartiles {s['q25_s']:.1f} / {s['median_s']:.1f} / {s['q75_s']:.1f} s, "
                              f"min {s['min_s']:.2f} s, max {s['max_s']:.1f} s. Row version {version}."),
           "has_arc": version == "v1" and core.has_arc_columns(df), "has_movement": core.has_movement(df)}
    if out["has_arc"]:
        out["kf_caption"] = (f"All seven predicted attributes agree across the start, mid and end keyframes in {100 * df['kf_all_agree'].mean():.1f}% of shots "
                             "(read before any series: low agreement means the labels depend on which frame is scored).")
        out["panels"] = [t for t, _ in PANELS]
        out["window_counts"] = {}
        for mode, tbl in arcs.items():
            counts = tbl.drop_duplicates("window")[["window", "window_start_min", "n_shots", "flag_sparse"]]
            out["window_counts"][mode] = "Shots per window: " + ", ".join(
                f"{int(r.window_start_min)}' n={r.n_shots}{' (flagged)' if r.flag_sparse else ''}" for r in counts.itertuples())
        out["html"] = {t: viz.overview_html(root, film, True, vdf, t) for t in THEMES}
        out["fingerprint_tips"] = tips = fingerprint_tips(vdf)
        for t in THEMES:  # the fingerprint's hover titles are rebuilt by the page from fingerprint_tips (same words, same numbers)
            want = [html.escape(fingerprint_title(x, len(df))) for x in tips]
            if TITLE_RE.findall(out["html"][t]["fingerprint"]) != [f' title="{w}"' for w in want]:
                raise ValueError("fingerprint titles and tips differ")
            out["html"][t]["fingerprint"] = TITLE_RE.sub("", out["html"][t]["fingerprint"])
        out["strip"] = "strip.png"
    if out["has_movement"]:
        out["movement_caption"] = core.movement_caption(film, df, root)
    if hero is not None:
        hid = int(hero["shot_id"])
        out["hero"] = {"shot_id": hid, "kf_mid": kf_name(root, film, hero["kf_mid"]), "token_row": hero["token_row"],
                       "caption": f"{label}, shot {hid} at {hero['start_min']:.1f} min, mid keyframe",
                       "chips": style.chips_html(hero["token_row"], "lg"), "home_html": {t: viz.home_html(vdf, label, hid, t) for t in THEMES}}
    return out


TITLE_RE = re.compile(r' title="[^"]*"(?=><span class="cgt-vfv">)')


def fingerprint_tips(vdf: pd.DataFrame) -> list[list]:
    """[slot, value, n, share %, disagree % or None] per fingerprint bar, in the order of viz.fingerprint_html (its title text, as data)."""
    fp = viz.fingerprint(vdf)
    out = []
    for slot, g in fp.groupby("slot", sort=False):
        for r in g.sort_values("order").itertuples():
            out.append([slot, str(r.value), int(r.n), f"{100 * r.share:.1f}",
                        f"{100 * r.share_low / r.share:.0f}" if r.share_low and r.share else None])
    return out


def fingerprint_title(tip: list, total: int) -> str:  # = the title text of viz.fingerprint_html
    slot, value, n, pct, dis = tip
    return f"{slot} = {value}: {n} of {total} shots ({pct}%)" + (f"; keyframes disagree on {dis}% of them" if dis else "")


def strip_png(root: Path, film: str, vdf: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    viz.contact_strip(root, film, True, vdf).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---- site --------------------------------------------------------------------------------------------------------------------------------------------

def site_json(root: Path, films: list[dict]) -> dict:
    labels = [f["label"] for f in films]
    eligible = [f["label"] for f in films if f["has_arc"]]
    lib_paths = list(core.PAPER_PUBLIC) + [core.PAPER_PUBLIC_PDF]
    return {
        "format": FORMAT_VERSION, "public": True, "films": films,
        "text": {"predictions": core.CAPTION_PREDICTIONS, "model": core.CAPTION_MODEL, "intervals": core.CAPTION_INTERVALS, "shapes": core.CAPTION_SHAPES,
                 "similar": core.CAPTION_SIMILAR, "movement_generic": core.CAPTION_MOVEMENT_GENERIC,
                 "explorer": core.CAPTION_PREDICTIONS + EXPLORER_CAVEAT_SUFFIX,
                 "compare_info": COMPARE_INFO.format(n=len(eligible), have=", ".join(eligible) or "none")},
        "slot_help": [list(x) for x in core.SLOT_HELP], "marker_help": [list(x) for x in core.MARKER_HELP],
        "chip_key_html": style.chip_key_html(), "brand_html": style.brand_html(),
        "compare_html": style.compare_pair_html(eligible[0], None) if len(eligible) == 1 else "",
        "reports_html": style.report_library_html(lib_paths, None),
        "series_labels": {f"arc_{i}": names for i, (_, names) in enumerate(PANELS)},
        "film_labels": labels,
    }


def build(root: Path, out: Path, wanted: list[str] | None = None, n_boot_arc: int = 1000, n_boot_move: int = 500) -> dict:
    root = Path(root).resolve()
    films = public_films(root, wanted)  # raises Refused before anything is written
    staged: dict[str, bytes] = {}
    entries = []
    for film in films:
        df, version = core.load_table(root, film, True)
        if "film_id" in df.columns and not (df["film_id"] == film).all():
            raise Refused(f"{film}: the table holds rows of another film")
        label = core.film_label(root, film)
        vdf = viz.prepare(df)
        hero = hero_row(root, film, df)
        has_arc = version == "v1" and core.has_arc_columns(df)
        arcs = {m: core.arc_series(df, m, n_boot=n_boot_arc) for m in MODES} if has_arc else {}
        moves = {m: core.movement_series(df, m, n_boot=n_boot_move) for m in MODES} if has_arc and core.has_movement(df) else {}
        top = core.similar_table(root, film, True, df)
        fj = film_json(root, film, label, df, version, vdf, hero, arcs, moves)
        staged[f"{film}/film.json"] = _dump(fj)
        staged[f"{film}/shots.json"] = _dump(shots_json(root, film, df, top))
        if has_arc:
            groups = split_groups(film_charts(film, df, vdf, None if hero is None else int(hero["shot_id"]), arcs, moves))
            for group, obj in groups.items():
                staged[f"{film}/{group}.json"] = _dump(obj)
            staged[f"{film}/strip.png"] = strip_png(root, film, vdf)
        s = core.length_stats(df)
        entries.append({"id": film, "label": label, "shots": int(s["shots"]), "running_min": float(s["running_min"]),
                        "card": f"{s['shots']:,} shots · {s['running_min']:.1f} min", "has_arc": has_arc,
                        "hero": None if hero is None else {"shot_id": int(hero["shot_id"]), "kf_mid": kf_name(root, film, hero["kf_mid"])}})
    staged["site.json"] = _dump(site_json(root, entries))
    out = Path(out)
    if out.exists():
        for p in sorted(out.iterdir()):  # only what this script writes: site.json and one folder per film
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            elif p.name == "site.json":
                p.unlink()
    for rel, b in staged.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    return {"films": films, "files": {rel: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for rel, b in sorted(staged.items())}}


def _dump(obj, sep: tuple = (", ", ":")) -> bytes:
    """Compact JSON; data files keep a space after each comma so a list of numbers never reads as one thousands-grouped number."""
    return (json.dumps(obj, ensure_ascii=False, separators=sep, allow_nan=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="public demo data root (default: env CGT_DEMO_ROOT, else demo_data/, else data/release/demo_bundle/)")
    ap.add_argument("--out", default=str(ROOT / "app/static/data"))
    ap.add_argument("--film", action="append", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root) if a.root else default_root()
    try:
        res = build(root, Path(a.out), a.film)
    except (Refused, PermissionError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    print(json.dumps({"films": res["films"], "files": {k: v["bytes"] for k, v in res["files"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
