"""Demo v2.0: one front end, two builds. Home, films hub, film overview with visual arc, shot explorer, compare (side-by-side within-film shapes), reports, about; (local only) tokens vs my essay, add your own clip. Spec: docs/demo_spec.md.
Run: uv run streamlit run app/demo/app.py --client.toolbarMode minimal   (the flag hides the deploy toolbar from the first session; data root: env CGT_DEMO_ROOT, else this checkout; public mode: CGT_DEMO_PUBLIC=1).
Everything shown about shots is a MODEL PREDICTION or an unverified heuristic; nothing here is accuracy. The app writes nothing to disk."""
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import style
import viz

from cgt.tokens import filters
from cgt.tokens.format import PEOPLE_MAX

PAGES = ["Home", "Films", "Film overview", "Shot explorer", "Side by side", "Reports", "About"]  # both builds
ESSAY_PAGE = "Tokens vs my essay"  # local only, and only when an overlay film has a claims file and a report
LOCAL_PAGES = ["Add your own clip"]  # local mode only (never listed in public mode)
PAGE_LABELS = {"Side by side": "Compare (side by side)"}  # displayed name; the page key stays (tests and session state use it)
PAGE_SIZE = 8
PAGE_ICONS = {"Films": ":material/movie:", "Side by side": ":material/compare:", "Reports": ":material/description:", "About": ":material/info:",
              ESSAY_PAGE: ":material/edit_note:", "Add your own clip": ":material/add_circle:"}
RUN_COMMAND = "uv run streamlit run app/demo/app.py --client.toolbarMode minimal"
SERIES = alt.Scale(range=style.CHART_COLORS)  # v3: one categorical palette (= theme.chartCategoricalColors), each colour >= 3:1 on both grounds
ACCENT = style.ACCENT["light"]  # the light theme accent (config.toml primaryColor; v3 #146964, >= 4.5:1 on every light surface): single-series bars use it
ICON_WARN, ICON_LOCK = ":material/warning:", ":material/lock:"  # one icon set (Material), no emoji glyphs as icons


def themed(chart):
    """One chart look on every page (v2.1): readable axis labels and titles, light grid, no view frame. Applied to the top-level chart only."""
    return (chart.configure(font="IBM Plex Sans")
            .configure_axis(labelFontSize=12, titleFontSize=12, titleFontWeight="normal", titlePadding=8, gridOpacity=0.28, domain=False, ticks=False, labelPadding=6)
            .configure_legend(labelFontSize=12, symbolSize=120).configure_view(strokeWidth=0))


def caveat(text: str) -> None:
    """A required caveat: always visible (never in a collapsed expander), full text, warning style. Pages that show predictions call this at the top."""
    st.warning(text, icon=ICON_WARN)


def legend(columns: int = 2) -> alt.Legend:
    """Bottom legend that wraps (two columns, one column for more than three series or a long label (legend_columns) so labels are never cut at 800 px) instead of clipping labels; full-opacity line swatches. Every layer that encodes the colour uses this same legend:
    a `legend=None` on the faint interval band would drop the merged legend of the layered chart, and without symbolOpacity the band's 0.15 opacity would show."""
    return alt.Legend(orient="bottom", columns=columns, labelLimit=360, symbolOpacity=1, symbolType="stroke", symbolStrokeWidth=3)


LEGEND_LONG_LABEL = 28  # characters: a longer label does not fit half of the legend at 800 px with the sidebar open (demo review A7)


def legend_columns(labels: list[str]) -> int:
    """Two legend columns for up to three series with short labels; one column for more series or when any label is longer than LEGEND_LONG_LABEL characters."""
    return 2 if len(labels) <= 3 and max((len(x) for x in labels), default=0) <= LEGEND_LONG_LABEL else 1


STATIC = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
PANELS = [("Shot rhythm (heuristic)", ["cuts_per_min", "median_len_s"]),
          ("Scale (model prediction)", ["close_share", "mean_size_ordinal"]),
          ("Tone and light (key: heuristic; light: model prediction)", ["low_key_share", "high_contrast_share", "daylight_share"]),
          ("People and sound (heuristics)", ["people_mean", "multi_person_share", "dialogue_share"]),
          ("Angle (model; no eye-level class) and keyframe stability", ["low_angle_share", "keyframes_all_agree_share"])]


@st.cache_data(show_spinner=False, persist=False)
def _load(root: str, film: str, public: bool, stamp: int) -> tuple[pd.DataFrame, str]:  # `stamp` (table mtime) invalidates the in-memory cache
    return core.load_table(Path(root), film, public)


def load(root: Path, film: str, public: bool) -> tuple[pd.DataFrame, str]:
    core._check(root, film, public)
    p, _ = core.table_path(root, film)
    return _load(str(root), film, public, p.stat().st_mtime_ns)


@st.cache_data(show_spinner=False, persist=False)
def _arc(root: str, film: str, public: bool, stamp: int, mode: str) -> pd.DataFrame:
    return core.arc_series(_load(root, film, public, stamp)[0], mode)


@st.cache_data(show_spinner=False, persist=False)
def _move(root: str, film: str, public: bool, stamp: int, mode: str) -> pd.DataFrame:
    return core.movement_series(_load(root, film, public, stamp)[0], mode)


@st.cache_data(show_spinner=False, persist=False)
def _shapes(root: str, film: str, public: bool, stamp: int) -> pd.DataFrame:
    return core.shapes(_load(root, film, public, stamp)[0])


@st.cache_data(show_spinner=False, persist=False)
def _similar(root: str, film: str, public: bool, stamp: int, cstamp: int) -> pd.DataFrame:  # cstamp: centroid file mtime (0 = absent)
    t = core.similar_table(Path(root), film, public, _load(root, film, public, stamp)[0])
    t.attrs = dict(t.attrs)
    return t


@st.cache_data(show_spinner=False, persist=False)
def _viz_df(root: str, film: str, public: bool, stamp: int) -> pd.DataFrame:  # v3.2: the guarded table, prepared for app/demo/viz.py
    return viz.prepare(_load(root, film, public, stamp)[0])


@st.cache_data(show_spinner="Building the contact strip from the keyframes", persist=False)
def _strip_png(root: str, film: str, public: bool, stamp: int) -> bytes:  # keyframes only through core.frame_path (viz.contact_strip)
    import io
    buf = io.BytesIO()
    viz.contact_strip(Path(root), film, public, _viz_df(root, film, public, stamp)).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False, persist=False)
def _overview_html(root: str, film: str, public: bool, stamp: int, theme: str) -> dict:
    return viz.overview_html(Path(root), film, public, _viz_df(root, film, public, stamp), theme)


def theme_type() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - tests, old clients
        return "light"


def html_block(snippet: str, container=None) -> None:
    """A v3.2 presentational snippet of app/demo/viz.py (always '<div class="cgt-viz ...'); no caveat or banner is ever passed here."""
    (container or st).markdown(snippet, unsafe_allow_html=True)


def viz_chart(chart, container=None) -> None:
    (container or st).altair_chart(chart, width="stretch", theme=None)  # theme=None: viz.themed() styles the chart for the active theme


def stamp(root: Path, film: str) -> int:
    return core.table_path(root, film)[0].stat().st_mtime_ns


def banner(root: Path, film: str) -> None:
    if core.is_local_only(root, film):
        st.error(f"**{core.BANNER_LOCAL_ONLY}.** `{film}` is not public domain: do not screenshot, export, commit, host or publish anything shown for it.", icon=ICON_LOCK)


def series_chart(tbl: pd.DataFrame, names: list[str], x: str, x_title: str) -> alt.Chart:
    d = tbl[tbl.series.isin(names)].copy()
    d["label"] = d["series"].map(core.series_label)
    cols = legend_columns([core.series_label(n) for n in names])
    base = alt.Chart(d).encode(x=alt.X(f"{x}:Q", title=x_title))
    line = base.mark_line(point=True).encode(y=alt.Y("value:Q", title=None), color=alt.Color("label:N", title=None, scale=SERIES, legend=legend(cols)), tooltip=[
        alt.Tooltip("label:N", title="series"), alt.Tooltip("value:Q", title="value", format=".3f"), alt.Tooltip("ci_lo:Q", title="interval low", format=".3f"),
        alt.Tooltip("ci_hi:Q", title="interval high", format=".3f"), alt.Tooltip("n_shots:Q", title="shots in window"), alt.Tooltip("flag_sparse:N", title="fewer than 15 shots")])
    band = base.mark_area(opacity=0.15).encode(y="ci_lo:Q", y2="ci_hi:Q", color=alt.Color("label:N", title=None, scale=SERIES, legend=legend(cols))).transform_filter(
        "isValid(datum.ci_lo)")
    sparse = alt.Chart(d[d.flag_sparse]).mark_point(shape="cross", size=60, color="gray").encode(x=f"{x}:Q", y="value:Q")
    rows = -(-len(names) // cols)
    return themed((band + line + sparse).properties(height=220 + 20 * rows))  # the bottom legend takes part of the chart height: keep the plot area for long legends


def film_select(root: Path, label: str, films: list[str], key: str, index: int = 0) -> str:
    return st.sidebar.selectbox(label, films, index=index, key=key, format_func=lambda f: core.film_label(root, f))


def token_row(container, row, chips: bool = False, size: str = "md") -> None:
    """The complete token row as wrapped text (no horizontal scrolling, nothing cut off); with `chips`, first the same row as labelled chips (v3)."""
    if chips:
        h = style.chips_html(row, size)
        if h:
            container.markdown(h, unsafe_allow_html=True)
    container.code("" if row is None else str(row), language=None, wrap_lines=True)


def slot_legend() -> str:
    lines = ["| slot | source | meaning |", "|---|---|---|"]
    lines += [f"| `{s}` | {src} | {txt} |" for s, src, txt in core.SLOT_HELP]
    lines += ["", "| marker | meaning |", "|---|---|"]
    lines += [f"| `{m}` | {txt} |" for m, txt in core.MARKER_HELP]
    return "\n".join(lines)


def length_bin_chart(df: pd.DataFrame) -> alt.Chart:
    d = df["len_tok"].value_counts(normalize=True).rename_axis("bin").reset_index(name="share")

    def lower(b: str) -> float:
        t = str(b).rstrip("s")
        try:
            return -1.0 if t.startswith("<") else float(t.split("-")[0].rstrip("+"))
        except ValueError:
            return float("inf")
    order = sorted(d["bin"], key=lower)
    return themed(alt.Chart(d).mark_bar(color=ACCENT).encode(x=alt.X("bin:N", sort=order, title="shot length bin (seconds)", axis=alt.Axis(labelAngle=0)), y=alt.Y("share:Q", title="share of shots", axis=alt.Axis(format="%")),
                                          tooltip=[alt.Tooltip("bin:N", title="length bin"), alt.Tooltip("share:Q", title="share of shots", format=".1%")]).properties(height=220))


def page_about(root: Path, public: bool, films: list[str]) -> None:
    st.header("About this demo")
    st.markdown(
        "The **Cinematic Grammar Tokenizer** turns a film into **one readable line of tokens per shot**. Film students describe a shot in a shared vocabulary: how big the "
        "subject is, where the camera sits, how it moves, how the scene is lit, how long the shot lasts. This project writes that description for every shot automatically, "
        "with fixed slots in a fixed order, so a film can be searched, counted and plotted without watching it again (for example: every low-angle close-up with one person).")
    caveat(core.CAPTION_PREDICTIONS)
    st.subheader("What a shot token row is")
    st.markdown("Each shot gets one row with 13 slots, always in this order (row version v1, `configs/tokens.yaml`). Model slots come from trained models; the others come from "
                "hand-written rules and pretrained detectors whose thresholds are not checked against labels.")
    st.markdown(style.chip_key_html(), unsafe_allow_html=True)
    st.markdown(slot_legend())
    st.subheader("Which model writes the static slots")
    st.markdown(core.CAPTION_MODEL)
    st.subheader("How it was verified")
    st.markdown(
        "- **Pre-registration**: plans, hyper-parameters, selection and stopping rules were committed before each training run; later changes were written down as dated amendments.\n"
        "- **Sealed evaluators**: test labels sat behind an evaluator that refused uncommitted code and logged every evaluation once.\n"
        "- **Independent checks**: each change was checked by a separate verification pass that did not write it.\n"
        "- **Mutation checks**: each guard was broken on purpose to check that a test failed (this demo's public-mode guard included).\n"
        "- **Byte-identical reproductions** of model runs and token tables where possible.\n\n"
        "None of this measures how accurate the tokens are on the films shown here: no labels exist for them yet.")
    st.subheader("Caveats (read before any page)")
    st.markdown("**Predictions of unmeasured accuracy.**")
    st.caption(core.CAPTION_PREDICTIONS)
    st.markdown("**Within-film shapes only.**")
    st.caption(core.CAPTION_SHAPES)
    st.markdown("**Intervals.**")
    st.caption(core.CAPTION_INTERVALS)
    st.markdown("**Inflated movement sets.**")
    st.caption(core.CAPTION_MOVEMENT_GENERIC)
    st.markdown("**Shots like this.**")
    st.caption(core.CAPTION_SIMILAR)
    st.subheader("Films in this mode")
    if films:
        st.markdown("\n".join(f"- {core.film_label(root, f)}" for f in films))
    else:
        st.markdown("No film is available in this mode.")
    if public:
        st.caption("Public mode: only films marked public domain in the committed film registry are listed.")
    else:
        st.caption("Local mode: films that are not public domain carry a LOCAL ONLY banner and must never be screenshotted, exported or published.")
    st.caption(f"Run command: `{'CGT_DEMO_PUBLIC=1 ' if public else ''}{RUN_COMMAND}` (the flag hides Streamlit's deploy toolbar from the first session on).")


def page_add_clip(root: Path, public: bool, films: list[str]) -> None:
    if public:  # never reached: the page is not listed in public mode
        return
    st.header("Add your own clip (local only)")
    st.markdown("This app does **not** run any command and accepts **no upload**. To tokenize your own movie or clip, run this in a terminal at the root of the checkout:")
    st.code(core.ADD_CLIP_COMMAND, language="bash", wrap_lines=True)
    st.markdown("It runs the whole pipeline (shots, heuristics, static attributes, movement, tokens) and writes `data/processed/<id>_tokens_v1.parquet`; reload this app "
                "and the clip appears in the film lists in local mode. `--dry-run` prints the plan and writes nothing.")
    st.warning("The default shot-cut threshold is UNCHECKED for a new input: no labeled data exists for it and nothing is tuned automatically. Hand-check about 50 cuts after "
               "the first stage (`scripts/spotcheck_cuts.py`, as the command prints), then re-run with `--cut-threshold N` if needed.", icon=ICON_WARN)
    st.markdown(f"Then start this app locally with `{RUN_COMMAND}`.")
    st.info("Your clip stays local: it is registered in the gitignored overlay `configs/films.local.yaml`, its outputs stay under the gitignored `data/` tree, it is shown "
            "with the LOCAL ONLY banner, and it is never listed in public mode or published.", icon=ICON_LOCK)


def page_overview(root: Path, public: bool, films: list[str]) -> None:
    film = film_select(root, "Film", films, "ov_film")
    banner(root, film)
    df, version = load(root, film, public)
    st.header(f"Film overview: {core.film_label(root, film)}")
    caveat(core.CAPTION_PREDICTIONS)
    s = core.length_stats(df)
    c = st.container(key="ov_metrics").columns(4)  # keyed: two tiles per row on phones (style.py)
    c[0].metric("shots", f"{s['shots']:,}", border=True)
    c[1].metric("runtime (min)", f"{s['running_min']:.1f}", border=True, help="running time in minutes")
    c[2].metric("median shot (s)", f"{s['median_s']:.1f}", border=True)
    c[3].metric("mean shot (s)", f"{s['mean_s']:.1f}", border=True)
    st.caption(f"Shot length (heuristic cut table): quartiles {s['q25_s']:.1f} / {s['median_s']:.1f} / {s['q75_s']:.1f} s, min {s['min_s']:.2f} s, max {s['max_s']:.1f} s. Row version {version}.")
    if "len_tok" in df.columns:
        st.altair_chart(length_bin_chart(df), width="stretch")
    if version != "v1" or not core.has_arc_columns(df):
        st.info("No v1 token table with the arc columns for this film: the visual arc needs `<film>_tokens_v1.parquet`.")
        return
    st.caption(f"All seven predicted attributes agree across the start, mid and end keyframes in {100 * df['kf_all_agree'].mean():.1f}% of shots "
               "(read before any series: low agreement means the labels depend on which frame is scored).")
    overview_charts(root, film, public)
    st.subheader("Visual arc")
    mode = st.segmented_control("windows", ["equal20", "min5"], default="equal20", required=True, key="ov_win",
                                format_func=lambda m: "20 equal windows of running time (primary)" if m == "equal20" else "5-minute windows (secondary)")
    st.caption(core.CAPTION_INTERVALS)
    st.caption(core.CAPTION_SHAPES)
    tbl = _arc(str(root), film, public, stamp(root, film), mode)
    for title, names in PANELS:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.altair_chart(series_chart(tbl, names, "window_mid_min", "minutes into the film (window centre)"), width="stretch")
    counts = tbl.drop_duplicates("window")[["window", "window_start_min", "n_shots", "flag_sparse"]]
    st.caption("Shots per window: " + ", ".join(f"{int(r.window_start_min)}' n={r.n_shots}{' (flagged)' if r.flag_sparse else ''}" for r in counts.itertuples()))
    if core.has_movement(df):
        st.subheader("Predicted camera movement")
        caveat(core.movement_caption(film, df, root))
        mv = _move(str(root), film, public, stamp(root, film), mode)
        st.container(border=True).altair_chart(series_chart(mv, ["move_static_share", "move_static_alone_share", "move_contradictory_share"], "window_mid_min",
                                                            "minutes (window centre)"), width="stretch")
        st.container(border=True).altair_chart(series_chart(mv, [f"move_{g}_share" for g in core.arc.MOVE_GROUPS], "window_mid_min", "minutes (window centre)"),
                                               width="stretch")


def overview_charts(root: Path, film: str, public: bool) -> None:
    """v3.2 (docs/demo_spec.md): the six film charts of app/demo/viz.py in three sections, before the visual arc. Same guarded table as the page."""
    theme = theme_type()
    vdf = _viz_df(str(root), film, public, stamp(root, film))
    h = _overview_html(str(root), film, public, stamp(root, film), theme)
    html_block(h["glance"])
    with st.container(border=True, key="viz_page"):
        html_block(h["title_page"])
        viz_chart(viz.page_chart(vdf, theme))
        html_block(h["legend_lanes"])
        html_block(h["note"])
    with st.container(border=True, key="viz_strip"):
        html_block(h["title_strip"])
        st.image(_strip_png(str(root), film, public, stamp(root, film)), width="stretch")
        viz_chart(viz.size_lane_chart(vdf, theme))
        html_block(h["legend_size"])
        html_block(h["strip_cap"])
        html_block(h["note"])
    html_block(h["rhythm_sec"])
    with st.container(border=True, key="viz_pace"):
        html_block(h["title_pace"])
        for ch in viz.pace_charts(vdf, theme):  # two single views on one time axis
            viz_chart(ch)
        html_block(h["legend_pace"])
        html_block(h["note"])
    with st.container(border=True, key="viz_rhythm"):
        html_block(h["title_rhythm"])
        viz_chart(viz.rhythm_chart(vdf, theme))
        html_block(h["legend_rhythm"])
        html_block(h["note"])
    html_block(h["grammar_sec"])
    with st.container(border=True, key="viz_grammar"):
        html_block(h["title_grammar"])
        viz_chart(viz.transition_chart(vdf, theme))
        html_block(h["note"])
    with st.container(border=True, key="viz_fingerprint"):
        html_block(h["title_fingerprint"])
        html_block(h["fingerprint"])
        html_block(h["note"])


def page_explorer(root: Path, public: bool, films: list[str]) -> None:
    film = film_select(root, "Film", films, "ex_film")
    banner(root, film)
    df, _ = load(root, film, public)
    st.header(f"Shot explorer: {core.film_label(root, film)}")
    caveat(core.CAPTION_PREDICTIONS + " A * marks an attribute whose start/mid/end keyframes give different predictions.")
    st.markdown(style.chip_key_html(), unsafe_allow_html=True)
    with st.expander("How to read a token row (slots, `*`, `?`)"):
        st.markdown(slot_legend())
    st.sidebar.header("Filter by token slot")
    sel: dict[str, list] = {}
    for slot, col in core.slot_options(df).items():
        sel[col] = st.sidebar.multiselect(slot, sorted(df[col].dropna().unique().tolist()), key=f"ex_{col}")
    people = st.sidebar.multiselect("people", [str(i) for i in range(PEOPLE_MAX)] + [f"{PEOPLE_MAX}+"], key="ex_people")
    move_sel: list = []
    if core.has_movement(df):
        atoms = sorted({a for xs in df["move_atoms"] if xs is not None for a in xs})
        move_sel = st.sidebar.multiselect("move (any atom of the predicted set)", atoms, key="ex_move")
    lo, hi = float(df.len_s.min()), float(df.len_s.max())
    len_range = st.sidebar.slider("shot length (s)", lo, hi, (lo, hi), key="ex_len") if hi > lo else (lo, hi)
    kf = (st.sidebar.segmented_control("keyframes", list(filters.KEYFRAME_MODES), default=filters.KEYFRAME_MODES[0], required=True, key="ex_kf")
          if "n_kf_disagree" in df.columns else "any")
    view = filters.apply_filters(df, sel, None if tuple(len_range) == (lo, hi) else tuple(len_range), people, kf, move_sel)
    n_pages = filters.page_count(len(view), PAGE_SIZE)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**{len(view):,}** of {len(df):,} shots match the filters")  # plain text: a metric "N / M" was cut at 800 px
    page = c2.number_input(f"page (1-{n_pages})", 1, n_pages, 1, key="ex_page")
    c3.download_button(f"CSV ({len(view)} rows)", core.csv_bytes(root, film, public, view), file_name=f"{film}_token_rows.csv", mime="text/csv",
                       key="ex_csv", on_click="ignore", help="All columns of the loaded token table except keyframe file paths; only the rows matching the current filters.")
    for r in filters.get_page(view, int(page), PAGE_SIZE).to_dict("records"):
        card = st.container(border=True, key=f"shotcard_{r['shot_id']}")  # one bordered card per shot (v2.1); keyed for the v3 style (st-key-shotcard_*)
        card.markdown(f"**shot {r['shot_id']}** · start {r['start_min']:.1f} min · length {r['len_s']:.1f} s")
        cols = card.columns(3)
        for col, key, name in zip(cols, ("kf_start", "kf_mid", "kf_end"), ("start", "mid", "end")):
            p = core.frame_path(root, film, r.get(key), public)
            if p is None:
                col.caption(f"{name}: frame not available")
            else:
                col.image(str(p), caption=name, width="stretch")
        token_row(card, r.get("token_row", ""), chips=True)
        if all(f"{a}_p1" in r for a in STATIC):
            detail = " · ".join(f"{a} {r[f'{a}_p1']:.2f}/{r[f'{a}_margin']:.2f}" for a in STATIC)
            card.caption(f"p1/margin per attribute (the model's own softmax, not calibrated accuracy): {detail}")
        with card.expander("Shots like this"):
            similar_panel(root, film, public, df, int(r["shot_id"]))


def similar_panel(root: Path, film: str, public: bool, df: pd.DataFrame, shot_id: int) -> None:
    """Top-5 of the same (already loaded, guarded) film: keyframe through core.frame_path, token row, score. Spec: docs/similar_shots_spec.md."""
    cp = root / core.CENTROIDS_REL
    top = _similar(str(root), film, public, stamp(root, film), cp.stat().st_mtime_ns if cp.is_file() else 0)
    mode = top.attrs.get("palette_mode", "centroid")
    note = "" if mode == "centroid" else " Palette centroids are not available under this data root: the palette slot is an exact id match here."
    st.caption(core.CAPTION_SIMILAR + note)
    by_id = df.set_index("shot_id")
    for s in top[top.shot_id == shot_id].itertuples():
        o = by_id.loc[s.other_shot_id]
        c1, c2 = st.columns([1, 3])
        p = core.frame_path(root, film, o.get("kf_mid"), public)
        if p is None:
            c1.caption("similar shot: mid frame not available")
        else:
            c1.image(str(p), caption="mid", width="stretch")
        c2.markdown(f"**#{s.rank} shot {s.other_shot_id}** · start {s.other_start_s / 60:.1f} min · gap {s.gap_min:.1f} min · similarity {s.score:.3f}")
        token_row(c2, o.get("token_row", ""))


def page_side_by_side(root: Path, public: bool, films: list[str]) -> None:
    st.header("Compare: within-film shapes side by side")
    caveat(core.CAPTION_SHAPES)
    st.caption(core.CAPTION_PREDICTIONS)
    eligible = []
    for f in films:
        df, v = load(root, f, public)
        if v == "v1" and core.has_arc_columns(df):
            eligible.append(f)
    if len(eligible) < 2:
        if eligible:  # v3.1: the picker summary shows the one film and an honest empty slot (labels of listed films only)
            st.markdown(style.compare_pair_html(core.film_label(root, eligible[0]), None), unsafe_allow_html=True)
        if public:
            have = ", ".join(core.film_label(root, f) for f in eligible) or "none"
            st.info(f"The side-by-side view needs two listed films with a v1 token table. Public mode lists only films marked public domain in the committed film registry, "
                    f"and only {len(eligible)} such film is available here ({have}), so there is nothing to compare it with. Its within-film arc is on the Film overview page.")
        else:
            st.info("The side-by-side view needs two listed films with a v1 token table; fewer are available in this mode.")
        return
    a = film_select(root, "Film A", eligible, "sb_a")
    b = film_select(root, "Film B", [f for f in eligible if f != a], "sb_b")
    for f in (a, b):
        banner(root, f)
    st.markdown(style.compare_pair_html(core.film_label(root, a), core.film_label(root, b)), unsafe_allow_html=True)  # v3.1: which two films, which colour
    sh = pd.concat([_shapes(str(root), f, public, stamp(root, f)).assign(film=core.film_label(root, f)) for f in (a, b)], ignore_index=True)
    st.caption("x = fraction of each film's running time (20 equal windows); y = z-score of the series within its own film. Crosses mark windows with fewer than 15 shots.")
    names = list(dict.fromkeys(sh["series"]))
    pair = alt.Scale(domain=sorted(sh["film"].unique()), range=style.CHART_COLORS)  # v3.1: the picker summary above is the legend (same colours, same order)
    for i in range(0, len(names), 2):
        cols = st.columns(2)
        for col, name in zip(cols, names[i:i + 2]):
            d = sh[sh.series == name]
            kind = d["kind"].iloc[0]
            ch = alt.Chart(d).mark_line(point=True).encode(x=alt.X("frac:Q", title="fraction of running time"), y=alt.Y("z_within_film:Q", title="z within film"),
                                                           color=alt.Color("film:N", title=None, scale=pair, legend=None),
                                                           tooltip=[alt.Tooltip("film:N"), alt.Tooltip("frac:Q", title="fraction of running time", format=".3f"),
                                                                    alt.Tooltip("z_within_film:Q", title="z within film", format=".2f"), alt.Tooltip("n_shots:Q", title="shots in window")])
            sp = alt.Chart(d[d.flag_sparse]).mark_point(shape="cross", size=60, color="gray").encode(x="frac:Q", y="z_within_film:Q")
            box = col.container(border=True, key=f"cmp_{name}")  # keyed (v3.1): equal-height title rows so paired charts line up (style.py)
            box.markdown(f"**{core.series_label(name)}** ({kind})")
            box.altair_chart(themed((ch + sp).properties(height=180)), width="stretch")
    n = sh.drop_duplicates(["film", "frac"]).pivot(index="frac", columns="film", values="n_shots")
    st.markdown("**Shots per window**")
    st.dataframe(n)


def goto(page: str, **state) -> None:
    """Button callback: switch page (and preselect a film) before the next run."""
    st.session_state["page"] = page
    for k, v in state.items():
        st.session_state[k] = v


def film_card_stats(root: Path, film: str, public: bool) -> dict:
    df, _ = load(root, film, public)
    s = core.length_stats(df)
    return {"shots": s["shots"], "running_min": s["running_min"]}


HOME_TITLE = "One readable token row per shot"


def hero_shot(root: Path, public: bool, films: list[str]):
    """(film, row) of the Home example: the first shot (by shot id) of the first listed film that is NOT local only whose row parses, is a close shot
    (size CU or MCU) of one person with no `*` marker, and has a mid keyframe; None if there is none. Never a LOCAL ONLY film, in either mode."""
    for film in films:
        if core.is_local_only(root, film):
            continue
        try:
            df, _ = load(root, film, public)
        except (FileNotFoundError, PermissionError, KeyError, ValueError):
            continue
        if "token_row" not in df.columns:
            continue
        for r in df.sort_values("shot_id").to_dict("records"):
            slots = {t["slot"]: t for t in style.parse_row(r.get("token_row"))}
            if (slots.get("size", {}).get("value") in ("CU", "MCU") and slots.get("people", {}).get("value") == "1"
                    and not any(t["low"] for t in slots.values()) and core.frame_path(root, film, r.get("kf_mid"), public) is not None):
                return film, r
    return None


def page_home(root: Path, public: bool, films: list[str]) -> None:
    st.header(HOME_TITLE)
    st.markdown(
        "This project turns a film into **one readable line of tokens per shot**: shot size, lens, angle, framing, light, composition, key, depth of field, people, sound, "
        "length, palette and camera movement, in fixed slots. The rows can be searched, filtered, counted and plotted over a film's running time.")
    caveat(core.CAPTION_PREDICTIONS)  # above the example row so it is on the first screen at phone width too
    hero = hero_shot(root, public, films)
    if hero is not None:
        film, r = hero  # v3.2: the barcode hero shows the example's film (never LOCAL ONLY: hero_shot skips them in both modes)
        theme = theme_type()
        vdf = _viz_df(str(root), film, public, stamp(root, film))
        parts = viz.home_html(vdf, core.film_label(root, film), int(r["shot_id"]), theme)
        html_block(parts["head"])
        viz_chart(viz.barcode_chart(vdf, theme, mark_shot=int(r["shot_id"])), st.container(key="viz_hero"))  # keyed: scrolls in its own box on phones
        html_block(parts["legend"])
        html_block(parts["note"])
        html_block(parts["zoom"])
        c1, c2 = st.container(key="hero").columns([5, 6], gap="large", vertical_alignment="center")  # keyed: wraps to one column when narrow (style.py)
        c1.image(str(core.frame_path(root, film, r.get("kf_mid"), public)), width="stretch")
        c1.markdown(f'<div class="cgt-hero-cap">{core.film_label(root, film)}, shot {r["shot_id"]} at {r["start_min"]:.1f} min, mid keyframe</div>',
                    unsafe_allow_html=True)
        c2.markdown(style.chip_key_html(), unsafe_allow_html=True)
        token_row(c2, r.get("token_row", ""), chips=True, size="lg")
    st.subheader("Films in this build")
    if films:
        st.markdown("\n".join(f"- {core.film_label(root, f)}{' (LOCAL ONLY)' if not public and core.is_local_only(root, f) else ''}" for f in films))
    else:
        st.markdown("No film is available in this mode.")
    st.subheader("Where to go")
    links = [("Films", "every film with its shots and running time"), ("Side by side", "within-film shapes of two films"),
             ("Reports", "the paper and the phase reports"), ("About", "how to read a token row, how it was verified, caveats")]
    if not public:
        links += [(p, d) for p, d in ((ESSAY_PAGE, "the tokens of your clip next to your essay's claims"), ("Add your own clip", "how to tokenize a new clip"))]
    allowed = page_list(root, public)
    shown = [(p, d) for p, d in links if p in allowed]
    for i in range(0, len(shown), 2):
        cols = st.columns(2, gap="small")
        for col, (page, desc) in zip(cols, shown[i:i + 2]):
            box = col.container(border=True, key=f"homelink_{page.replace(' ', '_')}")
            box.button(PAGE_LABELS.get(page, page), key=f"home_{page}", on_click=goto, args=(page,), icon=PAGE_ICONS.get(page), type="tertiary")
            box.caption(desc)
    st.caption("Public build: public-domain films only, the public paper and the reports that pass the release checker." if public else
               "Local build: every film with a token table and every report; anything marked LOCAL ONLY must never be screenshotted, exported or published.")


def page_films(root: Path, public: bool, films: list[str]) -> None:
    st.header("Films")
    st.caption("One card per film available in this build. Shots and running time come from the heuristic cut table (no labels exist for these films).")
    for i in range(0, len(films), 3):
        cols = st.columns(3)
        for col, f in zip(cols, films[i:i + 3]):
            box = col.container(border=True, key=f"filmcard_{f}")
            ex = hero_shot(root, public, [f])  # None for LOCAL ONLY films: their cards carry no picture
            if ex is not None:
                box.image(str(core.frame_path(root, f, ex[1].get("kf_mid"), public)), width="stretch")
            box.markdown(f"**{core.film_label(root, f)}**")
            if core.is_local_only(root, f):
                box.error("LOCAL ONLY", icon=ICON_LOCK)
            try:
                s = film_card_stats(root, f, public)
                box.caption(f"{s['shots']:,} shots · {s['running_min']:.1f} min")
            except (FileNotFoundError, KeyError, ValueError):
                box.caption("token table not readable")
            box.button("Film overview", key=f"films_ov_{f}", on_click=goto, args=("Film overview",), kwargs={"ov_film": f}, width="stretch", icon=":material/timeline:")
            box.button("Shot explorer", key=f"films_ex_{f}", on_click=goto, args=("Shot explorer",), kwargs={"ex_film": f}, width="stretch", icon=":material/grid_view:")


@st.cache_data(show_spinner=False, persist=False)
def _release_clean(root: str, files: tuple) -> set:  # files = ((rel, mtime_ns, size), ...): recomputed when a file changes
    return core.release_clean(Path(root), [f[0] for f in files])


def clean_set(root: Path) -> set:
    rels = core.public_candidates(root)
    return _release_clean(str(root), tuple((r, (root / r).stat().st_mtime_ns, (root / r).stat().st_size) for r in rels))


def page_reports(root: Path, public: bool, films: list[str]) -> None:
    st.header("Reports")
    clean = clean_set(root) if public else None
    paper = core.paper_files(root, public, clean)
    reports = core.report_list(root, public, clean)
    allowed = [x for x in (paper["main"], paper["pdf"]) if x] + reports
    current = st.session_state.get("rep_sel") if st.session_state.get("rep_sel") in reports else (reports[0] if reports else None)
    lib = style.report_library_html(allowed, current)  # v3.1: the listed documents as a library (exactly `allowed`: same set as before)
    if lib:
        st.markdown(lib, unsafe_allow_html=True)
    st.subheader("The paper")
    if not public:
        st.error(f"**{core.BANNER_LOCAL_REPORT}.** The local paper names the owned film and the user's clip and includes the internal matched ShotBench results (2B and 4B).", icon=ICON_LOCK)
    if paper["main"] is None:
        st.info("The paper is not available in this build.")
    else:
        data = core.read_file(root, paper["main"], public, allowed)
        st.caption(f"Source: `{paper['main']}`" + (" (the HTML paper is not built yet: showing the draft)" if paper["main"].endswith(".md") else ""))
        if paper["main"].endswith(".html"):
            st.iframe(style.strip_emoji(data.decode("utf-8", "replace")), height=900)  # scrollable frame (st.components.v1.html is deprecated in streamlit 1.64)
            st.download_button("Download the paper (HTML)", data, file_name=Path(paper["main"]).name, mime="text/html", key="dl_html", on_click="ignore")
        else:
            with st.container(height=700, key="paper_doc"):  # keyed (v3.1): reading measure and type scale (style.py)
                st.markdown(style.strip_emoji(data.decode("utf-8", "replace")))
            st.download_button("Download the draft (Markdown)", data, file_name=Path(paper["main"]).name, mime="text/markdown", key="dl_md", on_click="ignore")
        if paper["pdf"]:
            st.download_button("Download the paper (PDF)", core.read_file(root, paper["pdf"], public, allowed), file_name=Path(paper["pdf"]).name,
                               mime="application/pdf", key="dl_pdf", on_click="ignore")
    st.caption(core.CAPTION_MODEL)
    st.subheader("Phase reports")
    if public:
        st.caption("Public build: only reports that pass the release checker are listed; internal results are never shown.")
    if not reports:
        st.info("No report is available in this build.")
        return
    rel = st.selectbox("Report", reports, key="rep_sel")
    if core.is_local_report(rel):
        st.error(f"**{core.BANNER_LOCAL_REPORT}.**", icon=ICON_LOCK)
    with st.container(height=700, key="rep_doc"):  # keyed (v3.1): reading measure, type scale, tables scroll in place (style.py)
        st.markdown(style.strip_emoji(core.read_file(root, rel, public, allowed).decode("utf-8", "replace")))


def page_essay(root: Path, public: bool, films: list[str]) -> None:
    efilms = core.essay_films(root, public)
    if public or not efilms:  # never reached: the page is not listed then
        return
    film = film_select(root, "Clip", efilms, "es_film")
    banner(root, film)
    st.header(f"Tokens vs my essay: {core.film_label(root, film)}")
    st.warning(f"**{core.CAPTION_ESSAY}** The candidate shots and verdicts are those of the local report; the token rows are model predictions and heuristics "
               "whose accuracy on this clip is UNMEASURED.", icon=ICON_WARN)
    essay = core.load_essay(root, film, public)
    df, _ = load(root, film, public)
    by_id = df.set_index("shot_id")
    with st.expander("The essay's claims (reference only)", expanded=True):
        st.markdown(style.strip_emoji("\n".join(f"- {c}" for c in essay["claims"])) or "No claim in the claims file.")
    counts = pd.Series([r["verdict_class"] for r in essay["rows"]]).value_counts()
    st.caption("Verdicts in the report: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for i, r in enumerate(essay["rows"]):
        st.markdown(style.verdict_html(r["verdict_class"]), unsafe_allow_html=True)  # v3: a text label replaces the old emoji marks
        st.markdown(f"#### {style.strip_emoji(r['claim'])}")
        st.markdown(f"**Verdict ({r['verdict_class']}):** {style.strip_emoji(r['verdict'])}")
        st.markdown(f"**What the tokens say:** {style.strip_emoji(r['tokens'])}")
        ids = [s for s in r["shot_ids"] if s in by_id.index]
        st.caption(f"Candidate shots (report): {r['candidates']}" + (f" · times {', '.join(r['times'])}" if r["times"] else "") + f" · {len(ids)} shot(s) in the table")
        shown = ids[:core.ESSAY_MAX_SHOTS]
        with st.expander(f"Keyframes and token rows ({len(shown)} of {len(ids)})", expanded=i == 0):
            for j in range(0, len(shown), 4):
                cols = st.columns(4)
                for col, s in zip(cols, shown[j:j + 4]):
                    o = by_id.loc[s]
                    p = core.frame_path(root, film, o.get("kf_mid"), public)
                    if p is None:
                        col.caption(f"shot {s}: mid frame not available")
                    else:
                        col.image(str(p), caption=f"shot {s} · {o['start_s']:.0f}-{o['end_s']:.0f} s", width="stretch")
                    token_row(col, o.get("token_row", ""))
            if len(ids) > len(shown):
                st.caption(f"{len(ids) - len(shown)} more candidate shots not shown.")
        st.divider()


def page_list(root: Path, public: bool) -> list[str]:
    if public:
        return list(PAGES)
    return list(PAGES) + ([ESSAY_PAGE] if core.essay_films(root, public) else []) + LOCAL_PAGES


def main() -> None:
    st.set_page_config(page_title="Cinematic Grammar Tokenizer demo", page_icon=":material/movie:", layout="wide", menu_items={"Get help": None, "Report a bug": None, "About": None},
                       initial_sidebar_state="auto")  # v3: "auto" keeps the page list expanded above 768 px and collapsed on phones
    st.set_option("client.toolbarMode", "minimal")  # hides the deploy button and developer menu (the app needs no app/demo config file)
    root, public = core.data_root(), core.public_mode()
    style.inject()  # v3 look: the one style module (app/demo/style.py)
    st.sidebar.markdown(style.brand_html(), unsafe_allow_html=True)  # the brand sits once in the sidebar; each page has one title (its header)
    if public:
        st.sidebar.badge("PUBLIC MODE: public-domain films only", icon=":material/public:", color="green")
    else:
        st.sidebar.badge("local mode", icon=":material/computer:", color="orange")
    pages = page_list(root, public)
    page = st.sidebar.radio("Page", pages, key="page", format_func=lambda p: PAGE_LABELS.get(p, p))
    if page not in pages:  # fail closed: a page carried in session state from another build is never rendered
        return
    films = core.listed_films(root, public)
    if page in ("Home", "About", "Reports", "Add your own clip", ESSAY_PAGE):
        {"Home": page_home, "About": page_about, "Reports": page_reports, "Add your own clip": page_add_clip, ESSAY_PAGE: page_essay}[page](root, public, films)
        return
    if not films:
        st.warning("No film is available in this mode (no token table, or the film registry could not be read).")
        return
    {"Films": page_films, "Film overview": page_overview, "Shot explorer": page_explorer, "Side by side": page_side_by_side}[page](root, public, films)


if __name__ == "__main__":  # streamlit runs the script as __main__; importing the module (unit tests) renders nothing
    main()
