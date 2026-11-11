"""Local browser for the token rows of a film (Streamlit). Run: uv run streamlit run app/browse/app.py
Everything shown about shots is a MODEL PREDICTION or an unverified heuristic; nothing here is accuracy. Local only: nothing is published from this app."""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(os.environ.get("CGT_BROWSE_ROOT") or Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "src"))

from cgt.tokens import filters
from cgt.tokens.format import PEOPLE_MAX

PAGE_SIZE = 8
STATIC = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]


def films_with_tokens() -> list[str]:
    d = ROOT / "data/processed"
    names = {p.name.removesuffix("_tokens.parquet") for p in d.glob("*_tokens.parquet")}
    names |= {p.name.removesuffix("_tokens_v1.parquet") for p in d.glob("*_tokens_v1.parquet")}
    return sorted(names)


def versions(film: str) -> list[str]:
    d = ROOT / "data/processed"
    return [v for v, suffix in (("v1", "_tokens_v1.parquet"), ("v0", "_tokens.parquet")) if (d / f"{film}{suffix}").exists()]


@st.cache_data(show_spinner=False)
def _read(path: str, stamp: int) -> pd.DataFrame:  # `stamp` (file mtime) invalidates the cache when the table is rebuilt
    return pd.read_parquet(path)


def load(film: str, version: str = "v0") -> pd.DataFrame:
    suffix = "_tokens_v1.parquet" if version == "v1" else "_tokens.parquet"
    p = ROOT / f"data/processed/{film}{suffix}"
    return _read(str(p), p.stat().st_mtime_ns)


def frame_path(film: str, rel: str) -> Path | None:
    """Keyframes are shown only from data/interim/<film>/keyframes (resolved; anything else is not displayed)."""
    root = (ROOT / "data/interim" / film / "keyframes").resolve()
    p = (ROOT / rel).resolve()
    return p if root in p.parents and p.is_file() else None


def main() -> None:
    st.set_page_config(page_title="Shot token browser", page_icon=":material/movie:", layout="wide")
    st.title("Shot token browser")
    st.error("**All shot attributes shown here are MODEL PREDICTIONS whose accuracy on this film is UNMEASURED** (no labels exist). "
             "size, lens, angle, framing, light and comp come from a ShotQA-trained Qwen3-VL-4B adapter (the film pipeline's default since 2026-10-22; earlier tables used a 2B) applied to a 1968 black-and-white film; key, DoF, people, sound, len "
             "and palette are unverified Phase 1/2 heuristics. A * marks an attribute whose start/mid/end keyframes give different predictions. "
             "Consistency is not correctness.", icon=":material/warning:")
    from cgt.pipeline import registry as films_registry
    try:  # owned films of configs/films.yaml plus every ad-hoc film of the local overlay configs/films.local.yaml
        owned = films_registry.local_only_ids(ROOT)
    except OSError:
        owned = set()
    except films_registry.RegistryClash as e:
        st.error(f"Film registry error: {e}")
        return
    films = films_with_tokens()
    if not films:
        st.warning("No token tables found. Run: uv run python scripts/make_token_rows.py --film <film_id>")
        return
    film = st.sidebar.selectbox("Film", films)
    if film in owned:
        st.error("**OWNED FILM: LOCAL ONLY.** Do not screenshot, export, commit or publish anything shown here.", icon=":material/lock:")
    vers = versions(film)
    version = st.sidebar.radio("row version", vers, horizontal=True, key="version", help="v1: soft vote over the 3 keyframes, length bins, global palette") if len(vers) > 1 else vers[0]
    df = load(film, version)
    if version == "v1":
        st.info("Row v1: each primary label is the argmax of the mean probability over the start, mid and end keyframes (soft vote); `*` marks attributes whose three keyframes "
                "disagree; palette ids `gNN` come from a global palette fitted on ShotQA stills; length is a log bin. "
                "`move` is a PREDICTION of the Phase 4 geometry model (unordered atom set, accuracy on this film unmeasured) or `n/a` when no movement table exists.")

    st.sidebar.header("Filters")
    sel = {}
    for col in [*filters.FILTER_COLUMNS, *[c for c in ("len_tok", "palette_tok") if c in df.columns]]:
        opts = sorted(df[col].dropna().unique().tolist())
        sel[col] = st.sidebar.multiselect(col, opts, key=f"f_{col}")
    move_sel: list = []
    if "move_atoms" in df.columns and bool(df.get("move_present", pd.Series([False])).iloc[0]):
        atoms = sorted({a for xs in df["move_atoms"] if xs is not None for a in xs})
        move_sel = st.sidebar.multiselect("movement (prediction, accuracy unmeasured)", atoms, key="f_move")
    people = st.sidebar.multiselect("people", [str(i) for i in range(PEOPLE_MAX)] + [f"{PEOPLE_MAX}+"], key="f_people")
    lo, hi = float(df.len_s.min()), float(df.len_s.max())
    len_range = st.sidebar.slider("shot length (s)", lo, hi, (lo, hi), key="f_len")
    kf = st.sidebar.radio("keyframes", list(filters.KEYFRAME_MODES), horizontal=True, key="f_kf",
                          help="disagree: at least one of the 7 predicted attributes differs between the start, mid and end keyframes")
    view = filters.apply_filters(df, sel, None if len_range == (lo, hi) else len_range, people, kf, move_sel)

    c1, c2, c3 = st.columns([2, 2, 3])
    c1.metric("shots matching", f"{len(view)} / {len(df)}")
    if "seed" not in st.session_state:
        st.session_state.seed = None
    if c2.button("Random sample", key="random"):
        st.session_state.seed = int(pd.Timestamp.now().value % 2**31)
    if c2.button("Back to ordered list", key="ordered"):
        st.session_state.seed = None
    n_pages = filters.page_count(len(view), PAGE_SIZE)
    if st.session_state.seed is not None:
        shown = filters.random_sample(view, PAGE_SIZE, st.session_state.seed)
        c3.caption(f"random sample of {len(shown)} (seed {st.session_state.seed})")
    else:
        page = c3.number_input(f"page (1-{n_pages})", 1, n_pages, 1, key="page")
        shown = filters.get_page(view, int(page), PAGE_SIZE)

    for r in shown.to_dict("records"):
        st.markdown(f"**shot {r['shot_id']}**  ·  start {r['start_min']:.1f} min  ·  length {r['len_s']:.1f} s  ·  palette ids are film-specific")
        cols = st.columns(3)
        for col, key, name in zip(cols, ("kf_start", "kf_mid", "kf_end"), ("start", "mid", "end")):
            p = frame_path(film, r[key])
            if p is None:
                col.caption(f"{name}: frame not available")
            else:
                col.image(str(p), caption=name, width="stretch")
        st.code(r["token_row"], language=None)
        detail = " · ".join(f"{a} {r[f'{a}_p1']:.2f}/{r[f'{a}_margin']:.2f}{'*' if r[f'{a}_consistent'] is False else ''}" for a in STATIC)
        st.caption(f"p1/margin per attribute (model's own softmax, not calibrated accuracy): {detail}")
        st.divider()


main()
