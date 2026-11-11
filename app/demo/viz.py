"""Demo v3.2 charts (docs/design/viz_lab.md, docs/demo_spec.md section v3.2): the seven headline charts the user picked from an earlier change viz lab, as one
module of the demo. Home leads with 1 (grammar barcode); Film overview shows 2-7 in three sections.

Pure: no streamlit import, writes nothing. Tables come from the demo's guarded loaders (app.load -> core.load_table) and keyframes only through
core.frame_path, so the public-mode guard of core applies unchanged. Every value drawn here is a model prediction or a heuristic whose accuracy on the
film is UNMEASURED (core.CAPTION_PREDICTIONS); every chart repeats a visible pointer to that note (NOTE) and draws low confidence as faded marks.
Every HTML snippet starts with '<div class="cgt-viz' and is produced by home_html() / overview_html() only, so the guard tests can rebuild each one
exactly. Colours: only the palettes pinned in style.py (CHART_COLORS and the VIZ_* sets), selected per theme."""
import bisect
import html
from itertools import pairwise
from pathlib import Path

import altair as alt
import core
import numpy as np
import pandas as pd
import style
import yaml

MOVE_CLASSES = ["static alone", "static + other atoms", "movement only", "?"]  # 'static + other' is a contradictory set (core.CAPTION_MOVEMENT_GENERIC)
PEOPLE_VALUES = ["0", "1", "2", "3", "4+"]
LEN_VALUES = ["<1s", "1-2s", "2-4s", "4-8s", "8-16s", "16-32s", "32+s"]
SOUND_VALUES = ["dialogue", "score", "ambient", "silence"]
PALETTE_TOP = 6  # palette clusters shown by name in the fingerprint; the rest fold into "other"
LINE_MIN = 4.0  # the film as a page: minutes of running time per line
LOW_OPACITY = 0.35  # a slot whose start, mid and end keyframes disagree is drawn at this opacity (not hidden)
LANE_H = 30  # hero lanes (Home), incl. the gap between lanes
PACE_DOT_OPACITY = 0.8  # v3.3: the accent at 0.8 is 4.0:1 on the light ground, 5.2:1 on the dark (0.6 was 2.67 light)
RHYTHM_OPACITY = (0.7, 0.22)  # shot-size rhythm: shot steps where keyframes agree / disagree (quiet, so the trend leads)
STRIP_W, STRIP_H = 1600, 160  # contact-sheet barcode image (time-proportional)
NOTE = "Model predictions and heuristics; accuracy on this film is unmeasured (full note at the top of the page)."

# (lane label, column, provenance, consistency column or None): the six slots of the barcode, also the colour choices of the page chart
LANES = [("size", "size", "model", "size_consistent"), ("angle", "angle", "model", "angle_consistent"),
         ("light", "light_type", "model", "light_type_consistent"), ("key", "key", "heuristic", None),
         ("sound", "sound", "heuristic", None), ("move", "move_class", "model", None)]
FINGERPRINT = [("size", "size", "model", "size_consistent"), ("lens", "lens", "model", "lens_consistent"), ("angle", "angle", "model", "angle_consistent"),
               ("framing", "framing", "model", "framing_consistent"), ("light type", "light_type", "model", "light_type_consistent"),
               ("light condition", "light_cond", "model", "light_cond_consistent"), ("comp", "composition", "model", "composition_consistent"),
               ("move", "move_class", "model", None), ("key", "key", "heuristic", None), ("DoF", "dof", "heuristic", None),
               ("people", "people_tok", "heuristic", None), ("sound", "sound", "heuristic", None), ("len", "len_tok", "heuristic", None),
               ("palette", "palette_tok", "heuristic", None)]


# ---- data ----------------------------------------------------------------------------------------------------------------------------------------------

def vocab() -> dict[str, list[str]]:
    """Value order of each slot from configs/schema.yaml of the code checkout (the only home of attribute vocabularies)."""
    try:
        attrs = (yaml.safe_load((core.CODE_ROOT / "configs/schema.yaml").read_text()) or {}).get("attributes", {})
    except OSError:
        attrs = {}
    return {k: [str(x) for x in v.get("values", [])] for k, v in attrs.items() if isinstance(v, dict) and "values" in v}


def _sizes() -> list[str]:
    return vocab().get("size") or ["ECU", "CU", "MCU", "MS", "MWS", "WS", "EWS"]


def _move_class(atoms) -> str:
    if atoms is None or isinstance(atoms, float) or len(atoms) == 0:
        return "?"
    s = {str(a) for a in atoms}
    if s == {"static"}:
        return "static alone"
    return "static + other atoms" if "static" in s else "movement only"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """The loaded (guarded) token table sorted by time, with the derived columns the charts use. Missing optional columns become '?'."""
    d = df.sort_values("start_s").reset_index(drop=True).copy()
    d["start_min"], d["end_min"] = d["start_s"] / 60.0, d["end_s"] / 60.0
    sizes = _sizes()
    d["size_ord"] = d["size"].map({s: i for i, s in enumerate(sizes)}) if "size" in d.columns else np.nan
    d["closeness"] = len(sizes) - 1 - d["size_ord"]  # 6 = ECU (closest) ... 0 = EWS
    d["move_class"] = [_move_class(a) for a in d["move_atoms"]] if "move_atoms" in d.columns else "?"
    if "people" in d.columns:
        d["people_tok"] = [("4+" if p >= 4 else str(int(p))) if pd.notna(p) else "?" for p in d["people"]]
    d["time"] = [f"{int(s // 60)}:{int(s % 60):02d}" for s in d["start_s"]]
    for _, col, _, _ in LANES:
        if col not in d.columns:
            d[col] = "?"
    if "token_row" not in d.columns:
        d["token_row"] = ""
    return d


def lane_values(df: pd.DataFrame, col: str) -> list[str]:
    """Values of a slot in schema order, only those present (unknown values appended in table order)."""
    order = {"move_class": MOVE_CLASSES, "people_tok": PEOPLE_VALUES, "len_tok": LEN_VALUES, "sound": SOUND_VALUES}.get(col) or vocab().get(col, [])
    present = list(dict.fromkeys(df[col].astype(str)))
    return [v for v in order if v in present] + [v for v in present if v not in order]


def _low(df: pd.DataFrame, cons: str | None) -> pd.Series:
    return ~df[cons].astype(bool) if cons and cons in df.columns else pd.Series(False, index=df.index)


def barcode(df: pd.DataFrame) -> pd.DataFrame:
    """One row per shot: x from start_min to end_min (width = shot length), one value column per lane and a 'low_<lane>' flag (keyframes disagree)."""
    out = df[["shot_id", "start_min", "end_min", "len_s", "time", "token_row"]].copy()
    for lane, col, _, cons in LANES:
        out[lane] = df[col].astype(str)
        out[f"low_{lane}"] = _low(df, cons)
    return out


def per_second(df: pd.DataFrame, col: str) -> np.ndarray:
    """The value of `col` for every second of running time (the shot covering that second's midpoint): time-weighted, not shot-weighted."""
    t = np.arange(int(np.ceil(df["end_s"].max()))) + 0.5
    idx = np.clip(np.searchsorted(df["start_s"].to_numpy(), t, side="right") - 1, 0, len(df) - 1)
    return df[col].to_numpy(dtype=float)[idx]


def size_trend(df: pd.DataFrame, sigma_s: float = 100.0, step_s: int = 10) -> pd.DataFrame:
    """Gaussian-smoothed closeness (0 = EWS ... 6 = ECU) over running time, sampled every step_s seconds; seconds of unknown size are left out."""
    v = per_second(df, "closeness")
    ok = np.isfinite(v)
    k = np.arange(-int(4 * sigma_s), int(4 * sigma_s) + 1)
    w = np.exp(-0.5 * (k / sigma_s) ** 2)
    num = np.convolve(np.where(ok, v, 0.0), w, mode="same")
    den = np.convolve(ok.astype(float), w, mode="same")  # edge- and gap-corrected
    s = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    i = np.arange(0, len(s), step_s)
    return pd.DataFrame({"t_min": (i + 0.5) / 60.0, "closeness": s[i]})


def swings(trend: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The n largest monotone runs of the smoothed trend between consecutive turning points (start, end, change in size steps)."""
    tr = trend.dropna()
    y, t = tr["closeness"].to_numpy(), tr["t_min"].to_numpy()
    cols = ["start_min", "end_min", "from", "to", "delta", "label"]
    if len(y) < 2:
        return pd.DataFrame(columns=cols)
    d = np.sign(np.diff(y))
    turns = [0] + [i for i in range(1, len(d)) if d[i] != 0 and d[i] != d[i - 1] and d[i - 1] != 0] + [len(y) - 1]
    out = pd.DataFrame([{"start_min": t[a], "end_min": t[b], "from": y[a], "to": y[b], "delta": y[b] - y[a]} for a, b in pairwise(turns)])
    out = out.reindex(out["delta"].abs().sort_values(ascending=False).index).head(n).sort_values("start_min").reset_index(drop=True)
    out["label"] = [f"{'closer' if r.delta > 0 else 'wider'} by {abs(r.delta):.1f} steps, {r.start_min:.0f}-{r.end_min:.0f} min" for r in out.itertuples()]
    return out


def strip_edges(df: pd.DataFrame, width: int) -> np.ndarray:
    """Pixel edges of every shot on a strip `width` pixels wide (x proportional to running time); widths sum exactly to `width`."""
    total = float(df["end_s"].max())
    e = np.round(np.concatenate([df["start_s"].to_numpy(), [total]]) / total * width).astype(int)
    e[0], e[-1] = 0, width
    return e


def strip_counts(root: Path, film: str, public: bool, df: pd.DataFrame, width: int = STRIP_W) -> tuple[int, int, int]:
    """(shots with a keyframe, shots without one, shots narrower than one pixel) of the contact strip; keyframes only via core.frame_path."""
    have = sum(1 for rel in df["kf_mid"] if core.frame_path(root, film, rel, public) is not None) if "kf_mid" in df.columns else 0
    return have, len(df) - have, int((np.diff(strip_edges(df, width)) == 0).sum())


def contact_strip(root: Path, film: str, public: bool, df: pd.DataFrame, width: int = STRIP_W, height: int = STRIP_H):
    """Movie barcode: every shot's mid keyframe averaged across its width into one column of tone, drawn as wide as the shot is long (PIL image).
    Keyframes only via core.frame_path (the public guard); a shot without a keyframe stays the dark ground."""
    from PIL import Image
    img = Image.new("RGB", (width, height), (24, 24, 24))
    e = strip_edges(df, width)
    for i, rel in enumerate(df["kf_mid"] if "kf_mid" in df.columns else []):
        w = int(e[i + 1] - e[i])
        if w <= 0:
            continue
        p = core.frame_path(root, film, rel, public)
        if p is None:
            continue
        with Image.open(p) as im:
            im.draft("RGB", (max(8, w * 4), height))
            col = im.convert("RGB").resize((1, height), Image.BOX)
            img.paste(col.resize((w, height), Image.NEAREST), (int(e[i]), 0))
    return img


def fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: slot, provenance, value, n, share (of all shots), share_low (of all shots: keyframes disagree on this slot). Shares sum to 1 per slot."""
    rows = []
    for label, col, src, cons in FINGERPRINT:
        if col not in df.columns:
            continue
        vals = df[col].astype(str)
        if col == "palette_tok":
            top = vals.value_counts().head(PALETTE_TOP).index
            vals = vals.where(vals.isin(top), "other")
            order = list(top) + (["other"] if (vals == "other").any() else [])
        else:
            order = lane_values(df.assign(**{col: vals}), col)
        low = _low(df, cons)
        for i, v in enumerate(order):
            m = vals == v
            rows.append({"slot": label, "source": src, "value": v, "order": i, "n": int(m.sum()), "share": float(m.mean()), "share_low": float((m & low).mean())})
    return pd.DataFrame(rows)


def transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Shot size -> next shot size over consecutive shots: n, share of the row (rows with cuts sum to 1), diag flag."""
    sizes = _sizes()
    a, b = df["size"].astype(str).iloc[:-1].to_numpy(), df["size"].astype(str).iloc[1:].to_numpy()
    rows = []
    for f in sizes:
        tot = int((a == f).sum())
        for t in sizes:
            n = int(((a == f) & (b == t)).sum())
            rows.append({"from": f, "to": t, "n": n, "row_n": tot, "share": n / tot if tot else 0.0, "diag": f == t})
    return pd.DataFrame(rows)


def cut_rate(df: pd.DataFrame, window_min: int = 3) -> pd.DataFrame:
    """Shot starts in each whole minute of running time, and a centred rolling mean over `window_min` minutes."""
    n_min = max(1, int(np.ceil(df["end_s"].max() / 60.0)))
    counts = np.bincount(np.floor(df["start_s"].to_numpy() / 60.0).astype(int), minlength=n_min)[:n_min]
    r = pd.Series(counts, dtype=float).rolling(window_min, center=True, min_periods=1).mean()
    return pd.DataFrame({"minute": np.arange(n_min), "mid_min": np.arange(n_min) + 0.5, "starts": counts, "rolling": r.to_numpy()})


def fastest_minutes(rate: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The n whole minutes with the most shot starts, no two adjacent."""
    picked: list[int] = []
    for m in rate.sort_values(["starts", "minute"], ascending=[False, True])["minute"]:
        if all(abs(m - p) > 1 for p in picked):
            picked.append(int(m))
        if len(picked) == n:
            break
    out = rate[rate["minute"].isin(picked)].copy()
    out["x0"], out["x1"] = out["minute"].astype(float), out["minute"] + 1.0
    out["label"] = [f"min {m}: {s} cuts" for m, s in zip(out["minute"], out["starts"], strict=True)]
    return out.reset_index(drop=True)


def page_pieces(df: pd.DataFrame, line_min: float = LINE_MIN) -> pd.DataFrame:
    """Shots laid out like words on a page: running time wrapped into lines of `line_min` minutes; a shot crossing a line end continues on the next
    line. Columns: line, x0, x1 (seconds within the line), and the shot's fields. Piece lengths sum to the film length."""
    L = line_min * 60.0
    keep = ["shot_id", "time", "len_s"] + [c for _, c, _, _ in LANES] + [c for _, _, _, c in LANES if c]
    keep = [k for k in dict.fromkeys(keep) if k in df.columns]
    rows = []
    for r in df[["start_s", "end_s", *keep]].to_dict("records"):
        s = r["start_s"]
        while s < r["end_s"] - 1e-9:
            line = int(s // L)
            e = min(r["end_s"], (line + 1) * L)
            rows.append({"line": line, "x0": s - line * L, "x1": e - line * L, **{k: r[k] for k in keep}})
            s = e
    out = pd.DataFrame(rows)
    out["line_label"] = [f"{int(i * line_min)}:00" for i in out["line"]]
    return out


# ---- colour (style.py palettes only) -------------------------------------------------------------------------------------------------------------------

def _t(theme: str) -> str:
    return "dark" if theme == "dark" else "light"


def ink(theme: str) -> str:
    return style.VIZ_INK[_t(theme)]


def muted(theme: str) -> str:
    return style.VIZ_MUTED[_t(theme)]


def ramp_pick(n: int, theme: str) -> list[str]:
    """n steps of the pinned 7-step ramp (index 0 = most intense on this theme's ground), evenly spread."""
    r = style.VIZ_RAMP[_t(theme)]
    return [r[0]] if n <= 1 else [r[round(i * (len(r) - 1) / (n - 1))] for i in range(n)]


def lane_colors(df: pd.DataFrame, col: str, theme: str) -> tuple[list[str], list[str]]:
    """(values in order, colours) of one slot: ordinal slots on the teal ramp (closest / most = most intense), key on literal greys, nominal slots
    on the v3 hues (re-ordered: style.VIZ_NOMINAL; values past the fifth fold to the neutral grey), movement classes neutral / amber / teal."""
    vals = lane_values(df, col)
    t, voc, neutral = _t(theme), vocab(), style.VIZ_NEUTRAL
    if col == "move_class":
        fixed = {"static alone": neutral, "static + other atoms": style.VIZ_NOMINAL[1], "movement only": style.VIZ_NOMINAL[0]}
        return vals, [fixed.get(v, style.VIZ_MISSING[t]) for v in vals]
    if col == "key":
        order = voc.get("key") or ["low-key", "mid-key", "high-key"]
        return vals, [style.VIZ_GREYS[t][order.index(v)] if v in order[:3] else style.VIZ_MISSING[t] for v in vals]
    ordinal = {"size": voc.get("size", []), "lens": voc.get("lens", []), "dof": voc.get("dof", []), "len_tok": LEN_VALUES[::-1], "people_tok": PEOPLE_VALUES[::-1]}
    if ordinal.get(col):
        order = ordinal[col]
        r = ramp_pick(len(order), theme)
        return vals, [r[order.index(v)] if v in order else style.VIZ_MISSING[t] for v in vals]
    cats = style.VIZ_NOMINAL
    known = [v for v in vals if v not in ("?", "n/a")]
    return vals, [style.VIZ_MISSING[t] if v in ("?", "n/a") else cats[known.index(v)] if known.index(v) < len(cats) else neutral for v in vals]


def palette_sets() -> list[str]:
    """Every colour a chart of this module may use as a data colour (for the guard tests)."""
    out = list(style.CHART_COLORS) + list(style.VIZ_NOMINAL) + [style.VIZ_HIGHLIGHT, style.VIZ_NEUTRAL]
    for t in ("light", "dark"):
        out += style.VIZ_RAMP[t] + style.VIZ_GREYS[t] + [style.VIZ_MISSING[t], style.VIZ_INK[t], style.VIZ_MUTED[t], style.VIZ_SURFACE[t],
                                                       style.VIZ_GROUND[t], style.ACCENT[t]]
    return out


def text_on(fill: str) -> str:
    """Ink or white, whichever clears more contrast on a filled cell."""
    return "#16181B" if style.contrast(fill, "#16181B") >= style.contrast(fill, "#FFFFFF") else "#FFFFFF"


# ---- chart look ----------------------------------------------------------------------------------------------------------------------------------------

def themed(chart, theme: str):
    """The demo's chart look with theme-aware ink (the charts are drawn with theme=None, so this is the only styling). usermeta marks a v3.2 chart."""
    i, m, line = ink(theme), muted(theme), style.VIZ_LINE[_t(theme)]
    return (chart.properties(usermeta={"cgt_viz": "v3.2"}).configure(font="IBM Plex Sans", background="transparent")
            .configure_axis(labelFontSize=11, titleFontSize=11, titleFontWeight="normal", titlePadding=8, gridColor=line, gridOpacity=1, gridWidth=1,
                            domain=False, ticks=False, labelPadding=6, labelColor=m, titleColor=m)
            .configure_legend(labelFontSize=11, titleFontSize=11, labelColor=i, titleColor=m, titleFontWeight="normal")
            .configure_text(color=i, font="IBM Plex Sans").configure_view(strokeWidth=0).configure_concat(spacing=2))


def x_minutes(total_min: float, axis: bool = True, title: str | None = "minutes into the film", field: str = "start_min") -> alt.X:
    ax = alt.Axis(values=list(range(0, int(total_min) + 1, 10)), grid=False, title=title) if axis else None
    return alt.X(f"{field}:Q", scale=alt.Scale(domain=[0, total_min], nice=False, zero=True), axis=ax)


def _shot_tooltip(extra: list | None = None) -> list:
    return [alt.Tooltip("shot_id:Q", title="shot"), alt.Tooltip("time:N", title="starts at"), alt.Tooltip("len_s:Q", title="length (s)", format=".1f"),
            *(extra or []), alt.Tooltip("token_row:N", title="token row")]


# ---- 1. grammar barcode (Home hero) --------------------------------------------------------------------------------------------------------------------

def barcode_chart(df: pd.DataFrame, theme: str, mark_shot: int | None = None, label_width: int = 52):
    """Stacked lanes, one per token slot; x = running time, each shot as wide as it is long; colour = the slot's value; faded = keyframes disagree.
    Hover lights the shot in every lane and gives its full token row. `mark_shot`: the Home example shot, outlined in every lane and pointed at.
    One layered view (the lanes are a fold of one row per shot), so width="container" fits the column exactly at every width."""
    bc = barcode(df)
    total = float(df["end_min"].max())
    lanes = [x[0] for x in LANES]
    domain, rng = [], []
    for lane, col, _, _ in LANES:
        vals, cols = lane_colors(df, col, theme)
        domain += [f"{lane}={v}" for v in vals]
        rng += cols
    srcs = "{" + ",".join(f"'{lane}':'{src}'" for lane, _, src, _ in LANES) + "}"
    pinned = mark_shot is not None and bool((bc["shot_id"] == mark_shot).any())
    rows = (["pin"] if pinned else []) + lanes
    ys = alt.Scale(domain=rows, paddingInner=0.16, paddingOuter=0.04)
    yax = alt.Axis(title=None, labelLimit=label_width, labelFontSize=11, labelPadding=8, minExtent=label_width, maxExtent=label_width, labelColor=ink(theme),
                   labelFontWeight=500, labelExpr="datum.value == 'pin' ? '' : datum.value", grid=False)
    hover = alt.selection_point(name="shot_hover", fields=["shot_id"], on="pointerover", clear="pointerout", empty=False)
    lit = f"datum.shot_id == {int(mark_shot)}" if mark_shot is not None else "false"
    on = {"or": [{"param": "shot_hover", "empty": False}, lit]}  # hovered shot, or the Home example shot
    lanes_layer = alt.Chart(bc).transform_fold(lanes, as_=["lane", "value"]).transform_calculate(
        ckey="datum.lane + '=' + datum.value", low="datum['low_' + datum.lane]", source=f"{srcs}[datum.lane]").mark_rect().encode(
        x=x_minutes(total), x2="end_min:Q", y=alt.Y("lane:N", scale=ys, axis=yax),
        color=alt.Color("ckey:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
        opacity=alt.condition("datum.low", alt.value(LOW_OPACITY), alt.value(1.0)),
        stroke={"condition": {"test": on, "value": ink(theme)}, "value": None},
        strokeWidth={"condition": {"test": on, "value": 1.5}, "value": 0},
        tooltip=[alt.Tooltip("shot_id:Q", title="shot"), alt.Tooltip("time:N", title="starts at"), alt.Tooltip("len_s:Q", title="length (s)", format=".1f"),
                 alt.Tooltip("lane:N", title="slot"), alt.Tooltip("value:N", title="value"), alt.Tooltip("source:N", title="source"),
                 alt.Tooltip("token_row:N", title="token row")]).add_params(hover)
    layers = [lanes_layer]
    if pinned:
        r = bc[bc["shot_id"] == mark_shot].iloc[0]
        right = r.start_min > total * 0.7
        pin = pd.DataFrame({"mid": [(r.start_min + r.end_min) / 2], "lane": ["pin"], "label": [f"shot {int(mark_shot)}, shown below"]})
        xs = alt.Scale(domain=[0, total], nice=False, zero=True)
        layers.append(alt.Chart(pin).mark_point(shape="triangle-down", filled=True, size=60, color=ink(theme), opacity=1).encode(
            x=alt.X("mid:Q", scale=xs), y=alt.Y("lane:N", scale=ys)))
        layers.append(alt.Chart(pin).mark_text(align="right" if right else "left", dx=-8 if right else 8, fontSize=11, font="IBM Plex Sans",
                                               color=ink(theme)).encode(x=alt.X("mid:Q", scale=xs), y=alt.Y("lane:N", scale=ys), text="label:N"))
    return themed(alt.layer(*layers).properties(height=LANE_H * len(rows), width="container"), theme)


# ---- 2. shot-size rhythm -------------------------------------------------------------------------------------------------------------------------------

def rhythm_chart(df: pd.DataFrame, theme: str, height: int = 240):
    """Every shot's size as a step over running time (closer = higher), faded where keyframes disagree; a time-weighted smoothed trend; the three
    largest swings banded and numbered."""
    total = float(df["end_min"].max())
    trend = size_trend(df)
    sw = swings(trend)
    sw = sw.assign(n=[str(i + 1) for i in range(len(sw))], mid=(sw["start_min"] + sw["end_min"]) / 2) if len(sw) else sw
    labels = _sizes()[::-1]
    expr = "[" + ",".join(f"'{s}'" for s in labels) + "][datum.value]"
    y = alt.Y("closeness:Q", scale=alt.Scale(domain=[-0.3, len(labels) + 0.1], nice=False),
              axis=alt.Axis(values=list(range(len(labels))), labelExpr=expr, title=None, grid=True, labelOverlap=False))
    cons = df["size_consistent"].astype(bool) if "size_consistent" in df.columns else pd.Series(True, index=df.index)
    shots = df[["shot_id", "time", "len_s", "token_row", "start_min", "end_min", "closeness", "size"]].assign(
        agree=cons.to_numpy(), size_kf=np.where(cons, "agree", "disagree"))
    xs = alt.Scale(domain=[0, total], nice=False, zero=True)
    layers = []
    if len(sw):
        layers.append(alt.Chart(sw).mark_rect(color=style.ACCENT[_t(theme)], opacity=0.09).encode(x=alt.X("start_min:Q", scale=xs), x2="end_min:Q"))
    layers.append(alt.Chart(shots).mark_line(interpolate="step-after", strokeWidth=0.5, color=style.VIZ_NEUTRAL, opacity=0.35).encode(x=x_minutes(total), y=y))
    layers.append(alt.Chart(shots).mark_rule(strokeWidth=2, color=muted(theme)).encode(
        x=x_minutes(total), x2="end_min:Q", y=y, opacity=alt.condition("datum.agree", alt.value(RHYTHM_OPACITY[0]), alt.value(RHYTHM_OPACITY[1])),
        tooltip=_shot_tooltip([alt.Tooltip("size:N", title="size (model)"), alt.Tooltip("size_kf:N", title="keyframes")])))
    layers.append(alt.Chart(trend.dropna()).mark_line(strokeWidth=2.5, color=style.ACCENT[_t(theme)], strokeCap="round", strokeJoin="round").encode(
        x=alt.X("t_min:Q", scale=xs), y=y,
        tooltip=[alt.Tooltip("t_min:Q", title="minute", format=".1f"), alt.Tooltip("closeness:Q", title="smoothed size (0 EWS - 6 ECU)", format=".2f")]))
    if len(sw):
        layers.append(alt.Chart(sw).mark_text(fontSize=12, fontWeight=600, color=style.ACCENT[_t(theme)], y=9).encode(
            x=alt.X("mid:Q", scale=xs), text="n:N", tooltip=[alt.Tooltip("label:N", title="swing")]))
    return themed(alt.layer(*layers).properties(height=height, width="container"), theme)


# ---- 3. contact-sheet barcode: the size lane under the image -------------------------------------------------------------------------------------------

def size_lane_chart(df: pd.DataFrame, theme: str, height: int = 24):
    """The size lane of the barcode on its own, with no left gutter, so it lines up with the full-width keyframe strip above it."""
    bc = barcode(df)
    total = float(df["end_min"].max())
    vals, cols = lane_colors(df, "size", theme)
    c = alt.Chart(bc).mark_rect().encode(
        x=x_minutes(total), x2="end_min:Q", color=alt.Color("size:N", scale=alt.Scale(domain=vals, range=cols), legend=None),
        opacity=alt.condition("datum.low_size", alt.value(LOW_OPACITY), alt.value(1.0)),
        tooltip=_shot_tooltip([alt.Tooltip("size:N", title="size (model)")])).properties(
        height=height, width="container", padding=0, autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    return themed(c, theme)


# ---- 5. cutting grammar --------------------------------------------------------------------------------------------------------------------------------

# v3.3: the share of the row in six equal bins on the pinned ramp (exact ramp colours, so a live theme switch can re-map every cell,
# style.viz_theme_css). Ramp step 5 is left out: on it neither ink nor white reaches 4.5:1 for the cell label (4.2 / 4.25), on either theme.
GRAMMAR_RAMP_IDX = (6, 4, 3, 2, 1, 0)  # smallest share -> largest: VIZ_RAMP indices (0 = most intense on the theme's ground)
GRAMMAR_STEPS = len(GRAMMAR_RAMP_IDX)


def grammar_bins(mx: float) -> list[float]:
    """Upper-open bin edges of the cutting-grammar colour (a Vega threshold scale: share < edge[0] -> bin 0, ...; >= edge[-1] -> the last bin)."""
    return [mx * k / GRAMMAR_STEPS for k in range(1, GRAMMAR_STEPS)]


def grammar_fill(share: float, mx: float, theme: str) -> str:
    """The cell colour Vega draws for `share` (bisect-right on the same edges as the threshold scale): the ramp from its quiet end (small share)
    to its intense end (largest change of size)."""
    return grammar_range(theme)[bisect.bisect_right(grammar_bins(mx), share)]


def grammar_range(theme: str) -> list[str]:
    return [style.VIZ_RAMP[_t(theme)][i] for i in GRAMMAR_RAMP_IDX]


def transition_chart(df: pd.DataFrame, theme: str, height: int = 290):
    """Row-normalised size -> next size matrix. Changes of size on a one-hue scale (six bins of the pinned ramp, every step >= 3:1 on the ground);
    the diagonal (same size again) and empty cells drawn on the neutral surface. Cell labels are ink or white, whichever reads on the cell; the label
    layers are named by that choice in both themes (cgt_lab_<light><dark>) so a live theme switch re-colours them with their cells."""
    t = transitions(df).rename(columns={"from": "this_size", "to": "next_size"})
    sizes = _sizes()
    off = t[~t["diag"]]
    mx = float(off["share"].max()) or 1.0
    flat = [d or n == 0 for d, n in zip(t["diag"], t["n"], strict=True)]
    lab = ["mm" if f else "".join("k" if text_on(grammar_fill(min(s, mx), mx, th)) == style.LABEL_ON["k"] else "w" for th in ("light", "dark"))
           for s, f in zip(t["share"], flat, strict=True)]
    t = t.assign(flat=flat, ink=[muted(theme) if f else text_on(grammar_fill(min(s, mx), mx, theme)) for s, f in zip(t["share"], flat, strict=True)],
                 pct=[("<1%" if s < 0.005 else f"{100 * s:.0f}%") if n else "" for s, n in zip(t["share"], t["n"], strict=True)],  # every non-empty cell labelled
                 kind=np.where(t["diag"], "same size again (de-emphasised)", "change of size"), lab=lab)
    x = alt.X("next_size:N", sort=sizes, title="next shot's size", axis=alt.Axis(orient="top", labelAngle=0, grid=False))
    y = alt.Y("this_size:N", sort=sizes, title="this shot's size", axis=alt.Axis(grid=False))
    tips = [alt.Tooltip("this_size:N", title="from"), alt.Tooltip("next_size:N", title="to"), alt.Tooltip("n:Q", title="cuts"),
            alt.Tooltip("row_n:Q", title="cuts from this size"), alt.Tooltip("share:Q", title="share of the row", format=".1%"), alt.Tooltip("kind:N", title="cell")]
    cells = alt.Chart(t).mark_rect(cornerRadius=3, stroke=style.VIZ_GROUND[_t(theme)], strokeWidth=2).encode(
        x=x, y=y, color=alt.condition("datum.flat", alt.value(style.VIZ_SURFACE[_t(theme)]),
                                      alt.Color("share:Q", scale=alt.Scale(type="threshold", domain=grammar_bins(mx), range=grammar_range(theme)),
                                                legend=alt.Legend(title="share of the row", format=".0%", orient="bottom", direction="horizontal",
                                                                  symbolType="square", symbolSize=110, labelOverlap=False))),
        tooltip=tips)
    texts = [alt.Chart(t).transform_filter(f"datum.lab == '{k}'").mark_text(fontSize=11, font="IBM Plex Sans").encode(
        x=x, y=y, text="pct:N", color=alt.Color("ink:N", scale=None), tooltip=tips).properties(name=f"cgt_lab_{k}")
        for k in GRAMMAR_LABEL_LAYERS if k in set(lab)]
    return themed(alt.layer(cells, *texts).resolve_scale(color="independent").properties(height=height, width="container"), theme)


GRAMMAR_LABEL_LAYERS = ("mm", "kk", "kw", "wk", "ww")  # m = muted (diagonal / empty cells: re-mapped as a theme pair), k = ink, w = white


# ---- 6. pace -------------------------------------------------------------------------------------------------------------------------------------------

def pace_charts(df: pd.DataFrame, theme: str, height: int = 180) -> tuple:
    """Two charts on one time axis, never two y-scales on one plot: (top) every shot's length (log scale) at its start; (bottom) shot starts per minute
    with a 3-minute rolling mean. The three fastest-cut minutes are banded in both. Two single views (not a concat) so each fits the column; equal
    y-axis gutters keep the time axes aligned."""
    total = float(df["end_min"].max())
    rate = cut_rate(df)
    fast = fastest_minutes(rate)
    xs = alt.Scale(domain=[0, total], nice=False, zero=True)
    gutter = {"minExtent": 44, "maxExtent": 44}
    hl = alt.Chart(fast).mark_rect(color=style.VIZ_HIGHLIGHT, opacity=0.16).encode(x=alt.X("x0:Q", scale=xs), x2="x1:Q",
                                                                                  tooltip=[alt.Tooltip("label:N", title="fastest-cut minute")])
    dots = alt.Chart(df[["shot_id", "time", "len_s", "token_row", "start_min"]]).mark_circle(size=16, color=style.ACCENT[_t(theme)], opacity=PACE_DOT_OPACITY).encode(
        x=alt.X("start_min:Q", scale=xs, axis=alt.Axis(labels=False, title=None, grid=False)),
        y=alt.Y("len_s:Q", scale=alt.Scale(type="log", domain=[0.2, 200], nice=False, clamp=True), title="shot length (s, log)",
                axis=alt.Axis(values=[0.5, 1, 2, 5, 10, 30, 100], **gutter)),
        tooltip=_shot_tooltip())
    med = alt.Chart(rate.assign(med=60.0 / rate["rolling"].clip(lower=0.1))).mark_line(strokeWidth=1.5, color=muted(theme)).encode(
        x=alt.X("mid_min:Q", scale=xs), y=alt.Y("med:Q", scale=alt.Scale(type="log", domain=[0.2, 200], nice=False, clamp=True)),
        tooltip=[alt.Tooltip("minute:Q"), alt.Tooltip("med:Q", title="mean shot length implied by the rolling rate (s)", format=".1f")])
    top = themed(alt.layer(hl, dots, med).properties(height=height, width="container"), theme)
    bars = alt.Chart(rate).mark_bar(color=style.VIZ_NEUTRAL, width=3).encode(
        x=alt.X("mid_min:Q", scale=xs, axis=alt.Axis(values=list(range(0, int(total) + 1, 10)), grid=False, title="minutes into the film")),
        y=alt.Y("starts:Q", title="starts / min", axis=alt.Axis(tickCount=4, **gutter)),
        tooltip=[alt.Tooltip("minute:Q"), alt.Tooltip("starts:Q", title="shot starts"), alt.Tooltip("rolling:Q", title="3-min rolling mean", format=".1f")])
    roll = alt.Chart(rate).mark_line(strokeWidth=2, color=style.ACCENT[_t(theme)]).encode(x=alt.X("mid_min:Q", scale=xs), y="rolling:Q")
    bottom = themed(alt.layer(hl, bars, roll).properties(height=int(height * 0.7), width="container"), theme)
    return top, bottom


# ---- 7. the film as a page -----------------------------------------------------------------------------------------------------------------------------

def page_chart(df: pd.DataFrame, theme: str, row_h: int = 15):
    """Running time wrapped into lines of four minutes like text on a page: each shot a word as long as the shot, coloured by one slot, chosen with
    the chart's own radio (a Vega-Lite binding, not a Streamlit widget); a 1-px gap in the page colour separates shots; faded = keyframes disagree."""
    p = page_pieces(df)
    data = p[["shot_id", "line_label", "x0", "x1"]].copy()
    domain, rng, src_of = [], [], {}
    for lane, col, src, cons in LANES:
        vals, cols = lane_colors(df, col, theme)
        domain += [f"{lane}={v}" for v in vals]
        rng += cols
        src_of[lane] = src
        data[f"v_{lane}"] = p[col].astype(str)
        data[f"low_{lane}"] = (~p[cons].astype(bool)) if cons and cons in p.columns else False
    shots = df[["shot_id", "time", "len_s", "token_row"]]
    lines = list(dict.fromkeys(p.sort_values("line")["line_label"]))
    L = LINE_MIN * 60
    slot = alt.param(name="colour_by", value="size", bind=alt.binding_radio(options=[x[0] for x in LANES], name="colour by "))
    srcs = "{" + ",".join(f"'{k}':'{v}'" for k, v in src_of.items()) + "}"
    c = alt.Chart(data).mark_rect(stroke=style.VIZ_GROUND[_t(theme)], strokeWidth=1, cornerRadius=1.5).encode(
        x=alt.X("x0:Q", scale=alt.Scale(domain=[0, L], nice=False),
                axis=alt.Axis(values=[0, 60, 120, 180, 240], labelExpr="'+' + datum.value / 60 + ' min'", title=None, grid=False, orient="top")),
        x2="x1:Q",
        y=alt.Y("line_label:N", sort=lines, axis=alt.Axis(title=None, labelFontSize=10.5)),
        color=alt.Color("ckey:N", scale=alt.Scale(domain=domain, range=rng), legend=None),
        opacity=alt.condition("datum.low", alt.value(LOW_OPACITY), alt.value(1.0)),
        tooltip=[alt.Tooltip("shot_id:Q", title="shot"), alt.Tooltip("time:N", title="starts at"), alt.Tooltip("len_s:Q", title="length (s)", format=".1f"),
                 alt.Tooltip("slot:N", title="slot"), alt.Tooltip("value:N", title="value"), alt.Tooltip("source:N", title="source"),
                 alt.Tooltip("token_row:N", title="token row")]).add_params(slot).transform_calculate(
        slot="colour_by", value="datum['v_' + colour_by]", low="datum['low_' + colour_by]", source=f"{srcs}[colour_by]").transform_calculate(
        ckey="datum.slot + '=' + datum.value").transform_lookup(
        lookup="shot_id", from_=alt.LookupData(data=shots, key="shot_id", fields=["time", "len_s", "token_row"])
    ).properties(height=int(row_h * len(lines) * 1.35), width="container")
    return themed(c, theme)


# ---- HTML snippets (each starts with '<div class="cgt-viz') ---------------------------------------------------------------------------------------------

def _e(s) -> str:
    return html.escape(str(s))


def _sw(color: str, low: bool = False, line: bool = False) -> str:
    st = f"background:{color}" + (f";opacity:{LOW_OPACITY:.2f}" if low else "") + (";height:3px" if line else "")
    return f'<span class="cgt-vsw {swatch_class(color)}" style="{st}"></span>'


def swatch_class(color: str) -> str:
    """The colour of a legend swatch as a class: Streamlit re-serialises inline styles as rgb(), so style.viz_theme_css finds a
    theme-paired swatch by this class to re-map it on a live theme switch."""
    return "cgt-c-" + color.lstrip("#").lower()


def legend_html(df: pd.DataFrame, lanes: list, theme: str, note_low: bool = True, extra: str = "") -> str:
    """One legend line per lane: provenance chip (model teal-tinted, heuristic dashed) + swatches with their values; then the faded-mark key."""
    rows = []
    for lane, col, src, _ in lanes:
        vals, cols = lane_colors(df, col, theme)
        items = "".join(f'<span class="cgt-vli">{_sw(c)}{_e(v)}</span>' for v, c in zip(vals, cols, strict=True))
        rows.append(f'<div class="cgt-vlrow"><span class="cgt-chip cgt-{src} cgt-mini cgt-vlane"><span class="cgt-val">{_e(lane)}</span></span>{items}</div>')
    acc = style.ACCENT[_t(theme)]
    low = (f'<div class="cgt-vlrow cgt-vlnote"><span class="cgt-vli">{_sw(acc)}{_sw(acc, True)}faded = the start, mid and end keyframes disagree on this '
           'slot (low confidence; heuristic slots have no such check)</span></div>') if note_low else ""
    return f'<div class="cgt-viz cgt-vlegend">{"".join(rows)}{low}{extra}</div>'


def note_html() -> str:
    return f'<div class="cgt-viz cgt-vnote">{_e(NOTE)}</div>'


def title_html(title: str, what: str) -> str:
    return f'<div class="cgt-viz cgt-vtitle"><h4>{_e(title)}</h4><p>{_e(what)}</p></div>'


def section_html(kicker: str, title: str, text: str) -> str:
    return f'<div class="cgt-viz cgt-vsec"><p class="cgt-vkick">{_e(kicker)}</p><h3>{_e(title)}</h3><p>{_e(text)}</p></div>'


def fingerprint_html(df: pd.DataFrame, theme: str) -> str:
    """Every slot's distribution as horizontal proportion bars; model slots teal, heuristic slots neutral grey (chip language on each header); the
    keyframe-disagreement part of a model bar faded. Exact shares as text in every row."""
    fp = fingerprint(df)
    panels = []
    for slot, g in fp.groupby("slot", sort=False):
        src = g["source"].iloc[0]
        color = style.CHART_COLORS[0] if src == "model" else style.VIZ_NEUTRAL
        mx = max(g["share"].max(), 1e-9)
        bars = []
        for r in g.sort_values("order").itertuples():
            solid, faded = 100 * (r.share - r.share_low) / mx, 100 * r.share_low / mx
            tip = f"{slot} = {r.value}: {r.n} of {len(df)} shots ({100 * r.share:.1f}%)" + (
                f"; keyframes disagree on {100 * r.share_low / r.share:.0f}% of them" if r.share_low and r.share else "")
            bars.append(f'<div class="cgt-vfr" title="{_e(tip)}"><span class="cgt-vfv">{_e(r.value)}</span><span class="cgt-vft">'
                        f'<span class="cgt-vfb" style="width:{solid:.2f}%;background:{color}"></span>'
                        f'<span class="cgt-vfb" style="width:{faded:.2f}%;background:{color};opacity:{LOW_OPACITY:.2f}"></span></span>'
                        f'<span class="cgt-vfp">{100 * r.share:.0f}%</span></div>')
        panels.append(f'<div class="cgt-vfpanel"><div class="cgt-vfhead"><span class="cgt-chip cgt-{src} cgt-mini"><span class="cgt-val">{_e(slot)}</span></span>'
                      f'<span class="cgt-vfsrc">{"model" if src == "model" else "heuristic"}</span></div>{"".join(bars)}</div>')
    key = (f'<div class="cgt-vlrow cgt-vlnote"><span class="cgt-vli cgt-vli-long">{_sw(style.CHART_COLORS[0])}{_sw(style.CHART_COLORS[0], True)}model slot; the faded '
           f'part = shots whose keyframes disagree</span><span class="cgt-vli">{_sw(style.VIZ_NEUTRAL)}heuristic slot (no confidence available)</span></div>')
    return f'<div class="cgt-viz cgt-vfgrid-wrap"><div class="cgt-vfgrid">{"".join(panels)}</div>{key}</div>'


def rhythm_legend_html(df: pd.DataFrame, theme: str) -> str:
    sw = swings(size_trend(df))
    cons = df["size_consistent"].astype(bool) if "size_consistent" in df.columns else pd.Series(True, index=df.index)
    pct = 100 * (~cons).mean()
    items = "".join(f"<li>{_e(r.label)}</li>" for r in sw.itertuples())
    m = muted(theme)
    return (f'<div class="cgt-viz cgt-vlegend"><div class="cgt-vlrow"><span class="cgt-vli">{_sw(style.ACCENT[_t(theme)], line=True)}smoothed trend '
            f'(time-weighted, Gaussian, sigma 100 s)</span><span class="cgt-vli">{_sw(m, line=True)}each shot\'s size</span>'
            f'<span class="cgt-vli"><span class="cgt-vsw {swatch_class(m)}" style="background:{m};opacity:{RHYTHM_OPACITY[1]};height:3px"></span>faded: keyframes disagree '
            f'({pct:.0f}% of shots)</span></div>' + (f'<div class="cgt-vlrow cgt-vlnote"><span class="cgt-vli">{_e(SWING_RULE)}</span></div>'
                                                 f'<ol class="cgt-vswings">{items}</ol>' if items else "") + "</div>")


# v3.3: the selection rules, stated next to what they select
PACE_RULE = ("Fastest-cut minutes: the three whole minutes with the most shot starts, no two adjacent (a minute next to one already banded is "
             "skipped, even if it has more cuts than the next pick).")
SWING_RULE = ("Numbered swings: the smoothed trend split at its turning points into runs that only rise or only fall; the three runs with the "
              "largest change in size steps.")


def pace_legend_html(df: pd.DataFrame, theme: str) -> str:
    fast = fastest_minutes(cut_rate(df))
    return (f'<div class="cgt-viz cgt-vlegend"><div class="cgt-vlrow"><span class="cgt-vli">{_sw(style.ACCENT[_t(theme)])}each shot (top); 3-minute rolling '
            f'cut rate (bottom)</span><span class="cgt-vli">{_sw(muted(theme), line=True)}mean length implied by that rate</span>'
            f'<span class="cgt-vli cgt-vli-long">{_sw(style.VIZ_HIGHLIGHT)}fastest-cut minutes: {_e(", ".join(fast["label"]))}</span></div>'
            f'<div class="cgt-vlrow cgt-vlnote"><span class="cgt-vli">{_e(PACE_RULE)}</span></div></div>')


def strip_caption_html(have: int, missing: int, thin: int) -> str:
    miss = f" {missing:,} shots have no keyframe here and stay dark." if missing else ""
    return (f'<div class="cgt-viz cgt-vcap">Each shot\'s mid keyframe, averaged across its width into one column of tone, drawn as wide as the shot '
            f'is long ({have:,} keyframes; {thin:,} shots are narrower than one pixel at this width and their neighbours cover them).{_e(miss)} '
            'The size lane underneath shares the time axis.</div>')


HOME_TEXT = {
    "kicker": "The whole film, shot by shot",
    "lede": "{label}: {n:,} shots over {m:.0f} minutes. Six slots of every shot's token row are stacked as lanes, and each shot is as wide as it is long. "
            "Hover any sliver to light that shot in every lane and read its full token row.",
    "zoom": "Zoom into one shot",
    "zoom_text": "Shot {sid}, marked in the barcode above: its mid keyframe and its complete token row, slot by slot.",
}


def home_html(df: pd.DataFrame, label: str, shot_id: int, theme: str) -> dict[str, str]:
    """Every HTML snippet of the Home hero, by name (rendered in this order: head, chart, legend, note, zoom)."""
    m = float(df["end_s"].max() - df["start_s"].min()) / 60.0
    head = (f'<div class="cgt-viz cgt-vhero"><p class="cgt-vkick">{_e(HOME_TEXT["kicker"])}</p>'
            f'<p class="cgt-vlede">{_e(HOME_TEXT["lede"].format(label=label, n=len(df), m=m))}</p></div>')
    mark = f'<span class="cgt-vli"><span class="cgt-vpin" aria-hidden="true">&#9660;</span>outlined and pointed at: shot {int(shot_id)}, shown below</span>'
    return {"head": head, "legend": legend_html(df, LANES, theme, extra=f'<div class="cgt-vlrow cgt-vlnote">{mark}</div>'), "note": note_html(),
            "zoom": (f'<div class="cgt-viz cgt-vzoom"><p class="cgt-vkick">{_e(HOME_TEXT["zoom"])}</p>'
                     f'<p>{_e(HOME_TEXT["zoom_text"].format(sid=int(shot_id)))}</p></div>')}


OVERVIEW_TEXT = {
    "glance": ("At a glance", "The whole film on one screen",
               "Two time-proportional views of every shot: the running time wrapped like lines of text, and the real keyframes as a strip of tone."),
    "page": ("The film as a page", ("Running time wrapped into lines of four minutes, like text: each shot is a word as long as the shot, coloured by "
              "the slot you choose in the chart. Hover a word for the shot's token row.")),
    "strip": ("Contact-sheet barcode", ("Every shot's mid keyframe, averaged into one column of tone and as wide as the shot is long, with the "
               "predicted size lane aligned beneath.")),
    "rhythm_sec": ("Rhythm and pace", "How fast it cuts, and how close it gets",
                   "Per shot and per minute. The same quantities in 20 windows with intervals are in the visual arc further down."),
    "pace": ("Pace", ("Every shot's length on a log scale, and below it the shot starts per minute with a 3-minute rolling mean; the three fastest-cut "
              "minutes are banded.")),
    "rhythm": ("Shot-size rhythm", ("Each shot's predicted size as a step over running time (closer is higher), a time-weighted smoothed trend, and "
                "the three largest swings numbered.")),
    "grammar_sec": ("Grammar", "Which tokens it uses, and in which order",
                    "The cutting habits between shot sizes, and the distribution of every slot of the row."),
    "grammar": ("Cutting grammar", ("How often each predicted shot size is followed by each other size, as a share of the row. Repeats of the same size "
                 "and empty cells are drawn neutral so the changes stand out; every other cell carries its share.")),
    "fingerprint": ("Film fingerprint", "The distribution of every slot of the token row; model slots and heuristic slots marked as in the chips."),
}


def overview_html(root: Path, film: str, public: bool, df: pd.DataFrame, theme: str) -> dict[str, str]:
    """Every HTML snippet of the Film overview sections, by name (the app renders them in OVERVIEW_ORDER)."""
    out = {}
    for k in ("glance", "rhythm_sec", "grammar_sec"):
        out[k] = section_html(*OVERVIEW_TEXT[k])
    for k in ("page", "strip", "pace", "rhythm", "grammar", "fingerprint"):
        out[f"title_{k}"] = title_html(*OVERVIEW_TEXT[k])
    out["legend_lanes"] = legend_html(df, LANES, theme)
    out["legend_size"] = legend_html(df, LANES[:1], theme)
    out["strip_cap"] = strip_caption_html(*strip_counts(root, film, public, df))
    out["legend_pace"] = pace_legend_html(df, theme)
    out["legend_rhythm"] = rhythm_legend_html(df, theme)
    out["fingerprint"] = fingerprint_html(df, theme)
    out["note"] = note_html()
    return out
