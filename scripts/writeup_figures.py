"""Write-up figures for a PUBLIC-DOMAIN film. CPU only, matplotlib Agg, seconds.
Reads only the film's audio-corrected v1 token table (sha256-checked) and committed Phase 6 results; writes four PNGs + figures.md to --out.
No film frame, keyframe, owned film or benchmark result is read or drawn. Refuses any film that is not public_domain: true in the committed
configs/films.yaml (or is in the local overlay), a token table whose sha256 differs from --expect-sha, and a sample row that does not match the table.
Usage: PYTHONPATH=src python scripts/writeup_figures.py --film night_of_the_living_dead --tokens PATH --expect-sha SHA256 [--out docs/writeup/figures]
       [--numbers-json results/phase6/notld_writeup_numbers.json]   (also writes the write-up's section 6 numbers, recomputed from the same table)
`section6_numbers` / `contrast_numbers` are the recompute of the write-up's case-study numbers and of the text-only owned-film contrast
(a recorded decision: aggregates only, no id, path, frame or per-shot row; tests/test_writeup_4b.py recomputes them from the local tables)."""
import argparse
import hashlib
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN, OVERLAY = "configs/films.yaml", "configs/films.local.yaml"
NOTLD_SHA = "ee2a20921147fb1bc1ea4fdf2a50d8f2f47fed43a8a4eca9b1ddf3530990c135"  # live 4B table (p3c-v2-4b, the recorded rule); 2B table was 0e818c9d...
NOTLD_SHA_2B = "0e818c9d48546fc087609c539e9b9dd007b74100fd58b467c42f1350eaee9fb7"  # *_pre_4b, kept for the gold scorer (0023)
ARC_CAPTION = "model predictions / heuristics of unmeasured accuracy; within-film shapes only"
# Okabe-Ito (colour-blind safe)
BLUE, ORANGE, GREEN, VERMIL, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#555555", "#56B4E9"
RC = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
      "legend.fontsize": 8, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "axes.spines.top": False,
      "axes.spines.right": False, "axes.edgecolor": "#333333", "text.color": "#111111", "axes.labelcolor": "#111111", "path.simplify": False}
DPI = 150
GAP_BIN_MIN = 4.0
SMALL_WINDOW = 10  # windows with fewer shots are drawn hollow
ARC_PANELS = [  # (series, title, y label with unit, interval kind)
    ("cuts_per_min", "Shot starts per minute (heuristic cuts; no band)", "shots / min", None),
    ("median_len_s", "Median shot length (heuristic cuts; band: IQR)", "seconds", "IQR"),
    ("close_share", "Close-shot share, ECU/CU/MCU (model; band: 95% CI)", "share of shots", "95% CI"),
    ("mean_size_ordinal", "Mean shot size (model; band: 95% CI)", "0 = ECU ... 6 = EWS", "95% CI"),
    ("low_key_share", "Low-key share (heuristic; band: 95% CI)", "share of shots", "95% CI"),
    ("people_mean", "Mean people count, cap 4 (heuristic; 95% CI)", "people / shot", "95% CI"),
    ("multi_person_share", "Multi-person share, 2+ (heuristic; 95% CI)", "share of shots", "95% CI"),
    ("dialogue_share", "Dialogue share (audio heuristic; 95% CI)", "share of shots", "95% CI"),
]
MODEL_SLOTS = {"size", "lens", "angle", "framing", "light", "comp", "move"}
SLOT_TEXT = {
    "size": "shot scale, ECU (tightest) ... EWS (widest)",
    "lens": "apparent focal length: long / medium / wide / ultra-wide",
    "angle": "camera angle: low / high / overhead / aerial / dutch",
    "framing": "who/what is framed: single, 2-shot, group, insert ...",
    "light": "light type / light condition",
    "comp": "composition: center, balanced, right-heavy ...",
    "key": "luminance key: low / mid / high-key",
    "DoF": "depth of field: shallow / medium / deep",
    "people": "number of people detected (capped at 4)",
    "sound": "audio class of the shot: dialogue / score / ambient ...",
    "len": "shot length bin (seconds)",
    "palette": "colour palette cluster id (32 global clusters)",
    "move": "camera-movement atom set (atoms joined by +)",
}


class Refused(Exception):
    pass


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def public_entry(film: str, code_root: Path = ROOT) -> dict | None:
    """The committed films.yaml entry of `film` if it is public_domain: true and not in the local overlay, else None."""
    main = yaml.safe_load((Path(code_root) / MAIN).read_text()) or {}
    ov_p = Path(code_root) / OVERLAY
    overlay = {e.get("id") for e in ((yaml.safe_load(ov_p.read_text()) or {}).get("films") or [])} if ov_p.exists() else set()
    hits = [f for f in main.get("films") or [] if f.get("id") == film and f.get("public_domain") is True]
    return hits[0] if hits and film not in overlay else None


# ---------- data (pure, tested) ----------
def length_bins(len_s: np.ndarray, per_decade: int = 12) -> np.ndarray:
    lo = np.floor(np.log10(max(float(np.min(len_s)), 1e-3)) * per_decade) / per_decade
    hi = np.ceil(np.log10(float(np.max(len_s))) * per_decade) / per_decade
    return 10 ** np.arange(lo, hi + 1e-9 + 1 / per_decade, 1 / per_decade)


def eligible_gaps_min(start_s: np.ndarray, exclude: int = 2) -> np.ndarray:
    """|start_i - start_j| / 60 over every ORDERED pair of shots whose positions differ by more than `exclude` (the similar-shots candidate rule)."""
    s = np.asarray(start_s, dtype=float)
    out = [np.abs(s[k:] - s[:-k]) / 60.0 for k in range(exclude + 1, len(s))]
    if not out:
        return np.zeros(0)
    one = np.concatenate(out)
    return np.concatenate([one, one])  # both orders


def gap_hist(gaps: np.ndarray, edges: np.ndarray) -> np.ndarray:
    c, _ = np.histogram(np.clip(gaps, edges[0], edges[-1] - 1e-9), bins=edges)
    return c / max(len(gaps), 1)


def parse_row(row: str) -> list[tuple[str, str]]:
    body = row.strip()
    if not (body.startswith("[") and body.endswith("]")):
        raise Refused(f"not a token row: {row[:40]!r}")
    out = []
    for part in body[1:-1].split(" | "):
        k, _, v = part.partition("=")
        out.append((k.strip(), v.strip()))
    return out


def sample_row(md: str, shot_id: int) -> tuple[str, str]:
    """(start mmm:ss.s, token row) of `shot_id` from the code block of notld_token_rows_sample.md."""
    m = re.search(rf"^{shot_id:05d} (\d{{3}}:\d{{2}}\.\d) (\[.*\])$", md, flags=re.MULTILINE)
    if not m:
        raise Refused(f"shot {shot_id} not in the token-row sample")
    return m.group(1), m.group(2)


def start_label(start: str) -> str:
    """'mmm:ss.s' -> minutes without leading zeros: '045:12.3' -> '45:12.3', '000:12.3' -> '0:12.3' (figures A5)."""
    mins, rest = start.split(":", 1)
    return f"{int(mins)}:{rest}"


# ---------- write-up numbers (pure, tested) ----------
STATIC = ("size", "lens", "angle", "framing", "light_type", "light_cond", "composition")
SINGLE_LABEL = ("size", "lens", "angle", "framing")  # categorical in configs/schema.yaml; the other three are multilabel
CLOSE = ("ECU", "CU", "MCU", "MS")
ONE_PERSON_FRAMING = ("single", "insert", "ots")


def _share(mask) -> float:
    return round(float(np.mean(mask)), 4)


def section6_numbers(df: pd.DataFrame) -> dict:
    """Section 6.1 numbers of the write-up, per shot of the token table (soft-vote label over the three keyframes)."""
    n = len(df)
    one = df[df["people"].astype(int) == 1]
    lt = df["light_type"].value_counts()
    return {
        "shots": n,
        "size_ecu_to_ms_share": _share(df["size"].isin(CLOSE)),
        "angle_share": {k: _share(df["angle"] == k) for k in ("low", "high", "dutch", "overhead", "aerial")},
        "light_type_share": {k: round(float(v) / n, 4) for k, v in lt.items()},
        "keyframe_consistent_share": {a: _share(df[f"{a}_consistent"].astype(bool)) for a in STATIC},
        "four_single_label_agree_share": _share(df[[f"{a}_consistent" for a in SINGLE_LABEL]].astype(bool).all(axis=1)),
        "all_seven_agree": {"n": int(df["kf_all_agree"].astype(bool).sum()), "share": _share(df["kf_all_agree"].astype(bool))},
        "one_person_framing": {"n_shots": len(one), "share_single_insert_ots": _share(one["framing"].isin(ONE_PERSON_FRAMING))},
    }


def contrast_numbers(df: pd.DataFrame) -> dict:
    """Text-only aggregates of the section 7 contrast table: no id, path or per-shot value."""
    from cgt.tokens import arc
    n = len(df)
    contra = arc.contradictory_share(df["move_atoms"])
    agree = df["kf_all_agree"].astype(bool)
    return {"shots": n, "median_len_s": round(float(np.median(df["len_s"])), 2),
            "contradictory_move_sets": {"n": round(contra * n), "share": round(contra, 4)},
            "all_seven_agree": {"n": int(agree.sum()), "share": _share(agree)}}


# ---------- figures ----------
def _save(fig, path: Path, dpi: int = DPI) -> None:
    fig.savefig(path, dpi=dpi, metadata={"Software": None})
    plt.close(fig)


def fig_lengths(len_s: np.ndarray, path: Path, title: str) -> dict:
    edges = length_bins(len_s)
    med = float(np.median(len_s))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(len_s, bins=edges, color=SKY, edgecolor=BLUE, linewidth=0.6)
    ax.set_xscale("log")
    ax.axvline(med, color=VERMIL, linestyle="--", linewidth=1.4)
    ax.text(med * 1.06, ax.get_ylim()[1] * 0.93, f"median {med:.1f} s", color=VERMIL, fontsize=9, va="top")
    ticks = [t for t in (0.5, 1, 2, 4, 8, 16, 32, 64, 128) if edges[0] <= t <= edges[-1]]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("shot length (seconds, log scale)")
    ax.set_ylabel("number of shots")
    ax.set_title(f"{title}: shot lengths (n = {len(len_s):,}; automatic cuts, heuristic)")
    fig.tight_layout()
    _save(fig, path)
    return {"n": len(len_s), "median_s": med, "q25_s": float(np.quantile(len_s, 0.25)), "q75_s": float(np.quantile(len_s, 0.75))}


def fig_arc(arc: pd.DataFrame, path: Path, title: str) -> dict:
    fig, axes = plt.subplots(4, 2, figsize=(7.6, 8.6), sharex=True)
    for ax, (series, ttl, ylab, ivl) in zip(axes.flat, ARC_PANELS, strict=True):
        d = arc[arc["series"] == series].sort_values("window_start_min")
        if d.empty:
            raise Refused(f"visual-arc series missing: {series}")
        x = d["window_start_min"].to_numpy(float) + 2.5  # window centre (5-min windows)
        y = d["value"].to_numpy(float)
        small = d["n_shots"].to_numpy(int) < SMALL_WINDOW
        col = BLUE if d["kind"].iloc[0] == "prediction" else ORANGE
        if ivl and d["ci_lo"].notna().any():
            lo, hi = d["ci_lo"].to_numpy(float), d["ci_hi"].to_numpy(float)
            ax.fill_between(x, lo, hi, color=col, alpha=0.18, linewidth=0, step=None)
        ax.plot(x, y, color=col, linewidth=1.3)
        ax.plot(x[~small], y[~small], "o", color=col, markersize=3.5)
        ax.plot(x[small], y[small], "o", markerfacecolor="white", markeredgecolor=col, markersize=4)
        ax.set_title(ttl, loc="left")
        ax.set_ylabel(ylab)
        if ylab.startswith("share"):
            ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("film time (minutes, 5-min window centre)")
    fig.suptitle(f"{title}: visual arc in 5-minute windows\n{ARC_CAPTION}", fontsize=10)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, marker="o", label="model prediction"), Line2D([], [], color=ORANGE, marker="o", label="heuristic"),
               Line2D([], [], color=GREY, marker="o", markerfacecolor="white", linestyle="", label=f"window with < {SMALL_WINDOW} shots")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, path, dpi=130)
    return {"windows": int(arc["window"].nunique()), "series": [p[0] for p in ARC_PANELS]}


def fig_gaps(top1_gaps: np.ndarray, ref_gaps: np.ndarray, path: Path, title: str) -> dict:
    hi = GAP_BIN_MIN * np.ceil(max(float(np.max(top1_gaps)), float(np.max(ref_gaps))) / GAP_BIN_MIN)
    edges = np.arange(0.0, hi + GAP_BIN_MIN / 2, GAP_BIN_MIN)
    h1, h0 = gap_hist(top1_gaps, edges), gap_hist(ref_gaps, edges)
    m1, m0 = float(np.median(top1_gaps)), float(np.median(ref_gaps))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.stairs(h0, edges, color=GREY, linewidth=1.4, linestyle="--", label=f"all eligible candidate pairs (n = {len(ref_gaps):,})")
    ax.stairs(h1, edges, color=VERMIL, linewidth=1.8, label=f"each shot's top-1 similar shot (n = {len(top1_gaps):,})")
    ax.axvline(m0, color=GREY, linestyle=":", linewidth=1.2)
    ax.axvline(m1, color=VERMIL, linestyle=":", linewidth=1.2)
    top = max(h0.max(), h1.max())
    ax.text(m1 - 0.8, top * 0.62, f"top-1\nmedian\n{m1:.1f} min", color=VERMIL, fontsize=8, ha="right")
    ax.text(m0 + 0.8, top * 0.62, f"all-pairs median\n{m0:.1f} min", color=GREY, fontsize=8, ha="left")
    ax.set_ylim(0, top * 1.12)
    ax.set_xlim(0, edges[-1])
    ax.set_xlabel(f"time gap between the two shots (minutes, {GAP_BIN_MIN:g}-min bins)")
    ax.set_ylabel("share of shots / pairs per bin")
    ax.set_title(f"{title}: 'shots like this', gap to the top-1 similar shot\n(token similarity of predictions; not a visual callback claim)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    _save(fig, path)
    return {"n_top1": len(top1_gaps), "n_ref": len(ref_gaps), "median_top1_min": m1, "median_ref_min": m0,
            "share_top1_le_1min": float(np.mean(top1_gaps <= 1.0)), "share_ref_le_1min": float(np.mean(ref_gaps <= 1.0))}


def fig_anatomy(shot_id: int, start: str, row: str, path: Path, title: str) -> dict:
    slots = parse_row(row)
    n = len(slots)
    fig = plt.figure(figsize=(7.6, 1.55 + 0.36 * n))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    H = fig.get_size_inches()[1]
    dy = 100 * 0.36 / H
    y = 100 - 100 * 0.32 / H
    ax.text(2, y, f"{title}: anatomy of one real token row (shot {shot_id}, starts {start_label(start)})", fontsize=10.5, weight="bold", va="center")
    y -= dy * 0.95
    half = (n + 1) // 2
    for line in (" | ".join(f"{k}={v}" for k, v in slots[:half]), " | ".join(f"{k}={v}" for k, v in slots[half:])):
        ax.text(2, y, line, family="DejaVu Sans Mono", fontsize=7.6, va="center", color="#111111")
        y -= dy * 0.7
    y -= dy * 0.35
    for k, v in slots:
        model = k in MODEL_SLOTS
        col = BLUE if model else ORANGE
        ax.text(2.5, y, k, family="DejaVu Sans Mono", fontsize=8.5, va="center", ha="left", color="white", weight="bold",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": col, "edgecolor": col})
        ax.text(14, y, v, family="DejaVu Sans Mono", fontsize=8.5, va="center", ha="left")
        ax.text(50.5, y, ("MODEL  " if model else "HEUR.  ") + SLOT_TEXT.get(k, ""), fontsize=7.8, va="center", ha="left", color="#222222")
        y -= dy
    ax.text(2, y - dy * 0.1, "MODEL = model prediction (blue); HEUR. = heuristic (orange). All slots: accuracy on this film unmeasured.\n"
            "*  = the three keyframes disagree (near-tie);  ?  = missing.  Text-only figure: no film frame.", fontsize=7.8, va="center", color="#222222")
    _save(fig, path)
    return {"slots": [k for k, _ in slots]}


# ---------- driver ----------
NAMES = {"lengths": "{f}_fig1_shot_lengths.png", "arc": "{f}_fig2_visual_arc.png", "gaps": "{f}_fig3_similar_gap.png",
         "anatomy": "{f}_fig4_token_row_anatomy.png"}


def build(film: str, tokens: Path, expect_sha: str, out: Path, results: Path, code_root: Path = ROOT, anatomy_shot: int = 538,
          numbers_json: Path | None = None) -> dict:
    with plt.rc_context(RC):
        r = _build(film, tokens, expect_sha, out, results, code_root, anatomy_shot)
    if numbers_json is not None:  # only reached after _build's public-film and sha checks passed
        import json
        df = pd.read_parquet(tokens)
        doc = {"film": film, "token_table": {"name": tokens.name, "sha256": r["inputs"]["token table"][1]},
               "generated_by": "scripts/writeup_figures.py --numbers-json", "section6": section6_numbers(df), "contrast": contrast_numbers(df),
               "caveat": "descriptive shares of predictions and heuristics; accuracy on this film is unmeasured"}
        Path(numbers_json).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        r["numbers"] = doc
    return r


def _build(film, tokens, expect_sha, out, results, code_root, anatomy_shot) -> dict:
    entry = public_entry(film, code_root)
    if entry is None:
        raise Refused(f"{film}: not a public_domain: true film of the committed {MAIN} (or in the local overlay); figures are never made for it")
    got = sha256(tokens)
    if got != expect_sha:
        raise Refused(f"token table sha256 {got} != expected {expect_sha}")  # full hashes (figures A1)
    arc_p = results / f"visual_arc_{film}.csv"
    sim_p = results / f"similar_shots_{film}.csv"
    smp_p = results / "notld_token_rows_sample.md"
    df = pd.read_parquet(tokens, columns=["film_id", "shot_id", "start_s", "len_s", "token_row"]).sort_values("shot_id").reset_index(drop=True)
    if set(df["film_id"]) != {film}:
        raise Refused("token table holds another film")
    arc = pd.read_csv(arc_p)
    sim = pd.read_csv(sim_p)
    start, row = sample_row(smp_p.read_text(), anatomy_shot)
    tab = df.loc[df["shot_id"] == anatomy_shot, "token_row"]
    if tab.empty or tab.iloc[0] != row:
        raise Refused(f"sample row of shot {anatomy_shot} does not match the token table")
    title = str(entry.get("title") or film)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / v.format(f=film) for k, v in NAMES.items()}
    top1 = sim.loc[sim["rank"] == 1, "gap_min"].to_numpy(float)
    ref = eligible_gaps_min(df["start_s"].to_numpy(float))
    stats = {
        "lengths": fig_lengths(df["len_s"].to_numpy(float), paths["lengths"], title),
        "arc": fig_arc(arc, paths["arc"], title),
        "gaps": fig_gaps(top1, ref, paths["gaps"], title),
        "anatomy": fig_anatomy(anatomy_shot, start, row, paths["anatomy"], title),
    }
    inputs = {"token table": (tokens.name, got), "visual arc CSV": (arc_p.name, sha256(arc_p)),
              "similar shots CSV": (sim_p.name, sha256(sim_p)), "token-row sample": (smp_p.name, sha256(smp_p))}
    (out / "figures.md").write_text(figures_md(film, title, paths, stats, inputs, anatomy_shot))
    return {"paths": paths, "stats": stats, "inputs": inputs}


def figures_md(film, title, paths, st, inputs, shot) -> str:
    L, A, G = st["lengths"], st["arc"], st["gaps"]
    sz = {k: p.stat().st_size for k, p in paths.items()}
    inp = "\n".join(f"- {k}: `{n}` sha256 `{h}`" for k, (n, h) in inputs.items())
    return f"""# Write-up figures: {title}

Produced by `scripts/writeup_figures.py` (plan: `docs/writeup/figures_plan.md`). {title} is public domain
(`configs/films.yaml`). No figure contains a film frame or keyframe; no owned film and no benchmark result is drawn. Regenerate with
`PYTHONPATH=src python scripts/writeup_figures.py --film {film} --tokens data/processed/{film}_tokens_v1.parquet --expect-sha {inputs['token table'][1]}`.

## Inputs
{inp}

General caveat for every figure: shot boundaries are automatic (heuristic cut detector); model slots are predictions and heuristic slots are
unverified, and their accuracy on this film is unmeasured. Everything is descriptive and within this one film.

## Figure 1. Shot-length distribution
File: `{paths['lengths'].name}` ({sz['lengths'] / 1024:.0f} KB). Inputs: token table (`len_s`).

Caption: Distribution of the {L['n']:,} automatically detected shot lengths of {title} (log-spaced bins, log seconds axis). Dashed line:
median {L['median_s']:.1f} s (interquartile range {L['q25_s']:.1f}-{L['q75_s']:.1f} s).

Caveats: lengths depend on the cut detector (threshold fixed in `configs/films.yaml`, hand-checked on 50 cuts); missed or false cuts move
shots between bins.

## Figure 2. Visual arc
File: `{paths['arc'].name}` ({sz['arc'] / 1024:.0f} KB). Inputs: visual arc CSV.

Caption: {title} in {A['windows']} five-minute windows: shot starts per minute, median shot length (band: within-window IQR),
close-shot share, mean shot size (0 = ECU ... 6 = EWS), low-key share, mean people count (capped at 4), multi-person share and dialogue share
(bands: 95% bootstrap intervals over the window's shots). Blue: model predictions; orange: heuristics. Hollow markers: windows with fewer
than {SMALL_WINDOW} shots (the last window is partial).
All series: "{ARC_CAPTION}".

Caveats: intervals describe resampling of the window's shots, not label accuracy; shot starts per minute has no interval and counts the
first shot of the film; dialogue share is from the audio-corrected table. No narrative, causal or cross-film claim.

## Figure 3. "Shots like this": time gap to the top-1 similar shot
File: `{paths['gaps'].name}` ({sz['gaps'] / 1024:.0f} KB). Inputs: similar shots CSV (rank-1 rows), token table (`start_s`, for the reference).

Caption: For each of the {G['n_top1']:,} shots, the time gap to its most similar other shot by token similarity (solid vermilion line), against the gap of
all {G['n_ref']:,} eligible candidate pairs (ordered pairs more than two shots apart; dashed grey line), as the share per
{GAP_BIN_MIN:g}-minute bin. Medians {G['median_top1_min']:.1f} min (top-1) and {G['median_ref_min']:.1f} min (all pairs). Gaps of at most 1 min:
{100 * G['share_top1_le_1min']:.1f}% of top-1 partners vs {100 * G['share_ref_le_1min']:.1f}% of all pairs.

Caveats: similarity is computed on predicted / heuristic tokens with equal slot weights (a choice, docs/similar_shots_spec.md); a high
score means the pipeline wrote similar tokens, not that the shots look alike, and a long-range pair is not evidence of a deliberate
callback. The reference is a scale reference only; no test.

## Figure 4. Token-row anatomy
File: `{paths['anatomy'].name}` ({sz['anatomy'] / 1024:.0f} KB). Inputs: token-row sample (shot {shot}), checked equal to the token table's row.

Caption: One real token row of {title} (shot {shot}, the median-position row of the deterministic sample), with each of its
{len(st['anatomy']['slots'])} slots labelled as a model prediction or a heuristic. `*` marks a value on which the three keyframes disagree; `?` a
missing value.

Caveats: the row was selected by position, not by content; its values are not checked against human labels. Text-only: no image of the shot.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--film", required=True)
    ap.add_argument("--tokens", required=True, type=Path)
    ap.add_argument("--expect-sha", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/writeup/figures")
    ap.add_argument("--results", type=Path, default=ROOT / "results/phase6")
    ap.add_argument("--anatomy-shot", type=int, default=538)
    ap.add_argument("--numbers-json", type=Path, default=None, help="also write the section 6 / contrast numbers of this (public) film as JSON")
    a = ap.parse_args(argv)
    try:
        r = build(a.film, a.tokens, a.expect_sha, a.out, a.results, anatomy_shot=a.anatomy_shot, numbers_json=a.numbers_json)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    for k, p in r["paths"].items():
        print(f"{k}\t{p}\t{p.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
