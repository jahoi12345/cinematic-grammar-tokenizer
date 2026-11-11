"""Side-by-side visual arc of two films (NOTLD and an owned film). LOCAL ONLY: the outputs go to data/interim/arc_compare/ (gitignored) because the owned film is local only;
only NOTLD-only figures (scripts/visual_arc.py) are publishable. Descriptive: distributions of PREDICTIONS and unverified heuristics, never accuracy. CPU only.
Usage: uv run python scripts/visual_arc_compare.py --films night_of_the_living_dead <owned_film_id>
The owned film(s) among --films are read from the registry (not public_domain, or overlay); the report names them by id and states their facts only from the registry entry
(cut_threshold, note, path) and the loaded tables (shots, running time): no film fact is hard-coded here. When the video runs more than 5 min past
the last shot end, the report says that the rest is not analysed (derived at runtime from the video file, not stored in the registry)."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cgt.pipeline import registry
from cgt.tokens import arc

ROOT = Path(__file__).resolve().parents[1]
N_WIN, MIN_SHOTS = 20, 15
SERIES = ["cuts_per_min", "median_len_s", "low_key_share", "people_mean", "dialogue_share", "close_share", "mean_size_ordinal", "high_contrast_share", "daylight_share", "low_angle_share",
          "keyframes_all_agree_share"]
HEURISTIC = {"cuts_per_min", "median_len_s", "low_key_share", "people_mean", "dialogue_share"}


def per_film(film: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    df = pd.read_parquet(ROOT / f"data/processed/{film}_tokens_v1.parquet")
    dur = float(df.end_s.max() / 60.0)
    tbl = arc.series_table(df, dur / N_WIN, n_boot=500)  # 20 equal windows of the film's running time
    tbl["flag_sparse"] = tbl.n_shots < MIN_SHOTS
    return df, tbl, dur


TAIL_GAP_MIN = 5.0  # minutes of video after the last shot end before the report says that the rest is not analysed


def video_minutes(path: Path) -> float | None:
    """Running time of the video file in minutes (frame count / fps), None when the file is missing or unreadable."""
    if not path.is_file():
        return None
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()
    return n / fps / 60.0 if fps and n > 0 else None


def tail_caveat(root: Path, entry: dict, analysed_min: float) -> str:
    """Derived at runtime (no registry key): when the registered video runs more than TAIL_GAP_MIN minutes past the last shot end, say so; otherwise ''."""
    total = video_minutes(Path(root) / entry["path"]) if entry.get("path") else None
    if total is None or total - analysed_min <= TAIL_GAP_MIN:
        return ""
    return f"the analysed part ends at {analysed_min:.1f} of {total:.1f} min (the rest is not analysed)"


def z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std() if v.std() > 0 else v * 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", nargs=2, required=True)
    a = ap.parse_args()
    out = ROOT / "data/interim/arc_compare"
    out.mkdir(parents=True, exist_ok=True)
    films_reg = registry.load_films(ROOT)
    owned = [f for f in a.films if f in registry.local_only_ids(ROOT)]
    own = ", ".join(owned) or "the owned film"
    data = {f: per_film(f) for f in a.films}
    fig, axes = plt.subplots(len(SERIES), 2, figsize=(13, 2.4 * len(SERIES)), sharex=True)
    for i, name in enumerate(SERIES):
        for f, (df, tbl, dur) in data.items():
            d = tbl[tbl.series == name].sort_values("window")
            x = (d.window + 0.5) / N_WIN
            axes[i, 0].plot(x, d.value, marker="o", ms=3, label=f)
            if d.ci_lo.notna().all():
                axes[i, 0].fill_between(x, d.ci_lo, d.ci_hi, alpha=0.12)
            axes[i, 1].plot(x, z(d.value.to_numpy()), marker="o", ms=3, label=f)
            sp = d[d.flag_sparse]
            axes[i, 0].scatter((sp.window + 0.5) / N_WIN, sp.value, marker="x", color="k", s=25, zorder=5)
        axes[i, 0].set_ylabel(name + ("\n(heuristic)" if name in HEURISTIC else "\n(model)"), fontsize=7)
        axes[i, 0].grid(alpha=0.3)
        axes[i, 1].grid(alpha=0.3)
    axes[0, 0].set_title("levels (x = window with fewer than 15 shots)", fontsize=9)
    axes[0, 1].set_title("shape: z-scored within each film", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    axes[-1, 0].set_xlabel("fraction of running time")
    axes[-1, 1].set_xlabel("fraction of running time")
    fig.suptitle(f"LOCAL ONLY (contains an owned film). Predictions and unverified heuristics; accuracy unmeasured on both films; PROVISIONAL for {own} (heuristics unvalidated on it, no labels)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "arc_compare.png", dpi=100)

    tails = {f: tail_caveat(ROOT, films_reg.get(f, {}), data[f][2]) for f in a.films}
    provisional = [f"{f} is PROVISIONAL: cut threshold {films_reg.get(f, {}).get('cut_threshold', 'not registered')} (configs/films.yaml), {len(data[f][0])} shots over "
                   f"{data[f][2]:.1f} min in the loaded table; registry note: {films_reg.get(f, {}).get('note') or 'none'}. Its heuristics and model labels are unvalidated domain-shift hypotheses. "
                   for f in owned]
    L = [f"# {a.films[0]} vs {a.films[1]}: side-by-side visual arc (LOCAL ONLY, PROVISIONAL)", "",
         f"Contains {own}-derived outputs (owned film): never commit or publish. Everything is a prediction of a model trained on live-action ShotQA stills or an unverified heuristic; **no accuracy is known on either film**. "
         + "".join(provisional) + "".join(f"{f}: {t}. " for f, t in tails.items() if t) +
         f"Windows: {N_WIN} equal windows of each film's running time; windows with fewer than {MIN_SHOTS} shots are flagged (x), not merged. Model-series LEVELS are not compared across films; shapes are shown z-scored within each film and are descriptive only (two films, strongly autocorrelated windows: no similarity statistic is reported).", ""]
    L += ["| film | shots | running time (min) | median shot (s) | windows < 15 shots | all 7 attributes agree across keyframes | global palette ids used (of 32) |", "|---|---|---|---|---|---|---|"]
    for f, (df, tbl, dur) in data.items():
        L.append(f"| {f} | {len(df)} | {dur:.1f} | {df.len_s.median():.1f} | {int(tbl[tbl.series == 'cuts_per_min'].flag_sparse.sum())} | {100 * df.kf_all_agree.mean():.1f}% | {df.palette_global.nunique()} |")
    L += ["", "Keyframe agreement is shown before any series is read: a low value means the model's labels depend strongly on which frame of the shot is scored.", ""]
    for col, title in [("size", "size"), ("lens", "lens"), ("angle", "angle (no eye-level class)"), ("framing", "framing"), ("light_type", "light type"), ("light_cond", "light condition (primary)"),
                       ("composition", "composition"), ("key", "key (Phase 1 heuristic)"), ("dof", "depth of field (heuristic)"), ("sound", "sound (heuristic)"), ("len_tok", "length bin")]:
        t = pd.concat({f: d[col].value_counts(normalize=True) for f, (d, _, _) in data.items()}, axis=1).fillna(0).sort_values(by=a.films[0], ascending=False)
        L += [f"### {title}: share of shots", "", "| value | " + " | ".join(a.films) + " |", "|---|" + "---|" * len(a.films)]
        L += [f"| {v} | " + " | ".join(f"{100 * t.loc[v, f]:.1f}%" for f in a.films) + " |" for v in t.index]
        L.append("")
    L += ["### people (heuristic; unvalidated on these films): share of shots", "", "| count | " + " | ".join(a.films) + " |", "|---|" + "---|" * len(a.films)]
    pe = pd.concat({f: d.people.clip(upper=4).value_counts(normalize=True) for f, (d, _, _) in data.items()}, axis=1).fillna(0).sort_index()
    L += [f"| {v if v < 4 else '4+'} | " + " | ".join(f"{100 * pe.loc[v, f]:.1f}%" for f in a.films) + " |" for v in pe.index]
    mv_ok = all("move_atoms" in df.columns and bool(df["move_present"].iloc[0]) for df, _, _ in data.values())
    if mv_ok:
        mv_rows = ["move_static_share", "move_static_alone_share", "move_mean_set_size", "move_contradictory_share", "move_pan_share", "move_tilt_share", "move_push/pull_share", "move_lateral_share", "move_boom_share"]
        figm, axm = plt.subplots(len(mv_rows), 2, figsize=(12, 2.0 * len(mv_rows)), sharex=True)
        shares = {}
        for j, (f, (df, _, dur)) in enumerate(data.items()):
            minutes = dur / N_WIN
            tm = pd.concat([arc.movement_series_table(df, minutes, n_boot=300), arc.contradictory_series(df, minutes, n_boot=300)])
            shares[f] = arc.contradictory_share(df["move_atoms"])
            for i, name in enumerate(mv_rows):
                d = tm[tm.series == name].sort_values("window")
                x = (d.window + 0.5) / N_WIN
                axm[i, j].plot(x, d.value, marker="o", ms=3, color=f"C{j}")
                axm[i, j].fill_between(x, d.ci_lo, d.ci_hi, alpha=0.15, color=f"C{j}")
                sp = d[d.n_shots < MIN_SHOTS]
                axm[i, j].scatter((sp.window + 0.5) / N_WIN, sp.value, marker="x", color="k", s=25, zorder=5)
                axm[i, j].set_ylabel(name.replace("move_", ""), fontsize=7)
                axm[i, j].grid(alpha=0.3)
            axm[0, j].set_title(f"{f} (own y-axes; shape only)", fontsize=9)
        figm.suptitle("LOCAL ONLY. Predicted camera movement (frozen geometry model; accuracy UNMEASURED); sets are inflated; levels are NOT comparable across films", fontsize=9)
        figm.tight_layout(rect=(0, 0, 1, 0.98))
        figm.savefig(out / "arc_compare_movement.png", dpi=100)
        L += ["## Movement panel (predictions of the frozen Phase 4 geometry model; accuracy on these films is UNMEASURED)", "",
              "Each film is drawn on its OWN axes; movement-series LEVELS are never compared across the two films (model-based series, plan section 1.4), only each film's shape over its running time. "
              "The predicted sets are INFLATED: " + ", ".join(f"{100 * v:.1f}% of {f}'s shots" for f, v in shares.items()) + " have a contradictory set (an opposite pair or `static` together with another atom), which no real camera move produces. "
              f"Windows with fewer than 15 shots are marked x. Every value is a prediction of unmeasured accuracy; {own} is provisional and this file is local only.", "", "![movement](arc_compare_movement.png)", ""]
    L += ["", "Reading notes: a series that is almost constant in one film (for example a share that is about 0 in every window but one) makes its z-scored shape meaningless; read the level panel. "
          "" if mv_ok else "The movement panel is not shown: a film lacks a movement table.", "", "![arc compare](arc_compare.png)"]
    (out / "arc_compare.md").write_text("\n".join(L) + "\n")
    print(f"wrote {out}/arc_compare.md and arc_compare.png")


if __name__ == "__main__":
    main()
