# ruff: noqa: ISC004
"""Visual-arc analysis #1 on a film's v1 token table (Phase 6 start). CPU only. Writes results/phase6/visual_arc_<film>.{md,csv,png}.
Descriptive series of PREDICTIONS and unverified heuristics; no accuracy, nothing tuned on the film. Usage: uv run python scripts/visual_arc.py --film night_of_the_living_dead [--note "<refresh note>"]\nThe NOTLD md committed on 2026-10-14 was written with --note set to its third line (the refresh note after the F2 audio fix)."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cgt.tokens import arc

ROOT = Path(__file__).resolve().parents[1]
PANELS = [("Shot rhythm (heuristic): shot starts/min and median length", ["cuts_per_min", "median_len_s"]),
          ("Scale (model: size)", ["close_share", "mean_size_ordinal"]),
          ("Tone and light", ["low_key_share", "high_contrast_share", "daylight_share"]),
          ("People and sound (heuristics)", ["people_mean", "multi_person_share", "dialogue_share"]),
          ("Angle (model; no eye-level class) and keyframe stability", ["low_angle_share", "keyframes_all_agree_share"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default="night_of_the_living_dead")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--note", default=None, help="optional refresh note, written as its own paragraph under the md title (so a rerun reproduces a refreshed md exactly)")
    a = ap.parse_args()
    df = pd.read_parquet(ROOT / f"data/processed/{a.film}_tokens_v1.parquet")
    tbl = arc.series_table(df, a.minutes)
    sp = arc.spearman_with_time(tbl).sort_values("spearman_with_time")
    out = ROOT / "results/phase6"
    out.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out / f"visual_arc_{a.film}.csv", index=False)

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(10, 3 * len(PANELS)), sharex=True)
    for ax, (title, names) in zip(axes, PANELS):
        for n in names:
            d = tbl[tbl.series == n]
            x = d.window_start_min + a.minutes / 2
            ax.plot(x, d.value, marker="o", ms=3, label=n)
            if d.ci_lo.notna().all():
                ax.fill_between(x, d.ci_lo, d.ci_hi, alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("minutes into the film (window centre)")
    fig.suptitle(f"{a.film}: visual arc, {a.minutes:.0f}-min windows. Model predictions and unverified heuristics; accuracy unmeasured.\n"
                 "Mean-based series: 95% bootstrap intervals over each window's shots; median length: within-window interquartile range; shot-start rate: no interval", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / f"visual_arc_{a.film}.png", dpi=110)

    # robustness: recompute the model-based series with the kf_mid label instead of the soft vote
    alt = df.assign(size=df["size_kfmid"], light_cond=df["light_cond_kfmid"], light_type=df["light_type_kfmid"], angle=df["angle_kfmid"])
    t2 = arc.series_table(alt, a.minutes, n_boot=10)
    rob = []
    for n in ("close_share", "mean_size_ordinal", "high_contrast_share", "daylight_share", "low_angle_share"):
        v1_, v2 = tbl[tbl.series == n].value.to_numpy(), t2[t2.series == n].value.to_numpy()
        rob.append((n, float(np.corrcoef(v1_, v2)[0, 1]), float(np.abs(v1_ - v2).max())))

    mv = None
    if "move_atoms" in df.columns and bool(df["move_present"].iloc[0]):
        mv = arc.movement_series_table(df, a.minutes)
        mv.to_csv(out / f"visual_arc_{a.film}_movement.csv", index=False)
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, (title, names) in zip(axes2, [("Static share and set size (model prediction)", ["move_static_share", "move_static_alone_share"]),
                                              ("Mean predicted set size", ["move_mean_set_size"]),
                                              ("Atom-group shares (model prediction)", [f"move_{g}_share" for g in arc.MOVE_GROUPS])]):
            for n in names:
                d = mv[mv.series == n]
                ax.plot(d.window_start_min + a.minutes / 2, d.value, marker="o", ms=3, label=n.replace("move_", ""))
                ax.fill_between(d.window_start_min + a.minutes / 2, d.ci_lo, d.ci_hi, alpha=0.12)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7, ncol=3)
            ax.grid(alpha=0.3)
        axes2[-1].set_xlabel("minutes into the film (window centre)")
        fig2.suptitle(f"{a.film}: predicted camera movement (Phase 4 geometry model; accuracy on this film unmeasured). 95% bootstrap over each window's shots", fontsize=9)
        fig2.tight_layout(rect=(0, 0, 1, 0.97))
        fig2.savefig(out / f"visual_arc_{a.film}_movement.png", dpi=110)

    n_win = tbl.window.nunique()
    fast = tbl[tbl.series == "cuts_per_min"].sort_values("value")
    L = [f"# Visual arc #1: {a.film} ({len(df)} shots, {n_win} windows of {a.minutes:.0f} min)", ""] + ([a.note, ""] if a.note else []) + [
         "**Everything here is descriptive.** Model-based series (scale, light, angle) are predictions whose accuracy on this film is unmeasured; heuristic series (rhythm, key, people, sound) are unverified Phase 1/2 outputs. "
         "Windows hold between {} and {} shots. Mean-based series show 95% bootstrap intervals from resampling each window's shots; median shot length shows its within-window interquartile range; "
         "shot-start rate has no interval. None measures label accuracy. `cuts_per_min` counts shot STARTS per minute (including the film's first shot), not strictly inter-shot cuts. "
         "No threshold, window width or series was tuned on the film. The figure PNG is not tracked in git: regenerate it with `scripts/visual_arc.py`.".format(
             tbl.groupby("window").n_shots.first().min(), tbl.groupby("window").n_shots.first().max()), "",
         f"![visual arc](visual_arc_{a.film}.png)", "", "## Drift with time (Spearman of the window series with window position; descriptive, not a test)", "",
         "| series | source | Spearman with time |", "|---|---|---|"]
    L += [f"| {r.series} | {r.kind} | {r.spearman_with_time:+.2f} |" for r in sp.itertuples()]
    small = tbl.groupby("window").agg(n=("n_shots", "first"), t=("window_start_min", "first"))
    small = small[small.n < 10]
    if len(small):
        L += ["", "Windows with fewer than 10 shots (intervals there are very wide; the last window is partial): " + ", ".join(f"min {int(r.t)} (n={int(r.n)})" for r in small.itertuples()) + "."]
    L += ["", "## Rhythm", "", "Fastest windows by shot starts per minute: " + ", ".join(f"min {int(r.window_start_min)}-{int(r.window_start_min + a.minutes)} ({r.value:.1f})" for r in fast.tail(3).itertuples()) +
          "; slowest: " + ", ".join(f"min {int(r.window_start_min)}-{int(r.window_start_min + a.minutes)} ({r.value:.1f})" for r in fast.head(3).itertuples()) + ". The fast windows around minutes 65-85 and the slow ones around 25-30 and 55-60 match the earlier Phase 1 rolling-ASL observation (status log), which used the same cut table, so this is not independent evidence.", "",
          "## Robustness of the model-based series to the keyframe aggregation", "",
          "Correlation (across windows) and largest window difference between the soft-vote series and the same series from the kf_mid label only:", "",
          "| series | correlation | max abs window difference |", "|---|---|---|"]
    L += [f"| {n} | {c:.2f} | {m:.3f} |" for n, c, m in rob]
    L += ["", f"Keyframe stability: all seven attributes agree across the three keyframes in {df.kf_all_agree.mean() * 100:.1f}% of shots; the share per window is the last series in the figure. "
          "Series built from labels that disagree between keyframes should be read as trends in what the model tends to say, not as measurements of the film.", "",
          "## Not claimed", "", "No accuracy, no causal or narrative claim, no comparison with other films. Narrative readings (acts, climax) need the human-checked labels and a transcript; the NOTLD subtitle file is unreliable (status log)."]
    if mv is not None:
        sizes = df["move_n_atoms"].value_counts().sort_index()
        L += ["", "## Movement panel (predictions of the Phase 4 geometry model; accuracy on this film is unmeasured)", "",
              f"![movement](visual_arc_{a.film}_movement.png)", "",
              f"Model `{df['move_model'].iloc[0]}`, table `{a.film}_move.parquet` (run 2). The slot is an unordered atom set; shots whose set contains `static`: {100 * df['move_atoms'].map(lambda x: 'static' in x).mean():.1f}%, "
              f"`static` alone: {100 * df['move_atoms'].map(lambda x: list(x) == ['static']).mean():.1f}%, median set size {df['move_n_atoms'].median():.0f}; set-size counts: "
              + ", ".join(f"{k} atoms {v}" for k, v in sizes.items()) + ". A set that contains `static` together with motion atoms is what the registered decision rule (every atom with probability >= 0.5) produced; "
              f"it is reported as is. Shots under 1 s ({int(df['move_short_lt_1s'].sum()) if 'move_short_lt_1s' in df else 'n/a'}) and top-1 fallbacks ({int(df['move_fallback_top1'].sum()) if 'move_fallback_top1' in df else 'n/a'}) are the less reliable ones. "
              "Series: share of shots per window whose set contains the atom group, with bootstrap intervals over the window's shots; no claim about what the camera really did.",
              f"**The predicted sets are inflated:** {100 * arc.contradictory_share(df['move_atoms']):.1f}% of the shots have a contradictory set (an opposite pair such as pan-left with pan-right, or `static` together with another atom), which no real camera move produces; read the panel as the model's tendencies, never as camera work. Movement levels are not compared with any other film."]
    (out / f"visual_arc_{a.film}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
