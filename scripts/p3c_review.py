# ruff: noqa: ISC004
"""Phase 3C step 4: review aids for the film predictions (no labels; everything here is a prediction summary or an AGREEMENT with a heuristic, never accuracy).
(a) predicted label distribution vs the ShotQA train prior (results/phase3c/review.md), (b) 8 contact sheets x 12 random shots under
data/interim/<film>/p3c_review/ (gitignored), (c) agreement of predictions with overlapping Phase 1/2 heuristic columns.
Usage: uv run python scripts/p3c_review.py --film night_of_the_living_dead"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from cgt.p3b_vlm.data import load_chat_samples
from cgt.p3c_tokens.selection import gold_labels
from cgt.p3c_tokens.vocab import ATTRS, MULTI, labels

ROOT = Path(__file__).resolve().parents[1]
RARE = 0.005  # a label predicted for < 0.5% of shots is flagged "almost never predicted"


def md(ct: pd.DataFrame) -> str:
    """Crosstab as a markdown table (no tabulate dependency)."""
    head = "| " + (ct.index.name or "") + " \\ " + (ct.columns.name or "") + " | " + " | ".join(map(str, ct.columns)) + " |"
    rows = ["| " + str(i) + " | " + " | ".join(str(v) for v in r) + " |" for i, r in zip(ct.index, ct.to_numpy())]
    return "\n".join([head, "|---" * (ct.shape[1] + 1) + "|", *rows])


def prior_from_shotqa() -> dict[str, dict[str, float]]:
    """Label frequencies in ShotQA TRAIN answers (each label of an answer's set counted once), normalised per attribute."""
    cnt = {a: Counter() for a in ATTRS}
    for s in load_chat_samples("train"):
        for lab in gold_labels(s):
            cnt[s["attribute"]][lab] += 1
    return {a: {lab: cnt[a][lab] / max(1, sum(cnt[a].values())) for lab in labels(a)} for a in ATTRS}


def distribution_section(df: pd.DataFrame, prior: dict) -> list[str]:
    L = ["## (a) Predicted label distribution (kf_mid primary label) vs the ShotQA train prior", "",
         "Prior = share of each label among ShotQA train answers (a multi-label answer counts each member). NOTLD shares = share of the "
         f"{len(df)} shots whose primary label it is (set share = share of shots whose predicted set contains it). Flag: predicted for < {100 * RARE:.1f}% of shots.", ""]
    for a in ATTRS:
        vc = df[f"{a}_pred"].value_counts()
        rows = ["| label | ShotQA prior % | NOTLD primary % | n | " + ("NOTLD in-set % | " if a in MULTI else "") + "flag |",
                "|---|---|---|---|" + ("---|" if a in MULTI else "") + "---|"]
        for lab in labels(a):
            n = int(vc.get(lab, 0))
            share = n / len(df)
            flag = "NEVER" if n == 0 else ("rare" if share < RARE else "")
            ins = f" {100 * df[f'{a}_set'].map(lambda s, lab=lab: lab in s).mean():.1f} |" if a in MULTI else ""
            rows.append(f"| {lab} | {100 * prior[a][lab]:.1f} | {100 * share:.1f} | {n} |{ins} {flag} |")
        mean_set = f"; mean predicted set size {df[f'{a}_set'].map(len).mean():.2f}" if a in MULTI else ""
        L += [f"### {a}", "", *rows, "", f"Mean top probability {df[f'{a}_p1'].mean():.2f}, mean margin {df[f'{a}_margin'].mean():.2f}{mean_set}.", ""]
    return L


def consistency_section(df: pd.DataFrame) -> list[str]:
    L = ["## Keyframe self-consistency (kf_start / kf_mid / kf_end agree on the primary label)", "",
         "Consistency is NOT accuracy: a model can be consistently wrong, and shots with camera or subject motion legitimately differ between keyframes. "
         "Chance level for k labels drawn from the model's own marginal is shown for context (sum of squared marginal^2 across 3 independent frames).", "",
         "| attribute | consistent % | mid-vs-start % | mid-vs-end % | chance-consistent % (independent frames) |", "|---|---|---|---|---|"]
    for a in ATTRS:
        p = df[f"{a}_pred"].value_counts(normalize=True)
        chance = float((p ** 3).sum())
        L.append(f"| {a} | {100 * df[f'{a}_consistent'].mean():.1f} | {100 * (df[f'{a}_pred'] == df[f'{a}_pred_start']).mean():.1f} | "
                 f"{100 * (df[f'{a}_pred'] == df[f'{a}_pred_end']).mean():.1f} | {100 * chance:.1f} |")
    L += ["", "| multi-label | mean pairwise Jaccard of the predicted sets |", "|---|---|"]
    for a in MULTI:
        L.append(f"| {a} | {df[f'{a}_set_jaccard'].mean():.2f} |")
    return L + [""]


def heuristic_section(df: pd.DataFrame, p1: pd.DataFrame, p2: pd.DataFrame) -> list[str]:
    d = df.merge(p1[["shot_id", "key", "contrast", "len_s"]], on="shot_id").merge(p2[["shot_id", "people", "faces", "light_dir", "dof", "light_asym"]], on="shot_id")
    L = ["## (c) Agreement of predictions with the Phase 1/2 heuristics that overlap in meaning", "",
         "The heuristics are themselves unverified and uncalibrated (status log); this table shows AGREEMENT between two unlabelled systems, not accuracy of either. "
         "'Compatible' = the predicted label is one that the heuristic reading makes plausible, defined here in advance: "
         "people=0 -> establishing/insert; people=1 -> single/insert/ots; people>=2 -> 2-shot/3-shot/group/ots/establishing.", ""]
    d["people_g"] = d["people"].clip(upper=2).map({0: "0", 1: "1", 2: "2+"})
    ct = pd.crosstab(d["people_g"], d["framing_pred"])
    L += ["### people count (heuristic) x predicted framing (counts)", "", md(ct), ""]
    comp = {"0": {"establishing", "insert"}, "1": {"single", "insert", "ots"}, "2+": {"2-shot", "3-shot", "group", "ots", "establishing"}}
    ok = d.apply(lambda r: r["framing_pred"] in comp[r["people_g"]], axis=1)
    L += ["| people (heuristic) | n shots | framing prediction compatible % |", "|---|---|---|"]
    for g in ("0", "1", "2+"):
        m = d["people_g"] == g
        L.append(f"| {g} | {int(m.sum())} | {100 * ok[m].mean():.1f} |")
    L += ["", "### key (Phase 1 luminance) x predicted lighting conditions (share of shots whose predicted SET contains the label)", "",
          "| key | n | high-contrast | low-contrast | soft | hard | silhouette |", "|---|---|---|---|---|---|---|"]
    for k in ("low-key", "mid-key", "high-key"):
        m = d["key"] == k
        if m.sum():
            L.append(f"| {k} | {int(m.sum())} | " + " | ".join(f"{100 * d.loc[m, 'light_cond_set'].map(lambda s, x=x: x in s).mean():.1f}" for x in
                                                                ("high-contrast", "low-contrast", "soft", "hard", "silhouette")) + " |")
    hc = d["light_cond_set"].map(lambda s: "high-contrast" in s)
    lc = d["light_cond_set"].map(lambda s: "low-contrast" in s)
    L += ["", f"Mean Phase 1 `contrast` (RMS-style, higher = more contrast): shots with high-contrast in the predicted set {d.loc[hc, 'contrast'].mean():.2f} (n={int(hc.sum())}); "
          f"low-contrast {d.loc[lc, 'contrast'].mean():.2f} (n={int(lc.sum())}); neither {d.loc[~hc & ~lc, 'contrast'].mean():.2f}. "
          f"Spearman(contrast, p(high-contrast) from the score vector): {pd.Series(d['contrast']).corr(d['light_cond_scores'].map(lambda v: v[labels('light_cond').index('high-contrast')]), method='spearman'):.2f}.", ""]
    L += ["### Phase 2 light direction (only where a face was found) x predicted lighting", "",
          "| light_dir | n | side in set % | back/edge/silhouette in set % |", "|---|---|---|---|"]
    for k in ("front", "side", "back"):
        m = d["light_dir"] == k
        if m.sum():
            L.append(f"| {k} | {int(m.sum())} | {100 * d.loc[m, 'light_cond_set'].map(lambda s: 'side' in s).mean():.1f} | "
                     f"{100 * d.loc[m, 'light_cond_set'].map(lambda s: bool({'back', 'edge', 'silhouette'} & set(s))).mean():.1f} |")
    L += ["", "### depth-of-field heuristic x predicted lens (counts)", "", md(pd.crosstab(d["dof"], d["lens_pred"])), "",
          "### Phase 2 faces found x predicted size (counts; faces>0 expected mostly at CU-MS)", "",
          md(pd.crosstab(d["faces"].clip(upper=1).map({0: "no face", 1: "face"}), d["size_pred"])), ""]
    return L


def contact_sheets(df: pd.DataFrame, shots: pd.DataFrame, out_dir: Path, n_sheets: int = 8, per: int = 12, seed: int = 0) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pick = np.random.default_rng(seed).permutation(len(df))[: n_sheets * per]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    except OSError:
        font = ImageFont.load_default()
    tw, th, cols = 200, 150, 2
    cell_w, cell_h = 3 * tw + 8, th + 52
    names = []
    for k in range(n_sheets):
        rows = [df.iloc[i] for i in pick[k * per:(k + 1) * per]]
        sheet = Image.new("RGB", (cols * cell_w, (per // cols) * cell_h), (24, 24, 24))
        dr = ImageDraw.Draw(sheet)
        for j, r in enumerate(rows):
            x0, y0 = (j % cols) * cell_w, (j // cols) * cell_h
            srow = shots[shots.shot_id == r.shot_id].iloc[0]
            for t, kf in enumerate(("kf_start", "kf_mid", "kf_end")):
                im = Image.open(ROOT / srow[kf]).convert("RGB").resize((tw, th))
                sheet.paste(im, (x0 + t * tw, y0))
            tok = (f"#{int(r.shot_id)} {srow.start_s:.0f}s len {srow.len_s:.1f}s  size {r.size_pred} lens {r.lens_pred} angle {r.angle_pred} frm {r.framing_pred}")
            tok2 = (f"light {'+'.join(r.light_type_set)} | {'+'.join(r.light_cond_set)} | comp {'+'.join(r.composition_set)}")
            dr.text((x0 + 2, y0 + th + 4), tok, fill=(235, 235, 235), font=font)
            dr.text((x0 + 2, y0 + th + 20), tok2, fill=(180, 220, 255), font=font)
            cons = [a for a in ("size", "lens", "angle", "framing") if not r[f"{a}_consistent"]]
            dr.text((x0 + 2, y0 + th + 36), "keyframes disagree: " + (",".join(cons) if cons else "none (4 single-label)"), fill=(255, 200, 120), font=font)
        p = out_dir / f"sheet_{k + 1:02d}.jpg"
        sheet.save(p, quality=88)
        names.append(str(p.relative_to(ROOT)))
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(ROOT / f"data/processed/{a.film}_p3c.parquet")
    shots = pd.read_parquet(ROOT / f"data/processed/{a.film}_shots.parquet")
    p2 = pd.read_parquet(ROOT / f"data/processed/{a.film}_p2.parquet")
    prior = prior_from_shotqa()
    L = [f"# Phase 3C review aids: {a.film} ({len(df)} shots). Predictions only; no accuracy is known on this film.", "",
         f"Source: run key `{df['p3c_run_key'].iloc[0]}`, method `{df['p3c_method'].iloc[0]}` (see `results/phase3c/report.md` for the code SHA, adapter hash and the determinism check against the first run).", ""]
    L += distribution_section(df, prior) + consistency_section(df) + heuristic_section(df, shots, p2)
    sheets = contact_sheets(df, shots, ROOT / f"data/interim/{a.film}/p3c_review")
    L += ["## (b) Contact sheets", "", "8 sheets x 12 random shots (seed 0; 3 keyframes start/mid/end + predicted tokens), saved under "
          f"`data/interim/{a.film}/p3c_review/` (gitignored, frames are not committed): " + ", ".join(f"`{Path(s).name}`" for s in sheets) + "."]
    out = ROOT / "results/phase3c/review.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
