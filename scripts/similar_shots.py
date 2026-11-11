# ruff: noqa: ISC004
"""Similar shots on a PUBLIC-DOMAIN film's v1 token table: the pre-declared descriptive analysis of docs/similar_shots_spec.md. CPU only, seconds.
Writes results/phase6/similar_shots_<film>.{md,csv}. Refuses any film that is not public_domain: true in the committed configs/films.yaml (or is in the local overlay):
nothing about an owned film is ever written under results/.
Usage: uv run python scripts/similar_shots.py --film night_of_the_living_dead [--root DATA_ROOT] [--centroids PATH] [--expect-sha SHA256]"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cgt.pipeline import registry
from cgt.tokens import palette_global as pg
from cgt.tokens import similar

ROOT = Path(__file__).resolve().parents[1]


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def public_film(film: str, code_root: Path = ROOT) -> bool:
    main = yaml.safe_load((code_root / registry.MAIN).read_text()) or {}
    overlay = {e.get("id") for e in registry.read_overlay(code_root)}
    return any(f.get("id") == film and f.get("public_domain") is True for f in main.get("films") or []) and film not in overlay


def mmss(sec: float) -> str:
    s = round(float(sec))
    return f"{s // 60}:{s % 60:02d}"


def cell(row: str) -> str:
    """A token row inside a markdown table cell: its ' | ' separators are escaped so the table keeps its columns."""
    return row.replace("|", "\\|")


def fmt_stats(st: dict) -> list[str]:
    return [f"| n | {st['n']:,} |", f"| min / q25 / median / q75 / max (min) | {st['min']:.2f} / {st['q25']:.2f} / {st['median']:.2f} / {st['q75']:.2f} / {st['max']:.2f} |",
            f"| mean (min) | {st['mean']:.2f} |",
            "| share > 1 / 5 / 10 / 30 min | " + " / ".join(f"{100 * st[f'share_gt_{t}']:.1f}%" for t in (1, 5, 10, 30)) + " |",
            "| histogram " + ", ".join(similar.GAP_BIN_NAMES) + " min | " + ", ".join(f"{v:,}" for v in st["hist"].values()) + " |"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default="night_of_the_living_dead")
    ap.add_argument("--root", type=Path, default=ROOT, help="data root holding data/processed/<film>_tokens_v1.parquet (read only)")
    ap.add_argument("--centroids", type=Path, default=None, help="frozen palette centroids (default ROOT/data/cache/tokens/palette_global_k32.npz); required")
    ap.add_argument("--expect-sha", default=None, help="refuse unless the token table has this sha256")
    a = ap.parse_args()
    if not public_film(a.film):
        raise SystemExit(f"{a.film} is not a public-domain film of configs/films.yaml: nothing is written under results/")
    tp = a.root / f"data/processed/{a.film}_tokens_v1.parquet"
    tsha = sha256(tp)
    if a.expect_sha and tsha != a.expect_sha:
        raise SystemExit(f"token table sha256 {tsha} != expected {a.expect_sha}")
    cp = a.centroids or (a.root / "data/cache/tokens/palette_global_k32.npz")
    if not cp.exists():
        raise SystemExit(f"palette centroids {cp} missing: the analysis requires palette_mode=centroid")
    meta = tp.with_name(f"{a.film}_tokens_v1.meta.json")
    csha = sha256(cp)
    if meta.exists():
        want = json.loads(meta.read_text())["inputs"].get("palette_centroids")
        if want and want != csha:
            raise SystemExit(f"centroids {csha[:12]} are not the ones the token table was built with ({want[:12]})")
    cents = pg.load(cp)
    df = pd.read_parquet(tp).sort_values("shot_id").reset_index(drop=True)
    top = similar.top_k_table(df, k=5, centroids=cents)
    assert top.attrs["palette_mode"] == "centroid"
    t1 = top[top["rank"] == 1]
    st = similar.gap_stats(t1["gap_min"])
    ref = similar.gap_stats(similar.all_candidate_gaps(df["start_s"].to_numpy()))
    sc = t1["score"].to_numpy()
    ties = int(t1["tie_next"].sum())
    mutual = similar.mutual_top1_share(top)
    pairs = similar.long_range_pairs(top, 10.0, 10)
    n_long = int((t1["gap_min"] > 10).sum())
    n_long_pairs = len(similar.long_range_pairs(top, 10.0, 10**9))
    rows = df.set_index("shot_id")["token_row"]
    starts = df.set_index("shot_id")["start_s"]

    out = ROOT / "results/phase6"
    out.mkdir(parents=True, exist_ok=True)
    top.to_csv(out / f"similar_shots_{a.film}.csv", index=False, float_format="%.6f")
    L = [f"# Similar shots within the film: {a.film}",
         "",
         "Pre-declared descriptive analysis of docs/similar_shots_spec.md (definition committed before the code). Every slot value is a MODEL PREDICTION or an unverified heuristic "
         "whose accuracy on this film is UNMEASURED: a high score means the pipeline wrote similar tokens, not that the shots look alike, and a long-range pair is NOT evidence of a "
         "deliberate visual callback. Equal slot weights and the slot rules are choices, not findings. No p-values; description only.",
         "",
         f"Inputs: token table sha256 `{tsha}`; frozen palette centroids file sha256 `{csha}` (the value in the table's meta.json; ShotQA-derived, not committed). "
         f"Slots used: {top.attrs['n_slots']} (movement {'included' if top.attrs['move_used'] else 'absent'}), palette_mode `{top.attrs['palette_mode']}`. "
         f"The shot itself and its +-{similar.WINDOW} neighbours are never candidates; S is rounded to {similar.S_DECIMALS} decimals before ranking so mathematically equal scores tie, and ties go to the smaller shot id. CSV: `similar_shots_{a.film}.csv` (top-5 per shot).",
         "",
         "## Time gap from each shot to its top-1 similar shot (minutes)",
         "",
         "| statistic | top-1 partner | reference: all eligible candidate pairs |",
         "|---|---|---|"]
    for x, y in zip(fmt_stats(st), fmt_stats(ref), strict=True):
        name, v1_ = x.strip("|").split("|", 1)
        L.append(f"| {name.strip()} | {v1_.strip(' |')} | {y.strip('|').split('|', 1)[1].strip(' |')} |")
    L += ["",
          "The reference column is what the gap would look like if the partner were drawn uniformly from the eligible candidates; it is shown for scale only (no test).",
          "",
          f"Top-1 score S (0-1): min {sc.min():.3f}, median {np.median(sc):.3f}, max {sc.max():.3f}. "
          f"Shots whose top-1 score ties with their next candidate (the smaller-id tie-break decides): {ties} / {len(t1):,}. "
          f"Mutual top-1 pairs (a's top-1 is b and b's is a): {100 * mutual:.1f}% of shots. Shots with a top-1 gap > 10 min: {n_long:,} ({n_long_pairs:,} unordered pairs).",
          "",
          "## The 10 strongest long-range pairs (top-1 pairs with gap > 10 min, counted once, S descending)",
          "",
          "| # | shot a | start a | shot b | start b | gap (min) | S | mutual | token row a | token row b |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(pairs.itertuples(), 1):
        L.append(f"| {i} | {int(r.a)} | {mmss(starts[r.a])} | {int(r.b)} | {mmss(starts[r.b])} | {r.gap_min:.1f} | {r.score:.3f} | {'yes' if r.mutual else 'no'} | "
                 f"`{cell(rows[r.a])}` | `{cell(rows[r.b])}` |")
    L += ["",
          "Inside the table the row separators are escaped as `\\|`. A `*` marks an attribute whose start/mid/end keyframes disagree (it counts half in the score). Times are shot starts (mm:ss) in the film file used by the pipeline.",
          ""]
    (out / f"similar_shots_{a.film}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
