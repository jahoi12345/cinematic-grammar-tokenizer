"""Deterministic dev/gold shot selection. Shot ids only; no labels are read.
Usage: uv run python scripts/make_splits.py --films night_of_the_living_dead <owned_film_id> --dev 100 --gold 300
Timeline is cut into 240 s segments, each wholly dev or gold; shots within 30 s of a cross-role edge are dropped.
Writes data/{dev,gold}/shots.json.
Run AFTER the cut threshold is final: shot ids change with the threshold."""
import argparse
import json

import numpy as np
import pandas as pd

from cgt.p1_shots_color.pipeline import ROOT

GAP, MIN_LEN, SEG = 30.0, 1.0, 240.0

def roles(df, dev_frac, rng):
    """Split the film timeline into SEG-second segments, each wholly dev or gold. Random gold shots over the whole
    film cannot keep dev >= GAP s away (300 gold shots x ~60 s exclusion > film length), so separate by segment,
    then drop shots within GAP s of a segment edge that borders the other role."""
    nseg = int(df.end_s.max() // SEG) + 1
    role = np.where(rng.permutation(nseg) < round(nseg * dev_frac), "dev", "gold")
    out = []
    for r in df.itertuples():
        i, j = int(r.start_s // SEG), int((r.end_s - 1e-6) // SEG)
        if i != j: out.append(None); continue          # straddles a segment edge
        ok = role[i]
        if i > 0 and role[i - 1] != ok and r.start_s - i * SEG < GAP: ok = None
        if i + 1 < nseg and role[i + 1] != ok and role[i + 1] != role[i] and (i + 1) * SEG - r.end_s < GAP: ok = None
        out.append(ok)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--films", nargs="+", required=True)
    ap.add_argument("--dev", type=int, default=100); ap.add_argument("--gold", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    dev, gold = [], []
    for k, film in enumerate(a.films):
        df = pd.read_parquet(ROOT / f"data/processed/{film}_shots.parquet")
        df = df[df.len_s >= MIN_LEN].reset_index(drop=True)
        df["role"] = roles(df, a.dev / (a.dev + a.gold), rng)
        ng = a.gold // len(a.films) + (k < a.gold % len(a.films))
        nd = a.dev // len(a.films) + (k < a.dev % len(a.films))
        for name, n, out in (("gold", ng, gold), ("dev", nd, dev)):
            pool = df[df.role == name]
            assert len(pool) >= n, f"{film}: only {len(pool)} eligible {name} shots, need {n}"
            out += [(film, int(r.shot_id), float(r.start_s), float(r.end_s)) for r in pool.sample(n, random_state=int(rng.integers(1 << 31))).itertuples()]
    for g_ in gold:
        for d_ in dev:
            if g_[0] == d_[0]: assert d_[2] - g_[3] >= GAP or g_[2] - d_[3] >= GAP
    for name, rows in (("dev", dev), ("gold", gold)):
        (ROOT / f"data/{name}").mkdir(parents=True, exist_ok=True)
        (ROOT / f"data/{name}/shots.json").write_text(json.dumps([{"film": f, "shot_id": i, "start_s": s, "end_s": e} for f, i, s, e in rows]))
    print(f"dev={len(dev)} gold={len(gold)}")

if __name__ == "__main__":
    main()
