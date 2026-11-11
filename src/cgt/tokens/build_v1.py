"""Build the v1 token table of a film: soft-vote primaries from the Phase 3C per-keyframe score vectors, length bins, global palette.
CPU only. Inference targets only: nothing here is trained or tuned on the film."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cgt.tokens import build, palette, v1
from cgt.tokens import palette_global as pg
from cgt.tokens.build import IncompletePhase3C, require_complete_p3c, sha256_file

ROOT = Path(__file__).resolve().parents[3]
KFS = ("kf_start", "kf_mid", "kf_end")
CODE_FILES = ["v1.py", "build_v1.py", "palette_global.py", "palette.py", "format.py", "build.py"]


def run_dir(film: str, p3c: pd.DataFrame, root: Path) -> Path:
    return root / "data/cache/p3c/runs" / f"{film}_{p3c['p3c_run_key'].iloc[0]}"


def load_scores(rd: Path) -> dict[str, dict[int, dict]]:
    return {kf: {r["shot_id"]: r for r in map(json.loads, (rd / f"scores_{kf}.jsonl").read_text().splitlines())} for kf in KFS}


def check_complete(film: str, root: Path) -> pd.DataFrame:
    paths = build.input_paths(film, root)
    p3c = pd.read_parquet(paths["p3c"])
    require_complete_p3c(film, p3c, len(pd.read_parquet(paths["shots"], columns=["shot_id"])), root)
    return p3c


def build_tokens_v1(film: str, root: Path = ROOT, centroids: np.ndarray | None = None) -> pd.DataFrame:
    p3c = check_complete(film, root)
    paths = build.input_paths(film, root)
    shots, p2 = pd.read_parquet(paths["shots"]), pd.read_parquet(paths["p2"])
    if set(shots.shot_id) != set(p2.shot_id) or set(shots.shot_id) != set(p3c.shot_id):
        raise ValueError("the Phase 1, 2 and 3C tables cover different shots")
    cfg = v1.load_config(root / "configs/tokens.yaml")
    cents = pg.load(root / cfg["palette"]["centroids"]) if centroids is None else centroids
    labels_ = build.labels_by_attr()
    scores = load_scores(run_dir(film, p3c, root))
    for kf in KFS:
        if set(scores[kf]) != set(shots.shot_id):
            raise IncompletePhase3C(f"scores for {kf} do not cover every shot")
    df = shots[["film_id", "shot_id", "start_s", "end_s", "len_s", "kf_start", "kf_mid", "kf_end", "key"]].copy().sort_values("shot_id").reset_index(drop=True)
    df["start_min"] = df["start_s"] / 60.0
    df["len_bin"] = [v1.len_bin(x, cfg["len_bins"]["edges"]) for x in df["len_s"]]
    sh = shots.set_index("shot_id").loc[df.shot_id]
    df["palette_global"] = pg.assign(np.stack([np.asarray(p, dtype=float) for p in sh["palette_lab"]]), cents)
    df["len_tok"] = [v1.len_token(x, cfg["len_bins"]["names"]) for x in df["len_s"]]
    df["palette_tok"] = [v1.palette_token(g) for g in df["palette_global"]]
    df["palette_film"] = palette.fit_palette_ids([list(p) for p in sh["palette_lab"]])
    df = df.merge(p2[["shot_id", "dof", "people", "sound"]], on="shot_id", how="left", validate="1:1")
    cols: dict[str, list] = {}
    for a in build.P3C_ATTRS:
        labs = labels_[a]
        prim, p1, mg, cons, kfmid, p1mid, probs, votes = [], [], [], [], [], [], [], []
        for sid in df.shot_id:
            vs = [scores[kf][int(sid)][a] for kf in KFS]
            mean = v1.soft_vote(vs)
            i, a1, a2 = v1.primary_and_margin(mean)
            per = [int(np.argmax(v)) for v in vs]
            prim.append(labs[i]); p1.append(a1); mg.append(a2); probs.append([float(x) for x in mean])
            cons.append(len(set(per)) == 1)
            kfmid.append(labs[per[1]]); p1mid.append(float(v1.softmax(vs[1]).max())); votes.append(int(np.argmax(v1.softmax(vs[1])) == i))
        cols.update({a: prim, f"{a}_p1": p1, f"{a}_margin": mg, f"{a}_consistent": cons, f"{a}_kfmid": kfmid,
                     f"{a}_p1_kfmid": p1mid, f"{a}_probs_mean": probs})
    df = pd.concat([df, pd.DataFrame(cols)], axis=1)
    df = _add_movement(df, film, root)
    cons = df[[f"{a}_consistent" for a in v1.STATIC]]
    df["n_kf_disagree"] = (~cons.astype(bool)).sum(axis=1).astype(int)
    df["kf_all_agree"] = df["n_kf_disagree"] == 0
    df["token_row"] = [v1.format_row(r) for r in df.to_dict("records")]
    df["p3c_run_key"] = p3c["p3c_run_key"].iloc[0]
    df["tokens_version"] = v1.VERSION
    return df


def move_path(film: str, root: Path) -> Path:
    return root / f"data/processed/{film}_move.parquet"


def _add_movement(df: pd.DataFrame, film: str, root: Path) -> pd.DataFrame:
    """Movement slot from the Phase 4 table `<film>_move.parquet` (a PREDICTION of unmeasured accuracy), when it exists; otherwise the slot is `n/a` for every shot.
    The table must cover exactly the film's shots and its atoms must be schema move classes; `move_atoms` is a string of atoms joined by ' + ' in schema order."""
    mp = move_path(film, root)
    df["move_present"] = mp.exists()
    if not mp.exists():
        df["move_atoms"] = [None] * len(df)
        df["move"] = v1.move_token(None, False)
        return df
    mv = pd.read_parquet(mp)
    atoms_ok = v1.schema_move_atoms()
    if set(mv.shot_id) != set(df.shot_id) or mv.shot_id.duplicated().any():
        raise ValueError("the movement table does not cover exactly the film's shots")
    parsed = {}
    for sid, txt in zip(mv.shot_id, mv["move_atoms"], strict=True):
        atoms = [a.strip() for a in str(txt).split("+")] if isinstance(txt, str) and txt.strip() else []
        bad = set(atoms) - set(atoms_ok)
        if not atoms or bad:
            raise ValueError(f"shot {sid}: movement atoms {atoms} are empty or outside the schema move classes ({sorted(bad)})")
        parsed[int(sid)] = sorted(atoms, key=atoms_ok.index)  # schema order, unordered set
    df["move_atoms"] = [parsed[int(s)] for s in df.shot_id]
    df["move"] = [v1.move_token(a, True) for a in df["move_atoms"]]
    extra = mv.set_index("shot_id").loc[df.shot_id]
    df["move_n_atoms"] = [len(a) for a in df["move_atoms"]]
    for c in ("fallback_top1", "short_lt_1s", "few_pairs_lt_8", "no_features"):
        if c in extra:
            df[f"move_{c}"] = extra[c].to_numpy()
    for c in extra.columns:
        if c.startswith("p_"):
            df[f"move_{c}"] = extra[c].to_numpy()
    df["move_model"] = str(extra["model"].iloc[0]) if "model" in extra else "unknown"
    return df


def input_hashes(film: str, root: Path = ROOT) -> dict:
    p = build.input_paths(film, root)
    h = {k: sha256_file(v) for k, v in p.items()}
    cfg_path = root / "configs/tokens.yaml"
    h["config"] = sha256_file(cfg_path)
    h["palette_centroids"] = sha256_file(root / v1.load_config(cfg_path)["palette"]["centroids"])
    key = pd.read_parquet(p["p3c"], columns=["p3c_run_key"])["p3c_run_key"].iloc[0]
    rd = root / "data/cache/p3c/runs" / f"{film}_{key}"
    for kf in KFS:
        h[f"scores_{kf}"] = sha256_file(rd / f"scores_{kf}.jsonl")
    mp = move_path(film, root)
    h["movement"] = sha256_file(mp) if mp.exists() else "absent"
    h["code"] = hashlib.sha256("".join(sha256_file(Path(__file__).parent / f) for f in CODE_FILES).encode()).hexdigest()
    return h


def write_tokens_v1(film: str, root: Path = ROOT) -> tuple[Path, Path, bool]:
    """Idempotent; the Phase 3C completion check runs before any cached return."""
    check_complete(film, root)
    out = root / f"data/processed/{film}_tokens_v1.parquet"
    txt = root / f"data/processed/{film}_tokens_v1.txt"
    meta = root / f"data/processed/{film}_tokens_v1.meta.json"
    hashes = input_hashes(film, root)
    if out.exists() and txt.exists() and meta.exists() and json.loads(meta.read_text()).get("inputs") == hashes:
        return out, txt, False
    df = build_tokens_v1(film, root)
    df.to_parquet(out, index=False)
    txt.write_text("\n".join(build.text_lines(df)) + "\n")
    meta.write_text(json.dumps({"inputs": hashes, "rows": len(df), "tokens_version": v1.VERSION}, indent=1))
    return out, txt, True
