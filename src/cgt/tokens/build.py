"""Join the Phase 1 / 2 / 3C tables into the token-row table of one film (docs/tokens_interface.md). CPU only; inference targets only (nothing is trained or tuned)."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from cgt.tokens import palette
from cgt.tokens.format import STATIC, VERSION, format_row

ROOT = Path(__file__).resolve().parents[3]
P3C_ATTRS = ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]
CODE_FILES = ["format.py", "palette.py", "build.py"]


class IncompletePhase3C(RuntimeError):
    """The Phase 3C run behind the table is not a complete, fully covered film run (fail closed)."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_complete_p3c(film: str, p3c: pd.DataFrame, n_shots: int, root: Path = ROOT) -> dict:
    keys = set(p3c["p3c_run_key"])
    if len(keys) != 1:
        raise IncompletePhase3C(f"the Phase 3C table mixes run keys {sorted(keys)}")
    key = keys.pop()
    mf = root / "data/cache/p3c/runs" / f"{film}_{key}/manifest.json"
    if not mf.exists():
        raise IncompletePhase3C(f"no Phase 3C manifest at {mf}")
    man = json.loads(mf.read_text())
    if man.get("state") != "complete" or "finished" not in man:
        raise IncompletePhase3C(f"Phase 3C run {key} is '{man.get('state')}', not complete")
    if man.get("coverage") != {"kf_mid": n_shots, "kf_start": n_shots, "kf_end": n_shots}:
        raise IncompletePhase3C(f"Phase 3C coverage {man.get('coverage')} does not cover all {n_shots} shots")
    if len(p3c) != n_shots:
        raise IncompletePhase3C(f"Phase 3C table has {len(p3c)} rows for {n_shots} shots")
    return man


def labels_by_attr() -> dict[str, list[str]]:
    """Schema vocabularies of the Phase 3C attributes (score vectors are in this order)."""
    from cgt.p3c_tokens.vocab import labels
    return {a: labels(a) for a in P3C_ATTRS}


def input_paths(film: str, root: Path = ROOT) -> dict[str, Path]:
    d = root / "data/processed"
    return {"shots": d / f"{film}_shots.parquet", "p2": d / f"{film}_p2.parquet", "p3c": d / f"{film}_p3c.parquet"}


def build_tokens(film: str, root: Path = ROOT) -> pd.DataFrame:
    paths = input_paths(film, root)
    shots, p2, p3c = (pd.read_parquet(paths[k]) for k in ("shots", "p2", "p3c"))
    require_complete_p3c(film, p3c, len(shots), root)
    for name, t in (("shots", shots), ("p2", p2), ("p3c", p3c)):
        if t.shot_id.duplicated().any():
            raise ValueError(f"duplicate shot_id in the {name} table")
    if set(shots.shot_id) != set(p2.shot_id) or set(shots.shot_id) != set(p3c.shot_id):
        raise ValueError("the Phase 1, 2 and 3C tables cover different shots")
    df = shots[["film_id", "shot_id", "start_s", "end_s", "len_s", "kf_start", "kf_mid", "kf_end", "key"]].copy()
    df["start_min"] = df["start_s"] / 60.0
    df["palette_id"] = palette.fit_palette_ids([list(p) for p in shots["palette_lab"]])
    df = df.merge(p2[["shot_id", "dof", "people", "sound"]], on="shot_id", how="left", validate="1:1")
    cols = {"shot_id": p3c["shot_id"]}
    for a in P3C_ATTRS:
        cols[a] = p3c[f"{a}_pred"]
        cols[f"{a}_p1"] = p3c[f"{a}_p1"]
        cols[f"{a}_margin"] = p3c[f"{a}_margin"]
        cols[f"{a}_consistent"] = p3c[f"{a}_consistent"]
    df = df.merge(pd.DataFrame(cols), on="shot_id", how="left", validate="1:1")
    df["move"] = None
    cons = df[[f"{a}_consistent" for a in STATIC]]
    df["n_kf_disagree"] = (cons == False).sum(axis=1).astype(int)
    df["kf_all_agree"] = df["n_kf_disagree"] == 0
    df["token_row"] = [format_row(r) for r in df.to_dict("records")]
    df["p3c_run_key"] = p3c["p3c_run_key"].iloc[0]
    df["tokens_version"] = VERSION
    return df.sort_values("shot_id").reset_index(drop=True)


def input_hashes(film: str, root: Path = ROOT) -> dict:
    h = {k: sha256_file(p) for k, p in input_paths(film, root).items()}
    h["code"] = hashlib.sha256("".join(sha256_file(Path(__file__).parent / f) for f in CODE_FILES).encode()).hexdigest()
    return h


def text_lines(df: pd.DataFrame) -> list[str]:
    """One line per shot: `<shot id> <start mm:ss.s> <token row>`."""
    out = []
    for r in df.itertuples(index=False):
        m, s = divmod(round(float(r.start_s), 1), 60.0)  # round the total first, so 3059.993 s prints 051:00.0, never 050:60.0
        out.append(f"{int(r.shot_id):05d} {int(m):03d}:{s:04.1f} {r.token_row}")
    return out


def write_tokens(film: str, root: Path = ROOT) -> tuple[Path, Path, bool]:
    """Idempotent: if the stored input/code hashes equal the current ones the outputs are left untouched (returns rebuilt=False). The Phase 3C completion check
    runs first, so a cached result never bypasses it."""
    out = root / f"data/processed/{film}_tokens.parquet"
    txt = root / f"data/processed/{film}_tokens.txt"
    meta = root / f"data/processed/{film}_tokens.meta.json"
    paths = input_paths(film, root)
    require_complete_p3c(film, pd.read_parquet(paths["p3c"]), len(pd.read_parquet(paths["shots"], columns=["shot_id"])), root)  # before any cached return
    hashes = input_hashes(film, root)
    if out.exists() and txt.exists() and meta.exists() and json.loads(meta.read_text()).get("inputs") == hashes:
        return out, txt, False
    df = build_tokens(film, root)
    df.to_parquet(out, index=False)
    txt.write_text("\n".join(text_lines(df)) + "\n")
    meta.write_text(json.dumps({"inputs": hashes, "rows": len(df), "tokens_version": VERSION, "palette": {"k": palette.K, "seed": palette.SEED}}, indent=1))
    return out, txt, True
