"""Token-table rebuild for the p3c-v2-4b default (docs/plans/default_4b.md section 4): preflight, staged build, verify, swap with
`_pre_4b` backups, and the GPU runner's pure logic (projection, drop, crash cap). Nothing here deletes a file. CPU only (the GPU work is
the child `scripts/p3c_run_film.py --model 4b`). Owned/overlay film ids come from the registry and are written only under data/."""
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from cgt.p3b_vlm import guards
from cgt.p3c_tokens import models
from cgt.pipeline import registry as films_registry

ROOT = Path(__file__).resolve().parents[3]
NOTLD = "night_of_the_living_dead"
SELECTORS = ("notld", "owned", "clip")
KINDS = ("p3c", "tokens", "txt", "meta")
WORK = "data/cache/rebuild_4b"
STOPPED_EXIT = 22                       # child: stopped at the stop time (cache kept, no table)
FAIL_CLOSED = tuple(range(3, 21))       # child refused (guards.FailClosed codes): no resume
MAX_CRASH_RESUMES = 10
SEC_PER_SHOT_W27 = 4.45                 # an earlier change pipeline check, 4B method D, 3 keyframes, buffers 10/10
LOAD_S = 0.03 * 3600
MARGIN = 1.15
DEADLINE = "2026-10-22T10:30:00+02:00"
P3C_DERIVED = ("p1", "margin", "consistent", "kfmid", "p1_kfmid", "probs_mean")


class RebuildRefused(guards.FailClosed):
    exit_code = 13


def sha(p: Path) -> str | None:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------- films
def resolve_films(selectors: list[str], root: Path = ROOT) -> list[dict]:
    """notld | owned (the single public_domain:false entry of configs/films.yaml) | clip (the single overlay entry) | an explicit id."""
    films = films_registry.load_films(root)
    out = []
    for s in selectors:
        if s == "notld":
            ids = [NOTLD] if NOTLD in films else []
        elif s == "owned":
            ids = [f for f, e in films.items() if not films_registry.is_overlay(e) and not e.get("public_domain")]
        elif s == "clip":
            ids = [f for f, e in films.items() if films_registry.is_overlay(e)]
        else:
            ids = [s] if s in films else []
        if len(ids) != 1:
            raise RebuildRefused(f"selector {s!r} resolves to {len(ids)} registered films (need exactly one)")
        e = films[ids[0]]
        local = films_registry.is_overlay(e) or not e.get("public_domain")
        out.append({"selector": s, "film": ids[0], "allow_owned": local, "local_only": local})
    return out


def label(f: dict) -> str:
    """What may be printed to a committed place: the selector for local films, the id for NOTLD."""
    return f["film"] if not f["local_only"] else f"<{f['selector']}>"


# ---------------------------------------------------------------- paths
def live_paths(film: str, root: Path = ROOT) -> dict[str, Path]:
    d = Path(root) / "data/processed"
    return {"p3c": d / f"{film}_p3c.parquet", "tokens": d / f"{film}_tokens_v1.parquet", "txt": d / f"{film}_tokens_v1.txt",
            "meta": d / f"{film}_tokens_v1.meta.json"}


def backup_paths(film: str, root: Path = ROOT) -> dict[str, Path]:
    d = Path(root) / "data/processed"
    return {"p3c": d / f"{film}_p3c_pre_4b.parquet", "tokens": d / f"{film}_tokens_v1_pre_4b.parquet", "txt": d / f"{film}_tokens_v1_pre_4b.txt",
            "meta": d / f"{film}_tokens_v1_pre_4b.meta.json"}


def work_dir(film: str, root: Path = ROOT) -> Path:
    return Path(root) / WORK / film


def stage_root(film: str, root: Path = ROOT) -> Path:
    return work_dir(film, root) / "stage"


def staged_paths(film: str, root: Path = ROOT) -> dict[str, Path]:
    return live_paths(film, stage_root(film, root))


def run_key_4b(film: str, code_sha: str) -> str:
    from cgt.p3c_tokens import runner
    ident = models.model_identity("4b")
    return runner.run_key(film, ident["adapter_sha256"], code_sha, "D", model_identity=ident)


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------- preflight
def _inputs_without_scores(film: str, root: Path) -> dict:
    """build_v1.input_hashes without the score files (the old run cache may live in another checkout)."""
    from cgt.tokens import build, build_v1, v1
    p = build.input_paths(film, root)
    h = {k: sha(v) for k, v in p.items()}
    cfg = Path(root) / "configs/tokens.yaml"
    h["config"] = sha(cfg)
    h["palette_centroids"] = sha(Path(root) / v1.load_config(cfg)["palette"]["centroids"])
    mp = build_v1.move_path(film, root)
    h["movement"] = sha(mp) if mp.exists() else "absent"
    h["code"] = hashlib.sha256("".join(sha(Path(build_v1.__file__).parent / f) for f in build_v1.CODE_FILES).encode()).hexdigest()
    return h


def preflight(f: dict, root: Path = ROOT, write: bool = True, adapter: Path | None = None) -> dict:
    film, root = f["film"], Path(root)
    problems = []
    live = live_paths(film, root)
    before = {k: sha(p) for k, p in live.items()}
    for k in ("p3c", "tokens", "meta"):
        if before[k] is None:
            problems.append(f"live {k} file missing")
    n_shots = None
    sp = root / f"data/processed/{film}_shots.parquet"
    if not sp.exists():
        problems.append("shot table missing")
    else:
        shots = pd.read_parquet(sp)
        n_shots = len(shots)
        kf_root = root / "data/interim" / film / "keyframes"
        miss = sum(not (root / r).exists() and not Path(r).exists() for c in ("kf_start", "kf_mid", "kf_end") for r in shots[c])
        if miss:
            problems.append(f"{miss} keyframe files missing (expected under {kf_root.relative_to(root)})")
    for rel in (f"data/processed/{film}_p2.parquet", "configs/tokens.yaml", "data/cache/tokens/palette_global_k32.npz"):
        if not (root / rel).exists():
            problems.append(f"missing {rel.replace(film, '<film>')}")
    ad = Path(adapter) if adapter else models.adapter_path("4b")
    ad_sha = sha(ad)
    if ad_sha != models.expected_sha("4b"):
        problems.append(f"4B adapter sha {ad_sha} != pinned {models.expected_sha('4b')}")
    mismatch = {}
    if before["meta"] and not problems:
        meta_in = json.loads(live["meta"].read_text()).get("inputs", {})
        cur = _inputs_without_scores(film, root)
        mismatch = {k: [meta_in.get(k), v] for k, v in cur.items() if meta_in.get(k) != v}
        if mismatch:
            problems.append(f"the live token table was not built from the current inputs: {sorted(mismatch)}")
    rec = {"film": film, "selector": f["selector"], "local_only": f["local_only"], "n_shots": n_shots, "before": before,
           "adapter_4b_sha256": ad_sha, "meta_mismatch": mismatch, "problems": problems, "ok": not problems,
           "pipeline_version": models.PIPELINE_VERSION["4b"]}
    if write:
        _write(work_dir(film, root) / "preflight.json", rec)
    return rec


# ---------------------------------------------------------------- build (staged)
def build_stage(f: dict, root: Path = ROOT) -> dict:
    """Byte copies of the unchanged inputs + the 4B run into the stage root, then the UNCHANGED token builder on the stage root."""
    from cgt.tokens import build_v1
    film, root = f["film"], Path(root)
    st = stage_root(film, root)
    sp = staged_paths(film, root)
    if not sp["p3c"].exists():
        raise RebuildRefused("no staged Phase 3C table: the GPU run has not completed this film")
    key = pd.read_parquet(sp["p3c"], columns=["p3c_run_key"])["p3c_run_key"].unique().tolist()
    if len(key) != 1:
        raise RebuildRefused(f"staged Phase 3C table mixes run keys {key}")
    rd = root / "data/cache/p3c/runs" / f"{film}_{key[0]}"
    man = json.loads((rd / "manifest.json").read_text())
    if man.get("state") != "complete" or man.get("model_identity") != models.model_identity("4b"):
        raise RebuildRefused(f"run {key[0]} is not a complete {models.PIPELINE_VERSION['4b']} run")
    copies = [f"data/processed/{film}_shots.parquet", f"data/processed/{film}_p2.parquet", "configs/tokens.yaml",
              "data/cache/tokens/palette_global_k32.npz"]
    if build_v1.move_path(film, root).exists():
        copies.append(f"data/processed/{film}_move.parquet")
    for rel in copies:
        (st / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, st / rel)
    srd = st / "data/cache/p3c/runs" / rd.name
    srd.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "scores_kf_start.jsonl", "scores_kf_mid.jsonl", "scores_kf_end.jsonl"):
        shutil.copy2(rd / name, srd / name)
    build_v1.write_tokens_v1(film, st)
    rec = {"film": film, "run_key": key[0], "staged": {k: sha(p) for k, p in sp.items()}}
    _write(work_dir(film, root) / "build.json", rec)
    return rec


# ---------------------------------------------------------------- verify
def _vocab_problems(df: pd.DataFrame) -> list[str]:
    from cgt.tokens import build
    out = []
    for a, labs in build.labels_by_attr().items():
        bad = set(df[a]) - set(labs)
        if bad:
            out.append(f"{a}: values outside the schema vocabulary {sorted(map(str, bad))}")
    return out


def p3c_derived_columns() -> set[str]:
    from cgt.tokens import build
    cols = {"n_kf_disagree", "kf_all_agree", "token_row", "p3c_run_key"}
    for a in build.P3C_ATTRS:
        cols |= {a, *(f"{a}_{s}" for s in P3C_DERIVED)}
    return cols


def verify(f: dict, root: Path = ROOT, write: bool = True) -> dict:
    film, root = f["film"], Path(root)
    wd = work_dir(film, root)
    pre = json.loads((wd / "preflight.json").read_text())
    live, sp = live_paths(film, root), staged_paths(film, root)
    problems = []
    shots = pd.read_parquet(root / f"data/processed/{film}_shots.parquet", columns=["shot_id"])
    new_p3c, new_tok = pd.read_parquet(sp["p3c"]), pd.read_parquet(sp["tokens"])
    old_tok = pd.read_parquet(live["tokens"])
    ids = set(shots.shot_id)
    for name, t in (("p3c", new_p3c), ("tokens", new_tok)):
        if len(t) != len(shots) or set(t.shot_id) != ids or t.shot_id.duplicated().any():
            problems.append(f"staged {name} table does not cover exactly the shot table")
    keys = set(new_p3c.p3c_run_key) | set(new_tok.p3c_run_key)
    if len(keys) != 1:
        problems.append(f"staged tables mix run keys {sorted(keys)}")
    key = str(new_p3c.p3c_run_key.iloc[0]) if len(new_p3c) else ""
    mf = root / "data/cache/p3c/runs" / f"{film}_{key}" / "manifest.json"
    man = json.loads(mf.read_text()) if mf.exists() else {}
    if man.get("state") != "complete" or man.get("model_identity") != models.model_identity("4b"):
        problems.append("the staged run is not a complete p3c-v2-4b run")
    if key in set(old_tok.p3c_run_key):
        problems.append("the staged run key equals the live one (nothing new)")
    if list(new_tok.columns) != list(old_tok.columns):
        problems.append(f"token columns differ: {sorted(set(new_tok.columns) ^ set(old_tok.columns))}")
    problems += _vocab_problems(new_tok)
    a_new, a_old = new_tok.sort_values("shot_id").reset_index(drop=True), old_tok.sort_values("shot_id").reset_index(drop=True)
    for c in sorted((set(a_new.columns) & set(a_old.columns)) - p3c_derived_columns()):
        if not a_new[c].equals(a_old[c]):
            problems.append(f"column {c} changed although it does not depend on Phase 3C")
    smeta = json.loads(sp["meta"].read_text())["inputs"]
    ometa = json.loads(live["meta"].read_text())["inputs"]
    for k in set(smeta) | set(ometa):
        if k == "p3c" or k.startswith("scores_"):
            continue
        if smeta.get(k) != ometa.get(k):
            problems.append(f"token meta input {k} differs from the live table's")
    if smeta.get("p3c") != sha(sp["p3c"]):
        problems.append("staged meta does not hash the staged Phase 3C table")
    changes = {}
    from cgt.tokens import build
    for a in build.P3C_ATTRS:
        if a in a_new and a in a_old and len(a_new) == len(a_old):
            changes[a] = int((a_new[a] != a_old[a]).sum())
    rec = {"film": film, "run_key": key, "before": pre["before"], "staged": {k: sha(p) for k, p in sp.items()},
           "primary_changes_vs_live": changes, "n": len(a_new), "problems": problems, "ok": not problems and pre.get("ok", False)}
    if not pre.get("ok", False):
        rec["problems"].append("preflight did not pass")
    if write:
        _write(wd / "verify.json", rec)
    return rec


# ---------------------------------------------------------------- swap (never deletes)
def _copy_checked(src: Path, dst: Path, want: str) -> None:
    tmp = dst.with_name(dst.name + ".swap-tmp")
    shutil.copy2(src, tmp)
    if sha(tmp) != want:
        raise RebuildRefused(f"copy of {src.name} has the wrong sha256")
    os.replace(tmp, dst)
    if sha(dst) != want:
        raise RebuildRefused(f"{dst.name}: sha256 after replace != expected")


def swap(f: dict, root: Path = ROOT, log: Callable[[str], None] = print) -> dict:
    """Live -> `_pre_4b` backup (sha checked), staged -> live (sha checked). Refuses unless verify passed and every live file still has its
    BEFORE hash (or already the staged one: idempotent). Records BEFORE/AFTER/backup hashes in swap.json. Deletes nothing."""
    film, root = f["film"], Path(root)
    wd = work_dir(film, root)
    if not (wd / "verify.json").exists():
        raise RebuildRefused("verify has not been run")
    ver = json.loads((wd / "verify.json").read_text())
    if not ver.get("ok"):
        raise RebuildRefused(f"verify did not pass: {ver.get('problems')}")
    live, bk, sp = live_paths(film, root), backup_paths(film, root), staged_paths(film, root)
    staged_now = {k: sha(p) for k, p in sp.items()}
    if staged_now != ver["staged"]:
        raise RebuildRefused("the staged files changed since verify")
    before = ver["before"]
    for k in KINDS:
        cur = sha(live[k])
        if cur not in (before[k], ver["staged"][k]):
            raise RebuildRefused(f"live {k} changed since preflight (sha {cur})")
        b = sha(bk[k])
        if before[k] is not None and b not in (None, before[k]):
            raise RebuildRefused(f"backup {bk[k].name} exists with other content; refusing to overwrite it")
    rec = {"film": film, "run_key": ver["run_key"], "before": before, "backup": {}, "after": {}}
    for k in KINDS:
        if before[k] is not None and sha(bk[k]) is None:
            if sha(live[k]) != before[k]:
                raise RebuildRefused(f"live {k} is no longer the BEFORE file and has no backup")
            _copy_checked(live[k], bk[k], before[k])
            log(f"backup {k}: {bk[k].name}")
        rec["backup"][k] = sha(bk[k])
    for k in KINDS:
        if sha(live[k]) != ver["staged"][k]:
            _copy_checked(sp[k], live[k], ver["staged"][k])
            log(f"swapped {k}")
        rec["after"][k] = sha(live[k])
    rec["ok"] = rec["after"] == ver["staged"] and all(rec["backup"][k] == before[k] for k in KINDS if before[k] is not None)
    _write(wd / "swap.json", rec)
    return rec


def guard_live_replace(film: str, root: Path, new_key: str) -> None:
    """A 4B run must not overwrite a live Phase 3C table of ANOTHER run that has no `_pre_4b` byte backup (use the rebuild script)."""
    lp, bp = live_paths(film, root)["p3c"], backup_paths(film, root)["p3c"]
    if not lp.exists():
        return
    keys = set(pd.read_parquet(lp, columns=["p3c_run_key"]).p3c_run_key)
    if keys == {new_key}:
        return
    if bp.exists() and sha(bp) == sha(lp):
        return
    raise RebuildRefused(f"{lp.name} holds run(s) {sorted(keys)} with no _pre_4b backup; write the 4B table elsewhere (--out) and replace it with "
                         "scripts/rebuild_default_4b.py (backup, verify, swap)")


# ---------------------------------------------------------------- runner logic (pure; scripts/rebuild_default_4b.py drives it)
def projection_s(remaining_items: int, s_per_item: float) -> float:
    return remaining_items * s_per_item * MARGIN + LOAD_S


def plan_film(now: datetime, deadline: datetime, cap_end: datetime, remaining_items: int, s_per_item: float) -> dict:
    """Drop (do not start) a film whose projection does not fit the earlier of the deadline and the cap. Returns {fits, stop_at, need_s}."""
    stop = min(deadline, cap_end)
    need = projection_s(remaining_items, s_per_item)
    return {"fits": remaining_items == 0 or now + timedelta(seconds=need) <= stop, "stop_at": stop, "need_s": need}


def classify_exit(rc: int) -> str:
    if rc == 0:
        return "complete"
    if rc == STOPPED_EXIT:
        return "stopped"
    if rc in FAIL_CLOSED:
        return "refused"
    return "crash"
