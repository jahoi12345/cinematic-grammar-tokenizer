"""Hash-bound evidence of a COLD invocation of scripts/run_move_film.py: runs the runner on N evenly spaced shots of a film with a NEW cache directory and a scratch output, then compares the scratch
cache arrays with the accepted cache and the scratch table rows with the accepted table, and writes results/phase4/film_move/cold_invocation.{md,json} with the command, times, code commit, clean-tree state and the SHA-256 of
every file involved, including this script itself. Usage: uv run python scripts/run_move_film_cold_evidence.py [--film night_of_the_living_dead] [--shots 25]
The script FAILS (exit 1, evidence still written with "pass": false and the reasons) when a shot of the scratch table lacks its flow or depth cache file (scratch or accepted), when no array was
compared, when a file pair holds different arrays, when a difference exceeds the declared tolerances below, or when any non-probability column differs."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()  # F-V2: the evidence records the sha256 of this script (evidence_script_sha256)
ARRAY_TOL = 0.0     # cached flow/depth arrays of a cold run must equal the accepted cache exactly (committed evidence: 0.0, all files byte-identical)
PROB_TOL = 1e-12    # predicted probabilities may differ by float rounding only (the prediction batch is smaller; committed evidence: 5.55e-16)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(scratch_cache: Path, acc_cache: Path, new: pd.DataFrame, acc: pd.DataFrame) -> dict:
    """Compare a cold run (scratch cache, scratch table indexed by shot_id) with the accepted cache and table. Returns the measured differences and a list of failures (empty = pass)."""
    fail, worst, n_arr, files = [], 0.0, 0, {}
    for sid in new.index:
        for kind in ("flow", "depth"):
            found = sorted((scratch_cache / kind).glob(f"shot{int(sid)}.*.npz"))
            if len(found) != 1:
                fail.append(f"shot {sid}: {len(found)} scratch {kind} cache files (expected 1)")
                continue
            p = found[0]
            q = acc_cache / kind / p.name
            if not q.exists():
                fail.append(f"shot {sid}: no accepted {kind} cache file {p.name}")
                continue
            with np.load(p) as za, np.load(q) as zb:
                if set(za.files) != set(zb.files):
                    fail.append(f"{kind}/{p.name}: array names differ")
                    continue
                for k in za.files:
                    a, b = za[k].astype(float), zb[k].astype(float)
                    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
                        fail.append(f"{kind}/{p.name}[{k}]: shape or NaN mask differs")
                        continue
                    if a.size == 0:  # F-D1: an empty series proves nothing; it is not counted as compared
                        continue
                    d = float(np.nanmax(np.abs(a - b))) if not np.isnan(a).all() else 0.0
                    worst = max(worst, d)
                    n_arr += 1
            files[f"{kind}/{p.name}"] = {"scratch_sha256": sha(p), "accepted_sha256": sha(q)}
    if n_arr == 0:
        fail.append("no cache array was compared (zero-size arrays do not count; an empty comparison proves nothing)")
    if worst > ARRAY_TOL:
        fail.append(f"cache arrays: max abs difference {worst} > tolerance {ARRAY_TOL}")
    missing = [s for s in new.index if s not in acc.index]
    if missing or len(new) == 0:
        fail.append(f"scratch table rows absent from the accepted table: {missing}" if missing else "the scratch table is empty")
        return {"failures": fail, "worst": worst, "n_arrays": n_arr, "files": files, "pdiff": float("nan"), "equal_other": False}
    pc = [c for c in new.columns if c.startswith("p_")]
    pdiff = float(np.abs(new[pc].to_numpy(float) - acc.loc[new.index, pc].to_numpy(float)).max()) if pc else 0.0
    if not pc:
        fail.append("no probability column (p_*) in the scratch table")
    if not pdiff <= PROB_TOL:
        fail.append(f"table probabilities: max abs difference {pdiff} > tolerance {PROB_TOL}")
    other = [c for c in new.columns if c not in pc]
    absent = [c for c in other if c not in acc.columns]
    equal_other = not absent and bool(all(((new[c] == acc.loc[new.index, c]) | (new[c].isna() & acc.loc[new.index, c].isna())).all() for c in other))
    if not equal_other:
        fail.append(f"table columns other than probabilities differ{f' (absent from the accepted table: {absent})' if absent else ''}")
    return {"failures": fail, "worst": worst, "n_arrays": n_arr, "files": files, "pdiff": pdiff, "equal_other": equal_other}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default="night_of_the_living_dead")
    ap.add_argument("--shots", type=int, default=25)
    a = ap.parse_args()
    acc_cache = ROOT / ("data/cache/p4/notld" if a.film == "night_of_the_living_dead" else f"data/cache/p4/film_move/{a.film}")
    table = ROOT / f"data/processed/{a.film}_move.parquet"
    work = Path(tempfile.mkdtemp(prefix="cold_move_"))
    cmd = [sys.executable, str(ROOT / "scripts/run_move_film.py"), "--film", a.film, "--shots", str(a.shots), "--cache-dir", str(work / "cache"), "--out", str(work / "out.parquet"), "--report", str(work / "report.md")]
    t0 = datetime.now().astimezone().isoformat(timespec="seconds")
    r = subprocess.run(cmd, capture_output=True, text=True, check=False, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    t1 = datetime.now().astimezone().isoformat(timespec="seconds")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(r.returncode)
    new, acc = pd.read_parquet(work / "out.parquet").set_index("shot_id"), pd.read_parquet(table).set_index("shot_id")
    c = compare(work / "cache", acc_cache, new, acc)
    worst, n_arr, files, pdiff, equal_other = c["worst"], c["n_arrays"], c["files"], c["pdiff"], c["equal_other"]
    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    dirty = [x for x in git("status", "--porcelain", "--", "src", "scripts").splitlines() if x.strip()]
    ev = {"film": a.film, "command": " ".join(cmd), "started": t0, "ended": t1, "code_commit": git("rev-parse", "HEAD"), "clean_tree": not dirty, "n_shots": len(new), "n_cached_arrays": n_arr,
          "cache_arrays_max_abs_diff_vs_accepted": worst, "table_probability_max_abs_diff": pdiff, "table_other_columns_equal": equal_other, "accepted_table_sha256": sha(table), "scratch_table_sha256": sha(work / "out.parquet"),
          "scratch_report_sha256": sha(work / "report.md"), "runner_sha256": sha(ROOT / "scripts/run_move_film.py"), "film_move_sha256": sha(ROOT / "src/cgt/p4_camera_move/film_move.py"),
          "evidence_script_sha256": sha(SELF),
          "array_tolerance": ARRAY_TOL, "probability_tolerance": PROB_TOL, "pass": not c["failures"], "failures": c["failures"], "cache_files": files}
    out = ROOT / "results/phase4/film_move"
    (out / "cold_invocation.json").write_text(json.dumps(ev, indent=1))
    (out / "cold_invocation.md").write_text(
        f"# Cold invocation of `run_move_film.py` ({a.film}, {len(new)} evenly spaced shots, new cache directory, scratch output)\n\n- command: `{ev['command']}`\n- {t0} to {t1}; code commit `{ev['code_commit'][:12]}`, clean tree (src, scripts): {ev['clean_tree']}\n"
        f"- {n_arr} cache arrays (flow and depth series) vs the accepted cache: maximum absolute difference **{worst}**; the {len(new)} table rows vs the accepted table: probabilities max abs difference {pdiff:.2e} (the prediction batch has {len(new)} instead of the full number of rows), "
        f"all other columns (sets, flags, hashes) equal: **{equal_other}**\n- SHA-256: accepted table `{ev['accepted_table_sha256'][:16]}`, scratch table `{ev['scratch_table_sha256'][:16]}`, runner `{ev['runner_sha256'][:16]}`, film_move `{ev['film_move_sha256'][:16]}`, this evidence script `{ev['evidence_script_sha256'][:16]}`; every cache file pair in `cold_invocation.json`\n"
        f"- tolerances: arrays <= {ARRAY_TOL}, probabilities <= {PROB_TOL}, other columns equal; every shot needs its flow and depth cache files. **{'PASS' if ev['pass'] else 'FAIL'}**"
        + "".join(f"\n  - {m}" for m in ev["failures"]) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    print((out / "cold_invocation.md").read_text())
    if c["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
