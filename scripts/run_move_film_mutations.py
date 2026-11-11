"""Mutation checks of the film-level movement runner (docs/phase4_film_move.md acceptance 2): each mutation removes ONE protective statement in a scratch copy (src, scripts, tests copied; data linked), the
targeted test must FAIL on the mutant and PASS on the unmutated copy. CPU only. Writes results/phase4/film_move/mutations.md. Usage: uv run python scripts/run_move_film_mutations.py"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "tests/test_film_move.py"
FMP = "src/cgt/p4_camera_move/film_move.py"
MUTATIONS = [
    ("owned film refusal removed", FMP, '    if not entry.get("public_domain") and not allow_owned:', "    if False:", [f"{T}::test_owned_film_needs_the_explicit_flag"]),
    ("unregistered film refusal removed", FMP, "    if entry is None:\n        raise Refused", "    if False:\n        raise Refused", [f"{T}::test_unregistered_film_is_refused"]),
    ("missing shot table not refused", FMP, "    if not film.is_file() or not shots.is_file():", "    if False:", [f"{T}::test_missing_shot_table_or_film_is_refused"]),
    ("existing output overwritten silently", FMP, "    if Path(out).exists() and not overwrite:", "    if False:", [f"{T}::test_existing_output_is_refused_without_overwrite"]),
    ("memory guard threshold changed", FMP, "    if free < MIN_FREE_PERCENT:", "    if free < 5:", [f"{T}::test_memory_guard_refuses_below_25_percent"]),
    ("cache name ignores the film hash and the frame range", FMP, '    return f"{film_sha[:12]}.{round(start_s * 1000)}-{round(end_s * 1000)}.r24"', '    return "fixed.r24"', [f"{T}::test_cache_identity_follows_film_bytes_frame_range_and_code_keys"]),
    ("cache name ignores the depth code key", FMP, '    key = f"{F.cache_key()}.{D.cache_key()}.{fkey}"', '    key = f"{F.cache_key()}.{fkey}"', [f"{T}::test_cache_identity_follows_film_bytes_frame_range_and_code_keys"]),
    ("threshold changed to strictly greater", FMP, "        idx = [j for j in range(len(atoms)) if p[j] >= THRESHOLD]", "        idx = [j for j in range(len(atoms)) if p[j] > THRESHOLD]", [f"{T}::test_decision_rule_threshold_fallback_and_never_empty"]),
    ("top-1 fallback removed (empty sets)", FMP, "        sets.append([atoms[j] for j in (idx or [int(np.argmax(p))])])", "        sets.append([atoms[j] for j in idx])", [f"{T}::test_decision_rule_threshold_fallback_and_never_empty"]),
    ("static rule dropped from the contradictory definition", FMP, '    static_other = "static" in s and len(s) > 1', "    static_other = False", [f"{T}::test_static_with_another_atom_is_contradictory_but_static_alone_is_not"]),
    ("an opposite pair missing", FMP, '("boom-up", "boom-down"), ', "", [f"{T}::test_every_opposite_pair_is_contradictory_in_both_orders"]),
    ("shot clip written at the film rate (no conformity)", FMP, "TARGET_FPS = 24000 / 1001", "TARGET_FPS = 30000 / 1001", [f"{T}::test_cut_clip_resamples_in_time_and_equals_the_accepted_notld_function"]),
]


def scratch(tmp: Path) -> Path:
    for d in ("src", "scripts", "tests", "configs"):
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for d in ("data", "results", "docs"):
        os.symlink((ROOT / d).resolve(), tmp / d)
    return tmp


def pytest(tmp: Path, ids: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": str(tmp / "src")}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", *ids], cwd=tmp, env=env, capture_output=True, text=True, check=False).returncode


def main() -> None:
    rows, ok = [], True
    for desc, rel, old, new, ids in MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            tmp = scratch(Path(td))
            base = pytest(tmp, ids)
            f = tmp / rel
            text = f.read_text()
            assert text.count(old) == 1, (desc, "statement not found exactly once")
            f.write_text(text.replace(old, new))
            mut = pytest(tmp, ids)
        good = base == 0 and mut != 0
        ok &= good
        rows.append((desc, "passes" if base == 0 else "FAILS", "killed" if mut != 0 else "SURVIVED", good))
        print(("OK " if good else "BAD ") + f"{desc}: unmutated {rows[-1][1]}, mutant {rows[-1][2]}", flush=True)
    L = ["# Mutation checks of the film-level movement runner (CPU, scratch copies)", "", "| removed protection | unmutated code | mutant | verdict |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {'ok' if d else 'BAD'} |" for a, b, c, d in rows]
    L += ["", f"All {len(rows)} mutants killed and all unmutated runs passed: {ok}."]
    out = ROOT / "results/phase4/film_move/mutations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
