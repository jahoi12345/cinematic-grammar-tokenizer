"""Mutation checks of the Step A static-track guards (docs/acc_step_a_static_plan.md section 8): each mutation removes ONE protection in a
scratch copy (src, scripts, tests, configs copied; data/results/docs linked); the named test must FAIL on the mutant and PASS unmutated.
CPU only, synthetic fixtures only. Writes results/p3s_stepa/mutations.md. Usage: uv run python scripts/p3s_stepa_mutations.py"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CB, VW, SPL = "src/cgt/p3s_stepa/combine.py", "src/cgt/p3s_stepa/views.py", "src/cgt/p3s_stepa/split.py"
SC = "scripts/p3s_stepa_views.py"
T = "tests/test_p3s_stepa.py"
MUTATIONS = [
    ("shift mapped back with the wrong sign", VW, "    return np.array([v[(i - s) % 4] for i in range(4)])", "    return np.array([v[(i + s) % 4] for i in range(4)])",
     [f"{T}::test_shift_mapping_roundtrip", f"{T}::test_permutation_averaging_cancels_letter_bias"]),
    ("calibration/ensemble fitted on all questions (selection-half leakage)", CB, '    cal_qs = [q for q in qs if q["half"] == "cal"]\n    info = {}',
     "    cal_qs = list(qs)\n    info = {}", [f"{T}::test_fit_uses_calibration_half_only"]),
    ("ensemble weight fitted on all questions", CB, "        w = fit_ensemble_w(cal_qs, scores, sig)", "        w = fit_ensemble_w(qs, scores, sig)",
     [f"{T}::test_fit_uses_calibration_half_only"]),
    ("calibration fitted on all questions", CB, "        scores, info[\"calibration_params\"] = apply_calibration(qs, scores, cal_qs, cal)",
     "        scores, info[\"calibration_params\"] = apply_calibration(qs, scores, qs, cal)", [f"{T}::test_fit_uses_calibration_half_only"]),
    ("C4* cross-fit on all questions", CB, "        cal, cv = crossfit_pick_c4(cal_qs, scores)", "        cal, cv = crossfit_pick_c4(qs, scores)",
     [f"{T}::test_c2_star_picked_on_calibration_only"]),
    ("C2* picked on all questions", CB, '        c2 = {v: accuracy(cal_qs, averaged(cal_qs, views, [v]))["mean"]',
     '        c2 = {v: accuracy(qs, averaged(qs, views, [v]))["mean"]', [f"{T}::test_c2_star_picked_on_calibration_only"]),
    ("horizontal flip allowed for composition", VW, '"attrs": tuple(a for a in ALL_ATTRS if a != "composition")},', '"attrs": ALL_ATTRS},',
     [f"{T}::test_view_attribute_rules"]),
    ("crop allowed for every attribute", VW, '"attrs": ("light_type", "light_cond")},', '"attrs": ALL_ATTRS},', [f"{T}::test_view_attribute_rules"]),
    ("resolution views upscale small images", VW, "    return budget_size(w, h, min(budget_px, w * h), multiple)", "    return budget_size(w, h, budget_px, multiple)",
     [f"{T}::test_resolution_never_upscales"]),
    ("missing view output silently skipped", CB, "        use = [v for v in view_set if applies(v, q[\"attribute\"])]",
     "        use = [v for v in view_set if applies(v, q[\"attribute\"]) and q[\"key\"] in views[v]]", [f"{T}::test_missing_view_output_raises"]),
    ("dedup keeps the last occurrence", SPL, "        if r[\"id\"] in seen:\n            continue", "        if r[\"id\"] in seen:\n            out = [o for o in out if o[\"id\"] != r[\"id\"]]",
     [f"{T}::test_dedup_keeps_first_occurrence"]),
    ("decision ignores the bootstrap interval", CB, "    ok = gain >= min_gain and lo > 0", "    ok = gain >= min_gain", [f"{T}::test_decision_rule"]),
    ("decision ignores the 0.5-point minimum", CB, "    ok = gain >= min_gain and lo > 0", "    ok = lo > 0", [f"{T}::test_decision_rule"]),
    ("views: adapter hash not checked", SC, "    sha = require_frozen_adapter(root / ADAPTER_REL)  # before any data or model access", "    sha = 'x' * 64",
     [f"{T}::test_views_refuse_wrong_adapter_before_anything"]),
    ("views: memory guard disabled", SC, "    if free < MIN_FREE_PERCENT:", "    if False:", [f"{T}::test_views_memory_guard_and_lock"]),
    ("views: held GPU lock ignored", SC, "    except LockHeld as exc:\n        raise GpuBusy(str(exc)) from exc",
     "    except LockHeld:\n        lock = GpuLock('x', Path(str(a.lock) + '.other')).acquire()", [f"{T}::test_views_memory_guard_and_lock"]),
    ("views: GPU lock not released", SC, "    finally:\n        lock.release()", "    finally:\n        pass", [f"{T}::test_views_never_see_labels"]),
    ("views: gold answer reaches the GPU loop", SC, '    qs = [{k: q[k] for k in ("key", "id", "img_id", "attribute", "images", "messages", "options")} for q in qs]',
     "    qs = [dict(q) for q in qs]", [f"{T}::test_views_never_see_labels"]),
    ("views: no resume (done keys re-run)", SC, 'and q["key"] not in done]', "]", [f"{T}::test_views_never_see_labels"]),
]


def scratch(tmp: Path) -> Path:
    for d in ("src", "scripts", "tests", "configs"):
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for d in ("data", "results", "docs"):
        os.symlink((ROOT / d).resolve(), tmp / d)
    return tmp


def pytest(tmp: Path, ids: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": str(tmp / "src")}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *ids], cwd=tmp, env=env, capture_output=True,
                          text=True, check=False).returncode


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
        rows.append((desc, rel, ids[0].split("::")[1], "passes" if base == 0 else "FAILS", "killed" if mut != 0 else "SURVIVED", good))
        print(("OK " if good else "BAD ") + f"{desc}: unmutated {rows[-1][3]}, mutant {rows[-1][4]}", flush=True)
    L = ["# Mutation checks of the Step A static-track guards (CPU, scratch copies, synthetic fixtures)", "",
         "| removed protection | file | test | unmutated code | mutant | verdict |", "|---|---|---|---|---|---|"]
    L += [f"| {a} | `{b}` | `{c}` | {d} | {e} | {'ok' if g else 'BAD'} |" for a, b, c, d, e, g in rows]
    L += ["", f"All {len(rows)} mutants killed and all unmutated runs passed: {ok}."]
    out = ROOT / "results/p3s_stepa/mutations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
