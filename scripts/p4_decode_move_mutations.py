"""Mutation checks of the Step A movement decoders (docs/acc_step_a_move_plan.md section 6): each mutant changes ONE statement of src/cgt/p4_decode/move.py in a scratch copy (src, tests, configs copied);
tests/test_p4_decode_move.py must FAIL on every mutant and PASS on the unmutated copy. CPU, synthetic data only. Writes results/p4_stepa/move_mutations.md.
Usage: uv run python scripts/p4_decode_move_mutations.py"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
F = "src/cgt/p4_decode/move.py"
T = "tests/test_p4_decode_move.py"
MUTATIONS = [
    ("D0 threshold > instead of >=", "if p[j] >= 0.5]", "if p[j] > 0.5]"),
    ("D0 fallback dropped (empty sets allowed)", "(idx or [int(np.argmax(p))])", "idx"),
    ("D1 keeps the LOWER-p member of a pair", "s.discard(b if p[ai[a]] >= p[ai[b]] else a)", "s.discard(a if p[ai[a]] >= p[ai[b]] else b)"),
    ("D1 pair rule removed", "        if a in s and b in s:\n            s.discard", "        if False:\n            s.discard"),
    ("D1 static rule removed", '    if "static" in s and moves:', '    if False and moves:'),
    ("D1 static always wins", 's = {"static"} if p[ai["static"]] >= max(p[ai[x]] for x in moves) else set(moves)', 's = {"static"}'),
    ("D3 without constraints", 'if candidate in ("D1", "D3"):\n            s = constrain', 'if candidate in ("D1",):\n            s = constrain'),
    ("D2 ignores thresholds", '        Q = recalibrate(P, thresholds)\n    else:', '        Q = P\n    else:'),
    ("threshold fit maximises nothing (first grid value)", "        best = max(s for s, _ in scored)", "        best = scored[0][0]"),
    ("threshold tie not to 0.5", "key=lambda t: (abs(t - 0.5), t))", "key=lambda t: t)"),
    ("no-positive atom default is not 0.5", "out = np.full(P_cal.shape[1], 0.5)", "out = np.full(P_cal.shape[1], 0.9)"),
    ("recalibration sign flipped", "np.exp(-(_logit(P) - _logit(t)))", "np.exp((_logit(P) - _logit(t)))"),
    ("option mask off", "        if not bad.all():\n            s = np.where", "        if False:\n            s = np.where"),
    ("option mask even when all four are contradictory", "        if not bad.all():", "        if True:"),
    ("option tie to the highest letter", "        hits.append(int(np.argmax(s)) ==", "        hits.append(3 - int(np.argmax(s[::-1])) =="),
    ("decision: interval touching 0 accepted", 'ci_ok = boot_vs_d0[top]["ci95"][0] > 0', 'ci_ok = boot_vs_d0[top]["ci95"][0] >= 0'),
    ("decision: Jaccard condition ignored", "    jac_ok = jaccard[top] - jaccard[\"D0\"] >= -0.010 - 1e-12", "    jac_ok = True"),
    ("set metric: Jaccard as intersection over prediction", "len(set(p) & set(g)) / len(set(p) | set(g))", "len(set(p) & set(g)) / len(set(p))"),
    ("fusion tie to the larger weight sum", "key=lambda k: (k[0] + k[1], k))", "key=lambda k: (-(k[0] + k[1]), k))"),
]


def run(tmp: Path) -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", T], cwd=tmp, env={"PYTHONPATH": str(tmp / "src"), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, check=False)
    return r.returncode == 0


def main() -> None:
    rows = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for sub in ("src", "tests", "configs"):
            shutil.copytree(ROOT / sub, tmp / sub)
        shutil.copy(ROOT / "pyproject.toml", tmp / "pyproject.toml")
        orig = (tmp / F).read_text()
        base_ok = run(tmp)
        for name, old, new in MUTATIONS:
            if orig.count(old) != 1:
                rows.append((name, "TARGET NOT FOUND", "error"))
                continue
            (tmp / F).write_text(orig.replace(old, new))
            passed = run(tmp)
            rows.append((name, "passes" if passed else "fails", "killed" if not passed else "SURVIVED"))
            (tmp / F).write_text(orig)
    killed = sum(r[2] == "killed" for r in rows)
    L = ["# Step A movement decoders: mutation checks", "", f"Unmutated copy: tests {'pass' if base_ok else 'FAIL'}. Mutants killed: {killed} of {len(rows)}.", "",
         "| mutant | tests on mutant | result |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    out = ROOT / "results/p4_stepa"
    out.mkdir(parents=True, exist_ok=True)
    (out / "move_mutations.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if not base_ok or killed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
