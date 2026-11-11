"""Drive the REAL entry points into each failure and record the exit codes (results/phase3b/fail_closed_demo.md).
Scenarios use temporary inputs only; no training step runs in any of them. Usage: uv run python scripts/p3b_fail_closed_demo.py"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/p3b"
tmp = Path(tempfile.mkdtemp())
rows = [json.loads(x) for x in (CACHE / "train.jsonl").read_text().splitlines()[:60]]
results = []


def sh(args, label, expect, extra_env=None):
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    out = (r.stdout + r.stderr).strip().splitlines()
    line = next((x for x in reversed(out) if "FAIL CLOSED" in x or "Error" in x), out[-1] if out else "")
    results.append((label, " ".join(args[2:])[:150], r.returncode, expect, line[:230]))
    print(f"{label}: exit {r.returncode} (expected {expect})", flush=True)


def jl(path: Path, items) -> Path:
    path.write_text("\n".join(json.dumps(x) for x in items) + "\n")
    return path


train = ["uv", "run", "python", "scripts/p3b_train.py"]
# 1. image outside the ShotQA root (exit 5, before any model is loaded)
bad = [dict(rows[0], images=["/tmp/not_in_shotqa.jpg"])] + rows[1:5]
sh(train + ["--tag", "demo_badroot", "--train-jsonl", str(jl(tmp / "badroot.jsonl", bad)), "--n-samples", "5"], "image outside the ShotQA root", 5)
# 2. unreadable image (inside the root but missing) (exit 6)
missing = [dict(rows[0], images=[str(CACHE.parents[1] / "external/shotqa/DOES_NOT_EXIST.jpg")])] + rows[1:5]
sh(train + ["--tag", "demo_unreadable", "--train-jsonl", str(jl(tmp / "missing.jsonl", missing)), "--n-samples", "5"], "unreadable (missing) image", 6)
# 3. leak probe failure (simulated measurement 0.125; exit 4 after the model loads, before any step)
sh(train + ["--tag", "demo_leak", "--n-samples", "20", "--simulate-probe-failure"], "leak probe not exactly 0.0", 4)
# 4. over-long sample (exit 8)
long_row = dict(rows[1], messages=[{"role": "user", "content": "word " * 3000}, rows[1]["messages"][1]])
sh(train + ["--tag", "demo_toolong", "--train-jsonl", str(jl(tmp / "long.jsonl", [rows[0], long_row, *rows[2:6]])), "--n-samples", "6"], "sample over 1,024 teacher-forced tokens", 8)
# 5. dirty tree (exit 10): an untracked file under scripts/
dirty = ROOT / "scripts/_dirty_demo_tmp.py"
dirty.write_text("# demo\n")
try:
    sh(train + ["--tag", "demo_dirty", "--n-samples", "20"], "uncommitted code under scripts/", 10)
finally:
    dirty.unlink()
# 6. adapter hash: tampered checkpoint on resume (exit 7) and unregistered adapter in the evaluator (exit 7)
src_run = CACHE / "runs/pipecheck_5k"
if (src_run / "ckpt_025.safetensors").exists():
    dst = CACHE / "runs/demo_hash"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src_run, dst)
    logd = ROOT / "results/phase3b/runs/demo_hash"
    logd.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results/phase3b/runs/pipecheck_5k/manifest.json", logd / "manifest.json")
    with open(dst / "ckpt_025.safetensors", "ab") as f:
        f.write(b"\0")
    sh(train + ["--tag", "demo_hash", "--n-samples", "5000", "--resume-every", "50"], "checkpoint no longer matches its recorded sha256 (resume)", 7)
    odd = tmp / "unregistered.safetensors"
    shutil.copy(src_run / "ckpt_025.safetensors", odd)
    with open(odd, "ab") as f:
        f.write(b"\1")
    sh(["uv", "run", "python", "scripts/p3b_eval_val.py", "--adapter", str(odd), "--manifest", str(ROOT / "results/phase3b/runs/pipecheck_5k/manifest.json"),
        "--n", "5", "--out", str(tmp / "x")], "adapter not registered in the manifest (evaluator)", 7)
    sh(["uv", "run", "python", "scripts/p3b_eval_val.py", "--adapter", str(tmp / "nope.safetensors"), "--n", "5", "--out", str(tmp / "y")],
       "adapter file missing (evaluator)", 7)
lines = ["# Fail-closed demonstration (real entry points, temporary inputs, no training step runs)", "",
         "| scenario | command | exit code | expected | message |", "|---|---|---|---|---|"]
for label, cmd, code, exp, msg in results:
    lines.append(f"| {label} | `{cmd}` | **{code}** | {exp} | {'PASS' if code == exp else 'FAIL'}: {msg.replace('|', '/')} |")
lines += ["", f"All {len(results)} scenarios exited with the expected code: {all(r[2] == r[3] for r in results)}."]
(ROOT / "results/phase3b/fail_closed_demo.md").write_text("\n".join(lines) + "\n")
raise SystemExit(0 if all(r[2] == r[3] for r in results) else 1)
