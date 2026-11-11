"""Mutation checks of the matched evaluator's guards (docs/p3m_matched_protocol.md section 3.1): each mutation removes ONE protective statement
in a scratch copy (src, scripts, tests copied; data/results/docs linked); the named test must FAIL on the mutant and PASS on the unmutated copy.
CPU only, synthetic fixtures only. Writes results/p3m_matched/mutations.md. Usage: uv run python scripts/p3m_matched_mutations.py"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SE, RG, PA, PR, SB = (f"src/cgt/p3m_matched/{m}.py" for m in ("sealed_eval", "registry", "parser", "protocol", "shotbench"))
VAL = "scripts/p3m_matched_val.py"
EV = "scripts/p3m_matched_eval.py"
TS, TV, TP, TQ = (f"tests/test_p3m_matched_{m}.py" for m in ("sealing", "val", "parser", "prompt"))
MUTATIONS = [
    ("GO check removed from preflight", SE, '    require_go(go_file, tag, e["identity_sha256"], phash)  # (5)', "    pass  # (5)",
     [f"{TS}::test_no_go_refuses_before_data"]),
    ("GO binding (content) check removed", SE, "    if p.read_text() != go_text(tag, identity_sha256, protocol_hash):", "    if False:",
     [f"{TS}::test_go_bound_to_tag_identity_protocol"]),
    ("ledger single-use check removed", SE, "            sbmod.ledger_assert_unused(tag, led)", "            pass", [f"{TS}::test_used_ledger_tag_refused"]),
    ("real ledger not checked when a scratch --ledger is given", SE,
     "        for led in dict.fromkeys([Path(ledger), Path(real_ledger or sbmod.LEDGER)]):", "        for led in [Path(ledger)]:",
     [f"{TS}::test_scratch_ledger_or_raw_dir_cannot_bypass_single_use"]),
    ("canonical raw dir not checked", SE, "    for d in dict.fromkeys([Path(raw_dir), Path(real_raw_dir)]):", "    for d in [Path(raw_dir)]:",
     [f"{TS}::test_scratch_ledger_or_raw_dir_cannot_bypass_single_use"]),
    ("predictions-file CONSUMED marker ignored", SE, '        if (d / f"{tag}.predictions.jsonl").exists():', "        if False:",
     [f"{TS}::test_scratch_ledger_or_raw_dir_cannot_bypass_single_use"]),
    ("unacknowledged-crash check removed", SE, "    sealed.require_no_unacknowledged_crash(raw_dir, tag)  # (4)", "    pass  # (4)",
     [f"{TS}::test_unacknowledged_crash_refused"]),
    ("clean-tree check removed", SE, "    git = clean_tree(code_paths)  # (6)", '    git = {"commit": "0", "dirty_files": []}  # (6)',
     [f"{TS}::test_dirty_tree_refused"]),
    ("unpinned identity accepted", RG, '    if not e.get("identity_sha256") or not e.get("revision"):', "    if False:",
     [f"{TS}::test_unpinned_identity_refused"]),
    ("identity hash comparison removed", RG, '    if got != e["identity_sha256"]:', "    if False:", [f"{TS}::test_identity_mismatch_refused"]),
    ("loader path containment removed", SB, "        img = require_under_image_root(ast.literal_eval(r.path)[0], image_root)",
     "        img = Path(image_root) / ast.literal_eval(r.path)[0]", [f"{TS}::test_loader_refuses_path_escape"]),
    ("eval: held GPU lock ignored", EV,
     '    except LockHeld as exc:\n        log(f"REFUSED: {exc} (exit 11); nothing evaluated")\n        sys.exit(11)',
     "    except LockHeld:\n        lock = GpuLock('x', Path(a.lock + '.other')).acquire()", [f"{TS}::test_script_refuses_when_gpu_lock_held"]),
    ("eval: GPU lock not released", EV, "    finally:\n        lock.release()", "    finally:\n        pass",
     [f"{TS}::test_script_without_go_exits_13_before_data"]),
    ("eval: GPU lock not taken", EV, '        lock = GpuLock(f"p3m_matched_eval {a.tag}", Path(a.lock) if a.lock else None).acquire()',
     '        lock = GpuLock(f"p3m_matched_eval {a.tag}", Path(a.lock) if a.lock else None)', [f"{TS}::test_script_takes_gpu_lock_with_nonce"]),
    ("val: adapter hash not recomputed", VAL, "    e = registry.require_identity(tag, data_root)", "    e = dict(registry.entry(tag))",
     [f"{TV}::test_val_refuses_wrong_adapter_hash"]),
    ("val: other tags accepted", VAL, '    if tag != "matched_v1":', "    if False:", [f"{TV}::test_val_refuses_other_tags"]),
    ("val: held GPU lock ignored", VAL, "    except LockHeld as exc:\n        raise GpuBusy(str(exc)) from exc",
     "    except LockHeld:\n        from cgt.pipeline.gpulock import GpuLock as _G\n        lock = _G('x', Path(a.lock + '.other'))", [f"{TV}::test_val_refuses_when_gpu_lock_held"]),
    ("val: per-question key falls back to the non-unique id", VAL, """    return [{"key": f"{i}:{s['id']}", "id": s["id"],""",
     """    return [{"key": s["id"], "id": s["id"],""", [f"{TV}::test_duplicate_ids_get_distinct_keys"]),
    ("val: memory guard threshold disabled", VAL, "    if free < MIN_FREE_PERCENT:", "    if False:", [f"{TV}::test_val_memory_guard"]),
    ("parser: several letters accepted", PA, "    if len(hits) == 1:\n        return hits[0]\n    if not hits", "    if hits:\n        return hits[0]\n    if not hits",
     [f"{TP}::test_two_letters_is_invalid"]),
    ("parser: several option texts accepted", PA, "    return hits[0] if len(hits) == 1 else False", "    return hits[0] if hits else False",
     [f"{TP}::test_text_two_hits_invalid"]),
    ("parser: answer tag ignored", PA, "    return m.group(1).strip() if m else text.strip()", "    return text.strip()",
     [f"{TP}::test_answer_tag_takes_precedence"]),
    ("prompt: training tail used instead of the official one", PR, 'USER_TAIL = "Please select the most likely answer from the options above."',
     'USER_TAIL = "Please select the most likely answer from the options above. \\n"', [f"{TQ}::test_user_text_exact"]),
]


def scratch(tmp: Path) -> Path:
    for d in ("src", "scripts", "tests", "configs"):
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for d in ("data", "results", "docs"):
        os.symlink((ROOT / d).resolve(), tmp / d)
    return tmp


def pytest(tmp: Path, ids: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": str(tmp / "src")}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *ids], cwd=tmp, env=env, capture_output=True,
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
    L = ["# Mutation checks of the matched evaluator guards (CPU, scratch copies, synthetic fixtures)", "",
         "| removed protection | file | test | unmutated code | mutant | verdict |", "|---|---|---|---|---|---|"]
    L += [f"| {a} | `{b}` | `{c}` | {d} | {e} | {'ok' if g else 'BAD'} |" for a, b, c, d, e, g in rows]
    L += ["", f"All {len(rows)} mutants killed and all unmutated runs passed: {ok}."]
    out = ROOT / "results/p3m_matched/mutations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
