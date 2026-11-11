"""Mutation checks of the blockers E1-E5 of the sealed final evaluator (docs/phase4_video_plan.md Amendment 12): each mutation removes ONE protective statement in a scratch copy of the code (src, scripts,
tests are copied; data and results are linked read-only), the targeted regression test must FAIL on the mutant and PASS on the unmutated copy. CPU only, synthetic worlds, no real test data. Writes
results/phase4/final_eval_mutations.md. Usage: uv run python scripts/p4_final_eval_mutations.py"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "tests/test_phase4_final_eval_blockers.py"
# (finding, description, file, old text, new text, test node ids that must fail)
MUTATIONS = [
    ("E1", "ledger append without fsync", "src/cgt/p4_camera_move/final_stores.py", "        os.fsync(fh.fileno())\n    if new:", "        pass\n    if new:", [f"{T}::test_camerabench_ledger_is_fsynced_before_label_open"]),
    ("E1", "CameraBench labels opened after a failed fsync (append errors swallowed)", "src/cgt/p4_camera_move/final_stores.py",
     "        reserve_once(self.ledger, CAMERABENCH, model_tag, check, adapter_sha256=adapter_sha256, config_hash=config_hash, code_sha=code_sha)\n",
     "        try:\n            reserve_once(self.ledger, CAMERABENCH, model_tag, check, adapter_sha256=adapter_sha256, config_hash=config_hash, code_sha=code_sha)\n        except OSError:\n            pass\n",
     [f"{T}::test_labels_are_never_opened_after_a_failed_fsync[camerabench]"]),
    ("E3", "check + append without the lock", "src/cgt/p4_camera_move/final_stores.py", "    with ledger_lock(path):\n        check()", "    if True:\n        check()", [f"{T}::test_duplicate_concurrent_label_reads_reserve_one_dataset_tag"]),
    ("E3", "run reservation not taken", "src/cgt/p4_camera_move/final_run.py", "        cm.__enter__()\n    except BlockingIOError", "        pass\n    except BlockingIOError",
     [f"{T}::test_second_evaluation_process_refuses_while_another_holds_the_reservation", f"{T}::test_run_reservation_is_global_across_datasets_and_models"]),
    ("E2", "geometry configuration read from the editable working copy (no committed-bytes check)", "src/cgt/p4_camera_move/final_run.py", "    if not working.exists() or working.read_bytes() != cfg_bytes:", "    if False:",
     [f"{T}::test_geometry_edited_parameters_are_refused_for_geometry_and_combined"]),
    ("E2", "stated configuration hash not recomputed from the content", "src/cgt/p4_camera_move/final_run.py", '    if "c_grid" in cfg and cfg_hash(', '    if False and "c_grid" in cfg and cfg_hash(',
     [f"{T}::test_geometry_config_must_carry_the_hash_of_its_content"]),
    ("E2", "go file does not bind the geometry inputs", "src/cgt/p4_camera_move/final_eval.py", "f\"geometry_inputs={geometry_inputs or '-'}\\n\")", "f\"geometry_inputs=-\\n\")",
     [f"{T}::test_go_file_binds_the_geometry_inputs"]),
    ("E2b", "training split taken from the editable working copy", "src/cgt/p4_camera_move/final_run.py", "    if man_bytes and split_path.read_bytes() != man_bytes:", "    if False:",
     [f"{T}::test_edited_training_split_is_refused_and_the_committed_split_is_the_one_returned"]),
    ("E2b", "split not tied to the manifest hash of the frozen configuration", "src/cgt/p4_camera_move/final_run.py", '    if split is not None and "manifest_sha256" in cfg and', '    if False and split is not None and "manifest_sha256" in cfg and',
     [f"{T}::test_split_manifest_must_be_the_one_the_geometry_config_was_frozen_with"]),
    ("E5", "conformity target rate is the film rate (no resampling to the training rate)", "src/cgt/p4_camera_move/conform.py", "TARGET_FPS = 24000 / 1001", "TARGET_FPS = 30000 / 1001",
     [f"{T}::test_conformed_video_is_resampled_in_time_and_sampled_at_the_training_rate"]),
    ("E5", "GIF durations ignored (frame index instead of timestamp)", "src/cgt/p4_camera_move/conform.py", "    return [int(np.searchsorted(starts, j / TARGET_FPS, side=\"right\") - 1) for j in range(n_out)]",
     "    return [min(len(durations_ms) - 1, j * len(durations_ms) // max(1, n_out)) for j in range(n_out)]", [f"{T}::test_conformed_gif_follows_the_frame_durations"]),
    ("E4", "table digest not compared with the manifest", "src/cgt/p4_camera_move/final_inputs.py", 'sha256_file(Path(table)) != m["feature_table_sha256"]', "False", [f"{T}::test_prepared_feature_table_digest_is_checked_before_use_here"]),
    ("E4", "input bytes not hashed", "src/cgt/p4_camera_move/final_inputs.py", '            if not f.is_file() or sha256_file(f) != m["input_sha256"][name]:', "            if not f.is_file():", [f"{T}::test_input_bytes_are_hashed_every_time"]),
    ("E4", "coverage not checked (unique, complete)", "src/cgt/p4_camera_move/final_inputs.py", '    if len(set(clips)) != len(clips) or set(m["input_sha256"]) != set(clips) or m["n_files"] != len(clips):', "    if False:",
     [f"{T}::test_coverage_must_be_complete_and_unique"]),
    ("E4", "provenance and conformity version not compared", "src/cgt/p4_camera_move/final_inputs.py", '    if m["provenance"] != cur or m["provenance"].get("conform_version") != conform.CONFORM_VERSION:', "    if False:", [f"{T}::test_stale_or_missing_provenance_and_conformity_are_refused"]),
    ("E4", "column hash not compared with the training columns", "src/cgt/p4_camera_move/final_inputs.py", "        if columns_sha256(cols) != m[\"columns_sha256\"] or (train_columns is not None and columns_sha256(train_columns) != m[\"columns_sha256\"]):", "        if False:",
     [f"{T}::test_feature_columns_must_equal_the_training_columns"]),
    ("Frames", "fallback frame index rule changed (first K frames instead of the bin centres)", "src/cgt/p4_camera_move/frames_fallback.py", "enumerate(V.frame_indices(len(frames), k))", "enumerate(range(k))",
     [f"{T}::test_sequential_fallback_gives_the_same_frames_as_the_seeking_extractor"]),
    ("Frames", "a GIF with fewer frames than K is refused again (the clip would be dropped)", "src/cgt/p4_camera_move/gif.py", "    if not frames:\n        raise ValueError(f\"{path}: no decodable frame\")", "    if len(frames) < k:\n        raise ValueError(f\"{path}: too few\")",
     [f"{T}::test_a_gif_with_fewer_frames_than_k_is_kept_with_repeated_frames"]),
]

def scratch(tmp: Path) -> Path:
    for d in ("src", "scripts", "tests", "configs"):
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    for d in ("data", "results", "docs"):
        os.symlink((ROOT / d).resolve(), tmp / d)
    shutil.copy(ROOT / ".gitignore", tmp / ".gitignore")
    return tmp


def pytest(tmp: Path, ids: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": str(tmp / "src")}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", *ids], cwd=tmp, env=env, capture_output=True, text=True, check=False).returncode


def main() -> None:
    rows, ok = [], True
    for finding, desc, rel, old, new, ids in MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            tmp = scratch(Path(td))
            base = pytest(tmp, ids)
            f = tmp / rel
            text = f.read_text()
            assert text.count(old) == 1, (finding, desc, "the mutated statement was not found exactly once")
            f.write_text(text.replace(old, new))
            mut = pytest(tmp, ids)
        good = base == 0 and mut != 0
        ok &= good
        rows.append((finding, desc, "passes" if base == 0 else "FAILS", "killed" if mut != 0 else "SURVIVED", good))
        print(("OK " if good else "BAD ") + f"{finding}: {desc}: unmutated {rows[-1][2]}, mutant {rows[-1][3]}", flush=True)
    L = ["# Mutation checks of the evaluator blockers E1-E5 (CPU, scratch copies)", "", "| finding | removed protection | unmutated code | mutant | verdict |", "|---|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} | {'ok' if e else 'BAD'} |" for a, b, c, d, e in rows]
    L += ["", f"All {len(rows)} mutants killed and all unmutated runs passed: {ok}."]
    (ROOT / "results/phase4/final_eval_mutations.md").write_text("\n".join(L) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
