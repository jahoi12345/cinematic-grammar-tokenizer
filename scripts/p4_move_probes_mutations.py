"""Mutation checks of the label-free direction probes (docs/phase4_move_probes.md acceptance): each mutation removes ONE protective statement in a scratch copy (src, scripts, tests copied; data linked), the
targeted test must FAIL on the mutant and PASS on the unmutated copy. CPU only. Writes results/films/move_probes/mutations.md. Usage: uv run python scripts/run_move_film_mutations.py"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "tests/test_p4_move_probes.py"
PRB = "src/cgt/p4_camera_move/probes.py"
MUTATIONS = [
    ("mirror not applied (H and V clips written unchanged)", PRB, '            wr.write(cv2.flip(fr, _flip_code(t)) if t in ("H", "V") else fr)', "            wr.write(fr)",
     [f"{T}::test_horizontal_mirror_swaps_left_and_right_only", f"{T}::test_vertical_mirror_swaps_top_and_bottom_only"]),
    ("time reversal not applied", PRB, "            for fr in reversed(frames):", "            for fr in frames:", [f"{T}::test_time_reversal_reverses_the_frame_order_across_blocks"]),
    ("wrong partner map (zoom-in paired with push-in under reversal)", PRB, '"zoom-in": "zoom-out", "zoom-out": "zoom-in", "pan-left"', '"zoom-in": "push-in", "zoom-out": "pull-out", "pan-left"',
     [f"{T}::test_partner_maps_equal_the_declared_maps_and_the_excluded_atoms_are_printed", f"{T}::test_a_model_that_swaps_partners_reads_direction_and_keeps_invariant_atoms_stable"]),
    ("wrong mirror axis (H and V exchanged)", PRB, 'return {"H": 1, "V": 0}[t]', 'return {"H": 0, "V": 1}[t]', [f"{T}::test_horizontal_mirror_swaps_left_and_right_only", f"{T}::test_vertical_mirror_swaps_top_and_bottom_only"]),
    ("transformed table replaced by the forward table", PRB, "        P_t = P_by_t[t]", "        P_t = P_f", [f"{T}::test_a_model_that_swaps_partners_reads_direction_and_keeps_invariant_atoms_stable"]),
    ("shuffle (permutation null) control disabled", PRB, "d_t[ok][rng.permutation(int(ok.sum()))])  # transformed table permuted across shots", "d_t[ok])  # control disabled", [f"{T}::test_the_permutation_null_is_near_zero"]),
    ("groups S and M swapped", PRB, 'return np.array(["S" if s == ["static"] else ("X" if "static" in s else "M") for s in sets])', 'return np.array(["M" if s == ["static"] else ("X" if "static" in s else "S") for s in sets])',
     [f"{T}::test_groups_s_m_x_follow_the_declared_definition", f"{T}::test_check_b_reports_consistency_between_motion_size_and_the_static_decision"]),
    ("tripwire disabled (a label file could be opened)", PRB, "        if _denied(file):", "        if False:", [f"{T}::test_the_tripwire_makes_every_label_path_unopenable"]),
    ("flagged shots not excluded from the primary analysis", PRB, "    ok = ~np.asarray(flagged, dtype=bool)\n    idx = {a: i", "    ok = np.ones(len(P_f), dtype=bool)\n    idx = {a: i",
     [f"{T}::test_pairs_with_fewer_than_30_confident_shots_get_no_verdict_and_flagged_shots_are_excluded"]),
    ("train label file always denied OR always allowed: exception not limited to the declared context", PRB, "        return not (_train_ok[0] and p.endswith(TRAIN_LABEL_FILE))", "        return False",
     [f"{T}::test_train_label_file_is_denied_by_default_and_allowed_only_inside_the_declared_context", f"{T}::test_the_tripwire_makes_every_label_path_unopenable"]),
    ("train label exception leaks past its context", PRB, "    finally:\n        _train_ok[0] = False", "    finally:\n        pass", [f"{T}::test_train_label_file_is_denied_by_default_and_allowed_only_inside_the_declared_context"]),
    ("cache identity ignores the transform code bytes", PRB, "    h.update(Path(__file__).read_bytes())\n", "", [f"{T}::test_the_cache_identity_follows_the_bytes_of_the_transform_and_conformity_code"]),
    ("cache identity ignores the clip-conformity code bytes", PRB, "    h.update(Path(FM.__file__).read_bytes())\n", "", [f"{T}::test_the_cache_identity_follows_the_bytes_of_the_transform_and_conformity_code"]),
    ("cached series reused without the producing identity", "scripts/p4_move_probes.py", '    if "impl_identity" not in z.files or str(z["impl_identity"]) != impl:\n        return None', "    pass",
     ["tests/test_p4_move_probes_cache.py::test_a_cached_series_without_or_with_another_identity_is_not_reused"]),
    ("memory not checked before the transform feature batch", "scripts/p4_move_probes.py", "                    PR.write_transformed_clip(fwd, clip, t)\n                    wait_memory()", "                    PR.write_transformed_clip(fwd, clip, t)",
     ["tests/test_p4_move_probes_cache.py::test_memory_is_checked_before_every_feature_batch"]),
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
    L = ["# Mutation checks of the direction probes (CPU, scratch copies)", "", "| removed protection | unmutated code | mutant | verdict |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {'ok' if d else 'BAD'} |" for a, b, c, d in rows]
    L += ["", f"All {len(rows)} mutants killed and all unmutated runs passed: {ok}."]
    out = ROOT / "results/films/move_probes/mutations.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
