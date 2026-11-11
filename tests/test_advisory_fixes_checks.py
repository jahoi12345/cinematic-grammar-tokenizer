"""Verification pass an earlier review (2026-10-13) probes of the F-V1 fix in scripts/run_move_film_cold_evidence.py (branch w-advisories @f55a983). Synthetic npz files and tables; no data, no GPU.
Skipped on a tree whose cold-evidence script has no `compare` function (main before the merge)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_move_film_cold_evidence.py"


@pytest.fixture(scope="module")
def cold():
    if not SCRIPT.exists():
        pytest.skip("cold-evidence script not on this tree")
    spec = importlib.util.spec_from_file_location("cold_evidence_vd", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not hasattr(m, "compare"):
        pytest.skip("cold-evidence script without the F-V1 compare function")
    return m


def case(tmp_path, arrays):
    scratch, acc = tmp_path / "scratch", tmp_path / "acc"
    for sid in (0, 7):
        for kind in ("flow", "depth"):
            for base in (scratch, acc):
                (base / kind).mkdir(parents=True, exist_ok=True)
                np.savez(base / kind / f"shot{sid}.k.r24.npz", **arrays)
    t = pd.DataFrame({"shot_id": [0, 7], "p_static": [0.25, 0.5], "move_atoms": ["static", "pan"]}).set_index("shot_id")
    return scratch, acc, t, t.copy()


def test_missing_accepted_cache_file_fails(cold, tmp_path):
    """an earlier review mutation M5 (the failure for a missing ACCEPTED file deleted) survived the author's tests; this pins the criterion."""
    scratch, acc, new, old = case(tmp_path, {"a": np.arange(6.0)})
    next((acc / "flow").glob("shot7.*.npz")).unlink()
    r = cold.compare(scratch, acc, new, old)
    assert any("shot 7" in m and "accepted flow" in m for m in r["failures"])


def test_cache_of_only_empty_arrays_fails(cold, tmp_path):
    scratch, acc, new, old = case(tmp_path, {"a": np.zeros((0,))})
    r = cold.compare(scratch, acc, new, old)
    assert r["failures"]
