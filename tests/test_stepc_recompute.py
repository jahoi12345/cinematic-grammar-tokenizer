"""an earlier review (Step C results review): runs the independent recomputation driver tests/vscr_stepc_recompute.py against the real
gitignored Step C outputs; skipped on a machine without them. CPU only."""
import subprocess
import sys
from pathlib import Path

import pytest

from cgt.worktrees import sibling

DRV = Path(__file__).with_name("vscr_stepc_recompute.py")
PRED = sibling("stepc_4b") / "data/cache/stepc/predictions"


@pytest.mark.skipif(not PRED.exists(), reason="Step C predictions not on this machine")
def test_vscr_recompute_all_probes_pass(tmp_path):
    r = subprocess.run([sys.executable, str(DRV), str(tmp_path / "out.json")], capture_output=True, text=True, timeout=1800, check=False)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


@pytest.mark.skipif(not PRED.exists(), reason="Step C predictions not on this machine")
@pytest.mark.parametrize("mutant", ["M1", "M2", "M3"])
def test_vscr_mutants_are_caught(tmp_path, mutant):
    r = subprocess.run([sys.executable, str(DRV), str(tmp_path / "out.json"), "--mutant", mutant], capture_output=True, text=True,
                       timeout=1800, check=False)
    assert r.returncode != 0, f"mutant {mutant} survived"
