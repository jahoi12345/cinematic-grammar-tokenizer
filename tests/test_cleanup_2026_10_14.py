"""docs/cleanup_2026-10-14.md item 3: the cold-invocation evidence records the sha256 of the evidence script itself.
main() runs with the runner and git faked (no model, no film data), in a temporary ROOT; only the written evidence is checked."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_move_film_cold_evidence.py"


@pytest.fixture
def cold():
    spec = importlib.util.spec_from_file_location("cold_evidence_cleanup", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_evidence_records_own_sha256(cold, tmp_path, monkeypatch):
    film = "test_film"
    for rel in ("scripts/run_move_film.py", "src/cgt/p4_camera_move/film_move.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"stub {rel}\n")
    (tmp_path / "data/processed").mkdir(parents=True)
    table = pd.DataFrame({"shot_id": [0, 1], "p_static": [0.5, 0.25], "move_atoms": ["static", "pan-left"]})
    table.to_parquet(tmp_path / f"data/processed/{film}_move.parquet")
    (tmp_path / "results/phase4/film_move").mkdir(parents=True)

    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[0] == sys.executable and str(cmd[1]).endswith("run_move_film.py"):  # the runner: write its scratch outputs
            table.to_parquet(cmd[cmd.index("--out") + 1])
            Path(cmd[cmd.index("--report") + 1]).write_text("report\n")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cold, "ROOT", tmp_path)
    monkeypatch.setattr(cold.subprocess, "run", fake_run)
    monkeypatch.setattr(cold, "compare", lambda *a: {"failures": [], "worst": 0.0, "n_arrays": 1, "files": {}, "pdiff": 0.0, "equal_other": True})
    monkeypatch.setattr(sys, "argv", ["x", "--film", film, "--shots", "2"])
    cold.main()

    ev = json.loads((tmp_path / "results/phase4/film_move/cold_invocation.json").read_text())
    want = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert ev["evidence_script_sha256"] == want
    assert ev["evidence_script_sha256"] != ev["runner_sha256"]
    assert f"this evidence script `{want[:16]}`" in (tmp_path / "results/phase4/film_move/cold_invocation.md").read_text()
