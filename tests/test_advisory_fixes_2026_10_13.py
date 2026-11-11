"""Advisory fixes F-V4, F-V5 (orchestrator movement and tokens stages) and F-V1 (cold-invocation evidence script) from an earlier review's review, a verification record.
Synthetic films, stubbed runner and token builder; no data, no GPU. Criteria: docs/advisory_fixes_2026-10-13.md."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cgt.pipeline import orchestrator as orch
from cgt.pipeline import stages
from cgt.pipeline.orchestrator import Ctx, PipelineError

ROOT = Path(__file__).resolve().parents[1]
SHOTS = [0, 1, 2]


# ---------------------------------------------------------------- shared synthetic film
@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": [{"id": "pd", "public_domain": True, "path": "pd.mp4"}]}))
    (root / "data/processed").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "data/cache/p4").mkdir(parents=True)
    for rel in stages.MOVE_CODE_FILES + stages.MOVE_CONFIG_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"v1 {rel}".encode())
    for rel in stages.MOVE_PREREQ_FILES:  # fake empty ShotQA train label file (move prerequisite since F1, docs/move_prereq_fix_2026-10-13.md; never read)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    (root / "pd.mp4").write_bytes(b"video-pd")
    pd.DataFrame({"shot_id": SHOTS, "start_s": [0.0, 1.0, 2.0], "end_s": [1.0, 2.0, 3.0], "len_s": 1.0}).to_parquet(root / "data/processed/pd_shots.parquet")
    monkeypatch.setattr(stages, "_free_percent", lambda: 80)
    hook = {}

    def fake_script(ctx, cmd):
        pd.DataFrame({"shot_id": SHOTS, "move_atoms": ["static", "push-in + pan-left", "pan-left"], "fallback_top1": [False, False, True], "no_features": False}).to_parquet(stages._move_table(ctx))
        if "during" in hook:
            hook["during"](ctx)
        return 0
    monkeypatch.setattr(stages, "_run_move_script", fake_script)
    monkeypatch.setattr("cgt.p4_camera_move.film_move.describe", lambda sets, fb: {"n_shots": len(sets), "median_set_size": 1.0, "contradictory_share": 0.0})
    return root, hook


def ctx_for(root):
    return Ctx("pd", {"id": "pd", "public_domain": True, "path": "pd.mp4"}, root=root, allow_owned=False, log=lambda m: None)


def run_move(root):
    return orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None)


# ---------------------------------------------------------------- F-V4
EDITS = {  # each call writes different bytes (n counts the edits), so every run sees a real change
    "code file": lambda root, n: (root / "src/cgt/p4_camera_move/flow.py").write_bytes(f"edited during run {n}".encode()),
    "config file": lambda root, n: (root / "results/phase4/frozen_config_flow_depth.json").write_bytes(f'{{"edit": {n}}}'.encode()),
    "shot table": lambda root, n: pd.DataFrame({"shot_id": SHOTS, "start_s": [0.0, 1.0 + n / 10, 2.0], "end_s": [1.0 + n / 10, 2.0, 3.0], "len_s": 1.0}).to_parquet(root / "data/processed/pd_shots.parquet"),
    "video": lambda root, n: (root / "pd.mp4").write_bytes(b"another video" + b"x" * n),
}


def editing_hook(root, what):
    n = [0]

    def during(ctx):
        n[0] += 1
        EDITS[what](root, n[0])
    return during


@pytest.mark.parametrize("what", list(EDITS))
def test_fv4_input_changed_during_the_run_fails_and_writes_no_record(env, what):
    root, hook = env
    hook["during"] = editing_hook(root, what)
    ctx = ctx_for(root)
    with pytest.raises(PipelineError) as e:
        stages.move_run(ctx)
    assert e.value.code == "stage_failed" and "changed while the runner ran" in str(e.value)
    assert not stages._move_record_path(ctx).exists()
    assert not stages.move_table_verified(ctx)[0]
    assert run_move(root) != 0  # through the orchestrator too


def test_fv4_record_binds_the_inputs_seen_before_the_run(env):
    root, _ = env
    ctx = ctx_for(root)
    before = stages._move_inputs(ctx)
    assert run_move(root) == 0
    r = json.loads(stages._move_record_path(ctx).read_text())
    assert r["built_by"] == "move_stage" and {k: r[k] for k in before} == before
    assert r["table_sha256"] == stages._sha_file(stages._move_table(ctx))
    assert stages.move_table_verified(ctx) == (True, "")


def test_fv4_stale_record_from_an_earlier_run_does_not_verify_after_a_changed_input_run(env):
    root, hook = env
    ctx = ctx_for(root)
    assert run_move(root) == 0
    (root / "src/cgt/p4_camera_move/labels.py").write_bytes(b"v2 labels")  # a real input change: the stage must run again
    hook["during"] = editing_hook(root, "code file")
    assert run_move(root) != 0
    assert json.loads(stages._move_record_path(ctx).read_text())["code_identity"] != stages._move_inputs(ctx)["code_identity"]  # the earlier record stays and does not match
    assert not stages.move_table_verified(ctx)[0]


# ---------------------------------------------------------------- F-V5
@pytest.fixture
def tok_env(env, monkeypatch):
    """Tokens inputs present; the runner script is ABSENT from the tree; build_v1 is stubbed and records whether it was called."""
    root, _ = env
    for name in ("p3c", "p2"):
        pd.DataFrame({"shot_id": SHOTS}).to_parquet(root / f"data/processed/pd_{name}.parquet")
    (root / "data/cache/tokens").mkdir(parents=True)
    (root / "data/cache/tokens/palette_global_k32.npz").write_bytes(b"x")
    (root / "scripts/run_move_film.py").unlink()
    calls = []
    from cgt.tokens import build_v1

    def fake_write(film, r):
        calls.append(film)
        out = r / f"data/processed/{film}_tokens_v1.parquet"
        pd.DataFrame({"shot_id": SHOTS, "kf_all_agree": True}).to_parquet(out)
        return out, None, None
    monkeypatch.setattr(build_v1, "write_tokens_v1", fake_write)
    monkeypatch.setattr(build_v1, "check_complete", lambda film, r: None)
    monkeypatch.setattr(build_v1, "input_hashes", lambda film, r: {"x": "y"})
    return root, calls


def write_identityless_table(ctx):
    pd.DataFrame({"shot_id": SHOTS, "move_atoms": ["static"] * 3, "fallback_top1": False, "no_features": False}).to_parquet(stages._move_table(ctx))


def test_fv5_runner_absent_identityless_table_is_refused_everywhere(tok_env):
    root, calls = tok_env
    ctx = ctx_for(root)
    write_identityless_table(ctx)
    assert stages.move_available(ctx) is not None
    assert any(m.startswith("needs:move") for _, m in stages.tokens_prereq(ctx))
    st = stages.tokens_status(ctx)
    assert not st.hit and "unverified movement table" in st.detail
    with pytest.raises(PipelineError) as e:
        stages.tokens_run(ctx)
    assert e.value.code == "unavailable_or_prerequisite" and calls == []  # refused before build_v1 could read the table


def test_fv5_runner_absent_no_table_is_not_blocked(tok_env):
    root, calls = tok_env
    ctx = ctx_for(root)
    assert not any("move" in m for _, m in stages.tokens_prereq(ctx))
    stages.tokens_run(ctx)
    assert calls == ["pd"]


def test_fv5_verified_table_is_consumed_with_the_runner_absent(env, tok_env):
    root, calls = tok_env
    ctx = ctx_for(root)
    (root / "scripts/run_move_film.py").write_bytes(b"v1 scripts/run_move_film.py")
    assert run_move(root) == 0
    (root / "scripts/run_move_film.py").unlink()  # the code identity covers the runner: its removal makes the table stale, so it is refused
    assert not stages.move_table_verified(ctx)[0] and any(m.startswith("needs:move") for _, m in stages.tokens_prereq(ctx))
    (root / "scripts/run_move_film.py").write_bytes(b"v1 scripts/run_move_film.py")
    assert stages.move_table_verified(ctx)[0] and stages.tokens_prereq(ctx) == []
    stages.tokens_run(ctx)
    assert calls == ["pd"]


def test_fv5_runner_present_no_table_still_waits_for_move(tok_env):
    root, _ = tok_env
    (root / "scripts/run_move_film.py").write_bytes(b"v1 scripts/run_move_film.py")
    assert any(m.startswith("needs:move") for _, m in stages.tokens_prereq(ctx_for(root)))


# ---------------------------------------------------------------- F-V1
@pytest.fixture(scope="module")
def cold():
    spec = importlib.util.spec_from_file_location("cold_evidence", ROOT / "scripts/run_move_film_cold_evidence.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_case(tmp_path, shots=(0, 7)):
    scratch, acc = tmp_path / "scratch", tmp_path / "acc"
    rng = np.random.default_rng(0)
    for sid in shots:
        for kind in ("flow", "depth"):
            arrs = {"a": rng.normal(size=(5, 3)), "b": np.array([1.0, np.nan, 2.0])}
            for base in (scratch, acc):
                (base / kind).mkdir(parents=True, exist_ok=True)
                np.savez(base / kind / f"shot{sid}.k1.k2.f.0-10.r24.npz", **arrs)
    table = pd.DataFrame({"shot_id": list(shots), "p_static": [0.25, 0.5], "p_pan": [0.75, 0.5], "move_atoms": ["static", "pan-left"], "hash": [None, "x"]}).set_index("shot_id")
    return scratch, acc, table, table.copy()


def test_fv1_tolerances_declared(cold):
    assert cold.ARRAY_TOL == 0.0 and cold.PROB_TOL == 1e-12


def test_fv1_control_passes(cold, tmp_path):
    scratch, acc, new, old = make_case(tmp_path)
    old["p_static"] = old["p_static"] + 5e-16
    r = cold.compare(scratch, acc, new, old)
    assert r["failures"] == [] and r["n_arrays"] == 8 and r["worst"] == 0.0 and r["equal_other"]


def test_fv1_empty_scratch_cache_fails(cold, tmp_path):
    scratch, acc, new, old = make_case(tmp_path)
    for f in scratch.rglob("*.npz"):
        f.unlink()
    r = cold.compare(scratch, acc, new, old)
    assert r["n_arrays"] == 0 and any("no cache array" in m for m in r["failures"])


def test_fv1_missing_depth_file_fails(cold, tmp_path):
    scratch, acc, new, old = make_case(tmp_path)
    next((scratch / "depth").glob("shot7.*.npz")).unlink()
    r = cold.compare(scratch, acc, new, old)
    assert any("shot 7" in m and "depth" in m for m in r["failures"])


def test_fv1_array_difference_fails(cold, tmp_path):
    scratch, acc, new, old = make_case(tmp_path)
    p = next((scratch / "flow").glob("shot0.*.npz"))
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    d["a"][0, 0] += 1e-3
    np.savez(p, **d)
    r = cold.compare(scratch, acc, new, old)
    assert any("cache arrays" in m for m in r["failures"]) and r["worst"] == pytest.approx(1e-3)


def test_fv1_probability_difference_fails(cold, tmp_path):
    scratch, acc, new, old = make_case(tmp_path)
    old.loc[7, "p_pan"] += 1e-3
    assert any("probabilities" in m for m in cold.compare(scratch, acc, new, old)["failures"])


def test_fv1_other_column_difference_fails(cold, tmp_path):
    scratch, acc, new, old = make_case(tmp_path)
    old.loc[7, "move_atoms"] = "static"
    assert any("other than probabilities" in m for m in cold.compare(scratch, acc, new, old)["failures"])


def test_fv1_main_exits_nonzero_on_failure(cold, tmp_path, monkeypatch):
    """End to end with a stubbed runner subprocess that writes NO cache files: the script writes the evidence (pass false) and exits 1."""
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "src/cgt/p4_camera_move").mkdir(parents=True)
    (root / "scripts/run_move_film.py").write_text("# stub")
    (root / "src/cgt/p4_camera_move/film_move.py").write_text("# stub")
    (root / "results/phase4/film_move").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    (root / "data/cache/p4/notld").mkdir(parents=True)
    table = pd.DataFrame({"shot_id": [0], "p_static": [1.0], "move_atoms": ["static"]})
    table.to_parquet(root / "data/processed/night_of_the_living_dead_move.parquet")
    real_run = cold.subprocess.run

    class R:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        if cmd[0] == "git":
            return real_run(["true"], capture_output=True, text=True, check=False)
        out = Path(cmd[cmd.index("--out") + 1])
        table.to_parquet(out)
        Path(cmd[cmd.index("--report") + 1]).write_text("report")
        return R()
    monkeypatch.setattr(cold, "ROOT", root)
    monkeypatch.setattr(cold.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["x", "--shots", "1"])
    with pytest.raises(SystemExit) as e:
        cold.main()
    assert e.value.code == 1
    ev = json.loads((root / "results/phase4/film_move/cold_invocation.json").read_text())
    assert ev["pass"] is False and ev["n_cached_arrays"] == 0 and ev["failures"]
