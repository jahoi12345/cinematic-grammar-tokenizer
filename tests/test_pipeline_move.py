"""Movement stage of the film orchestrator (criteria 1, 2, 7 of docs/w3_orchestrator_move_criteria.md): order, CPU-only, identity-bound table, refusals. Synthetic film and a fake runner."""
import json

import pandas as pd
import pytest
import yaml

from cgt.pipeline import orchestrator as orch
from cgt.pipeline import stages
from cgt.pipeline.orchestrator import EXIT, Ctx

ATOMS = ["static", "push-in", "pan-left"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path
    (root / "configs").mkdir()
    (root / "configs/films.yaml").write_text(yaml.safe_dump({"films": [{"id": "pd", "public_domain": True, "path": "pd.mp4"}, {"id": "owned", "public_domain": False, "path": "owned.mp4"}]}))
    (root / "data/processed").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts/run_move_film.py").write_text("# fake runner")
    (root / "data/cache/p4").mkdir(parents=True)
    for f in stages.MOVE_CONFIG_FILES[2:]:
        (root / f).write_bytes(b"train features")
    for rel in stages.MOVE_PREREQ_FILES:  # fake empty ShotQA train label file (move prerequisite since F1, docs/move_prereq_fix_2026-10-13.md; never read)
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    for f in ("pd", "owned"):
        (root / f"{f}.mp4").write_bytes(f"video-{f}".encode())
        pd.DataFrame({"shot_id": [0, 1, 2], "start_s": [0.0, 1.0, 2.0], "end_s": [1.0, 2.0, 3.0], "len_s": 1.0}).to_parquet(root / f"data/processed/{f}_shots.parquet")
    monkeypatch.setattr(stages, "_free_percent", lambda: 80)
    calls = []

    def fake_script(ctx, cmd):
        calls.append(cmd)
        t = pd.DataFrame({"shot_id": [0, 1, 2], "move_atoms": ["static", "push-in + pan-left", "pan-left"], "fallback_top1": [False, False, True], "no_features": False})
        t.to_parquet(stages._move_table(ctx))
        return 0
    monkeypatch.setattr(stages, "_run_move_script", fake_script)
    monkeypatch.setattr("cgt.p4_camera_move.film_move.describe", lambda sets, fb: {"n_shots": len(sets), "median_set_size": 1.0, "contradictory_share": 0.0})
    return root, calls


def ctx_for(root, film="pd", allow_owned=False):
    entry = {"id": film, "public_domain": film == "pd", "path": f"{film}.mp4"}
    return Ctx(film, entry, root=root, allow_owned=allow_owned, log=lambda m: None)


def registry_without_gpu_work(order_log):
    """Registry of the REAL move/tokens stages with the other stages replaced by recorders (p1..p3c are not under test here)."""
    reg = stages.registry()
    for name in ("p1", "p2", "p3c"):
        reg[name] = orch.Stage(name, name == "p3c", lambda ctx: orch.Status(True, "k"), lambda ctx: {}, lambda ctx: [])
    inner_move, _inner_tok = reg["move"].run, reg["tokens"].run
    reg["move"].run = lambda ctx: (order_log.append("move"), inner_move(ctx))[1]
    reg["tokens"].run = lambda ctx: (order_log.append("tokens"), {"rows": 3})[1]
    reg["tokens"].status = lambda ctx: orch.Status("tokens" in order_log, "t")
    reg["tokens"].prerequisites = lambda ctx: [x for x in stages.tokens_prereq(ctx) if "needs:move" in x[1]]  # only the movement prerequisite is under test
    return reg


def test_order_is_p1_p2_p3c_move_tokens_and_move_is_cpu():
    assert orch.ORDER == ["p1", "p2", "p3c", "move", "tokens"]
    reg = stages.registry()
    assert reg["move"].gpu is False and reg["move"].needs_gpu is None and reg["p3c"].gpu is True


def test_run_executes_move_before_tokens_and_never_takes_the_gpu_lock(env, monkeypatch, tmp_path):
    root, calls = env
    log, lock = [], tmp_path / "gpu.lock"
    code = orch.run("pd", ["move", "tokens"], registry_without_gpu_work(log), root=root, lock_path=lock, log=lambda m: None)
    assert code == 0 and log == ["move", "tokens"]
    assert not lock.exists()  # a CPU-only plan never creates the lock
    assert "-n" in calls[0] and calls[0][calls[0].index("-n") + 1] == "19" and calls[0][calls[0].index("--workers") + 1] == "3" and "--allow-owned" not in calls[0]


def test_held_gpu_lock_does_not_block_the_cpu_movement_stage(env, tmp_path):
    root, calls = env
    lock = tmp_path / "gpu.lock"
    lock.write_text("other-job nonce=abc\n")
    code = orch.run("pd", ["move"], stages.registry(), root=root, lock_path=lock, log=lambda m: None)
    assert code == 0 and len(calls) == 1 and lock.read_text().startswith("other-job")


def test_owned_film_is_refused_without_the_flag_and_gets_the_runner_flag_with_it(env):
    root, calls = env
    assert orch.run("owned", ["move"], stages.registry(), root=root, log=lambda m: None) == EXIT["film_not_allowed"] and calls == []
    assert orch.run("owned", ["move"], stages.registry(), allow_owned=True, root=root, log=lambda m: None) == 0
    assert "--allow-owned" in calls[0]


def test_memory_guard_fails_closed_below_25_percent(env, monkeypatch):
    root, calls = env
    monkeypatch.setattr(stages, "_free_percent", lambda: 24)
    code = orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None)
    assert code == EXIT["memory_guard"] and calls == [] and not stages._move_table(ctx_for(root)).exists()
    monkeypatch.setattr(stages, "_free_percent", lambda: 25)
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0


def test_second_run_is_a_cache_hit_and_dry_run_shows_the_state(env, capsys):
    root, calls = env
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=print) == 0 and len(calls) == 1
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0 and len(calls) == 1  # idempotent
    orch.run("pd", ["move"], stages.registry(), dry_run=True, root=root)
    out = capsys.readouterr().out
    assert "move" in out and "cache HIT" in out and "verified movement table" in out
    (stages._move_table(ctx_for(root))).unlink()
    orch.run("pd", ["move"], stages.registry(), dry_run=True, root=root)
    assert "MISS" in capsys.readouterr().out


def test_identity_less_or_legacy_table_is_never_adopted_and_is_recomputed(env):
    root, calls = env
    ctx = ctx_for(root)
    pd.DataFrame({"shot_id": [0, 1, 2], "move_atoms": ["static"] * 3, "fallback_top1": False, "no_features": False}).to_parquet(stages._move_table(ctx))  # accepted-style table, no record
    st = stages.move_status(ctx)
    assert not st.hit and "legacy movement table" in st.detail
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0 and len(calls) == 1  # recomputed
    # a record that was not written by the movement stage is refused as well
    rec = stages._move_record_path(ctx)
    r = json.loads(rec.read_text())
    r["built_by"] = "hand"
    rec.write_text(json.dumps(r))
    assert not stages.move_status(ctx).hit


@pytest.mark.parametrize("what", ["video", "shots", "code", "table"])
def test_stale_table_is_refused_when_any_bound_input_changes(env, what):
    root, _calls = env
    ctx = ctx_for(root)
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0 and stages.move_status(ctx).hit
    if what == "video":
        (root / "pd.mp4").write_bytes(b"another video")
    elif what == "shots":
        s = pd.read_parquet(root / "data/processed/pd_shots.parquet")
        s.loc[0, "end_s"] = 5.0
        s.to_parquet(root / "data/processed/pd_shots.parquet")
    elif what == "code":
        (root / "scripts/run_move_film.py").write_text("# changed runner")
    else:
        t = pd.read_parquet(stages._move_table(ctx))
        t.loc[0, "move_atoms"] = "pan-left"
        t.to_parquet(stages._move_table(ctx))
    st = stages.move_status(ctx)
    assert not st.hit


@pytest.mark.parametrize("bad", [{"shot_id": [0, 1], "move_atoms": ["static", "static"]},                                   # missing shot
                                 {"shot_id": [0, 1, 2, 2], "move_atoms": ["static"] * 4},                                    # duplicated shot id / wrong count
                                 {"shot_id": [0, 1, 7], "move_atoms": ["static"] * 3},                                       # other shot ids
                                 {"shot_id": [0, 1, 2], "move_atoms": ["static", "", "static"]},                             # empty set
                                 {"shot_id": [0, 1, 2], "move_atoms": ["static", "fly-through", "static"]}])                 # atom outside the schema
def test_table_that_does_not_cover_exactly_the_shots_or_has_bad_atoms_fails_the_stage(env, monkeypatch, bad):
    root, _calls = env

    def bad_script(ctx, cmd):
        pd.DataFrame({**bad, "fallback_top1": False, "no_features": False}).to_parquet(stages._move_table(ctx))
        return 0
    monkeypatch.setattr(stages, "_run_move_script", bad_script)
    code = orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None)
    assert code == EXIT["stage_failed"] and not stages._move_record_path(ctx_for(root)).exists()  # no identity record for a bad table


def test_runner_failure_is_a_stage_failure_without_a_record(env, monkeypatch):
    root, _ = env
    monkeypatch.setattr(stages, "_run_move_script", lambda ctx, cmd: 3)
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == EXIT["stage_failed"]
    assert not stages._move_record_path(ctx_for(root)).exists()


def test_tokens_stage_requires_a_verified_movement_table(env):
    root, _calls = env
    ctx = ctx_for(root)
    for name in ("p3c", "p2"):
        pd.DataFrame({"shot_id": [0, 1, 2]}).to_parquet(root / f"data/processed/pd_{name}.parquet")
    (root / "data/cache/tokens").mkdir(parents=True)
    (root / "data/cache/tokens/palette_global_k32.npz").write_bytes(b"x")
    probs = stages.tokens_prereq(ctx)
    assert any("needs:move" in m for _, m in probs)  # no movement table at all
    pd.DataFrame({"shot_id": [0, 1, 2], "move_atoms": ["static"] * 3, "fallback_top1": False, "no_features": False}).to_parquet(stages._move_table(ctx))
    assert any("legacy movement table" in m for _, m in stages.tokens_prereq(ctx))  # present but identity-less: not consumed
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0
    assert not any("needs:move" in m for _, m in stages.tokens_prereq(ctx))


def test_verification_rechecks_coverage_even_when_the_recorded_table_bytes_match(env):
    root, _calls = env
    ctx = ctx_for(root)
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0
    t = pd.read_parquet(stages._move_table(ctx)).iloc[:2]  # a table that lost a shot but whose bytes were re-recorded
    t.to_parquet(stages._move_table(ctx))
    rec = stages._move_record_path(ctx)
    r = json.loads(rec.read_text())
    r["table_sha256"] = stages._sha_file(stages._move_table(ctx))
    rec.write_text(json.dumps(r))
    ok, why = stages.move_table_verified(ctx)
    assert not ok and "cover exactly" in why and not stages.move_status(ctx).hit


def test_tokens_status_is_a_miss_while_the_movement_table_is_unverified(env, monkeypatch):
    """Token rows built from an identity-less or stale movement table are never a cache hit."""
    root, _calls = env
    ctx = ctx_for(root)
    from cgt.tokens import build, build_v1
    monkeypatch.setattr(build_v1, "input_hashes", lambda film, r: {"x": "1"})
    monkeypatch.setattr(build_v1, "check_complete", lambda film, r: None)
    monkeypatch.setattr(build, "text_lines", lambda df: ["a"])
    (root / "data/cache/tokens").mkdir(parents=True)
    (root / "data/cache/tokens/palette_global_k32.npz").write_bytes(b"x")
    for n in ("shots", "p2", "p3c"):
        (root / f"data/processed/pd_{n}.parquet").exists() or pd.DataFrame({"shot_id": [0, 1, 2]}).to_parquet(root / f"data/processed/pd_{n}.parquet")
    pd.DataFrame({"kf_all_agree": [True] * 3}).to_parquet(root / "data/processed/pd_tokens_v1.parquet")
    (root / "data/processed/pd_tokens_v1.txt").write_text("a\n")
    (root / "data/processed/pd_tokens_v1.meta.json").write_text(json.dumps({"inputs": {"x": "1"}}))
    pd.DataFrame({"shot_id": [0, 1, 2], "move_atoms": ["static"] * 3, "fallback_top1": False, "no_features": False}).to_parquet(stages._move_table(ctx))  # no identity record
    st = stages.tokens_status(ctx)
    assert not st.hit and "unverified movement table" in st.detail
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0
    assert stages.tokens_status(ctx).hit


def test_missing_train_feature_tables_or_shot_table_are_prerequisite_failures(env):
    root, calls = env
    (root / stages.MOVE_CONFIG_FILES[2]).unlink()
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == EXIT["unavailable_or_prerequisite"] and calls == []
    (root / stages.MOVE_CONFIG_FILES[2]).write_bytes(b"x")
    (root / "data/processed/pd_shots.parquet").unlink()
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == EXIT["unavailable_or_prerequisite"] and calls == []


def test_changed_train_feature_table_invalidates_the_movement_table(env):
    root, _calls = env
    ctx = ctx_for(root)
    assert orch.run("pd", ["move"], stages.registry(), root=root, log=lambda m: None) == 0 and stages.move_status(ctx).hit
    (root / stages.MOVE_CONFIG_FILES[3]).write_bytes(b"other training features")
    assert not stages.move_status(ctx).hit
