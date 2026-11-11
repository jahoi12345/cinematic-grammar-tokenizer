"""CPU tests of the film pipeline orchestrator on synthetic stages (no model, no video) plus a dry-run check against NOTLD's existing caches when present."""
import json
from pathlib import Path

import pytest
import yaml

from cgt.pipeline import orchestrator as orch
from cgt.pipeline.gpulock import GpuLock, LockHeld
from cgt.pipeline.orchestrator import EXIT, Stage, Status

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def env(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/films.yaml").write_text(yaml.safe_dump({"films": [
        {"id": "pd", "title": "Public", "public_domain": True}, {"id": "owned", "title": "Owned", "public_domain": False}]}))
    return tmp_path


def make_registry(root, calls, fail=None, gpu_unavailable=False, bad_output=None):
    """p1 -> p2 -> g (GPU) -> t; each stage 'produces' a marker file; a stage is a hit iff its marker exists."""
    def stage(name, gpu=False, needs=None):
        marker = root / f"out_{name}"

        def status(ctx):
            return Status(marker.exists(), f"key-{name}-000000", "marker", [marker] if marker.exists() else [], {"n": 3}, 7.0 if marker.exists() else None)

        def run(ctx):
            calls.append(name)
            if fail == name:
                raise RuntimeError("boom")
            if bad_output != name:
                marker.write_text("x")
            return {"n": 3}

        def prereq(ctx):
            return [("unavailable_or_prerequisite", f"needs:{needs}")] if needs and not (root / f"out_{needs}").exists() else []
        return Stage(name, gpu, status, run, prereq, reference="ref")
    reg = {"p1": stage("p1"), "p2": stage("p2", needs="p1"), "p3c": stage("p3c", gpu=True, needs="p2"), "tokens": stage("tokens", needs="p3c")}
    mv = stage("move", gpu=True)
    mv.available = lambda ctx: "not integrated"
    reg["move"] = mv
    return reg


def go(env, reg, **kw):
    logs = []
    code = orch.run(kw.pop("film", "pd"), kw.pop("stages", None), reg, root=env, lock_path=env / "gpu.lock", log=logs.append, **kw)
    return code, "\n".join(logs)


def test_full_run_in_order_writes_manifests_and_summary_then_second_run_is_all_cache_hits(env):
    calls = []
    reg = make_registry(env, calls)
    code, _ = go(env, reg)
    assert code == 0 and calls == ["p1", "p2", "p3c", "tokens"]  # move is unavailable and silently skipped when not requested
    man = json.loads((env / "data/cache/pipeline/pd/p3c.json").read_text())
    assert man["status"] == "ran" and man["gpu"] is True and man["key"].startswith("key-p3c")
    summ = (env / "results/films/pd.md").read_text()
    assert "| p1 | CPU | ran" in summ and "| p3c | GPU | ran" in summ and "unavailable (not integrated)" in summ and "Cache hits: 0 of 4" in summ
    code, _ = go(env, reg)
    assert code == 0 and calls == ["p1", "p2", "p3c", "tokens"]  # nothing recomputed
    assert "Cache hits: 4 of 4" in (env / "results/films/pd.md").read_text()
    assert json.loads((env / "data/cache/pipeline/pd/p3c.json").read_text())["status"] in ("ran", "hit")


def test_dry_run_prints_the_plan_and_writes_and_runs_nothing(env):
    calls = []
    code, log = go(env, make_registry(env, calls), dry_run=True)
    assert code == 0 and calls == [] and "MISS: would RUN" in log and "UNAVAILABLE" in log and "dry run" in log
    assert not (env / "data").exists() and not (env / "results").exists() and not (env / "gpu.lock").exists()


def test_partial_hit_resumes_from_the_missing_stage(env):
    calls = []
    (env / "out_p1").write_text("x")
    (env / "out_p2").write_text("x")
    go(env, make_registry(env, calls))
    assert calls == ["p3c", "tokens"]


def test_stage_subset_and_prerequisite_failure(env):
    calls = []
    code, log = go(env, make_registry(env, calls), stages=["tokens"])
    assert code == EXIT["unavailable_or_prerequisite"] and calls == [] and "needs:p3c" in log
    code, _ = go(env, make_registry(env, calls), stages=["p1", "p2"])
    assert code == 0 and calls == ["p1", "p2"]


def test_explicit_request_of_an_unavailable_stage_fails_closed(env):
    calls = []
    code, log = go(env, make_registry(env, calls), stages=["move"])
    assert code == EXIT["unavailable_or_prerequisite"] and calls == [] and "unavailable" in log


def test_unknown_stage_is_a_usage_error(env):
    assert go(env, make_registry(env, []), stages=["nope"])[0] == EXIT["usage"]


def test_stage_failure_stops_the_run_records_the_error_and_releases_the_lock(env):
    calls = []
    code, _ = go(env, make_registry(env, calls, fail="p3c"))
    assert code == EXIT["stage_failed"] and calls == ["p1", "p2", "p3c"]  # tokens never ran
    man = json.loads((env / "data/cache/pipeline/pd/p3c.json").read_text())
    assert man["status"] == "failed" and "boom" in man["error"]
    assert not (env / "gpu.lock").exists()
    assert "failed" in (env / "results/films/pd.md").read_text()


def test_a_stage_whose_output_is_invalid_after_running_is_a_failure(env):
    code, _ = go(env, make_registry(env, [], bad_output="p1"))
    assert code == EXIT["stage_failed"]
    assert json.loads((env / "data/cache/pipeline/pd/p1.json").read_text())["status"] == "failed"


def test_running_state_is_on_disk_before_the_work_starts(env):
    reg = make_registry(env, [])
    seen = {}
    inner = reg["p1"].run

    def spy(ctx):
        seen["m"] = json.loads((env / "data/cache/pipeline/pd/p1.json").read_text())
        return inner(ctx)
    reg["p1"].run = spy
    go(env, reg, stages=["p1"])
    assert seen["m"]["status"] == "running" and "finished" not in seen["m"]


def test_held_gpu_lock_refuses_before_any_stage_starts(env):
    (env / "gpu.lock").write_text("other-job pid=1 nonce=abc\n")
    calls = []
    code, log = go(env, make_registry(env, calls))
    assert code == EXIT["gpu_lock_held"] and calls == [] and "other-job" in log
    assert (env / "gpu.lock").read_text().startswith("other-job")  # the other job's lock is untouched


def test_lock_is_not_taken_when_every_gpu_stage_is_a_cache_hit_or_only_cpu_stages_run(env):
    (env / "out_p3c").write_text("x")
    (env / "gpu.lock").write_text("other-job nonce=abc\n")
    calls = []
    for n in ("p1", "p2"):
        (env / f"out_{n}").write_text("x")
    assert go(env, make_registry(env, calls))[0] == 0 and calls == ["tokens"]
    calls.clear()
    (env / "out_tokens").unlink()
    assert go(env, make_registry(env, calls), stages=["p1", "p2"])[0] == 0  # CPU-only request never needs the lock


def test_lock_is_held_during_gpu_stage_and_only_released_by_its_owner(env):
    reg = make_registry(env, [])
    inner = reg["p3c"].run
    state = {}

    def spy(ctx):
        state["during"] = (env / "gpu.lock").read_text()
        (env / "gpu.lock").write_text("someone-else nonce=zzz\n")  # a foreign process replaced the lock while we ran
        return inner(ctx)
    reg["p3c"].run = spy
    go(env, reg)
    assert "run_film:pd" in state["during"] and "nonce=" in state["during"]
    assert (env / "gpu.lock").read_text().startswith("someone-else")  # not removed: the nonce is not ours


def test_gpulock_semantics(tmp_path):
    p = tmp_path / "l"
    a = GpuLock("a", p).acquire()
    with pytest.raises(LockHeld):
        GpuLock("b", p).acquire()
    assert GpuLock("b", p).release() is False and p.exists()  # never acquired: cannot remove
    assert a.release() is True and not p.exists() and a.release() is False
    with GpuLock("c", p):
        assert p.exists()
    assert not p.exists()


def test_owned_film_is_refused_unless_asked_and_then_runs_locally_without_a_results_file(env):
    calls = []
    assert go(env, make_registry(env, calls), film="owned")[0] == EXIT["film_not_allowed"] and calls == []
    # with the explicit flag every stage, including the GPU one, may run: local inference only
    code, log = go(env, make_registry(env, calls), film="owned", allow_owned=True)
    assert code == 0 and calls == ["p1", "p2", "p3c", "tokens"] and "OWNED: LOCAL INFERENCE ONLY" in log
    assert not (env / "results").exists()  # nothing is written under results/ for an owned film
    summary = (env / "data/cache/pipeline/owned/summary.md").read_text()
    assert "OWNED FILM: LOCAL ONLY" in summary and "| p3c | GPU | ran" in summary
    assert not (env / "gpu.lock").exists()


def test_owned_film_can_never_be_published(env):
    calls = []
    assert go(env, make_registry(env, calls), film="owned", allow_owned=True, stages=["p1"], publish=True)[0] == EXIT["film_not_allowed"] and calls == []


def test_unregistered_film_is_refused(env):
    assert go(env, make_registry(env, []), film="ghost")[0] == EXIT["film_not_allowed"]


def test_cli_usage_errors_return_exit_2():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_film_cli", REPO / "scripts/run_film.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main([]) == EXIT["usage"] and m.main(["--film", "x", "--bogus"]) == EXIT["usage"]


@pytest.mark.skipif(not (REPO / "data/processed/night_of_the_living_dead_shots.parquet").exists() or not (REPO / "data/cache/p3c/runs").exists(),
                    reason="NOTLD caches not present")
def test_notld_dry_run_refuses_legacy_caches_and_computes_nothing(tmp_path, capsys):
    """Real stage definitions against the real caches: dry-run only (no lock, no write, no model, no video). Caches that predate content identity are not adopted."""
    from cgt.pipeline.stages import registry
    before = {p: p.stat().st_mtime_ns for p in (REPO / "data/processed").glob("night_of_the_living_dead*")}
    orch.run("night_of_the_living_dead", ["p1", "p3c"], registry(), dry_run=True, lock_path=tmp_path / "x.lock", p3c_model="2b")
    out = capsys.readouterr().out
    assert "legacy" in out and "MISS" in out and not (tmp_path / "x.lock").exists()
    assert before == {p: p.stat().st_mtime_ns for p in (REPO / "data/processed").glob("night_of_the_living_dead*")}
