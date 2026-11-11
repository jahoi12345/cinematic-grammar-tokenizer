# ruff: noqa: F811
"""Input-identity tests of the film pipeline: keyframe content in the Phase 3C cache identity, source-video identity in the Phase 1 cache identity, legacy-cache
adoption by mtime ordering, Phase 2 input identity, CPU table rebuild, token text artifact. Synthetic films and FakeEngine only."""
import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
from _p3c_fixtures import KF, _run, film_env  # noqa: F401
from PIL import Image
from test_tokens_v1 import root as token_root  # noqa: F401

from cgt.p3c_tokens import runner
from cgt.pipeline import identity, stages
from cgt.pipeline.orchestrator import Ctx

ALPHAS = {"light_type": 0.5, "light_cond": 0.5, "composition": 0.5}


@pytest.fixture
def p3c_env(film_env, monkeypatch):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    monkeypatch.setattr(stages, "_frozen_sha", lambda ctx: "a" * 64)
    monkeypatch.setattr(stages, "_selected_method", lambda ctx: "D")
    monkeypatch.setattr(stages, "_git_equal", lambda *a: True)
    rd = _run(root)
    runner.build_table("synthetic", rd, ALPHAS, root / "data/processed/synthetic_p3c.parquet")
    return root, shots, rd, Ctx("synthetic", {"public_domain": True}, root=root, p3c_model="2b")  # 2B runs are an explicit opt-in


def _set_mtime(p: Path, t: float):
    os.utime(p, (t, t))


# ---- identity helpers ----------------------------------------------------------------------------------------------------------------------
def test_cached_sha256_reuses_only_while_size_and_mtime_match(tmp_path, monkeypatch):
    f, cache = tmp_path / "v.bin", tmp_path / "c.json"
    f.write_bytes(b"AAAA")
    sha1, _ = identity.cached_sha256(f, cache)
    calls = []
    real = identity.file_sha256
    monkeypatch.setattr(identity, "file_sha256", lambda p: calls.append(p) or real(p))
    assert identity.cached_sha256(f, cache)[0] == sha1 and calls == []  # cache hit: not re-hashed
    f.write_bytes(b"BBBB")  # same size, new mtime: re-hashed, so the new content is seen
    assert identity.cached_sha256(f, cache)[0] != sha1 and len(calls) == 1
    f.write_bytes(b"BBBBB")  # changed size: re-hashed as well
    assert identity.cached_sha256(f, cache)[0] != sha1 and len(calls) == 2


def test_dry_run_hashing_leaves_no_cache_file(tmp_path):
    f = tmp_path / "v.bin"
    f.write_bytes(b"x")
    identity.cached_sha256(f, tmp_path / "c.json", write=False)
    assert not (tmp_path / "c.json").exists()


def test_digest_is_content_and_order_sensitive(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    c = tmp_path / "c.json"
    d1 = identity.digest_of_files([a, b], c)
    assert d1 == identity.digest_of_files([a, b], c) != identity.digest_of_files([b, a], c)
    b.write_bytes(b"22")
    assert identity.digest_of_files([a, b], c) != d1


# ---- O1: Phase 3C cache identity includes the keyframe images --------------------------------------------------------------------------------
def test_run_manifest_records_the_keyframe_content_digest(p3c_env):
    _root, _, rd, ctx = p3c_env
    m = json.loads((rd / "manifest.json").read_text())
    assert m["inputs_digest"] == stages._keyframe_digest(ctx) and len(m["inputs_digest"]) == 64


def test_replaced_keyframe_invalidates_the_stage_and_the_next_run_archives_and_rescored(p3c_env):
    root, _, rd, ctx = p3c_env
    assert stages.p3c_status(ctx).hit
    Image.new("RGB", (32, 32), (255, 0, 0)).save(root / KF.format(0))
    st = stages.p3c_status(ctx)
    assert not st.hit and "keyframe images changed" in st.detail and stages.p3c_needs_gpu(ctx)
    from _p3c_fixtures import FakeEngine
    eng = FakeEngine()
    rd2 = runner.run_film("synthetic", "D", root / "adapter", engine=eng, log=lambda m: None)
    assert rd2 == rd and eng.passes == 2 * 3 * 7  # every shot and keyframe scored again, nothing reused
    assert list(rd.parent.glob(f"{rd.name}.stale-*")), "the stale run must be archived, not deleted"
    runner.build_table("synthetic", rd2, ALPHAS, root / "data/processed/synthetic_p3c.parquet")
    assert stages.p3c_status(ctx).hit


def test_identical_content_with_new_mtime_is_still_a_hit(p3c_env):
    root, _, _, ctx = p3c_env
    p = root / KF.format(0)
    p.write_bytes(p.read_bytes())  # rewritten with the same bytes
    assert stages.p3c_status(ctx).hit


def test_shrunk_image_cache_is_content_addressed(film_env, monkeypatch):
    root, shots = film_env
    shots(KF.format(0))
    seen = []
    monkeypatch.setattr(runner, "shrunk_path", lambda path, d: seen.append(Path(d)) or path)
    _run(root)
    d1 = seen[0]
    Image.new("RGB", (32, 32), (9, 9, 9)).save(root / KF.format(0))
    seen.clear()
    _run(root)
    assert seen[0] != d1 and d1.name != seen[0].name  # another image content -> another shrunk-copy directory


def test_legacy_run_without_identity_is_never_adopted_even_with_old_timestamps(p3c_env):
    root, _, rd, ctx = p3c_env
    mf = rd / "manifest.json"
    m = json.loads(mf.read_text())
    m.pop("inputs_digest")
    m["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() + 3600))  # every mtime ordering says "not newer"
    mf.write_text(json.dumps(m))
    st = stages.p3c_status(ctx)
    assert not st.hit and "legacy run" in st.detail and stages.p3c_needs_gpu(ctx)
    from _p3c_fixtures import FakeEngine
    eng = FakeEngine()
    runner.run_film("synthetic", "D", root / "adapter", engine=eng, log=lambda m: None)
    assert eng.passes == 2 * 3 * 7 and list(rd.parent.glob(f"{rd.name}.stale-*"))  # rescored from the current keyframes; the legacy run kept as evidence


def test_truncated_score_cache_is_not_trusted_and_missing_table_needs_no_gpu(p3c_env):
    root, _, rd, ctx = p3c_env
    (root / "data/processed/synthetic_p3c.parquet").unlink()
    st = stages.p3c_status(ctx)
    assert not st.hit and "without the GPU" in st.detail and not stages.p3c_needs_gpu(ctx)
    runner.build_table("synthetic", rd, ALPHAS, root / "data/processed/synthetic_p3c.parquet")
    (rd / "scores_kf_end.jsonl").write_text("")
    st = stages.p3c_status(ctx)
    assert not st.hit and stages.p3c_needs_gpu(ctx) and "do not cover every shot" in st.detail


def test_p3c_run_rebuilds_the_table_from_a_valid_cache_without_scoring(p3c_env, monkeypatch):
    root, _, _rd, ctx = p3c_env
    (root / "data/processed/synthetic_p3c.parquet").unlink()
    (root / "results/phase3c").mkdir(parents=True)
    (root / "results/phase3c/method_selection.json").write_text(json.dumps({"results": {"D": {"per_attribute": {"light_type": {"alpha": 0.5}, "light_cond": {"alpha": 0.5}, "composition": {"alpha": 0.5}}}}}))
    monkeypatch.setattr(runner, "run_film", lambda *a, **k: pytest.fail("a valid cache must not trigger scoring"))
    counts = stages.p3c_run(ctx)
    assert counts["shots"] == 2 and (root / "data/processed/synthetic_p3c.parquet").exists()


# ---- O2: Phase 1 cache identity includes the source video ------------------------------------------------------------------------------------
@pytest.fixture
def p1_env(film_env):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    video = root / "synthetic.mp4"
    video.write_bytes(b"video-A")
    cuts = root / "data/interim/synthetic/cuts_t27.json"
    cuts.write_text(json.dumps([[0, 1], [1, 2]]))
    return root, video, cuts, Ctx("synthetic", {"public_domain": True, "path": "synthetic.mp4"}, root=root)


def test_p1_legacy_cache_is_never_adopted_and_a_recorded_identity_must_match(p1_env):
    _root, video, _cuts, ctx = p1_env
    s0 = stages.p1_status(ctx)
    assert not s0.hit and "legacy cache" in s0.detail  # no record: timestamps (video older than cuts) are not evidence
    stages._p1_write_record(ctx)  # what a Phase 1 RUN writes at its end
    s1 = stages.p1_status(ctx)
    assert s1.hit and "video, cuts, table, keyframe bytes" in s1.detail and s1.key == s0.key
    video.write_bytes(b"video-B")  # the registered source is replaced: no HIT and another key
    s2 = stages.p1_status(ctx)
    assert not s2.hit and s2.key != s1.key and "differ from the identity recorded" in s2.detail


def test_p1_record_must_bind_the_keyframe_bytes_and_the_table(p1_env):
    root, _video, _cuts, ctx = p1_env
    stages._p1_write_record(ctx)
    assert stages.p1_status(ctx).hit
    Image.new("RGB", (32, 32), (7, 7, 7)).save(root / KF.format(1))  # a keyframe replaced after Phase 1 built it
    st = stages.p1_status(ctx)
    assert not st.hit and "keyframe images differ" in st.detail
    stages._p1_write_record(ctx)
    assert stages.p1_status(ctx).hit
    df = pd.read_parquet(root / "data/processed/synthetic_shots.parquet")
    df.loc[0, "start_s"] = 99.0  # the shot table edited
    df.to_parquet(root / "data/processed/synthetic_shots.parquet")
    assert not stages.p1_status(ctx).hit


@pytest.mark.parametrize("old_style", [{"attested": "legacy_mtime"}, {"verified_by": "re-derivation of the cuts from the registered video"}])
def test_p1_records_not_written_by_a_phase1_run_are_refused(p1_env, old_style):
    root, _video, _cuts, ctx = p1_env
    stages._p1_write_record(ctx)
    rec = root / "data/interim/synthetic/p1_cache.json"
    r = json.loads(rec.read_text())
    r.pop("built_by")
    r.update(old_style)
    rec.write_text(json.dumps(r))
    assert not stages.p1_status(ctx).hit


def test_counterexample_legacy_cache_with_matching_cuts_but_images_from_elsewhere_is_rebuilt_not_adopted(p1_env, monkeypatch):
    """the independent verification pass's counterexample: the cuts re-derived from the registered video equal the legacy cache's cuts, yet its keyframes, tables and predictions may come from another source."""
    root, _video, cuts, ctx = p1_env
    import cgt.p1_shots_color.detection as det
    import cgt.p1_shots_color.pipeline as pl
    ctx.log = lambda m: None
    monkeypatch.setattr(det, "detect_shots", lambda path, thr: [(0, 1), (1, 2)], raising=False)  # equal to the cached cuts
    ran = []

    def fake_run(film, thr, **k):
        ran.append(thr)
        cuts.write_text(json.dumps([[0, 1], [1, 2]]))  # the real Phase 1 re-extracts everything, including keyframes
        return pd.DataFrame({"shot_id": [0, 1]})
    monkeypatch.setattr(pl, "run_film", fake_run)
    Image.new("RGB", (32, 32), (200, 10, 10)).save(root / KF.format(0))  # the legacy keyframes came from somewhere else
    assert not stages.p1_status(ctx).hit  # equal cuts do not make it a hit
    stages.p1_run(ctx)
    assert ran == [27.0], "a legacy cache must be rebuilt, never adopted on cut equality"
    rec = json.loads((root / "data/interim/synthetic/p1_cache.json").read_text())
    assert rec["built_by"] == "p1_run" and "verified_by" not in rec and "attested" not in rec


def test_p1_missing_video_cannot_establish_identity(p1_env):
    _root, video, _cuts, ctx = p1_env
    video.unlink()
    st = stages.p1_status(ctx)
    assert not st.hit and "missing" in st.detail


def test_p1_status_in_a_dry_run_writes_no_cache_files(p1_env):
    root, _video, _cuts, ctx = p1_env
    ctx.dry_run = True
    stages.p1_status(ctx)
    assert not (root / "data/cache/pipeline").exists()


def test_p1_recorded_threshold_or_cuts_change_refuses_the_hit(p1_env):
    _, _, cuts, ctx = p1_env
    stages._p1_write_record(ctx)
    cuts.write_text(json.dumps([[0, 1], [1, 3]]))
    assert not stages.p1_status(ctx).hit


# ---- Phase 2 input identity ----------------------------------------------------------------------------------------------------------------
@pytest.fixture
def p2_env(p1_env, monkeypatch):
    root, video, _cuts, ctx = p1_env
    from cgt.p2_pretrained import pipeline as p2
    monkeypatch.setattr(p2, "stage_key", lambda film, stage, shots, f: f"key-{stage}")
    shots = pd.read_parquet(root / "data/processed/synthetic_shots.parquet")
    cols = {"shot_id": shots.shot_id, **{c: 1 for s in p2.STAGES for c in p2.MARKERS[s]}}
    pd.DataFrame(cols).to_parquet(root / "data/processed/synthetic_p2.parquet")
    (root / "data/processed/synthetic_p2.meta.json").write_text(json.dumps({s: f"key-{s}" for s in p2.STAGES}))
    now = time.time()
    for p in [video, *[root / x for x in shots.kf_mid]]:
        _set_mtime(p, now - 1000)
    return root, video, ctx


def _p2_record(ctx, root):
    stages._p2_write_record(ctx)


def test_p2_legacy_table_is_never_adopted_and_a_recorded_identity_is_invalidated_by_changed_inputs(p2_env):
    root, _video, ctx = p2_env
    ctx.film = {**ctx.film, "subtitles": None}
    st = stages.p2_status(ctx)
    assert not st.hit and "legacy Phase 2 table" in st.detail  # old mtimes are not evidence
    _p2_record(ctx, root)
    st = stages.p2_status(ctx)
    assert st.hit and "recorded input identity" in st.detail
    Image.new("RGB", (32, 32), (1, 2, 3)).save(root / KF.format(0))  # mid keyframe replaced
    st = stages.p2_status(ctx)
    assert not st.hit and "changed since Phase 2" in st.detail


def test_p2_video_replacement_invalidates_the_audio_dependent_table(p2_env):
    root, video, ctx = p2_env
    ctx.film = {**ctx.film, "subtitles": None}
    _p2_record(ctx, root)
    assert stages.p2_status(ctx).hit
    video.write_bytes(b"another video")
    assert not stages.p2_status(ctx).hit


# ---- tokens text artifact (O3) -------------------------------------------------------------------------------------------------------------
def test_tokens_status_requires_the_text_file_and_its_content(token_root):
    from cgt.tokens import build_v1
    _, txt, _ = build_v1.write_tokens_v1("f", token_root)
    ctx = Ctx("f", {"public_domain": True}, root=token_root)
    assert stages.tokens_status(ctx).hit
    good = txt.read_text()
    txt.write_text(good.replace("size=", "SIZE=", 1))  # edited text file: no longer the rendering of the table
    assert not stages.tokens_status(ctx).hit
    txt.unlink()
    assert not stages.tokens_status(ctx).hit


def test_p2_table_bytes_are_bound_to_the_record(p2_env):
    root, _video, ctx = p2_env
    ctx.film = {**ctx.film, "subtitles": None}
    _p2_record(ctx, root)
    assert stages.p2_status(ctx).hit
    t = pd.read_parquet(root / "data/processed/synthetic_p2.parquet")
    t.iloc[0, 1] = 99  # same columns and ids, another value: not the table Phase 2 computed
    t.to_parquet(root / "data/processed/synthetic_p2.parquet")
    st = stages.p2_status(ctx)
    assert not st.hit and "differs from the bytes recorded" in st.detail


def test_p2_run_forces_a_full_recompute_when_the_record_cannot_vouch_for_the_table(p2_env, monkeypatch):
    root, _video, ctx = p2_env
    ctx.film = {**ctx.film, "subtitles": None}
    import cgt.p2_pretrained.pipeline as p2
    calls = []

    def fake_run(film, stages, force=False, **k):
        calls.append(force)
        return pd.read_parquet(root / "data/processed/synthetic_p2.parquet")
    monkeypatch.setattr(p2, "run_film", fake_run)
    shots = pd.read_parquet(root / "data/processed/synthetic_shots.parquet")
    ctx.manifest_dir.mkdir(parents=True, exist_ok=True)
    (ctx.manifest_dir / "p2.inputs.json").write_text(json.dumps(stages._p2_inputs(ctx, shots)))  # an older record: inputs but no table bytes
    stages.p2_run(ctx)
    assert calls == [True]
    stages.p2_run(ctx)  # now a complete record exists: no force
    assert calls == [True, False]
