# ruff: noqa: F811
"""Verify actual stage cache decisions using synthetic frames/tables, never a real film."""
import json

from _p3c_fixtures import KF, _run, film_env  # noqa: F401
from PIL import Image
from test_tokens_v1 import root as token_root  # noqa: F401

from cgt.p3c_tokens import runner
from cgt.pipeline import stages
from cgt.pipeline.orchestrator import Ctx
from cgt.tokens import build_v1


def test_p3c_stage_invalidates_changed_input_keyframe(film_env, monkeypatch):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    rd = _run(root)
    runner.build_table('synthetic', rd, {'light_type': 0.5, 'light_cond': 0.5, 'composition': 0.5},
                       root / 'data/processed/synthetic_p3c.parquet')
    monkeypatch.setattr(stages, '_frozen_sha', lambda ctx: 'a' * 64)
    monkeypatch.setattr(stages, '_selected_method', lambda ctx: 'D')
    monkeypatch.setattr(stages, '_git_equal', lambda *args: True)  # scoring source unchanged
    ctx = Ctx('synthetic', {'public_domain': True}, root=root, p3c_model='2b')  # 2B runs are an explicit opt-in
    assert stages.p3c_status(ctx).hit
    Image.new('RGB', (32, 32), (255, 0, 0)).save(root / KF.format(0))
    assert not stages.p3c_status(ctx).hit, 'same shot ids hid a replaced image; old predictions reused'


def test_tokens_stage_requires_the_advertised_text_output(token_root):
    out, txt, _ = build_v1.write_tokens_v1('f', token_root)
    ctx = Ctx('f', {'public_domain': True}, root=token_root)
    assert out.exists() and stages.tokens_status(ctx).hit
    txt.unlink()
    assert not stages.tokens_status(ctx).hit, 'cache hit omits the required .txt artifact'


def test_p3c_stage_does_not_trust_complete_flag_after_cache_truncation(film_env, monkeypatch):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    rd = _run(root)
    runner.build_table('synthetic', rd, {'light_type': 0.5, 'light_cond': 0.5, 'composition': 0.5},
                       root / 'data/processed/synthetic_p3c.parquet')
    monkeypatch.setattr(stages, '_frozen_sha', lambda ctx: 'a' * 64)
    monkeypatch.setattr(stages, '_selected_method', lambda ctx: 'D')
    monkeypatch.setattr(stages, '_git_equal', lambda *args: True)
    ctx = Ctx('synthetic', {'public_domain': True}, root=root, p3c_model='2b')  # 2B runs are an explicit opt-in
    assert stages.p3c_status(ctx).hit
    (rd / 'scores_kf_mid.jsonl').write_text(json.dumps({'shot_id': 0}) + '\n')
    assert not stages.p3c_status(ctx).hit, 'complete manifest must not hide missing/invalid score rows'


def test_p1_stage_cache_key_changes_when_registered_video_is_replaced(film_env, monkeypatch):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    video = root / 'synthetic.mp4'
    video.write_bytes(b'synthetic-input-A')
    cuts = root / 'data/interim/synthetic/cuts_t27.json'
    cuts.write_text(json.dumps([[0, 1], [1, 2]]))
    ctx = Ctx('synthetic', {'public_domain': True, 'path': 'synthetic.mp4'}, root=root)
    # Establish a run-written identity; unknown legacy caches now correctly MISS.
    from cgt.p1_shots_color import pipeline

    def rebuild(film, threshold):
        import pandas as pd
        cuts.write_text(json.dumps([[0, 1], [1, 2]]))
        return pd.read_parquet(root / 'data/processed/synthetic_shots.parquet')

    monkeypatch.setattr(pipeline, 'run_film', rebuild)
    stages.p1_run(ctx)
    before = stages.p1_status(ctx)
    assert before.hit
    video.write_bytes(b'synthetic-input-B')
    after = stages.p1_status(ctx)
    assert not after.hit or after.key != before.key, 'replaced input video retained the cache hit and key'
