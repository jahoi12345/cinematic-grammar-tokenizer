# ruff: noqa: F811
"""Legacy caches must not invent the original input identity from mtimes."""
import json
import os
import time

from _p3c_fixtures import KF, FakeEngine, _run, film_env  # noqa: F401
from PIL import Image

from cgt.p3c_tokens import runner
from cgt.pipeline import stages
from cgt.pipeline.orchestrator import Ctx


def test_legacy_p1_cannot_attest_a_replaced_source_with_preserved_mtime(film_env):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    video = root / 'synthetic.mp4'
    video.write_bytes(b'original-source')
    cuts = root / 'data/interim/synthetic/cuts_t27.json'
    cuts.write_text(json.dumps([[0, 1], [1, 2]]))
    # A normal copy preserving the source timestamp gives no proof of which bytes
    # originally produced the old cuts; this is not a recorded-hash forgery.
    video.write_bytes(b'replacement-source')
    old = time.time() - 3600
    os.utime(video, (old, old))
    ctx = Ctx('synthetic', {'public_domain': True, 'path': 'synthetic.mp4'}, root=root)
    assert not stages.p1_status(ctx).hit, 'mtime ordering cannot establish the original source content'


def _changed_legacy_run(film_env, monkeypatch):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    rd = _run(root)
    runner.build_table('synthetic', rd, {'light_type': .5, 'light_cond': .5, 'composition': .5},
                       root / 'data/processed/synthetic_p3c.parquet')
    monkeypatch.setattr(stages, '_frozen_sha', lambda ctx: 'a' * 64)
    monkeypatch.setattr(stages, '_selected_method', lambda ctx: 'D')
    monkeypatch.setattr(stages, '_git_equal', lambda *args: True)
    mf = rd / 'manifest.json'
    manifest = json.loads(mf.read_text())
    manifest.pop('inputs_digest', None)
    mf.write_text(json.dumps(manifest))
    Image.new('RGB', (32, 32), (255, 0, 0)).save(root / KF.format(0))
    old = time.time() - 3600
    for sid in (0, 1):
        os.utime(root / KF.format(sid), (old, old))
    return root, rd, Ctx('synthetic', {'public_domain': True}, root=root)


def test_legacy_p3c_stage_refuses_changed_keyframe_with_preserved_mtime(film_env, monkeypatch):
    _root, _rd, ctx = _changed_legacy_run(film_env, monkeypatch)
    assert not stages.p3c_status(ctx).hit, 'missing original digest must not authorize old scores'


def test_legacy_p3c_runner_rescores_when_original_digest_is_unknown(film_env, monkeypatch):
    root, rd, _ctx = _changed_legacy_run(film_env, monkeypatch)
    engine = FakeEngine()
    runner.run_film('synthetic', 'D', root / 'adapter', engine=engine, log=lambda _: None)
    assert engine.passes == 2 * 3 * 7, 'all stale shot/keyframe scores must be recomputed'
    assert list(rd.parent.glob(f'{rd.name}.stale-*')), 'retain the unknown original run as evidence'
