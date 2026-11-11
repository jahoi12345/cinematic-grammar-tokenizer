"""Independent section-10 guard tests; all labels, registries and ledgers are synthetic."""
import hashlib
import importlib.util
import json
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Barrier
from time import sleep
from types import SimpleNamespace

import pandas as pd
import pytest

from cgt.p3b_vlm import guards
from cgt.p4_camera_move import final_eval as fe
from cgt.p4_camera_move import final_run as fr
from cgt.p4_camera_move import final_stores as fs
from cgt.p4_camera_move.camerabench import CameraBenchAlreadyEvaluated

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def world(tmp_path):
    demo = script('p4_final_eval_selfcheck')
    return demo.make_world(tmp_path), demo


def cb_store(w):
    return fs.CameraBenchFinalStore(root=w['cb_root'], ledger=w['ledger'],
                                  freeze_path=w['freeze'], repo_root=w['repo'])


def label_args():
    return {'final_evaluation': True, 'model_tag': fe.MODEL_TAGS['geometry'], 'config_hash': 'c' * 16}


def test_camerabench_ledger_is_fsynced_before_label_open(world, monkeypatch):
    w, _ = world
    synced = []
    real_sync, real_read = os.fsync, Path.read_text

    def sync(fd):
        stat = os.fstat(fd)
        ledger_stat = w['ledger'].stat()
        if (stat.st_dev, stat.st_ino) == (ledger_stat.st_dev, ledger_stat.st_ino):
            synced.append(fd)
        return real_sync(fd)

    def read(path, *args, **kwargs):
        if path == w['cb_root'] / 'test.jsonl':
            assert synced, 'ledger must be durable BEFORE labels are opened'
            assert fs.ledger_contains(w['ledger'], fs.CAMERABENCH, fe.MODEL_TAGS['geometry'])
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(os, 'fsync', sync)
    monkeypatch.setattr(Path, 'read_text', read)
    cb_store(w).labels(**label_args())


def test_duplicate_concurrent_label_reads_reserve_one_dataset_tag(world):
    w, _ = world
    barrier = Barrier(2)

    class ConcurrentStore(fs.CameraBenchFinalStore):
        def _evaluated(self, dataset, tag):
            found = super()._evaluated(dataset, tag)
            if not found:
                sleep(0.2)  # widen the legitimate check/append race; a serialized store still passes
            return found

    def read():
        store = ConcurrentStore(root=w['cb_root'], ledger=w['ledger'],
                                freeze_path=w['freeze'], repo_root=w['repo'])
        barrier.wait(timeout=10)
        try:
            store.labels(**label_args())
            return 'read'
        except CameraBenchAlreadyEvaluated:
            return 'refused'

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: read(), range(2)))
    assert sorted(outcomes) == ['read', 'refused'], outcomes
    assert len(fs.ledger_lines(w['ledger'])) == 1


def test_geometry_spec_does_not_accept_edited_parameters_with_old_hash(world):
    w, _ = world
    path = w['repo'] / 'results/phase4/frozen_config_flow_depth.json'
    path.write_text(json.dumps({'chosen_C': 0.01, 'config_hash': 'c' * 16}))
    subprocess.run(['git', 'add', str(path)], cwd=w['repo'], check=True, capture_output=True)
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'config'],
                   cwd=w['repo'], check=True, capture_output=True)
    before, registry, _ = fr.make_spec('shotbench_movement', 'geometry', root=w['repo'])
    path.write_text(json.dumps({'chosen_C': 1000.0, 'config_hash': 'c' * 16}))
    try:
        after, _, _ = fr.make_spec('shotbench_movement', 'geometry', root=w['repo'])
        from cgt.p4_camera_move.camerabench import require_registered
        require_registered(after.tag, registry, config_hash=after.config_hash)
    except (guards.FailClosed, RuntimeError):
        return  # refusal is sufficient; never release a changed geometry configuration
    assert after != before, 'changed C retained the authorized spec and old config hash'


def test_prepared_feature_table_digest_is_checked_before_use(tmp_path, monkeypatch):
    ev = script('p4_final_eval')
    table = tmp_path / 'features_camerabench_public_test.parquet'
    pd.DataFrame({'clip': ['toy.gif'], 'tx': [0.0]}).to_parquet(table, index=False)
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    manifest = tmp_path / 'results/phase4/final_eval/prepare_camerabench_public_test.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'dataset': 'camerabench_public_test',
                                    'feature_table_sha256': digest, 'input_sha256': {}}))
    pd.DataFrame({'clip': ['toy.gif'], 'tx': [1000.0]}).to_parquet(table, index=False)
    monkeypatch.setattr(ev, 'ROOT', tmp_path)
    monkeypatch.setattr(ev, 'FEATURES', {'camerabench_public_test': table})
    spec = fr.Spec('camerabench_public_test', 'geometry', None, 'c' * 16, None, None, 'f' * 64)
    backend = ev.RealBackend(spec, None, SimpleNamespace(atoms=['pan-left'], prior=[0.5]))
    with pytest.raises(guards.FailClosed):
        backend.preflight(SimpleNamespace(clips=lambda: ['toy.gif']))


def test_go_writer_only_scratch_and_all_six_bindings(world, monkeypatch):
    w, _ = world
    out = w['tmp'] / 'scratch_go'
    adapter = w['repo'] / 'data/cache/p4/video_runs/synthetic/final/adapter.safetensors'
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b'synthetic-weights-no-model')
    digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    combined = fe.combined_config_hash(3, 0.3, 0.7, digest)
    registry = {fe.MODEL_TAGS['geometry']: {'config_hash': 'c' * 16},
                fe.MODEL_TAGS['video']: {'adapter_sha256': digest},
                fe.MODEL_TAGS['combined']: {'adapter_sha256': digest, 'config_hash': combined}}
    w['freeze'].write_text(json.dumps({'tag': 'synthetic', 'adapter': {'file': adapter.name, 'sha256': digest},
                                      'selected_epoch_E': 3, 'combination_weights': {'w_s': 0.3, 'w_g': 0.7},
                                      'registered_models': registry}))
    config = w['repo'] / 'results/phase4/frozen_config_flow_depth.json'
    config.write_text(json.dumps({'chosen_C': 0.01, 'config_hash': 'c' * 16}))
    subprocess.run(['git', 'add', str(w['freeze']), str(config)], cwd=w['repo'], check=True, capture_output=True)
    subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'synthetic identity'],
                   cwd=w['repo'], check=True, capture_output=True)
    specs = {(d, m): fr.make_spec(d, m, root=w['repo'])[0] for d in fe.DATASETS for m in fe.MODELS}
    monkeypatch.setattr(fr, 'make_spec', partial(fr.make_spec, root=w['repo']))
    monkeypatch.setattr(sys, 'argv', ['p4_write_go.py', '--all', '--out-dir', str(out)])
    runpy.run_path(str(ROOT / 'scripts/p4_write_go.py'), run_name='__main__')
    assert len(list(out.iterdir())) == 6
    for spec in specs.values():
        go = out / f'GO_{spec.key}'
        fe.require_go(go, spec.go())
        for field in ('dataset', 'tag', 'adapter_sha256', 'config_hash', 'freeze_sha256'):
            lines = spec.go().splitlines()
            go.write_text('\n'.join(f'{field}=wrong' if line.startswith(f'{field}=') else line for line in lines) + '\n')
            with pytest.raises(guards.NotAuthorized):
                fe.require_go(go, spec.go())
        go.write_text(spec.go())


def test_question_parser_excludes_answers_and_refuses_duplicate_names(world, monkeypatch):
    w, _ = world
    calls, real_csv = [], pd.read_csv

    def csv(path, *args, **kwargs):
        calls.append(kwargs.get('usecols'))
        return real_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, 'read_csv', csv)
    store = fs.ShotBenchMovementStore(tsv=w['tsv'], expected_n=12)
    assert len(store.questions()) == 12
    assert calls == [fs.QUESTION_COLUMNS] and 'answer' not in calls[0]
    rows = real_csv(w['tsv'], sep='\t')  # SYNTHETIC answers only
    rows.loc[1, 'path'] = rows.loc[0, 'path']
    rows.to_csv(w['tsv'], sep='\t', index=False)
    with pytest.raises(ValueError, match='duplicate'):
        store.questions()


def test_generated_questions_follow_the_registered_gold_set_algorithm():
    import numpy as np

    names = [f's{i}.0.{i}.gif' for i in range(8)]
    tags = [['pan-left'], ['pan-right'], ['tilt-up'], ['zoom-in'], ['dolly-in'],
            ['pan-left', 'tilt-up'], ['side-tracking'], ['pan-left']]
    labels_by_stem = {Path(n).stem: t for n, t in zip(names, tags, strict=True)}
    mapping = {'pan-left': 'pan-left', 'pan-right': 'pan-right', 'tilt-up': 'tilt-up',
               'zoom-in': 'zoom-in', 'dolly-in': 'push-in'}
    order = {'pan-left': 0, 'pan-right': 1, 'tilt-up': 2, 'zoom-in': 3, 'push-in': 4}
    gold = [tuple(sorted({mapping[t] for t in ts if t in mapping}, key=order.get)) for ts in tags]
    questions, drops = fe.build_camerabench_questions(names, labels_by_stem, mapping)
    assert drops['no_mapped_atom'] == 1 and len(questions) == 7
    by_index = {q['index']: q for q in questions}
    for i, atoms in enumerate(gold):
        if not atoms:
            continue
        pool = [g for j, g in enumerate(gold) if j != i and g]
        rng = np.random.default_rng([0, i])
        distractors = []
        for index in rng.permutation(len(pool)):
            g = pool[index]
            if g != atoms and g not in distractors:
                distractors.append(g)
                if len(distractors) == 3:
                    break
        choices = [atoms, *distractors]
        permutation = rng.permutation(4)
        q = by_index[i]
        assert q['gold_letter'] == 'ABCD'[list(permutation).index(0)]
        # Sets avoid introducing an independent atom-order convention in the assertion.
        assert [set(q['option_atoms'][letter]) for letter in 'ABCD'] == [set(choices[j]) for j in permutation]


def test_paired_bootstrap_resamples_whole_sources_with_clip_weighting():
    import numpy as np

    a = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=float)
    b = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=float)
    groups = np.array(['a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])
    source_names = ['a', 'b', 'c']
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(1000):
        sample = rng.choice(source_names, 3, replace=True)
        numerator = sum((a[groups == s] - b[groups == s]).sum() for s in sample)
        denominator = sum(int((groups == s).sum()) for s in sample)
        diffs.append(numerator / denominator)
    expected = [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])]
    actual = fe.paired_bootstrap(a, b, groups)
    assert actual['ci95'] == expected and actual['n_sources'] == 3
    assert actual['diff'] == round(float((a-b).mean()), 4)
