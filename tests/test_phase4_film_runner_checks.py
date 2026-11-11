"""Synthetic coverage of registered-film row construction and local outputs."""
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner():
    root = Path(os.environ.get('CGT_TEST_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
    path = root / 'scripts/run_move_film.py'
    if not path.exists():
        pytest.skip('film runner absent; run against committed scratch source')
    spec = importlib.util.spec_from_file_location('synthetic_registered_film_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context():
    return types.SimpleNamespace(columns=['n_pairs', 'signal'], prior=np.array([.2, .6]),
                                 atoms=['static', 'push-in'], C=.01,
                                 Xtr=np.array([[0., 0.]]), Ytr=np.array([[0, 1]]))


def test_row_constructor_keeps_missing_shots_and_uses_prior(runner, monkeypatch):
    shots = pd.DataFrame({'shot_id': [8, 3, 99], 'len_s': [.99, 1., 2.]})
    rows = {8: {'n_pairs': 7, 'signal': 4}, 3: {'n_pairs': 8, 'signal': 6}}
    calls = []

    def probabilities(xtrain, ytrain, features, c):
        calls.append((features.copy(), c))
        return np.array([[.5, .3], [.1, .8]])

    monkeypatch.setattr(runner.FE, 'fit_atom_probs', probabilities)
    result = runner.build_table(shots, rows, context(), {'config_hash': 'synthetic-config'})
    assert result.shot_id.tolist() == [8, 3, 99]
    np.testing.assert_array_equal(calls[0][0], [[7, 4], [8, 6]])
    assert calls[0][1] == .01 and len(calls) == 1
    np.testing.assert_array_equal(result[['p_static', 'p_push-in']], [[.5, .3], [.1, .8], [.2, .6]])
    assert result.move_atoms.tolist() == ['static', 'push-in', 'push-in']
    assert result.n_atoms.tolist() == [1, 1, 1]
    assert result.no_features.tolist() == [False, False, True]
    assert result.short_lt_1s.tolist() == [True, False, False]
    assert result.few_pairs_lt_8.tolist() == [True, False, False]
    assert np.isnan(result.n_pairs.iloc[2])
    assert result.config_hash.tolist() == ['synthetic-config']*3
    assert result.model.tolist() == ['move_geom_flowdepth_v1']*3


def test_owned_default_outputs_remain_under_local_data(runner, monkeypatch, tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'data/processed').mkdir(parents=True)
    film = tmp_path/'data/synthetic.mp4'
    film.write_bytes(b'synthetic fixture; decoder is replaced')
    shots = tmp_path/'data/processed/owned_shots.parquet'
    pd.DataFrame({'shot_id': [0], 'start_s': [0.], 'end_s': [1.], 'len_s': [1.]}).to_parquet(shots)
    (tmp_path/'configs/films.yaml').write_text('films:\n  - id: owned\n    public_domain: false\n    path: data/synthetic.mp4\n')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.FR, 'geometry_identity', lambda: {'C': .01, 'split': {}, 'config_hash': 'fixture'})
    monkeypatch.setattr(runner.FM, 'memory_guard', lambda: None)
    monkeypatch.setattr(runner.FE, 'fit_atom_probs', lambda *a: np.array([[.8, .1]]))
    monkeypatch.setattr(runner, 'one', lambda args: (0, {'n_pairs': 8, 'signal': 0.}, None))
    train = types.ModuleType('p4_final_eval')
    train.TrainContext = lambda *a: context()
    monkeypatch.setitem(sys.modules, 'p4_final_eval', train)

    class SerialPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def imap_unordered(self, fn, jobs, **k):
            return map(fn, jobs)

    monkeypatch.setattr(runner, 'Pool', SerialPool)
    monkeypatch.setattr(sys, 'argv', ['run_move_film.py', '--film', 'owned', '--allow-owned'])
    runner.main()
    assert (tmp_path/'data/processed/owned_move.parquet').exists()
    report = (tmp_path/'data/processed/owned_move_report.md').read_text()
    assert 'UNMEASURED' in report and 'contradictory sets' in report
    assert not (tmp_path/'results').exists()
