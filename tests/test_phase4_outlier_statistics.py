"""Explicit TRAIN and pooled references must retain their declared meanings."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_outlier_reference_statistics_exclude_validation_features(tmp_path, monkeypatch):
    path = ROOT / 'scripts/p4_final_extremes.py'
    spec = importlib.util.spec_from_file_location('synthetic_extremes', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = tmp_path / 'data/cache/p4'
    cache.mkdir(parents=True)
    names = ['train0.mp4', 'train1.mp4', 'val.mp4']
    pd.DataFrame({'clip': names, 'fps_used': [0., 2., 999.]}).to_parquet(cache / 'features_shotqa.parquet', index=False)
    pd.DataFrame({'clip': names, 'px_tx_mean': [2., 4., 777.]}).to_parquet(cache / 'features_depth_shotqa.parquet', index=False)
    split = tmp_path / 'results/phase4/split_manifest.json'
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({'clips': {'train': names[:2], 'val': names[2:]}}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    from cgt.p4_camera_move import labels

    class SyntheticStore:
        def __init__(self, validation_clips):
            assert validation_clips == names[2:]

        def rows(self, clips):
            assert clips == names[:2], 'only synthetic TRAIN membership may be queried'
            return [{'clip': c, 'atoms': ('static',)} for c in clips]

    monkeypatch.setattr(labels, 'MovementLabelStore', SyntheticStore)
    mean, sd, columns = module.train_stats(usable_train=True)
    assert columns == ['fps_used', 'px_tx_mean']
    assert mean['fps_used'] == pytest.approx(1.), 'validation outlier must not alter TRAIN reference mean'
    assert mean['px_tx_mean'] == pytest.approx(3.)
    assert sd['fps_used'] == pytest.approx(2 ** .5)
    pooled, _, _ = module.train_stats()
    assert pooled['fps_used'] == pytest.approx(1001 / 3), 'the declared default reference is pooled'
