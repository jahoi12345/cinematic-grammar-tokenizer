"""Independent synthetic checks of the label-free probe and cache contracts."""
import importlib.util
import io
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def probe_modules():
    root = Path(os.environ.get('CGT_TEST_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
    path = root / 'scripts/p4_move_probes.py'
    if not path.exists():
        pytest.skip('probe script not present; run against pinned scratch source')
    from cgt.p4_camera_move import probes
    spec = importlib.util.spec_from_file_location('synthetic_movement_probe_script', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return probes, module


@pytest.mark.parametrize('reader', ['builtins', 'pathlib', 'io'])
def test_tripwire_denies_synthetic_train_labels(probe_modules, tmp_path, reader):
    probes, _ = probe_modules
    labels = tmp_path / 'data/external/shotqa/grpo.json'
    labels.parent.mkdir(parents=True)
    labels.write_text('{"synthetic_label_marker": true}')
    with probes.tripwire(), pytest.raises(PermissionError):
        if reader == 'pathlib':
            labels.read_text()
        else:
            opener = open if reader == 'builtins' else io.open
            with opener(labels) as fh:
                fh.read()


def _clip(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 24000 / 1001, (64, 48))
    assert writer.isOpened(), 'synthetic FFV1 writer must be usable'
    for i in range(9):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :32, 1] = 200
        frame[:24, :, 0] = 100
        frame[:, :, 2] = i
        writer.write(frame)
    writer.release()


def _fake_extraction(module, monkeypatch, source, events):
    def cut(_film, _start, _end, dest):
        shutil.copyfile(source, dest)
        return 9

    def flow(path):
        events.append('flow')
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        assert ok
        return {'left_green': np.array([float(frame[:, :32, 1].mean())])}

    def depth(_path):
        events.append('depth')
        return {}

    monkeypatch.setattr(module.FM, 'cut_clip', cut)
    monkeypatch.setattr(module.F, 'cache_key', lambda: 'frozen-flow')
    monkeypatch.setattr(module.D, 'cache_key', lambda: 'frozen-depth')
    monkeypatch.setattr(module.F, 'clip_series', flow)
    monkeypatch.setattr(module.D, 'clip_series', depth)
    monkeypatch.setattr(module.F, 'clip_features', lambda sf: {'left_green': float(sf['left_green'][0])})
    monkeypatch.setattr(module.D, 'clip_features', lambda _: {})


def test_transform_source_change_cannot_reuse_old_transform_cache(probe_modules, tmp_path, monkeypatch):
    probes, module = probe_modules
    source = tmp_path / 'synthetic.avi'
    _clip(source)
    _fake_extraction(module, monkeypatch, source, [])
    source_identity = tmp_path / 'probes_version.py'
    source_identity.write_text('version1: horizontal mirror uses the wrong axis')
    monkeypatch.setattr(probes, '__file__', str(source_identity))
    monkeypatch.setattr(probes, '_flip_code', lambda t: {'H': 0, 'V': 1}[t])
    args = (0, str(source), 0., 1., 'unchanged-film-and-boundaries',
            str(tmp_path / 'accepted'), str(tmp_path / 'cache'), False)
    _, bad, error = module.probe_shot(args)
    assert error is None and bad['rows']['H']['left_green'] == 200
    # Ordinary source correction, with film/feature code unchanged. Cached
    # transformed features must identify the transform that produced them.
    source_identity.write_text('version2: horizontal mirror now uses the correct axis')
    monkeypatch.setattr(probes, '_flip_code', lambda t: {'H': 1, 'V': 0}[t])
    _, corrected, error = module.probe_shot(args)
    assert error is None
    assert corrected['rows']['H']['left_green'] == 0, 'old wrong-axis features cannot be recertified under corrected probe code'


def test_memory_guard_precedes_each_transform_feature_batch(probe_modules, tmp_path, monkeypatch):
    _, module = probe_modules
    source = tmp_path / 'synthetic.avi'
    _clip(source)
    events = []
    _fake_extraction(module, monkeypatch, source, events)
    monkeypatch.setattr(module.FM, 'memory_guard', lambda: events.append('guard'))
    args = (0, str(source), 0., 1., 'synthetic-film',
            str(tmp_path / 'accepted'), str(tmp_path / 'cache'), False)
    _, rows, error = module.probe_shot(args)
    assert error is None and rows is not None
    for index, event in enumerate(events):
        if event == 'flow':
            assert index and events[index-1] == 'guard', 'memory must be rechecked before each expensive feature batch'
