"""Synthetic committed-split, preparation and temporal-conformity regressions."""
import hashlib
import json
import math
import subprocess

import numpy as np
import pandas as pd
import pytest

from cgt.p3b_vlm import guards
from cgt.p4_camera_move import final_run as fr


def test_geometry_consumes_committed_split_and_refuses_swapped_working_split(tmp_path):
    config = tmp_path / fr.GEOMETRY_CONFIG
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'chosen_C': .01, 'config_hash': 'c' * 16, 'manifest_sha256': 'synthetic-manifest'}))
    split = tmp_path / fr.SPLIT_MANIFEST
    original = {'sha256': 'synthetic-manifest', 'clips': {'train': ['a.mp4'], 'val': ['b.mp4']}}
    split.write_text(json.dumps(original))
    for args in [('init', '-q'), ('add', str(config), str(split)),
                 ('-c', 'user.name=test', '-c', 'user.email=test@invalid', 'commit', '-qm', 'synthetic')]:
        subprocess.run(['git', *args], cwd=tmp_path, capture_output=True, check=True)
    assert fr.geometry_identity(tmp_path)['split'] == original
    split.write_text(json.dumps({**original, 'clips': {'train': ['b.mp4'], 'val': ['a.mp4']}}))
    with pytest.raises(guards.DirtyTree):
        fr.geometry_identity(tmp_path)


@pytest.fixture
def preparation(tmp_path):
    from cgt.p4_camera_move import final_inputs as fi
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    clips = ['one.gif', 'two.gif']
    for name in clips:
        (inputs / name).write_bytes(name.encode())
    table = tmp_path / 'features.parquet'
    columns = ['tx_mean', 'ty_mean']
    pd.DataFrame({'clip': clips, 'tx_mean': [0., 1.], 'ty_mean': [2., 3.]}).to_parquet(table, index=False)
    record = fi.build_manifest('camerabench_public_test', [inputs / x for x in clips], table, columns, root=tmp_path)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(record))
    kwargs = {'dataset': 'camerabench_public_test', 'clips': clips, 'input_dir': inputs, 'table': table,
              'manifest_path': manifest, 'train_columns': columns, 'root': tmp_path}
    assert fi.validate_prepared(**kwargs)['labels_read'] is False
    return fi, kwargs, record


@pytest.mark.parametrize('fault', ['table', 'input', 'duplicate', 'columns', 'provenance', 'version', 'partial'])
def test_prepared_identity_refuses_stale_or_partial_artifacts(preparation, fault):
    fi, kw, record = preparation
    if fault == 'table':
        pd.DataFrame({'clip': kw['clips'], 'tx_mean': [999., 1.], 'ty_mean': [2., 3.]}).to_parquet(kw['table'], index=False)
    elif fault == 'input':
        (kw['input_dir'] / kw['clips'][0]).write_bytes(b'replaced file')
    elif fault == 'duplicate':
        pd.DataFrame({'clip': ['one.gif', 'one.gif'], 'tx_mean': [0., 1.], 'ty_mean': [2., 3.]}).to_parquet(kw['table'], index=False)
        record['feature_table_sha256'] = hashlib.sha256(kw['table'].read_bytes()).hexdigest()
    elif fault == 'columns':
        kw['train_columns'] = kw['train_columns'][::-1]
    elif fault in ('provenance', 'version'):
        field = 'flow_key' if fault == 'provenance' else 'conform_version'
        record['provenance'][field] = 'stale'
    else:
        record['n_files'] = 1
    kw['manifest_path'].write_text(json.dumps(record))
    with pytest.raises(guards.UnreadableImage):
        fi.validate_prepared(**kw)


@pytest.mark.parametrize('fps', [14.3, 25., 29.97, 60.])
def test_conformed_video_mapping_is_nearest_time_at_registered_rate(fps):
    from cgt.p4_camera_move import conform
    n = 93
    rate = 24000 / 1001
    expected = [min(n - 1, round(j * fps / rate)) for j in range(math.floor(n / fps * rate))]
    assert conform.video_source_indices(n, fps) == expected
    assert conform.TARGET_FPS == pytest.approx(rate, abs=1e-12)


def test_conformed_gif_uses_display_duration_instead_of_frame_rank():
    from cgt.p4_camera_move import conform
    durations = [40., 200., 60., 0., 100., 50.]
    lengths = np.array([40., 200., 60., 100., 100., 50.]) / 1000
    edges = np.concatenate([[0.], np.cumsum(lengths)])
    rate = 24000 / 1001
    expected = [int(np.flatnonzero(edges[:-1] <= j / rate)[-1]) for j in range(math.floor(edges[-1] * rate))]
    assert conform.gif_source_indices(durations) == expected
