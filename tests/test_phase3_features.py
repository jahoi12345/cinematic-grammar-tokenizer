"""Synthetic cache manifests; no encoder weights or dataset access."""
import json

import numpy as np
import pytest

from cgt.p3_static_attrs import features


def test_cache_provenance_and_source_root_fail_closed(tmp_path, monkeypatch):
    allowed = tmp_path/'shotqa'
    allowed.mkdir()
    cache = tmp_path/'cache'
    cache.mkdir()
    expected = {'model_name': 'synthetic', 'model_revision': 'rev1',
                'processor_sha256': 'resize-hash', 'feature': 'pooled', 'dtype': 'float16'}
    monkeypatch.setattr(features, 'current_provenance', lambda *a, **k: expected)
    with pytest.raises(features.CacheProvenanceError):
        features.verify_cache(cache, 'synthetic', allowed)
    shard = cache/'shard_00000.npz'
    def write(src=True, path=None):
        args = {'ids': np.array(['frame.png']), 'feats': np.ones((1, 3), dtype='float16')}
        if src:
            args['src'] = np.array([str(path or allowed/'frame.png')])
        np.savez(shard, **args)
    write()
    with pytest.raises(features.CacheProvenanceError):
        features.verify_cache(cache, 'synthetic', allowed)  # legacy, no manifest
    manifest = cache/'provenance.json'
    manifest.write_text(json.dumps(expected))
    result = features.verify_cache(cache, 'synthetic', allowed)
    assert result == {'provenance': expected, 'n_images': 1, 'source_roots': [str(allowed)]}
    for key in expected:
        manifest.write_text(json.dumps({**expected, key: 'wrong'}))
        with pytest.raises(features.CacheProvenanceError):
            features.verify_cache(cache, 'synthetic', allowed)
    manifest.write_text(json.dumps(expected))
    write(src=False)
    with pytest.raises(features.CacheProvenanceError):
        features.verify_cache(cache, 'synthetic', allowed)
    write(path=tmp_path/'shotbench/frame.png')
    with pytest.raises(features.CacheProvenanceError):
        features.verify_cache(cache, 'synthetic', allowed)
    escape = allowed/'escape'
    escape.symlink_to(tmp_path/'outside', target_is_directory=True)
    write(path=escape/'frame.png')
    with pytest.raises(features.CacheProvenanceError):
        features.verify_cache(cache, 'synthetic', allowed)
