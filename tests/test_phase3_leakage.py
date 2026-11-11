"""Synthetic hashes and frames; never load benchmark images."""
import numpy as np
from PIL import Image

from cgt.p3_static_attrs import leakage


def test_planted_exact_near_and_different_images(tmp_path):
    # Native hash resolution avoids resize ambiguities: ascending and descending ramps.
    for name, row in [('same', np.arange(9)), ('different', np.arange(9)[::-1])]:
        Image.fromarray(np.tile(row * 25, (8, 1)).astype('uint8')).save(tmp_path / f'{name}.png')
    same = leakage.dhash(tmp_path / 'same.png')
    different = leakage.dhash(tmp_path / 'different.png')
    assert (same ^ different).bit_count() == 64
    near = same ^ 15  # precisely four planted bit flips
    matches = leakage.near_duplicates(['exact', 'near', 'different'],
        np.array([same, near, different], dtype='uint64'), ['reference'],
        np.array([same], dtype='uint64'), max_distance=4, chunk=1)
    assert matches == {'exact': [('reference', 0)], 'near': [('reference', 4)]}
    assert leakage.exact_overlap({'shared', 'qa'}, {'shared', 'test'}) == {'shared'}
    assert leakage.exclusion_ids({'shared'}, matches) == {'shared', 'exact', 'near'}
    assert leakage.distance_counts(matches, 4) == [
        {'distance': d, 'exactly': int(d in (0, 4)), 'cumulative': 1 if d < 4 else 2}
        for d in range(5)]
