"""Path guards use scratch names; no real ShotQA, ShotBench or film data is read."""

import ast
from pathlib import Path

import pytest

from cgt.p3_static_attrs import data


def test_training_path_guard_rejects_holdouts_and_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    allowed = tmp_path / "data/external/shotqa"
    allowed.mkdir(parents=True)
    monkeypatch.setattr(data, "SHOTQA_ROOT", allowed)
    data.assert_shotqa_path(allowed / "sft.json")
    for relative in ("data/external/shotbench/test.tsv", "data/raw/films/movie.mp4",
                     "data/interim/frame.jpg", "data/processed/shots.parquet",
                     "data/dev/labels.json", "data/gold/labels.json"):
        # These are path strings only; nothing is opened or written at them.
        with pytest.raises(ValueError):
            data.assert_shotqa_path(tmp_path / relative)
    outside = tmp_path / "outside"
    outside.mkdir()
    (allowed / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        data.assert_shotqa_path(allowed / "escape/image.jpg")


def test_training_modules_do_not_import_shotbench_evaluator():
    package = Path(data.__file__).parent
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.extend(f"{node.module or ''}.{alias.name}" for alias in node.names)
        assert not any("shotbench" in imported for imported in imports), source.name


def test_split_by_image_disjoint_complete_excluded_and_deterministic():
    import numpy as np
    ids = [f'image_{i:02}.jpg' for i in range(20)]
    excluded = {ids[0], ids[1]}
    train, val = data.split_images(ids + ids, val_frac=.25, seed=7, exclude=excluded)
    eligible = sorted(set(ids) - excluded)
    perm = np.random.default_rng(7).permutation(len(eligible))
    expected_val = sorted(eligible[i] for i in perm[:round(len(eligible)*.25)])
    assert val == expected_val
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(eligible)
    assert train == sorted(train)
    assert (train, val) == data.split_images(reversed(ids), .25, 7, excluded)


def test_film_split_keeps_films_together_and_drops_unknowns():
    import numpy as np
    films = {f'{f}_{i}.jpg': f'film_{f:02}' for f in range(20) for i in range(3)}
    films['unknown.png'] = None
    train, val, _ = data.split_films(films)
    keys = sorted({f for f in films.values() if f is not None})
    expected = {keys[i] for i in np.random.default_rng(0).permutation(20)[:2]}
    assert {films[i] for i in val} == expected
    assert not {films[i] for i in train} & {films[i] for i in val}
    assert set(train) | set(val) == set(films)-{'unknown.png'}
    assert 'unknown.png' not in train + val
    assert train == sorted(train) and val == sorted(val)
    assert data.split_films(dict(reversed(list(films.items())))) == (train, val, _)


def test_film_identity_normalization():
    assert data.film_key('A & B/season/1/episode/2') == 'aandb'
    assert data.film_key('A AND B') == 'aandb'
    assert data.film_key('The Film: Part II!') == 'thefilmpartii'


def test_extraction_guards_annotation_before_read(tmp_path, monkeypatch):
    import runpy
    import shutil
    import sys

    from cgt.p3_static_attrs import features, labels
    root = tmp_path / 'data/external/shotqa'
    root.mkdir(parents=True)
    outside = tmp_path / 'outside.json'
    outside.write_text('[]')
    (root/'sft.json').symlink_to(outside)
    monkeypatch.setattr(data, 'ROOT', tmp_path)
    monkeypatch.setattr(data, 'SHOTQA_ROOT', root)
    monkeypatch.setattr(data, 'load_leak_ids', lambda: set())
    reads = []
    def read(path, **kwargs):
        reads.append(path)
        return [], {}
    monkeypatch.setattr(labels, 'build_examples', read)
    monkeypatch.setattr(features, 'embed_images', lambda *a, **k: {'n_new': 0})
    script = tmp_path / 'scripts/p3_extract_features.py'
    script.parent.mkdir()
    shutil.copyfile(Path(data.__file__).parents[3]/'scripts/p3_extract_features.py', script)
    monkeypatch.setattr(sys, 'argv', [str(script)])
    with pytest.raises(ValueError, match='training data must live under'):
        runpy.run_path(str(script), run_name='__main__')
    assert reads == []
