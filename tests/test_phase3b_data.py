"""Synthetic ShotQA trees only: no real annotations or holdouts are opened."""
import json

import numpy as np
import pytest

from cgt.p3_static_attrs import data as p3data
from cgt.p3b_vlm import data

QUESTION = ("<image>Question: What's the lens size of this shot?\nOptions:\n"
            'A. Long Lens\nB. Medium\nC. Wide\nD. Ultra Wide / Fisheye\n'
            'Please select the most likely answer from the options above. \n')


@pytest.fixture
def synthetic_qa(tmp_path, monkeypatch):
    root = tmp_path / 'data/external/shotqa'
    root.mkdir(parents=True)
    for module in (data, p3data):
        monkeypatch.setattr(module, 'SHOTQA_ROOT', root)
    monkeypatch.setattr(p3data, 'ROOT', tmp_path)
    monkeypatch.setattr(data, 'load_leak_ids', lambda: {'leak.jpg'})
    names = [f'image_{i:02}.jpg' for i in range(20)] + ['leak.jpg', 'unknown.jpg']
    samples = [{'images': [f'data/{name}'], 'messages': [
        {'role': 'user', 'content': QUESTION}, {'role': 'assistant', 'content': 'B'},
        {'role': 'user', 'content': QUESTION}, {'role': 'assistant', 'content': 'C'}]}
        for name in names]
    (root / 'sft.json').write_text(json.dumps(samples))
    meta = [{'img_id': name, 'title': f'Film {i:02}'} for i, name in enumerate(names[:-1])]
    (root / 'meta.jsonl').write_text('\n'.join(json.dumps(row) for row in meta))
    return root, names


def test_chat_turns_have_letters_options_and_guarded_image_paths(synthetic_qa):
    root, _ = synthetic_qa
    rows = data.load_chat_samples('train')
    assert rows
    first = rows[0]
    assert set(first) == {'id', 'img_id', 'attribute', 'images', 'messages', 'options', 'answer'}
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        assert a['id'] == f"{a['img_id']}#0" and b['id'] == f"{a['img_id']}#1"
        assert a['answer'] == 'B' and b['answer'] == 'C'
        assert a['messages'] == [{'role': 'user', 'content': QUESTION.replace('<image>', '', 1)},
                                 {'role': 'assistant', 'content': 'B'}]
        assert a['options'] == dict(zip('ABCD', ['Long Lens', 'Medium', 'Wide', 'Ultra Wide / Fisheye'], strict=True))
        assert a['attribute'] == 'lens'
        assert a['images'] == [str(root / a['img_id'])]


def test_builder_excludes_leaks_unknowns_and_nontrain_films(synthetic_qa):
    _, names = synthetic_qa
    train, val, info = data.film_split_ids()
    train_rows, val_rows = data.load_chat_samples('train'), data.load_chat_samples('val')
    assert {r['img_id'] for r in train_rows} == train
    assert {r['img_id'] for r in val_rows} == val
    assert len(train_rows) == 2 * len(train) and len(val_rows) == 2 * len(val)
    assert not train & val
    assert train | val == set(names) - {'leak.jpg', 'unknown.jpg'}
    films = {f'image_{i:02}.jpg': f'film{i:02}' for i in range(20)}
    assert not {films[i] for i in train} & {films[i] for i in val}
    assert info['n_val_films'] == 2
    with pytest.raises(KeyError):
        data.load_chat_samples('test')


def test_chat_answer_must_map_to_schema_vocabulary(synthetic_qa):
    root, _ = synthetic_qa
    samples = json.loads((root / 'sft.json').read_text())
    for row in samples:
        row['messages'][0]['content'] = QUESTION.replace('B. Medium', 'B. imaginary lens')
    (root / 'sft.json').write_text(json.dumps(samples))
    with pytest.raises(ValueError):
        data.load_chat_samples('train')


def test_phase3b_root_guard_rejects_holdouts_films_and_symlink_escape(synthetic_qa, tmp_path):
    root, _ = synthetic_qa
    assert data.assert_shotqa_path(root / 'synthetic.jpg') == root / 'synthetic.jpg'
    for relative in ('data/external/shotbench/test.tsv', 'data/gold/labels.json',
                     'data/dev/labels.json', 'data/raw/films/movie.mp4',
                     'data/processed/shots.parquet', 'data/interim/frame.png'):
        with pytest.raises(ValueError):
            data.assert_shotqa_path(tmp_path / relative)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        data.assert_shotqa_path(root / 'escape/image.jpg')


def test_subset_is_seeded_keeps_input_order_and_jsonl_roundtrips(tmp_path):
    samples = [{'id': f'synthetic_{i}'} for i in range(10)]
    expected = [samples[i] for i in sorted(np.random.default_rng(7).permutation(10)[:4])]
    assert data.subset(samples, 4, seed=7) == expected
    path = tmp_path / 'nested/samples.jsonl'
    data.write_jsonl(expected, path)
    assert path.read_text().endswith('\n')
    assert [json.loads(row) for row in path.read_text().splitlines()] == expected
