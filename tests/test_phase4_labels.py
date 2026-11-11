"""Phase 4 section 1 contract: synthetic text and tmp_path annotations only."""
import json
from dataclasses import FrozenInstanceError

import pytest

from cgt.p4_camera_move import labels


def test_single_atom_normalizes_case_spaces_and_punctuation():
    result = labels.parse_move_label('  PUSH IN.  ')
    assert (result.kind, result.atoms, result.order, result.flags, result.unmapped) == (
        'single', ('push-in',), ('push-in',), (), ())


def test_simultaneous_atoms_are_sorted_and_order_is_preserved():
    result = labels.parse_move_label('Tilt up and Push in')
    assert result.kind == 'simultaneous'
    assert result.atoms == ('push-in', 'tilt-up')
    assert result.order == ('tilt-up', 'push-in')


def test_sequential_atoms_keep_first_occurrence_order():
    result = labels.parse_move_label('Firstly Static shot, then Push in, finally Static shot.')
    assert result.kind == 'sequential'
    assert result.atoms == ('push-in', 'static')
    assert result.order == ('static', 'push-in')


def test_sequential_typo_markers_are_supported():
    for marker in ('Firstly', 'Fisrtly', 'Fristly', 'Finally'):
        result = labels.parse_move_label(f'{marker} Tilt up, then Pan right')
        assert result.kind == 'sequential'
        assert result.order == ('tilt-up', 'pan-right')


def test_alias_table_has_the_documented_schema_values():
    expected = {'trucking left': 'move-left', 'trucking right': 'move-right',
                'camera roll': 'roll', 'static shot': 'static', 'static': 'static',
                'dolly zoom': 'dolly-zoom'}
    for text, atom in expected.items():
        assert labels.ATOM_ALIASES[text] == atom
        assert labels.parse_move_label(text).atoms == (atom,)
    assert labels.FLAG_ALIASES == {'tracking': 'track', 'rack focus': 'rack'}


def test_flags_are_auxiliary_and_never_schema_atoms():
    result = labels.parse_move_label('Tracking and Pull out and Rack focus')
    assert result.kind == 'simultaneous'
    assert result.atoms == ('pull-out',)
    assert result.order == ('pull-out',)
    assert result.flags == ('rack', 'track')
    assert result.unmapped == ()
    assert labels.parse_move_label('Tracking').atoms == ()
    assert labels.parse_move_label('Rack focus').atoms == ()


def test_repeated_atom_and_trailing_delimiters_do_not_make_a_set():
    for text in ('Static shot,', 'Static shot and Static shot', 'Static and Static shot.'):
        result = labels.parse_move_label(text)
        assert result.kind == 'single'
        assert result.atoms == ('static',)
        assert result.order == ('static',)


def test_unknown_atoms_are_reported_without_guessing():
    result = labels.parse_move_label('Long and Push in')
    assert result.kind == 'simultaneous'
    assert result.atoms == ('push-in',)
    assert result.unmapped == ('long',)
    assert labels.parse_move_label('Long').atoms == ()


def test_move_label_is_frozen():
    result = labels.parse_move_label('Static')
    with pytest.raises(FrozenInstanceError):
        result.kind = 'sequential'


def test_source_id_stops_at_first_source_extension_boundary():
    for ext in ('webm', 'mp4', 'mkv', 'mov'):
        assert labels.source_id(f'X.{ext}_12.mp4') == 'X'
    assert labels.source_id('X.webm_12.mp4_3.mp4') == 'X'


def qa_row(clip, gold='A', first='Push in'):
    return {'videos': [f'data/{clip}'], 'messages': [{'role': 'user', 'content':
            f'Question: movement?\nOptions:\nA. {first}\nB. Static shot\n'
            'C. Tracking and Pull out\nD. Rack focus\nPlease choose.'}],
            'solution': f'<answer> {gold} </answer>'}


def test_loader_retains_file_order_and_separates_flag_only_options(tmp_path):
    path = tmp_path / 'grpo.json'
    path.write_text(json.dumps([qa_row('Z.webm_2.mp4', 'C'), {'images': ['fake.jpg']},
                                qa_row('A.webm_1.mp4')]))
    rows = labels.load_shotqa_movement(path)
    assert [r['clip'] for r in rows] == ['Z.webm_2.mp4', 'A.webm_1.mp4']
    first = rows[0]
    assert set(first) == {'clip', 'source', 'gold_text', 'gold_letter', 'options',
                         'kind', 'atoms', 'order', 'flags', 'unmapped', 'option_atoms'}
    assert first['source'] == 'Z'
    assert first['gold_letter'] == 'C'
    assert first['gold_text'] == 'Tracking and Pull out'
    assert first['atoms'] == ('pull-out',)
    assert first['flags'] == ('track',)
    assert first['option_atoms'] == {'A': ('push-in',), 'B': ('static',),
                                     'C': ('pull-out',), 'D': ()}


def test_fifteen_synthetic_flag_only_rows_have_no_scored_atom(tmp_path):
    path = tmp_path / 'grpo.json'
    flag_rows = [qa_row(f'F.webm_{i}.mp4', first='Tracking' if i < 12 else 'Rack focus')
                 for i in range(15)]
    path.write_text(json.dumps(flag_rows + [qa_row('M.webm_1.mp4')]))
    rows = labels.load_shotqa_movement(path)
    assert len(rows) == 16
    assert all(not row['atoms'] and row['flags'] for row in rows[:15])
    assert all(row['option_atoms']['A'] == () for row in rows[:15])
    assert rows[15]['atoms'] == ('push-in',)


def test_loader_refuses_unparseable_or_unoffered_answer(tmp_path):
    for solution in ('B', '<answer> E </answer>', '<answer> ??? </answer>'):
        row = qa_row('X.webm_1.mp4')
        row['solution'] = solution
        path = tmp_path / 'grpo.json'
        path.write_text(json.dumps([row]))
        with pytest.raises(ValueError):
            labels.load_shotqa_movement(path)


def test_schema_moves_reads_contract_values_from_a_synthetic_root(tmp_path, monkeypatch):
    folder = tmp_path / 'configs'
    folder.mkdir()
    expected = ['static', 'push-in', 'pull-out', 'boom-up', 'boom-down', 'pan-left',
                'pan-right', 'tilt-up', 'tilt-down', 'zoom-in', 'zoom-out',
                'move-left', 'move-right', 'arc', 'roll', 'dolly-zoom']
    (folder / 'schema.yaml').write_text(json.dumps({'attributes': {'move': {'values': expected}}}))
    monkeypatch.setattr(labels, 'ROOT', tmp_path)
    assert labels.schema_moves() == expected


def test_sequential_aliases_flags_and_unknowns_preserve_order_independently():
    label = labels.parse_move_label("Firstly Trucking right and Tracking, then Rack focus, finally Static shot and Mystery.")
    assert label.kind == "sequential"
    assert label.atoms == ("move-right", "static")
    assert label.order == ("move-right", "static")
    assert label.flags == ("rack", "track") and label.unmapped == ("mystery",)


def test_repeated_auxiliary_flags_do_not_create_extra_atoms_or_units():
    label = labels.parse_move_label("Tracking and Tracking, Rack focus and Rack focus")
    assert label.atoms == label.order == ()
    assert label.flags == ("rack", "track") and label.kind == "simultaneous"
