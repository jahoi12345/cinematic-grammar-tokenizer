"""Bug U1: the AudioSet Music subtree maps to "score"; dialogue rules first and unchanged; the rest unchanged.

See docs/p2_sound_groups_fix_2026-10-15.md. No model is run: only label strings and the (read-only) AST config id2label.
"""

import inspect
import json
from pathlib import Path

import pytest

from cgt.p2_pretrained import audio
from cgt.p2_pretrained.audio import (
    EXCLUDED_LABELS,
    GROUPS_VERSION,
    MUSIC_SUBTREE_LABELS,
    SINGING_LABELS,
    _group,
)

FIXTURE = Path(__file__).parent / "fixtures" / "audioset_music_subtree.json"
AST_CONFIG = (Path.home() / ".cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots"
              / "f826b80d28226b62986cc218e5cec390b1096902/config.json")
V1_AUDIO_SHA256 = "16166341137916202e41229632100b37338b71283e63f2db3ba9e43ad2c0f114"  # audio.py at main dbaa28d

DIALOGUE = ["Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking",
            "Conversation", "Narration, monologue", "Speech synthesizer", "Hubbub, speech noise, speech babble",
            "Shout", "Screaming", "Whispering"]
AMBIENT = ["Walk, footsteps", "Thump, thud", "Door", "Doorbell", "Silence", "Wind", "Rain", "Vehicle", "Dog",
           "Gunshot, gunfire", "Inside, small room"]
# a recorded decision Amendment 1
SINGING = ["Male singing", "Female singing", "Child singing", "Synthetic singing", "Rapping", "Humming", "Yodeling"]
EXCLUDED = ["Bicycle bell", "Church bell", "Wind chime", "Tuning fork"]
MUSICAL_BELLS = ["Cowbell", "Jingle bell", "Chime", "Change ringing (campanology)", "Gong", "Bell", "Tubular bells"]


def _old_group(label: str) -> str:  # v1 rule, verbatim
    l = label.lower()
    if "speech" in l or l in {"conversation", "narration, monologue", "shout", "screaming", "whispering"}:
        return "dialogue"
    if "music" in l or l in {"piano", "organ", "violin, fiddle", "orchestra", "synthesizer", "singing"}:
        return "score"
    return "ambient"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _id2label() -> set[str]:
    if not AST_CONFIG.exists():
        pytest.skip("AST config not in the local HF cache")
    return set(json.loads(AST_CONFIG.read_text())["id2label"].values())


@pytest.mark.parametrize("label", ["Opera", "Cello", "Choir", "Harp", "Orchestra"])
def test_u1_music_family_without_the_word_music_is_score(label):
    assert _group(label) == "score"


@pytest.mark.parametrize("label", DIALOGUE)
def test_dialogue_labels_still_dialogue(label):
    assert _group(label) == "dialogue"


@pytest.mark.parametrize("label", AMBIENT)
def test_known_ambient_labels_still_ambient(label):
    assert _group(label) == "ambient"


@pytest.mark.parametrize("label", SINGING)
def test_amendment1_singing_labels_outside_the_subtree_are_score(label):
    assert _group(label) == "score"


@pytest.mark.parametrize("label", EXCLUDED)
def test_amendment1_excluded_bells_and_tools_stay_ambient(label):
    assert _group(label) == "ambient"


@pytest.mark.parametrize("label", MUSICAL_BELLS)
def test_amendment1_musical_bells_and_percussion_stay_score(label):
    assert _group(label) == "score"


def test_amendment1_tuples():
    assert list(SINGING_LABELS) == SINGING and list(EXCLUDED_LABELS) == EXCLUDED
    assert set(EXCLUDED_LABELS) <= set(MUSIC_SUBTREE_LABELS)
    assert not set(SINGING_LABELS) & set(MUSIC_SUBTREE_LABELS)
    assert not set(SINGING_LABELS) & {n["name"] for n in _fixture()["nodes"]}


def test_singing_is_still_score():
    assert _group("Singing") == "score"


def test_every_subtree_label_except_excluded_is_score():
    assert len(MUSIC_SUBTREE_LABELS) == 150
    score = [lab for lab in MUSIC_SUBTREE_LABELS if lab not in EXCLUDED]
    assert len(score) == 146
    assert {lab: _group(lab) for lab in score} == {lab: "score" for lab in score}


def test_score_list_is_walk_minus_excluded_plus_singing():
    expected = (set(_fixture()["in_model"]) - set(EXCLUDED)) | set(SINGING)
    assert len(expected) == 153
    assert set(audio._SCORE_LABELS) == expected


def test_list_equals_the_recorded_ontology_walk():
    fx = _fixture()
    assert fx["root"] == {"id": "/m/04rlf", "name": "Music"}
    nodes = {n["id"]: n for n in fx["nodes"]}
    seen, queue = [], [fx["root"]["id"]]
    while queue:  # re-walk the recorded nodes: breadth-first over child_ids, each node once
        i = queue.pop(0)
        if i in seen:
            continue
        seen.append(i)
        queue += nodes[i]["child_ids"]
    assert seen == [n["id"] for n in fx["nodes"]]
    assert fx["walk_count"] == len(seen) == 189
    assert len(set(MUSIC_SUBTREE_LABELS)) == len(MUSIC_SUBTREE_LABELS)
    assert list(MUSIC_SUBTREE_LABELS) == fx["in_model"]
    assert fx["in_model_count"] == len(fx["in_model"])
    assert {n["name"] for n in fx["nodes"]} == set(fx["in_model"]) | {n["name"] for n in fx["not_in_model"]}


def test_list_names_are_exact_ast_labels():
    labels = _id2label()
    fx = _fixture()
    assert set(MUSIC_SUBTREE_LABELS) <= labels
    assert set(SINGING_LABELS) <= labels and set(EXCLUDED_LABELS) <= labels
    assert not {n["name"] for n in fx["not_in_model"]} & labels
    assert fx["in_model"] == [n["name"] for n in fx["nodes"] if n["name"] in labels]


def test_only_the_new_score_labels_change_vs_v1():
    labels = _id2label()
    assert len(labels) == 527
    changed = {lab for lab in labels if _group(lab) != _old_group(lab)}
    old_score = {lab for lab in labels if _old_group(lab) == "score"}
    assert changed == ((set(MUSIC_SUBTREE_LABELS) - set(EXCLUDED)) | set(SINGING)) - old_score
    assert all(_old_group(lab) == "ambient" and _group(lab) == "score" for lab in changed)


def test_no_subtree_label_matches_the_dialogue_rule():
    assert [lab for lab in MUSIC_SUBTREE_LABELS + SINGING_LABELS if _old_group(lab) == "dialogue"] == []


def test_dialogue_takes_precedence_over_the_music_list(monkeypatch):
    # decision: dialogue first. A label matching the dialogue rule stays dialogue even if it were in the Music list
    monkeypatch.setattr(audio, "_SCORE_LABELS", audio._SCORE_LABELS | {"Male speech, man speaking", "Conversation"})
    assert audio._group("Male speech, man speaking") == "dialogue"
    assert audio._group("Conversation") == "dialogue"


def test_list_lives_in_audio_py_so_the_stage_key_changes():
    src = inspect.getsource(audio)
    assert GROUPS_VERSION == 2
    assert "MUSIC_SUBTREE_LABELS = (" in src and '"Opera"' in src
    assert "SINGING_LABELS = (" in src and "EXCLUDED_LABELS = (" in src
    from cgt.p2_pretrained.pipeline import _source_hash
    assert _source_hash("audio") != V1_AUDIO_SHA256
