"""bug U1 sound groups, branch a sibling working copy.

The expected Music-subtree set below is the verification pass's OWN breadth-first walk of the AudioSet ontology from "Music" (/m/04rlf),
transcribed from https://raw.githubusercontent.com/audioset/ontology/master/ontology.json (read 2026-10-15, offsets 0 and 100000)
and intersected with the AST config id2label (snapshot f826b80d...): 189 nodes, 150 AST labels. It does not read the worker's
fixture or tuples to build the expectation. No model is run; config.json is read only when present in the local HF cache.
See a verification record.
"""

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from cgt.p2_pretrained import audio

AST_CONFIG = (Path.home() / ".cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots"
              / "f826b80d28226b62986cc218e5cec390b1096902/config.json")
MAIN_AUDIO_SHA256 = "16166341137916202e41229632100b37338b71283e63f2db3ba9e43ad2c0f114"  # audio.py at main ee5d1dc (= dbaa28d)

VS_WALK_IN_MODEL = frozenset({
    "A capella", "Accordion", "Acoustic guitar", "Afrobeat", "Ambient music", "Angry music", "Background music", "Bagpipes",
    "Banjo", "Bass drum", "Bass guitar", "Beatboxing", "Bell", "Bicycle bell", "Bluegrass", "Blues",
    "Bowed string instrument", "Brass instrument", "Carnatic music", "Cello", "Change ringing (campanology)", "Chant",
    "Chime", "Choir", "Christian music", "Christmas music", "Church bell", "Clarinet", "Classical music", "Country",
    "Cowbell", "Cymbal", "Dance music", "Didgeridoo", "Disco", "Double bass", "Drum", "Drum and bass", "Drum kit",
    "Drum machine", "Drum roll", "Dubstep", "Electric guitar", "Electric piano", "Electronic dance music",
    "Electronic music", "Electronic organ", "Electronica", "Exciting music", "Flamenco", "Flute", "Folk music",
    "French horn", "Funk", "Funny music", "Glockenspiel", "Gong", "Gospel music", "Grunge", "Guitar", "Hammond organ",
    "Happy music", "Harmonica", "Harp", "Harpsichord", "Heavy metal", "Hi-hat", "Hip hop music", "House music",
    "Independent music", "Jazz", "Jingle (music)", "Jingle bell", "Keyboard (musical)", "Lullaby", "Mallet percussion",
    "Mandolin", "Mantra", "Maraca", "Marimba, xylophone", "Middle Eastern music", "Music", "Music for children",
    "Music of Africa", "Music of Asia", "Music of Bollywood", "Music of Latin America", "Musical instrument",
    "New-age music", "Opera", "Orchestra", "Organ", "Percussion", "Piano", "Pizzicato", "Plucked string instrument",
    "Pop music", "Progressive rock", "Psychedelic rock", "Punk rock", "Rattle (instrument)", "Reggae", "Rhythm and blues",
    "Rimshot", "Rock and roll", "Rock music", "Sad music", "Salsa music", "Sampler", "Saxophone", "Scary music",
    "Scratching (performance technique)", "Shofar", "Singing bowl", "Sitar", "Ska", "Snare drum", "Song", "Soul music",
    "Soundtrack music", "Steel guitar, slide guitar", "Steelpan", "String section", "Strum", "Swing music", "Synthesizer",
    "Tabla", "Tambourine", "Tapping (guitar technique)", "Techno", "Tender music", "Theme music", "Theremin", "Timpani",
    "Traditional music", "Trance music", "Trombone", "Trumpet", "Tubular bells", "Tuning fork", "Ukulele", "Vibraphone",
    "Video game music", "Violin, fiddle", "Vocal music", "Wedding music", "Wind chime",
    "Wind instrument, woodwind instrument", "Wood block", "Zither",
})
EXCLUDED = frozenset({"Bicycle bell", "Church bell", "Wind chime", "Tuning fork"})
SINGING = frozenset({"Male singing", "Female singing", "Child singing", "Synthetic singing", "Rapping", "Humming", "Yodeling"})
EXPECTED_SCORE_LIST = (VS_WALK_IN_MODEL - EXCLUDED) | SINGING


def _v1(label: str) -> str:  # main ee5d1dc _group, verbatim
    l = label.lower()
    if "speech" in l or l in {"conversation", "narration, monologue", "shout", "screaming", "whispering"}:
        return "dialogue"
    if "music" in l or l in {"piano", "organ", "violin, fiddle", "orchestra", "synthesizer", "singing"}:
        return "score"
    return "ambient"


def _labels() -> list[str]:
    if not AST_CONFIG.exists():
        pytest.skip("AST config not in the local HF cache")
    return list(json.loads(AST_CONFIG.read_text())["id2label"].values())


def test_vs_walk_size():
    assert len(VS_WALK_IN_MODEL) == 150 and EXCLUDED <= VS_WALK_IN_MODEL and not SINGING & VS_WALK_IN_MODEL
    assert len(EXPECTED_SCORE_LIST) == 153


def test_vs_branch_tuples_equal_own_walk():
    assert set(audio.MUSIC_SUBTREE_LABELS) == VS_WALK_IN_MODEL
    assert set(audio.EXCLUDED_LABELS) == EXCLUDED and set(audio.SINGING_LABELS) == SINGING
    assert set(audio._SCORE_LABELS) == EXPECTED_SCORE_LIST


def test_vs_names_are_ast_labels():
    labels = set(_labels())
    assert len(labels) == 527
    assert VS_WALK_IN_MODEL | SINGING <= labels


def test_vs_all_527_only_expected_labels_change_and_only_ambient_to_score():
    labels = _labels()
    changed = {lab: (_v1(lab), audio._group(lab)) for lab in labels if _v1(lab) != audio._group(lab)}
    assert set(changed) == {lab for lab in EXPECTED_SCORE_LIST if _v1(lab) != "score"}
    assert len(changed) == 104
    assert set(changed.values()) == {("ambient", "score")}


def test_vs_final_score_set():
    labels = _labels()
    score = {lab for lab in labels if audio._group(lab) == "score"}
    # the v1 rule's "Singing" (parent of the singing labels, outside the Music subtree) stays score: unchanged rule
    assert score == EXPECTED_SCORE_LIST | {"Singing"}
    assert not {lab for lab in labels if audio._group(lab) == "score"} & EXCLUDED


def test_vs_dialogue_set_identical_to_v1():
    labels = _labels()
    assert {lab for lab in labels if audio._group(lab) == "dialogue"} == {lab for lab in labels if _v1(lab) == "dialogue"}


@pytest.mark.parametrize("label", ["Speech music", "Shout", "Whispering", "Conversation"])
def test_vs_dialogue_precedence_with_synthetic_overlap(monkeypatch, label):
    monkeypatch.setattr(audio, "_SCORE_LABELS", audio._SCORE_LABELS | {label})
    assert audio._group(label) == "dialogue"


@pytest.mark.parametrize("label", sorted(EXCLUDED))
def test_vs_excluded_ambient(label):
    assert audio._group(label) == "ambient"


@pytest.mark.parametrize("label", ["Opera", "Choir", "Mantra", "Cowbell", "Beatboxing", "Song", "Humming", "Rapping"])
def test_vs_spot_score(label):
    assert audio._group(label) == "score"


@pytest.mark.parametrize("label", ["Hum", "Whistling", "Jingle, tinkle", "Ding-dong", "Clapping", "Oboe", "Bassoon"])
def test_vs_near_misses_stay_ambient(label):
    assert audio._group(label) == "ambient"


def test_vs_stage_source_hash_changes():
    from cgt.p2_pretrained.pipeline import _source_hash
    src = inspect.getsource(audio)
    assert "MUSIC_SUBTREE_LABELS = (" in src and "EXCLUDED_LABELS = (" in src and "SINGING_LABELS = (" in src
    h = _source_hash("audio")
    assert h == hashlib.sha256(src.encode()).hexdigest()
    assert h != MAIN_AUDIO_SHA256
