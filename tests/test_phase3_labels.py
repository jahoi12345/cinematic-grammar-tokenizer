"""Synthetic strings exercise the normalization table in the Phase 3 contract."""

from pathlib import Path

import pytest
import yaml

from cgt.p3_static_attrs.labels import (
    UnmappedLabel,
    UnmappedQuestion,
    parse_label_set,
    parse_question,
)

NORMALIZATION = {
    "size": {"Extreme Close Up": "ECU", "Close Up": "CU", "Medium Close Up": "MCU",
             "Medium": "MS", "Medium Wide": "MWS", "Wide": "WS", "Extreme Wide": "EWS"},
    "lens": {"Long Lens": "long", "Medium": "medium", "Wide": "wide", "Ultra Wide / Fisheye": "ultra-wide"},
    "angle": {"Low angle": "low", "High angle": "high", "Overhead": "overhead", "Aerial": "aerial",
              "Dutch angle": "dutch"},
    "framing": {"Single": "single", "Insert": "insert", "2 shot": "2-shot", "3 shot": "3-shot",
                "Group shot": "group", "Establishing shot": "establishing", "Over the shoulder": "ots"},
    "light_type": {"Daylight": "daylight", "Artificial light": "artificial", "Mixed light": "mixed",
                   "Firelight": "firelight", "Overcast": "overcast", "Practical light": "practical",
                   "Sunny": "sunny", "Moonlight": "moonlight", "Fluorescent": "fluorescent",
                   "HMI": "hmi", "Tungsten": "tungsten", "LED": "led"},
    "light_cond": {"Side light": "side", "Backlight": "back", "High contrast": "high-contrast",
                   "Silhouette": "silhouette", "Edge light": "edge", "Underlight": "under",
                   "Top light": "top", "Hard light": "hard", "Soft light": "soft", "Low contrast": "low-contrast"},
    "composition": {"Center": "center", "Balanced": "balanced", "Symmetrical": "symmetrical",
                    "Right heavy": "right-heavy", "Left heavy": "left-heavy", "Short side": "short-side"},
}


def question(dimension):
    return (f"<image>Question: What's the {dimension} of this shot?\nOptions:\n"
            "A. First\nB. Second\nC. Third\nD. Fourth\n"
            "Please select the most likely answer from the options above. \n")


def test_all_question_dimensions_and_option_letters():
    dimensions = {"shot size": "size", "lens size": "lens", "shot framing": "framing",
                  "camera angle": "angle", "lighting type": "light_type", "lighting": "light_cond",
                  "composition": "composition"}
    for dimension, attribute in dimensions.items():
        assert parse_question(question(dimension)) == (
            attribute, {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"}
        )


def test_entire_written_normalization_table_and_schema_membership():
    schema = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/schema.yaml").read_text())
    for attribute, mapping in NORMALIZATION.items():
        for source, target in mapping.items():
            labels = parse_label_set(attribute, source)
            assert isinstance(labels, frozenset)
            assert labels == frozenset({target}), (attribute, source)
            assert labels <= set(schema["attributes"][attribute]["values"])


def test_compound_labels_are_sets_and_duplicates_removed():
    assert parse_label_set("size", "Extreme Wide, Wide") == frozenset({"EWS", "WS"})
    assert parse_label_set("lens", "Long Lens, Long Lens") == frozenset({"long"})
    assert parse_label_set("composition", "Balanced, Center") == parse_label_set("composition", "Center, Balanced")


def test_unknown_questions_and_labels_are_rejected():
    with pytest.raises(UnmappedQuestion):
        parse_question(question("camera movement"))
    with pytest.raises(UnmappedLabel):
        parse_label_set("size", "Made Up Shot Size")


def test_build_examples_synthetic_turn_order_and_drop_counts(tmp_path):
    import json

    from cgt.p3_static_attrs.labels import build_examples
    def question(dim):
        return (f"Question: What's the {dim} of this shot?\nOptions:\n"
                "A. Wide\nB. Medium\nC. Close Up\nD. Extreme Close Up\nPlease select.")
    messages = []
    for dim, answer in [('shot size', 'B'), ('camera movement', 'A'), ('shot size', 'A')]:
        messages.extend([{'role': 'user', 'content': question(dim)},
                         {'role': 'assistant', 'content': answer}])
    messages.append({'role': 'user', 'content': 'ignored unpaired'})
    path = tmp_path / 'sft.json'
    path.write_text(json.dumps([{'images': ['data/missing.jpg'], 'messages': messages}]))
    examples, stats = build_examples(path)
    assert [(e.img_id, e.attribute, e.answer_letter, e.target) for e in examples] == [
        ('missing.jpg', 'size', 'B', frozenset({'MS'})),
        ('missing.jpg', 'size', 'A', frozenset({'WS'}))]
    assert all(e.target == e.options[e.answer_letter] for e in examples)
    assert stats == {'samples': 1, 'turns': 3, 'unmapped_question': 1, 'unmapped_label': 0}
    assert build_examples(path) == (examples, stats)
