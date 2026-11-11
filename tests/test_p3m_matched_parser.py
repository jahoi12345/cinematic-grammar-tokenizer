"""Parser = the official ShotBench local matchers (docs/p3m_matched_protocol.md 1.4): documented formats and adversarial outputs."""
import math

import pytest

from cgt.p3m_matched.parser import (
    build_choices,
    can_infer_option,
    can_infer_text,
    extract_from_answer_tag,
    parse_answer,
)

OPTS = {"A": "Wide", "B": "Medium Close Up", "C": "Close Up", "D": "Extreme Wide"}


@pytest.mark.parametrize("text,want", [
    ("A", "A"), ("B.", "B"), ("(C)", "C"), ("[D]", "D"), ("Answer: B", "B"), ("**C**", "C"), ("{A}", "A"),
    ("The answer is C.", "C"), ("C. Close Up", "C"), ("<answer>D</answer>", "D"), ("<ANSWER> b </ANSWER>", None),
    ("<think>A or B? I pick B</think><answer>B</answer>", "B"), ("<answer>\n(A) Wide\n</answer>", "A"),
])
def test_documented_formats(text, want):
    letter, _ = parse_answer(text, OPTS)
    assert letter == want


def test_answer_tag_takes_precedence():
    # the reasoning mentions A, C and D; only the tag content counts
    assert parse_answer("<think>A, C or D? not A</think> <answer>C</answer>", OPTS) == ("C", "option")
    assert extract_from_answer_tag("x <answer> one </answer> <answer>two</answer>") == "one"
    assert extract_from_answer_tag("  no tag  ") == "no tag"


def test_two_letters_is_invalid():
    # exactly-one-hit rule: two option letters as tokens -> no option match; the text matcher then sees no unique option text
    assert can_infer_option("A or B", OPTS) is False
    assert parse_answer("A or B", OPTS) == (None, "invalid")


def test_letters_are_case_sensitive_whole_tokens():
    assert can_infer_option("a", OPTS) is False
    assert can_infer_option("Absolutely", OPTS) is False  # 'A' inside a word is not a token
    assert can_infer_option("A-ish", OPTS) is False  # '-' is not in the punctuation list


def test_z_rule():
    assert parse_answer("Z", OPTS) == ("Z", "option")
    assert can_infer_option("A Z", OPTS) == "A"  # a real hit wins over Z


def test_text_matching_and_substring_rule():
    assert can_infer_text("it is a wide shot", {"A": "Wide", "B": "Close Up"}) == "A"
    assert parse_answer("medium close up", {"A": "Wide", "B": "Medium Close Up"}) == ("B", "text")


def test_text_two_hits_invalid():
    # 'close up' is a substring of 'medium close up': two option texts match -> invalid (no GPT fallback)
    assert can_infer_text("medium close up", OPTS) is False
    assert parse_answer("medium close up", OPTS) == (None, "invalid")


@pytest.mark.parametrize("text", ["", "   ", "I cannot tell.", "<answer></answer>", "E", "option 5", "<answer>A B C D</answer>"])
def test_adversarial_invalid(text):
    assert parse_answer(text, OPTS)[0] is None


def test_build_choices_drops_empty_and_nan_keeps_order():
    ch = build_choices({"B": "x", "A": "", "C": float("nan"), "D": "y"})
    assert list(ch) == ["B", "D"]
    assert not any(isinstance(v, float) and math.isnan(v) for v in ch.values())
