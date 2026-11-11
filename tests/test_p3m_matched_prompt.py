"""Prompt equality with the documented template (docs/p3m_matched_protocol.md 1.2) and protocol hashing."""
import pytest

from cgt.p3m_matched import protocol as P


def test_user_text_exact():
    opts = {"A": "Wide", "B": "Medium", "C": "Close Up", "D": "Long Lens"}
    want = ("Question: What's the lens size of this shot?\nOptions:\nA. Wide\nB. Medium\nC. Close Up\nD. Long Lens\n"
            "Please select the most likely answer from the options above.")
    assert P.build_user_text("What's the lens size of this shot?", opts) == want


def test_options_keep_stored_order_and_question_unchanged():
    t = P.build_user_text("  q ", {"B": "x", "A": "y"})
    assert t == "Question:   q \nOptions:\nB. x\nA. y\nPlease select the most likely answer from the options above."


def test_differs_from_shotqa_training_tail():
    from cgt.p3b_vlm.shotbench import PROMPT_TAIL
    assert PROMPT_TAIL == P.USER_TAIL + " \n"  # the training turns carry an extra space + newline; the matched prompt does not


def test_system_prompts():
    assert P.SYSTEM_PLAIN == "You are a helpful assistant."
    r = P.SYSTEM_REASONING
    assert r.startswith("A conversation between User and Assistant. The user asks a question, and the Assistant solves it.")
    assert r.endswith("enclosed within <think> </think> and <answer> </answer> tags.")
    assert "  " not in r and "respectively" not in r


def test_messages_layout():
    m = P.build_messages("q", {"A": "x"}, "reasoning")
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[0]["content"][0]["text"] == P.SYSTEM_REASONING
    assert [c["type"] for c in m[1]["content"]] == ["image", "text"]
    with pytest.raises(ValueError):
        P.build_messages("q", {"A": "x"}, "other")


def test_decoding_and_hash():
    assert P.DECODING == {"do_sample": False, "temperature": 0.0, "max_new_tokens": 1024, "skip_special_tokens": True}
    assert P.protocol_sha256("plain") != P.protocol_sha256("reasoning")
    assert P.protocol_sha256("plain") == P.protocol_sha256("plain")


def test_question_from_shotqa_turn():
    t = "Question: What's the shot size of this shot?\nOptions:\nA. Wide\nPlease select the most likely answer from the options above. \n"
    assert P.question_from_shotqa_turn(t) == "What's the shot size of this shot?"
    with pytest.raises(ValueError):
        P.question_from_shotqa_turn("Options:\nA. x")
