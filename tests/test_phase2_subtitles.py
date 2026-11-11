"""Hand-authored cues with known timestamps and overlap, never film transcripts."""

import pytest

from cgt.p2_pretrained.subtitles import parse_srt, text_for_shot


def test_parse_handmade_srt_fractional_times_multiline_and_bad_block(tmp_path):
    path = tmp_path / "handmade.srt"
    path.write_bytes(b"1\n00:00:01,18 --> 00:00:02,01\nHello\nworld\n\n"
                     b"bad block without timestamp\n\n"
                     b"2\n00:00:03.250 --> 00:00:04.750\nSecond cue\n\n"
                     b"3\n00:00:05,000 --> 00:00:06,000\nBad byte \xff\n")
    cues = parse_srt(str(path))
    assert len(cues) == 3
    assert cues[0][:2] == pytest.approx((1.18, 2.01))
    assert cues[0][2] == "Hello world"
    assert cues[1] == (3.25, 4.75, "Second cue")
    assert cues[2][2] == "Bad byte \ufffd"
    assert parse_srt(str(path)) == cues
    path.write_text("")
    assert parse_srt(str(path)) == []


def test_subtitle_overlap_alignment_and_empty_string():
    cues = [(1.0, 3.0, "long"), (3.0, 4.0, "short"), (4.0, 4.2, "tiny")]
    assert text_for_shot(cues, 2.5, 3.5) == "long short"
    assert text_for_shot(cues, 2.6, 3.0) == "", "0.4 s is below the long-cue 0.5 s requirement"
    assert text_for_shot(cues, 3.0, 3.31) == "short"
    assert text_for_shot(cues, 4.0, 4.07) == "tiny"
    assert text_for_shot(cues, 4.2, 5.0) == ""
    assert text_for_shot([], 0, 10) == ""


def test_subtitle_nondefault_overlap_follows_parameter():
    cues = [(0., 1., 'cue')]
    assert text_for_shot(cues, .0, .2, min_overlap=.1) == 'cue'
    assert text_for_shot(cues, .0, .2, min_overlap=.4) == ''
    assert text_for_shot(cues, .0, .6, min_overlap=.9) == 'cue'  # cap .5
