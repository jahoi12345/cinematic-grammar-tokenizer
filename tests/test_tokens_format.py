"""CPU tests of the token-row formatter and the palette ids (docs/tokens_interface.md)."""
import numpy as np
import pytest

from cgt.tokens import format as fmt
from cgt.tokens.palette import fit_palette_ids

FULL = {"size": "MCU", "lens": "medium", "angle": "low", "framing": "single", "light_type": "daylight", "light_cond": "hard", "composition": "right-heavy",
        "key": "mid-key", "dof": "shallow", "people": 1, "sound": "dialogue", "len_s": 6.04, "palette_id": 7}


def test_row_matches_the_contract_example():
    assert fmt.format_row(FULL) == ("[size=MCU | lens=medium | angle=low | framing=single | light=daylight/hard | comp=right-heavy | key=mid-key | "
                                    "DoF=shallow | people=1 | sound=dialogue | len=6.0s | palette=p07 | move=n/a]")


def test_slot_order_is_fixed_and_parse_roundtrips():
    assert list(fmt.slot_values(FULL)) == fmt.SLOTS
    assert fmt.parse_row(fmt.format_row(FULL))["comp"] == "right-heavy"
    shuffled = "[lens=medium | size=MCU]"
    with pytest.raises(ValueError):
        fmt.parse_row(shuffled)
    with pytest.raises(ValueError):
        fmt.parse_row("size=MCU")


def test_disagreement_marker_only_on_false_and_per_component():
    rec = {**FULL, "size_consistent": False, "lens_consistent": True, "angle_consistent": None, "light_cond_consistent": False}
    v = fmt.slot_values(rec)
    assert v["size"] == "MCU*" and v["lens"] == "medium" and v["angle"] == "low"  # None (unknown) is not a disagreement
    assert v["light"] == "daylight/hard*"
    rec["light_type_consistent"] = False
    assert fmt.slot_values(rec)["light"] == "daylight*/hard*"
    assert "*" not in fmt.slot_values({**FULL})["size"]


def test_none_and_nan_become_question_marks_and_move_is_always_na():
    rec = {"size": None, "lens": float("nan"), "people": None, "len_s": None, "palette_id": None, "sound": None, "key": None, "dof": None}
    v = fmt.slot_values(rec)
    assert v["size"] == v["lens"] == v["angle"] == v["people"] == v["len"] == v["palette"] == v["sound"] == "?"
    assert v["light"] == "?/?" and v["move"] == "n/a"
    assert fmt.slot_values(FULL)["move"] == "n/a"


@pytest.mark.parametrize("n,tok", [(0, "0"), (3, "3"), (4, "4+"), (12, "4+")])
def test_people_is_capped_at_the_schema_maximum(n, tok):
    assert fmt.people_token(n) == tok


def test_palette_and_length_tokens():
    assert fmt.palette_token(0) == "p00" and fmt.palette_token(31) == "p31"
    assert fmt.length_token(37.26) == "37.3s" and fmt.length_token(0.04) == "0.0s"


def test_palette_ids_are_deterministic_lightness_ordered_and_bounded():
    rng = np.random.default_rng(1)
    dark = [np.tile([10.0, 0, 0], 5) + rng.normal(0, 0.5, 15) for _ in range(30)]
    light = [np.tile([90.0, 0, 0], 5) + rng.normal(0, 0.5, 15) for _ in range(30)]
    ids = fit_palette_ids(dark + light, k=2)
    again = fit_palette_ids(dark + light, k=2)
    assert (ids == again).all() and set(ids) == {0, 1}
    assert set(ids[:30]) == {0} and set(ids[30:]) == {1}  # p00 is the darkest cluster
    assert fit_palette_ids(dark[:3], k=32).max() < 3  # never more ids than distinct palettes


def test_parser_rejects_duplicated_slots_and_malformed_parts():
    row = fmt.format_row(FULL)
    with pytest.raises(ValueError, match="duplicate"):
        fmt.parse_row(row[:-1] + " | size=WS]")
    with pytest.raises(ValueError):
        fmt.parse_row(row[:-1] + " | junk]")
