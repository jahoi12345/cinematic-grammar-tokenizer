"""CPU tests for Step B data streams, question format and the pre-registered selection rule (no model is loaded)."""
import json
from pathlib import Path

import pytest

from cgt.stepb import data as D
from cgt.stepb import select as S
from cgt.stepb.evaluate import shotqa_points, summarize_cs2

ROOT = Path(__file__).resolve().parents[1]


def test_cs2_question_format_matches_shotqa_and_is_deterministic():
    q = D.cs2_question("cinescale2/angle/neutral/neutral_00001.jpg", "/x/neutral_00001.jpg", "neutral", 0)
    assert q == D.cs2_question("cinescale2/angle/neutral/neutral_00001.jpg", "/x/neutral_00001.jpg", "neutral", 0)
    text = q["messages"][0]["content"]
    assert text.startswith("Question: What's the camera angle of this shot?\nOptions:\nA. ")
    assert text.endswith("\nPlease select the most likely answer from the options above. \n")
    assert q["options"][q["answer"]] == "Eye level" and q["messages"][1]["content"] == q["answer"]
    assert len(set(q["options"].values())) == 4 and "Aerial" not in q["options"].values()
    assert set(q["options"].values()) <= {"Overhead", "High angle", "Eye level", "Low angle", "Dutch angle"}
    from cgt.p3_static_attrs.labels import parse_question
    attr, opts = parse_question("<image>" + text)
    assert attr == "angle" and opts == q["options"]
    q1 = D.cs2_question("cinescale2/angle/neutral/neutral_00001.jpg", "/x", "neutral", 1)
    assert q1["id"].endswith("#1")


def test_answer_letters_and_distractors_are_spread():
    letters, distract = [], set()
    for k in range(400):
        q = D.cs2_question(f"cinescale2/angle/high/high_{k:05d}.jpg", "/x", "high", 0)
        letters.append(q["answer"])
        distract |= set(q["options"].values()) - {"High angle"}
    assert {letters.count(x) for x in "ABCD"} and min(letters.count(x) for x in "ABCD") > 60
    assert distract == {"Overhead", "Eye level", "Low angle", "Dutch angle"}


def test_balanced_draws():
    by = {"overhead": ["o1"], "high": [f"h{k}" for k in range(100)], "low": ["l1", "l2"]}
    d = D.balanced_draws(by, 3000, seed=2)
    assert d == D.balanced_draws(by, 3000, seed=2)
    counts = {c: sum(1 for r, _ in d if r[0] == c[0]) for c in by}
    assert all(900 < v < 1100 for v in counts.values())
    o = [k for r, k in d if r == "o1"]
    assert o == list(range(len(o)))          # pass index counts earlier draws of the same image


def test_replay_mix_dedup_and_candidates():
    assert D.replay_subset(100, 10, 1) == D.replay_subset(100, 10, 1) and len(set(D.replay_subset(100, 10, 1))) == 10
    with pytest.raises(ValueError):
        D.replay_subset(5, 6)
    assert sorted(D.mix(list(range(20)), 0)) == list(range(20))
    s = [{"id": "a", "v": 1}, {"id": "b"}, {"id": "a", "v": 2}]
    assert D.dedup_first(s) == [{"id": "a", "v": 1}, {"id": "b"}]
    assert all(sum(v) == 13_734 for v in D.CANDIDATES.values()) and D.RUN_ORDER == ("B0", "B1", "B2")
    assert D.CANDIDATES["B0"][0] == 0


def _ck(cand, pct, primary, sec, **attr):
    per = {a: 80.0 for a in S.ATTRS}
    per.update(attr)
    return {"candidate": cand, "pct": pct, "primary": primary, "per_attribute": per, "secondary": sec}


BASE = {"primary": 81.0, "per_attribute": {a: 80.0 for a in S.ATTRS}}


def test_selection_eligibility_rules():
    assert S.eligibility(_ck("B1", 100, 81.5, 50), BASE)["eligible"]
    assert not S.eligibility(_ck("B1", 100, 81.49, 50), BASE)["eligible"]
    assert not S.eligibility(_ck("B1", 100, 82.0, 50, lens=77.9), BASE)["eligible"]
    assert S.eligibility(_ck("B1", 100, 82.0, 50, lens=78.0), BASE)["eligible"]
    assert not S.eligibility(_ck("B1", 100, 82.0, 50, angle=78.9), BASE)["eligible"]
    assert S.eligibility(_ck("B1", 100, 82.0, 50, angle=79.0), BASE)["eligible"]


def test_selection_tie_band_secondary_then_checkpoint_then_order():
    cks = [_ck("B0", 50, 82.00, 40), _ck("B1", 100, 82.05, 60), _ck("B2", 100, 81.96, 70), _ck("B2", 50, 81.80, 99)]
    r = S.select(cks, BASE)
    assert r["p_star"] == 82.05 and set(r["tie_set"]) == {("B0", 50), ("B1", 100), ("B2", 100)}
    assert (r["winner"]["candidate"], r["winner"]["pct"]) == ("B2", 100)
    r = S.select([_ck("B1", 100, 82.0, 60), _ck("B1", 50, 82.0, 60), _ck("B0", 50, 82.0, 60)], BASE)
    assert (r["winner"]["candidate"], r["winner"]["pct"]) == ("B0", 50)
    r = S.select([_ck("B2", 100, 82.0, 60), _ck("B1", 100, 82.0, 60.00001)], BASE)
    assert r["winner"]["candidate"] == "B1"           # secondary compared at 4 decimals: equal -> run order
    assert S.select([_ck("B1", 100, 81.2, 99)], BASE)["winner"] is None


def test_summaries():
    res = [{"cs2_class": "high", "pred": "A", "gold": "A"}, {"cs2_class": "high", "pred": "B", "gold": "A"},
           {"cs2_class": "dutch", "pred": "C", "gold": "C"}]
    s = summarize_cs2(res)
    assert s["n"] == 3 and abs(s["acc"] - 200 / 3) < 1e-9 and s["recall"] == {"dutch": 100.0, "high": 50.0}
    assert s["balanced_recall"] == 75.0 and summarize_cs2([])["acc"] is None
    p = shotqa_points({"mean_acc": 0.8, "n": 10, "per_attribute": {"size": {"acc": 0.75}}})
    assert p == {"primary": 80.0, "n": 10, "per_attribute": {"size": 75.0}}


def test_committed_data_summary():
    p = ROOT / "results/stepb/data_summary.json"
    if not p.exists():
        pytest.skip("data not built")
    s = json.loads(p.read_text())
    assert s["split"]["file"] == "results/cinescale2/split_v3.json"
    assert s["shotqa_val_dedup"] == 5762 and s["cs2_angle_val"] == 1178 and s["cs2_angle_train"] == 6709
    for c, (n_cs2, n_qa) in D.CANDIDATES.items():
        f = s["files"][c]
        assert f["n"] == 13_734 and f["cs2"] == n_cs2 and f["shotqa"] == n_qa
