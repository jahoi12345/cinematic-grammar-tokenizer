"""Fast CPU tests of the pure logic and the sealed protocol of the Phase 4 final evaluator (docs/phase4_interface.md section 10). Synthetic data only: no model, no test set, no real ledger."""
import itertools
import json

import numpy as np
import pytest

from cgt.p3b_vlm import guards
from cgt.p4_camera_move import final_eval as FE
from cgt.p4_camera_move.final_seal import FinalSealed
from cgt.p4_camera_move.labels import parse_move_label, schema_moves

ATOMS = [a for a in schema_moves() if a in FE.OPTION_WORDING]
TAGS = ["dolly-in", "pan-left", "pan-right", "tilt-up", "static", "zoom-in", "truck-left", "pedestal-up", "arc-CW", "roll-CW"]


def _world(n=40, seed=3):
    rng = np.random.default_rng(seed)
    names = sorted(f"V{i:03d}.{i % 4}.{i}.gif" for i in range(n))
    labels = {nm.rsplit(".", 1)[0]: [str(t) for t in rng.choice(TAGS, int(rng.integers(1, 3)), replace=False)] for nm in names}
    labels[names[0].rsplit(".", 1)[0]] = ["no-motion"]
    return names, labels


def test_option_wording_round_trips_through_the_label_parser():
    for r in (1, 2, 3):
        for c in itertools.combinations(ATOMS, r):
            m = parse_move_label(FE.option_text(c))
            assert set(m.atoms) == set(c) and not m.unmapped


def test_generator_invariants_determinism_and_mapping_variants():
    names, labels = _world()
    q1, d1 = FE.build_camerabench_questions(names, labels, FE.MAPPINGS["main"])
    q2, _ = FE.build_camerabench_questions(names, labels, FE.MAPPINGS["main"])
    assert FE.questions_digest(q1, True) == FE.questions_digest(q2, True)
    sets = FE.gold_sets(names, labels, FE.MAPPINGS["main"])
    for q in q1:
        assert len({q["option_atoms"][L] for L in "ABCD"}) == 4 and q["option_atoms"][q["gold_letter"]] == q["gold_atoms"]
        assert all(q["option_atoms"][L] in [s for j, s in enumerate(sets) if j != q["index"] and s] for L in "ABCD" if L != q["gold_letter"])
    assert d1["no_mapped_atom"] == 1  # the clip with only `no-motion`
    q_var, _ = FE.build_camerabench_questions(names, labels, FE.MAPPINGS["no_motion_as_static"])
    assert len(q_var) == len(q1) + 1 and next(q for q in q_var if q["index"] == 0)["gold_atoms"] == ("static",)


def test_generator_drops_a_question_with_too_few_distinct_distractors():
    labels = {"a": ["pan-left"], "b": ["pan-left"], "c": ["tilt-up"], "d": ["pan-left"]}
    q, d = FE.build_camerabench_questions(["a.gif", "b.gif", "c.gif", "d.gif"], labels, FE.MAPPINGS["main"])
    assert q == [] and d["fewer_than_3_distinct_distractors"] == 4


def test_tie_rule_and_scoring():
    assert FE.choose([1, 1, 1, 1]) == "A" and FE.choose([0, 2, 2, 1]) == "B"
    z, g = np.array([0.0, 1.0, 0.0, 0.0]), np.array([-5.0, -1.0, -3.0, -4.0])
    assert np.allclose(FE.combine("combined", z, g, 0.3, 0.7), 0.3 * FE.lsm(z) + 0.7 * g)
    assert np.allclose(FE.combine("geometry", None, g, 0.3, 0.7), g) and np.allclose(FE.combine("video", z, None, 0.3, 0.7), FE.lsm(z))


def test_go_text_binds_every_field():
    base = ("shotbench_movement", "t", "a" * 64, "c" * 16, "f" * 64)
    g = FE.go_text(*base)
    for i, other in enumerate(("camerabench_public_test", "u", "b" * 64, "d" * 16, "0" * 64)):
        changed = list(base)
        changed[i] = other
        assert FE.go_text(*changed) != g


def test_go_file_missing_or_unbound_authorizes_nothing(tmp_path):
    g = FE.go_text("shotbench_movement", "t", None, "c" * 16, "f" * 64)
    with pytest.raises(guards.NotAuthorized):
        FE.require_go(tmp_path / "nope", g)
    (tmp_path / "go").write_text(g + "x")
    with pytest.raises(guards.NotAuthorized):
        FE.require_go(tmp_path / "go", g)
    (tmp_path / "go").write_text(g)
    FE.require_go(tmp_path / "go", g)


def test_paired_bootstrap_is_zero_for_identical_hits_and_positive_for_a_better_model():
    rng = np.random.default_rng(1)
    src = np.repeat(np.arange(30), 4)
    a, b = rng.random(120) < 0.85, rng.random(120) < 0.55
    assert FE.paired_bootstrap(a, a, src)["ci95"] == [0.0, 0.0]
    assert FE.paired_bootstrap(a, b, src)["ci95"][0] > 0


def test_recalibration_uses_only_test_features():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(120, 6)), (rng.random((120, 3)) < 0.4).astype(int)
    Xte = rng.normal(size=(30, 6))
    base = FE.fit_atom_probs(X, Y, Xte, 0.1, recalibrate=True)
    assert np.allclose(base, FE.fit_atom_probs(X, Y, Xte * 3.0 + 5.0, 0.1, recalibrate=True), atol=1e-6)
    assert np.allclose(FE.fit_atom_probs(X, Y, X, 0.1, recalibrate=True), FE.fit_atom_probs(X, Y, X, 0.1))


def test_sealed_unknown_commit_publishes_nothing(tmp_path):
    log = []
    ev = FinalSealed("k", tmp_path / "raw", log.append, code_commit="c")
    ev.start(1)

    def boom():
        raise OSError("ledger unreadable")
    with pytest.raises(guards.CommitUnknown):
        ev.seal_after_labels(boom, boom, lambda: ({tmp_path / "x.json": "secret 0.4321"}, "echo 0.4321"))
    assert (tmp_path / "raw/k.UNKNOWN_COMMIT.json").exists() and not (tmp_path / "x.json").exists()
    assert "0.4321" not in "\n".join(log) + (tmp_path / "raw/k.UNKNOWN_COMMIT.json").read_text()


def test_sealed_post_commit_failure_is_consumed_and_writes_what_it_can(tmp_path):
    ev = FinalSealed("k", tmp_path / "raw", lambda m: None, code_commit="c")
    ev.start(1)
    bad = tmp_path / "isadir"
    bad.mkdir()
    with pytest.raises(guards.PostLedgerWriteFailed):
        ev.seal_after_labels(lambda: None, lambda: True, lambda: ({bad: "x", tmp_path / "ok.txt": "y"}, "echo"))
    rec = json.loads((tmp_path / "raw/k.CONSUMED.json").read_text())
    assert (tmp_path / "ok.txt").exists() and rec["ledger_consumed"] is True
