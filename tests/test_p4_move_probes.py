"""Tests of the label-free direction probes (docs/phase4_move_probes.md). Synthetic data only; the mutation checks of scripts/p4_move_probes_mutations.py remove one protective statement each."""
import cv2
import numpy as np
import pytest

from cgt.p4_camera_move import probes as PR
from cgt.p4_camera_move.labels import schema_moves

# the declared maps, written out here on purpose: the tests must not read the tables they check
DECLARED = {
    "T": {"push-in": "pull-out", "zoom-in": "zoom-out", "pan-left": "pan-right", "tilt-up": "tilt-down", "boom-up": "boom-down", "move-left": "move-right"},
    "H": {"pan-left": "pan-right", "move-left": "move-right"},
    "V": {"tilt-up": "tilt-down", "boom-up": "boom-down"},
}
ATOMS = [a for a in schema_moves() if a != "dolly-zoom"]


@pytest.fixture(autouse=True)
def small_bootstrap(monkeypatch):
    monkeypatch.setattr(PR, "N_BOOT", 40)


def test_partner_maps_equal_the_declared_maps_and_the_excluded_atoms_are_printed():
    for t, pairs in DECLARED.items():
        full = {**pairs, **{b: a for a, b in pairs.items()}}
        if t == "T":
            full["static"] = "static"
        assert PR.PARTNERS[t] == full
    PR.check_maps()
    assert PR.excluded_atoms() == ["arc", "roll", "dolly-zoom"]
    assert PR.invariant_atoms("H", ATOMS) == ["static", "push-in", "pull-out", "boom-up", "boom-down", "tilt-up", "tilt-down", "zoom-in", "zoom-out"]
    assert PR.invariant_atoms("T", ATOMS) == ["static"]


def _clip(path, n=40):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 23.976, (64, 48))
    for i in range(n):
        fr = np.zeros((48, 64, 3), np.uint8)
        fr[:, :, 2] = i  # frame index
        fr[:, :32, 1] = 200  # LEFT half green
        fr[:24, :, 0] = 200  # TOP half blue
        wr.write(fr)
    wr.release()


def _read(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, fr = cap.read()
        if not ok:
            return out
        out.append(fr)


def test_time_reversal_reverses_the_frame_order_across_blocks(tmp_path):
    _clip(tmp_path / "s.avi", 40)
    assert PR.write_transformed_clip(tmp_path / "s.avi", tmp_path / "t.avi", "T", block=16) == 40
    assert [int(f[0, 0, 2]) for f in _read(tmp_path / "t.avi")] == list(range(39, -1, -1))


def test_identity_through_the_same_writing_path_is_lossless(tmp_path):
    _clip(tmp_path / "s.avi")
    PR.write_transformed_clip(tmp_path / "s.avi", tmp_path / "i.avi", None)
    assert all(np.array_equal(a, b) for a, b in zip(_read(tmp_path / "s.avi"), _read(tmp_path / "i.avi"), strict=True))


def test_horizontal_mirror_swaps_left_and_right_only(tmp_path):
    _clip(tmp_path / "s.avi")
    PR.write_transformed_clip(tmp_path / "s.avi", tmp_path / "h.avi", "H")
    f = _read(tmp_path / "h.avi")[5]
    assert f[30, 10, 1] == 0 and f[30, 50, 1] == 200  # the green LEFT half is now on the right
    assert f[10, 10, 0] == 200 and f[40, 10, 0] == 0  # top stays top
    assert int(f[0, 0, 2]) == 5


def test_vertical_mirror_swaps_top_and_bottom_only(tmp_path):
    _clip(tmp_path / "s.avi")
    PR.write_transformed_clip(tmp_path / "s.avi", tmp_path / "v.avi", "V")
    f = _read(tmp_path / "v.avi")[5]
    assert f[10, 10, 0] == 0 and f[40, 10, 0] == 200  # the blue TOP half is now at the bottom
    assert f[30, 10, 1] == 200 and f[30, 50, 1] == 0  # left stays left


def _synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((n, len(ATOMS)))
    P[:, ATOMS.index("static")] = rng.random(n)
    swapped = {}
    for t in "THV":
        Q = P.copy()
        for a, b in DECLARED[t].items():  # transformed table = partner columns exchanged (the model behaves as it should)
            Q[:, ATOMS.index(a)] = P[:, ATOMS.index(b)]
            Q[:, ATOMS.index(b)] = P[:, ATOMS.index(a)]
        swapped[t] = Q
    return P, swapped


def test_a_model_that_swaps_partners_reads_direction_and_keeps_invariant_atoms_stable():
    P, Q = _synthetic()
    res = PR.evaluate_check_a(P, Q, ATOMS, np.zeros(len(P), dtype=bool))
    for t in "THV":
        pairs = res["transforms"][t]["pairs"]
        assert len(pairs) == len(DECLARED[t]) and all(p["band_primary"] == "reads direction" and p["rho_primary"] < -0.9 for p in pairs)
        assert all(x["band"] == "stable" and x["rho"] > 0.99 for x in res["transforms"][t]["invariant"])
    assert res["excluded_atoms"] == ["arc", "roll", "dolly-zoom"]


def test_the_permutation_null_is_near_zero():
    P, Q = _synthetic()
    res = PR.evaluate_check_a(P, Q, ATOMS, np.zeros(len(P), dtype=bool))
    nulls = [p["null_rho_permuted"] for t in "THV" for p in res["transforms"][t]["pairs"]] + [x["null_rho_permuted"] for t in "THV" for x in res["transforms"][t]["invariant"]]
    assert max(abs(v) for v in nulls) < 0.25


def test_a_model_that_ignores_the_transform_does_not_read_direction():
    P, _ = _synthetic()
    res = PR.evaluate_check_a(P, {t: P.copy() for t in "THV"}, ATOMS, np.zeros(len(P), dtype=bool))
    assert all(p["band_primary"] == "does not read direction" and p["rho_primary"] > 0.99 for t in "THV" for p in res["transforms"][t]["pairs"])


def test_pairs_with_fewer_than_30_confident_shots_get_no_verdict_and_flagged_shots_are_excluded():
    P, Q = _synthetic(n=40)
    res = PR.evaluate_check_a(P, Q, ATOMS, np.arange(40) < 10)
    assert res["n_flagged_excluded"] == 10
    assert any(p["band_primary"] == "insufficient support" for t in "THV" for p in res["transforms"][t]["pairs"])


def test_bands_are_the_declared_ones():
    assert [PR.band_pair(x) for x in (-0.9, -0.5, -0.3, -0.2, 0.4)] == ["reads direction", "reads direction", "weak", "does not read direction", "does not read direction"]
    assert [PR.band_invariant(x) for x in (0.95, 0.8, 0.79)] == ["stable", "stable", "unstable"]
    assert [PR.band_auc(x) for x in (0.9, 0.8, 0.7, 0.6, 0.59)] == ["consistent", "consistent", "weak", "weak", "inconsistent"]


def test_groups_s_m_x_follow_the_declared_definition():
    g = PR.groups([["static"], ["pan-left"], ["static", "push-in"], ["pan-left", "tilt-up"], ["static"]])
    assert g.tolist() == ["S", "M", "X", "M", "S"]


def test_auc_is_the_probability_of_a_larger_value_with_ties_counting_half():
    assert PR.auc_greater([3, 4], [1, 2]) == 1.0 and PR.auc_greater([1, 2], [3, 4]) == 0.0 and PR.auc_greater([1, 1], [1, 1]) == 0.5
    assert PR.auc_greater([2, 3], [1, 3]) == pytest.approx(0.625)


def test_motion_summaries_use_the_named_formulas():
    s = {"tx": np.array([3.0, 3.0, 0.0]), "ty": np.array([4.0, 4.0, 0.0]), "div": np.array([0.0, 6.0, 6.0]), "rot": np.array([0.0, 8.0, 8.0])}
    assert PR.motion_summaries(s) == (5.0, 10.0)  # medians of hypot(tx, ty) = [5, 5, 0] and hypot(div, rot) = [0, 10, 10]


def test_check_b_reports_consistency_between_motion_size_and_the_static_decision():
    rng = np.random.default_rng(0)
    n = 400
    moving = rng.random(n) < 0.5
    sets = [["pan-left"] if m else ["static"] for m in moving]
    motion = np.where(moving, rng.normal(2.0, 0.3, n), rng.normal(0.5, 0.3, n))
    res = PR.evaluate_check_b(sets, (~moving).astype(float) * 0.9, {"primary": motion, "secondary": motion}, np.zeros(n, dtype=bool))
    e = res["primary"]
    assert e["auc_M_vs_S"]["auc"] > 0.95 and e["auc_M_vs_S"]["band"] == "consistent" and e["rho_motion_vs_p_static"] < -0.7
    assert e["groups"]["S"]["n"] + e["groups"]["M"]["n"] == n and e["deciles"][0]["share_static_in_set"] > e["deciles"][9]["share_static_in_set"]
    assert e["null_rho_permuted"] > -0.2


def test_the_tripwire_makes_every_label_path_unopenable(tmp_path):
    ok = tmp_path / "ok.txt"
    ok.write_text("x")
    with PR.tripwire():
        assert ok.read_text() == "x"
        for bad in ("data/gold/shots.json", "data/dev/shots.json", "x/camerabench/test.jsonl", "data/external/shotbench/test.tsv", "something/videos.csv"):
            with pytest.raises(PermissionError):
                open(bad)  # noqa: SIM115
            with pytest.raises(PermissionError):
                PR.Path(bad).read_text()
    assert ok.read_text() == "x"


# ---------------------------------------------------------------------------- a recorded decision, P-C1, P-M1
def test_train_label_file_is_denied_by_default_and_allowed_only_inside_the_declared_context(tmp_path):
    grpo = tmp_path / "data/external/shotqa/grpo.json"
    grpo.parent.mkdir(parents=True)
    grpo.write_text("{}")
    with PR.tripwire():
        with pytest.raises(PermissionError):
            grpo.read_text()
        with PR.train_labels_allowed():
            assert grpo.read_text() == "{}"  # exactly this file, exactly here
            for other in ("data/gold/shots.json", "data/dev/shots.json", "x/camerabench/test.jsonl", "data/external/shotbench/test.tsv"):
                with pytest.raises(PermissionError):
                    open(other)  # noqa: SIM115
        with pytest.raises(PermissionError):  # the exception ends with the context
            grpo.read_text()


def test_the_cache_identity_follows_the_bytes_of_the_transform_and_conformity_code(tmp_path, monkeypatch):
    a = PR.impl_identity()
    assert a == PR.impl_identity()
    changed = tmp_path / "probes_copy.py"
    changed.write_bytes(PR.Path(PR.__file__).read_bytes() + b"\n# a source correction\n")
    monkeypatch.setattr(PR, "__file__", str(changed))
    assert PR.impl_identity() != a  # a corrected transform cannot reuse caches of the old one
    monkeypatch.undo()
    other = tmp_path / "film_move_copy.py"
    other.write_bytes(PR.Path(PR.FM.__file__).read_bytes() + b"\n# a conformity change\n")
    monkeypatch.setattr(PR.FM, "__file__", str(other))
    assert PR.impl_identity() != a  # nor can a changed clip conformity
    monkeypatch.undo()
    monkeypatch.setattr(PR.FM, "TARGET_FPS", 30.0)
    assert PR.impl_identity() != a
