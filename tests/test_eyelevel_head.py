"""CPU tests for the eye-level head. Synthetic data only."""
import numpy as np
import pytest

from cgt.eyelevel import head as H


def _split():
    return {"val": ["cinescale2/angle/neutral/neutral_00002.jpg", "cinescale2/angle/high/high_00001.jpg",
                    "cinescale2/level/eye/eye_00001.jpg"],
            "excluded": {"near_shotbench": ["cinescale2/angle/low/low_00009.jpg"], "synthetic": []}}


IDS = ["cinescale2/angle/neutral/neutral_00001.jpg", "cinescale2/angle/neutral/neutral_00002.jpg",
       "cinescale2/angle/high/high_00001.jpg", "cinescale2/angle/high/high_00002.jpg",
       "cinescale2/angle/low/low_00009.jpg", "cinescale2/angle/dutch/dutch_00001.jpg",
       "cinescale2/level/eye/eye_00001.jpg", "cinescale2/level/eye/eye_00002.jpg"]


# ---------------------------------------------------------------- split / groups
def test_angle_split_drops_excluded_level_and_val():
    train, val = H.angle_split(_split(), IDS)
    assert val == ["cinescale2/angle/high/high_00001.jpg", "cinescale2/angle/neutral/neutral_00002.jpg"]
    assert train == ["cinescale2/angle/dutch/dutch_00001.jpg", "cinescale2/angle/high/high_00002.jpg",
                     "cinescale2/angle/neutral/neutral_00001.jpg"]
    assert not set(train) & set(val)
    assert [H.label_of(i) for i in train] == [H.CLASSES.index("dutch"), H.CLASSES.index("high"), H.EYE]


def test_angle_split_refuses_pool_missing_val():
    with pytest.raises(ValueError, match="missing"):
        H.angle_split(_split(), [i for i in IDS if "neutral_00002" not in i])


def test_check_groups_detects_group_across_sides():
    g = {"a": "a", "b": "a", "c": "c"}
    H.check_groups(g, ["a", "b"], ["c"])
    with pytest.raises(AssertionError):
        H.check_groups(g, ["a"], ["b", "c"])


def test_folds_keep_groups_together_and_cover_all():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 300)
    groups = [f"g{k // 3}" for k in range(300)]  # groups of 3
    fold = H.folds(y, groups)
    assert set(fold) == set(range(H.N_FOLDS))
    for g in set(groups):
        assert len({fold[k] for k in range(300) if groups[k] == g}) == 1


def test_group_bootstrap_resamples_whole_groups():
    groups = ["a", "a", "b", "b", "c"]
    seen = []
    H.group_bootstrap(groups, lambda ix: seen.append(sorted(ix.tolist())) or 0.0, n_boot=50)
    for ix in seen:  # a group's members always appear together and equally often
        assert ix.count(0) == ix.count(1) and ix.count(2) == ix.count(3)


def test_build_groups_uses_hash_and_thumb_rules():
    th = np.random.default_rng(1).integers(0, 255, (3, 2, 32, 32)).astype(np.uint8)
    th[1] = th[0]  # same frame as 0
    full = np.array([0b1010, 0b1010, 2**63 - 1], np.uint64)
    ids = [f"/x/unpacked/cinescale2/angle/high/high_{k}.jpg" for k in range(3)]
    g = H.build_groups(ids, full, full.copy(), th)
    assert g["cinescale2/angle/high/high_0.jpg"] == g["cinescale2/angle/high/high_1.jpg"]
    assert g["cinescale2/angle/high/high_2.jpg"] == "cinescale2/angle/high/high_2.jpg"


def test_check_embedder():
    ok = '{"model": "google/siglip-base-patch16-224", "revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", ' \
         '"weights_sha256": "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"}'
    H.check_embedder(ok)
    with pytest.raises(ValueError):
        H.check_embedder(ok.replace("7fd15f", "000000"))


# ---------------------------------------------------------------- selection is train-only
def test_select_threshold_reaches_target_on_train():
    ids = [f"t{k}" for k in range(10)]
    scores = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .0])
    is_eye = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0], bool)
    r = H.select_threshold(ids, scores, is_eye, val_ids=["v1"], target=0.8)
    # flags >= .5: 4 tp / 5 = 0.8 -> recall 4/5; lower thresholds fall below 0.8
    assert r["reached_target"] and r["threshold"] == 0.5
    assert r["train_oof_precision"] == pytest.approx(0.8) and r["train_oof_recall"] == pytest.approx(0.8)


def test_select_threshold_fails_if_val_passed():
    ids = ["t1", "t2", "v1"]
    with pytest.raises(ValueError, match="train only"):
        H.select_threshold(ids, np.array([.9, .1, .5]), np.array([1, 0, 1], bool), val_ids=["v1"])
    with pytest.raises(ValueError):
        H.select_threshold(["t1"], np.array([.9]), np.array([1], bool), val_ids=[])


def test_select_threshold_not_reached_falls_back_to_f1():
    r = H.select_threshold(["a", "b", "c"], np.array([.9, .8, .1]), np.array([0, 1, 0], bool), val_ids=["v"], target=0.99)
    assert not r["reached_target"] and r["threshold"] == 0.8


def test_select_configs_train_only_and_ties():
    res = {"m1": {"family": "multinomial", "C": 0.1, "cw": None, "balanced_acc": 0.7, "eye_ap": 0.5},
           "m2": {"family": "multinomial", "C": 0.01, "cw": None, "balanced_acc": 0.7, "eye_ap": 0.4},
           "b1": {"family": "binary", "C": 0.01, "cw": None, "eye_ap": 0.6}}
    assert H.select_configs(res, ["t"], ["v"]) == ("m2", "b1")
    with pytest.raises(ValueError):
        H.select_configs(res, ["t", "v"], ["v"])


def test_grid_is_the_preregistered_one():
    g = H.grid()
    assert len(g) == 28 and {c.family for c in g} == {"multinomial", "binary"}
    assert H.TARGET_PRECISION == 0.80 and (H.ADOPT_PRECISION, H.ADOPT_RECALL, H.ADOPT_GUARD_CAP) == (0.75, 0.60, 0.05)


# ---------------------------------------------------------------- adoption rule
@pytest.mark.parametrize("p,r,g,reached,ok", [
    (0.75, 0.60, 0.05, True, True),
    (0.749, 0.9, 0.0, True, False),
    (0.9, 0.599, 0.0, True, False),
    (0.9, 0.9, 0.0501, True, False),
    (0.9, 0.9, 0.0, False, False),
    (float("nan"), 0.9, 0.0, True, False),
])
def test_adoption_rule(p, r, g, reached, ok):
    adopt, reasons = H.adoption(p, r, g, reached)
    assert adopt is ok and (len(reasons) == 0) == ok


def test_eye_prf_and_wilson():
    m = H.eye_prf(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)
    lo, hi = H.wilson(0, 838)
    assert lo == 0.0 and 0 < hi < 0.01


def test_tiny_end_to_end_cv_and_threshold():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 40)
    x = rng.normal(size=(200, 8)) + np.eye(5, 8)[y] * 3
    groups = [f"g{k // 2}" for k in range(200)]
    fold = H.folds(y, groups)
    cfg = H.Config("binary", 1.0, None)
    eye, probs = H.oof(cfg, H.l2norm(x), y, fold)
    assert probs is None and eye.shape == (200,)
    r = H.select_threshold([f"t{k}" for k in range(200)], eye, y == H.EYE, val_ids=["v"])
    assert r["reached_target"] and r["train_oof_precision"] >= 0.8
