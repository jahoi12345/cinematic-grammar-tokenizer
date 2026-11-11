"""Tests of the film-level movement runner (docs/phase4_film_move.md): refusals, cache identity, the unchanged decision rule, the contradictory-set definition, equality with the accepted NOTLD script. Synthetic only."""
import importlib.util
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cgt.p4_camera_move import film_move as FM

ROOT = Path(__file__).resolve().parents[1]
ATOMS = ["static", "push-in", "pull-out", "pan-left", "pan-right"]


def _registry(tmp_path, owned_file=True):
    import yaml
    (tmp_path / "configs").mkdir()
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/raw").mkdir()
    for fid, pd_flag in (("pd_film", True), ("owned_film", False)):
        (tmp_path / f"data/raw/{fid}.mp4").write_bytes(b"x")
        pd.DataFrame({"shot_id": [0], "start_s": [0.0], "end_s": [1.0], "len_s": [1.0]}).to_parquet(tmp_path / f"data/processed/{fid}_shots.parquet")
    films = [{"id": "pd_film", "public_domain": True, "path": "data/raw/pd_film.mp4"}, {"id": "owned_film", "public_domain": False, "path": "data/raw/owned_film.mp4"}]
    (tmp_path / "configs/films.yaml").write_text(yaml.safe_dump({"films": films}))
    return tmp_path


def test_unregistered_film_is_refused(tmp_path):
    root = _registry(tmp_path)
    with pytest.raises(FM.Refused) as e:
        FM.resolve_film("nope", root=root)
    assert e.value.code == 2


def test_owned_film_needs_the_explicit_flag(tmp_path):
    root = _registry(tmp_path)
    with pytest.raises(FM.Refused) as e:
        FM.resolve_film("owned_film", root=root)
    assert e.value.code == 2
    assert FM.resolve_film("owned_film", allow_owned=True, root=root)["id"] == "owned_film"
    assert FM.resolve_film("pd_film", root=root)["film_path"].name == "pd_film.mp4"


def test_missing_shot_table_or_film_is_refused(tmp_path):
    root = _registry(tmp_path)
    (root / "data/processed/pd_film_shots.parquet").unlink()
    with pytest.raises(FM.Refused):
        FM.resolve_film("pd_film", root=root)


def test_existing_output_is_refused_without_overwrite(tmp_path):
    out = tmp_path / "t.parquet"
    FM.refuse_existing(out, False)
    out.write_bytes(b"x")
    with pytest.raises(FM.Refused) as e:
        FM.refuse_existing(out, False)
    assert e.value.code == 2
    FM.refuse_existing(out, True)


def test_memory_guard_refuses_below_25_percent():
    with pytest.raises(FM.Refused) as e:
        FM.memory_guard(lambda: 24)
    assert e.value.code == 12
    FM.memory_guard(lambda: 25)


def test_cache_identity_follows_film_bytes_frame_range_and_code_keys(tmp_path):
    k = FM.film_key("a" * 64, 1.0, 2.0)
    f1, d1 = FM.cache_names(tmp_path, 5, k)
    assert (f1, d1) == FM.cache_names(tmp_path, 5, k)
    assert FM.cache_names(tmp_path, 5, FM.film_key("b" * 64, 1.0, 2.0)) != (f1, d1)  # another film file
    assert FM.cache_names(tmp_path, 5, FM.film_key("a" * 64, 1.0, 2.5)) != (f1, d1)  # another frame range
    assert FM.cache_names(tmp_path, 6, k) != (f1, d1) and f1.parent.name == "flow" and d1.parent.name == "depth"
    assert k.endswith(".r24")  # the conformity tag
    from cgt.p4_camera_move import depth_parallax as D
    from cgt.p4_camera_move import flow as F
    assert F.cache_key() in f1.name and D.cache_key() in f1.name  # the code keys are part of the name


def test_decision_rule_threshold_fallback_and_never_empty():
    P = np.array([[0.5, 0.2, 0.7, 0.1, 0.1], [0.3, 0.2, 0.1, 0.4, 0.1], [0.499999, 0.0, 0.0, 0.0, 0.0]])
    sets, fb = FM.predict(P, ATOMS)
    assert sets[0] == ["static", "pull-out"] and fb[0] is False  # 0.5 is included (>=)
    assert sets[1] == ["pan-left"] and fb[1] is True  # nothing reaches 0.5: the single top-1 atom
    assert sets[2] == ["static"] and fb[2] is True
    assert all(len(s) >= 1 for s in sets)


# the six pairs are written out here on purpose: the test must not read the tuple it checks
DECLARED_PAIRS = [("push-in", "pull-out"), ("pan-left", "pan-right"), ("tilt-up", "tilt-down"), ("boom-up", "boom-down"), ("zoom-in", "zoom-out"), ("move-left", "move-right")]


@pytest.mark.parametrize("a,b", DECLARED_PAIRS)
def test_every_opposite_pair_is_contradictory_in_both_orders(a, b):
    assert FM.is_contradictory([a, b])["opposite_pair"] and FM.is_contradictory([b, a])["contradictory"]
    assert not FM.is_contradictory([a])["contradictory"] and not FM.is_contradictory([b])["contradictory"]


def test_the_module_declares_exactly_the_six_pairs():
    assert sorted(map(sorted, FM.OPPOSITE_PAIRS)) == sorted(map(sorted, DECLARED_PAIRS))


def test_static_with_another_atom_is_contradictory_but_static_alone_is_not():
    assert FM.is_contradictory(["static", "push-in"]) == {"opposite_pair": False, "static_with_other": True, "contradictory": True}
    assert not FM.is_contradictory(["static"])["contradictory"]
    assert not FM.is_contradictory(["push-in", "pan-left"])["contradictory"]  # compatible atoms


def test_describe_counts_sizes_static_alone_and_contradictions():
    sets = [["static"], ["static", "push-in"], ["pan-left", "pan-right"], ["push-in"], ["tilt-up", "pan-left", "boom-down"]]
    d = FM.describe(sets, [False, False, False, True, False])
    assert d["set_sizes"] == {1: 2, 2: 2, 3: 1} and d["static_alone"] == 1 and d["fallback_top1"] == 1
    assert d["contradictory"] == 2 and d["opposite_pair"] == 1 and d["static_with_other"] == 1 and d["contradictory_share"] == 0.4


def _avi(path, n, fps):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (64, 48))
    for i in range(n):
        wr.write(np.full((48, 64, 3), (0, 0, i), np.uint8))
    wr.release()


def _frames(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, fr = cap.read()
        if not ok:
            return out
        out.append(int(fr[0, 0, 2]))


def test_cut_clip_resamples_in_time_and_equals_the_accepted_notld_function(tmp_path):
    spec = importlib.util.spec_from_file_location("p4_notld_move", ROOT / "scripts/p4_notld_move.py")
    notld = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notld)
    _avi(tmp_path / "film.avi", 120, 29.97)
    n1 = FM.cut_clip(tmp_path / "film.avi", 1.0, 3.0, tmp_path / "a.avi")
    n2 = notld.cut_clip(tmp_path / "film.avi", 1.0, 3.0, tmp_path / "b.avi")
    fa, fb = _frames(tmp_path / "a.avi"), _frames(tmp_path / "b.avi")
    assert n1 == n2 == len(fa) == int(2.0 * FM.TARGET_FPS) and fa == fb
    assert fa[0] == round(1.0 * 29.97) and all(y >= x for x, y in itertools.pairwise(fa))
    P = np.random.default_rng(0).random((50, 5))
    assert FM.predict(P, ATOMS) == notld.predict(P, ATOMS)


def test_the_runner_refuses_an_unregistered_film_from_the_command_line():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_move_film.py"), "--film", "not_a_film"], capture_output=True, text=True, check=False, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 2 and "not registered" in r.stdout and json is not None
