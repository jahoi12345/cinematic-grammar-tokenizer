"""CPU tests of token row v1: length bins, soft vote, global palette assignment, row format, v1 build (synthetic tables) and the fail-closed completion check."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgt.p3c_tokens.vocab import labels
from cgt.tokens import build, build_v1, v1
from cgt.tokens import palette_global as pg

REPO = Path(__file__).resolve().parents[1]
ATTRS = build.P3C_ATTRS


@pytest.mark.parametrize("sec,idx,tok", [(0.0, 0, "<1s"), (0.99, 0, "<1s"), (1.0, 1, "1-2s"), (3.9, 2, "2-4s"), (4.0, 3, "4-8s"), (8.0, 4, "8-16s"),
                                         (16.0, 5, "16-32s"), (31.9, 5, "16-32s"), (32.0, 6, "32+s"), (500.0, 6, "32+s")])
def test_length_bins_are_fixed_log_bins_with_inclusive_lower_edge(sec, idx, tok):
    assert v1.len_bin(sec) == idx and v1.len_token(sec) == tok


def test_len_token_none_is_question_mark():
    assert v1.len_token(None) == "?" and v1.len_token(float("nan")) == "?"


def test_soft_vote_is_the_mean_of_probabilities_not_of_scores():
    a = np.log([0.9, 0.05, 0.05])
    b = np.log([0.05, 0.9, 0.05])
    c = np.log([0.05, 0.9, 0.05])
    mean = v1.soft_vote([a, b, c])
    assert np.allclose(mean, [(0.9 + 0.05 + 0.05) / 3, (0.05 + 0.9 + 0.9) / 3, 0.05]) and mean.sum() == pytest.approx(1.0)
    i, p1, margin = v1.primary_and_margin(mean)
    assert i == 1 and p1 == pytest.approx(0.6167, abs=1e-3) and margin == pytest.approx(0.6167 - 0.3333, abs=1e-3)
    # a confident single frame can lose to two mild ones under probability averaging but not under score averaging of log-probabilities
    x, y, z = np.log([0.99, 0.01]), np.log([0.4, 0.6]), np.log([0.4, 0.6])
    assert v1.primary_and_margin(v1.soft_vote([x, y, z]))[0] == 0


def test_primary_ties_go_to_the_lowest_index():
    assert v1.primary_and_margin(np.array([0.5, 0.5, 0.0]))[0] == 0


def test_global_palette_assignment_and_lightness_order():
    rng = np.random.default_rng(0)
    dark = np.tile([10.0, 0, 0], 5) + rng.normal(0, 0.3, (40, 15))
    light = np.tile([90.0, 0, 0], 5) + rng.normal(0, 0.3, (40, 15))
    c = pg.fit_centroids(np.vstack([dark, light]), k=2)
    assert c[0, 0] < c[1, 0]  # g00 darkest
    ids = pg.assign(np.vstack([dark, light]), c)
    assert set(ids[:40]) == {0} and set(ids[40:]) == {1}
    assert (pg.assign(np.vstack([dark, light]), c) == ids).all()


def test_row_v1_format_slots_and_markers():
    rec = {"size": "MCU", "lens": "medium", "angle": "low", "framing": "single", "light_type": "daylight", "light_cond": "hard", "composition": "right-heavy",
           "key": "mid-key", "dof": "shallow", "people": 6, "sound": "dialogue", "len_s": 6.04, "palette_global": 7, "size_consistent": False}
    row = v1.format_row(rec)
    assert row == ("[size=MCU* | lens=medium | angle=low | framing=single | light=daylight/hard | comp=right-heavy | key=mid-key | DoF=shallow | "
                   "people=4+ | sound=dialogue | len=4-8s | palette=g07 | move=n/a]")
    assert list(v1.parse_row(row)) == v1.SLOTS and len(v1.SLOTS) == 13
    assert v1.palette_token(None) == "?"


def test_config_matches_code_constants():
    cfg = v1.load_config()
    assert cfg["slots"] == v1.SLOTS and cfg["len_bins"]["names"][0] == "<1" and len(cfg["len_bins"]["edges"]) == 6
    assert cfg["style"]["enabled"] is False and cfg["angle"]["reserved_extra"] == ["eye"] and cfg["palette"]["k"] == 32
    assert "eye" not in labels("angle")  # reserved in the tokenizer vocabulary only, never in the scored vocabulary


@pytest.fixture
def root(tmp_path):
    n = 5
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO / "configs/tokens.yaml", tmp_path / "configs/tokens.yaml")
    (tmp_path / "data/cache/tokens").mkdir(parents=True)
    cents = np.array([np.tile([10.0 + 20 * i, 0, 0], 5) for i in range(32)])
    pg.save(tmp_path / "data/cache/tokens/palette_global_k32.npz", cents, ["x.jpg"], 1)
    pal = [list(np.tile([10.0 + 20 * i, 0, 0], 5)) for i in range(n)]
    pd.DataFrame({"film_id": "f", "shot_id": range(n), "start_s": [60.0 * i for i in range(n)], "end_s": [60.0 * i + 5 for i in range(n)],
                  "len_s": [0.5, 1.5, 3.0, 9.0, 40.0], "kf_start": "a", "kf_mid": "b", "kf_end": "c", "palette_lab": pal, "key": "mid-key"}
                 ).to_parquet(tmp_path / "data/processed/f_shots.parquet")
    pd.DataFrame({"shot_id": range(n), "dof": "deep", "people": [0, 1, 2, 3, 9], "sound": "score"}).to_parquet(tmp_path / "data/processed/f_p2.parquet")
    p3 = {"shot_id": list(range(n)), "p3c_run_key": "k1"}
    for a in ATTRS:
        p3[f"{a}_pred"] = labels(a)[0]
    pd.DataFrame(p3).to_parquet(tmp_path / "data/processed/f_p3c.parquet")
    run = tmp_path / "data/cache/p3c/runs/f_k1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"state": "complete", "finished": "t", "coverage": {k: n for k in ("kf_mid", "kf_start", "kf_end")}}))
    rng = np.random.default_rng(0)
    for kf in ("kf_mid", "kf_start", "kf_end"):
        lines = []
        for i in range(n):
            rec = {"shot_id": i}
            for a in ATTRS:
                v = rng.normal(size=len(labels(a)))
                if i == 0:
                    v = np.zeros(len(labels(a)))
                    v[1 if kf == "kf_start" else 0] = 5.0  # shot 0: start keyframe disagrees with mid/end (primary 1 vs 0)
                rec[a] = [float(x) for x in v]
            lines.append(json.dumps(rec))
        (run / f"scores_{kf}.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_v1_build_columns_vote_bins_and_markers(root):
    df = build_v1.build_tokens_v1("f", root)
    assert df.shot_id.tolist() == list(range(5)) and df.tokens_version.eq("v1").all()
    assert df.len_bin.tolist() == [0, 1, 2, 4, 6]
    assert df.palette_global.tolist() == [0, 1, 2, 3, 4]  # nearest synthetic centroid
    r0 = df.iloc[0]
    assert r0["size"] == labels("size")[0] and r0["size_kfmid"] == labels("size")[0] and r0["size_consistent"] is False or not r0["size_consistent"]
    assert r0["n_kf_disagree"] == 7 and "size=" + labels("size")[0] + "*" in r0["token_row"]
    assert df["size_probs_mean"].map(len).eq(len(labels("size"))).all() and df["size_p1"].between(0, 1).all()
    assert df.token_row.map(lambda t: list(v1.parse_row(t))).map(tuple).nunique() == 1
    assert "len=<1s" in df.token_row[0] and "len=32+s" in df.token_row[4] and "palette=g00" in df.token_row[0]


def test_v1_write_is_idempotent_and_never_bypasses_completion(root):
    _, txt, rebuilt = build_v1.write_tokens_v1("f", root)
    assert rebuilt and txt.read_text().splitlines()[0].startswith("00000 000:00.0 [size=")
    assert build_v1.write_tokens_v1("f", root)[2] is False
    (root / "data/cache/p3c/runs/f_k1/manifest.json").write_text(json.dumps({"state": "running"}))
    with pytest.raises(build.IncompletePhase3C):
        build_v1.write_tokens_v1("f", root)


def test_v1_rebuilds_when_the_palette_or_a_score_file_changes(root):
    build_v1.write_tokens_v1("f", root)
    p = root / "data/cache/p3c/runs/f_k1/scores_kf_end.jsonl"
    p.write_text(p.read_text().replace("0.0", "0.5", 1))
    assert build_v1.write_tokens_v1("f", root)[2] is True


def test_frozen_centroid_artifact_is_verified_against_its_digest(tmp_path):
    c = np.arange(32 * 15, dtype=float).reshape(32, 15)
    pg.save(tmp_path / "p.npz", c, ["a.jpg"], 1, pg.fit_provenance(c))
    meta = json.loads((tmp_path / "p.json").read_text())
    assert meta["centroid_sha256"] == pg.centroid_digest(c) and meta["fit_provenance"]["input_dtype"] == "float64" and "threadpools" in meta["fit_provenance"]
    assert (pg.load(tmp_path / "p.npz") == c).all()
    np.savez(tmp_path / "p.npz", centroids=c + 1e-9)  # the artifact was replaced: refuse rather than silently change every id
    with pytest.raises(pg.CentroidMismatch):
        pg.load(tmp_path / "p.npz")


def _write_move(root, atoms_by_shot, extra=True):
    rows = {"shot_id": list(atoms_by_shot), "move_atoms": list(atoms_by_shot.values())}
    if extra:
        rows.update({"n_atoms": [len(str(a).split(" + ")) for a in atoms_by_shot.values()], "fallback_top1": [True] + [False] * (len(atoms_by_shot) - 1),
                     "short_lt_1s": False, "few_pairs_lt_8": False, "no_features": False, "model": "move_geom_flowdepth_v1", "p_static": 0.5, "p_pan-left": 0.4})
    pd.DataFrame(rows).to_parquet(root / "data/processed/f_move.parquet")


def test_move_slot_is_na_without_a_table_and_a_prediction_with_one(root):
    df = build_v1.build_tokens_v1("f", root)
    assert df["move_present"].eq(False).all() and df["move"].eq("n/a").all() and df.token_row.str.contains("move=n/a").all()
    _write_move(root, {0: "static", 1: "pan-left + push-in", 2: "push-in + pan-left", 3: "zoom-out", 4: "static"})
    df = build_v1.build_tokens_v1("f", root)
    assert df["move_present"].all() and df["move"].tolist() == ["static", "push-in+pan-left", "push-in+pan-left", "zoom-out", "static"]
    assert df["move_atoms"].map(tuple).tolist()[1:3] == [("push-in", "pan-left")] * 2  # unordered set stored in schema order: both orders give the same set
    assert "move=static]" in df.token_row[0] and "move=push-in+pan-left]" in df.token_row[1] and list(v1.parse_row(df.token_row[1]))[-1] == "move"
    assert df["move_model"].eq("move_geom_flowdepth_v1").all() and bool(df["move_fallback_top1"].iloc[0]) and not df["move_fallback_top1"].iloc[1:].any()
    assert "move_p_static" in df.columns and df["move_n_atoms"].tolist() == [1, 2, 2, 1, 1]


@pytest.mark.parametrize("bad", [{0: "static", 1: "pan-left", 2: "static", 3: "static"},                       # missing shot
                                 {0: "static", 1: "pan-left", 2: "static", 3: "static", 4: "fly-through"},   # atom outside the schema
                                 {0: "static", 1: "", 2: "static", 3: "static", 4: "static"}])               # empty set is never allowed
def test_invalid_movement_tables_fail_closed(root, bad):
    _write_move(root, bad, extra=False)
    with pytest.raises(ValueError):
        build_v1.build_tokens_v1("f", root)


def test_movement_table_changes_the_sidecar_hash_and_rebuilds(root):
    build_v1.write_tokens_v1("f", root)
    assert build_v1.write_tokens_v1("f", root)[2] is False
    _write_move(root, {i: "static" for i in range(5)})
    assert build_v1.write_tokens_v1("f", root)[2] is True
    assert build_v1.write_tokens_v1("f", root)[2] is False


def test_move_token_formats():
    assert v1.move_token(None, False) == "n/a" and v1.move_token(None, True) == "?" and v1.move_token([], True) == "?"
    assert v1.move_token(["static"], True) == "static" and v1.move_token(["pan-left", "push-in"], True) == "pan-left+push-in"
    assert len(v1.schema_move_atoms()) == 16 and v1.schema_move_atoms()[0] == "static"
