"""Token table build on synthetic Phase 1/2/3C tables: join, markers, fail-closed manifest, idempotence."""
import json

import numpy as np
import pandas as pd
import pytest

from cgt.tokens import build
from cgt.tokens.format import parse_row

ATTRS = build.P3C_ATTRS


@pytest.fixture
def root(tmp_path):
    n = 6
    (tmp_path / "data/processed").mkdir(parents=True)
    pal = [list(np.tile([10.0 + 15 * i, 0, 0], 5)) for i in range(n)]
    pd.DataFrame({"film_id": "f", "shot_id": range(n), "start_s": [60.0 * i for i in range(n)], "end_s": [60.0 * i + 5 for i in range(n)], "len_s": 5.0,
                  "kf_start": "a", "kf_mid": "b", "kf_end": "c", "palette_lab": pal, "key": "mid-key"}).to_parquet(tmp_path / "data/processed/f_shots.parquet")
    pd.DataFrame({"shot_id": range(n), "dof": "deep", "people": [0, 1, 2, 3, 4, 9], "sound": "score"}).to_parquet(tmp_path / "data/processed/f_p2.parquet")
    p3 = {"shot_id": list(range(n)), "p3c_run_key": "k1"}
    for a in ATTRS:
        p3[f"{a}_pred"] = "x"
        p3[f"{a}_p1"] = 0.5
        p3[f"{a}_margin"] = 0.2
        p3[f"{a}_consistent"] = [i != 2 for i in range(n)]
    pd.DataFrame(p3).to_parquet(tmp_path / "data/processed/f_p3c.parquet")
    run = tmp_path / "data/cache/p3c/runs/f_k1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"state": "complete", "finished": "t", "coverage": {k: n for k in ("kf_mid", "kf_start", "kf_end")}}))
    return tmp_path


def test_build_joins_marks_disagreement_and_formats(root):
    df = build.build_tokens("f", root)
    assert len(df) == 6 and df.shot_id.tolist() == list(range(6))
    assert df.n_kf_disagree.tolist() == [0, 0, 7, 0, 0, 0] and df.kf_all_agree.tolist() == [True, True, False, True, True, True]
    assert "*" in df.token_row[2] and "*" not in df.token_row[0]
    assert parse_row(df.token_row[5])["people"] == "4+"
    assert df.token_row.map(lambda t: list(parse_row(t))).map(tuple).nunique() == 1
    assert df.palette_id.between(0, 5).all() and df.tokens_version.eq("v0").all()


@pytest.mark.parametrize("manifest", [{"state": "running"}, {"state": "limited_complete", "finished": "t"},
                                      {"state": "complete", "finished": "t", "coverage": {"kf_mid": 5, "kf_start": 6, "kf_end": 6}}])
def test_incomplete_phase3c_run_fails_closed(root, manifest):
    (root / "data/cache/p3c/runs/f_k1/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(build.IncompletePhase3C):
        build.build_tokens("f", root)


def test_missing_manifest_and_mixed_keys_fail_closed(root):
    (root / "data/cache/p3c/runs/f_k1/manifest.json").unlink()
    with pytest.raises(build.IncompletePhase3C):
        build.build_tokens("f", root)


def test_write_is_idempotent_and_rebuilds_when_an_input_changes(root):
    out, txt, rebuilt = build.write_tokens("f", root)
    assert rebuilt and txt.read_text().splitlines()[0].startswith("00000 000:00.0 [size=")
    m = out.stat().st_mtime_ns
    assert build.write_tokens("f", root)[2] is False and out.stat().st_mtime_ns == m
    p2 = pd.read_parquet(root / "data/processed/f_p2.parquet")
    p2["sound"] = "dialogue"
    p2.to_parquet(root / "data/processed/f_p2.parquet")
    assert build.write_tokens("f", root)[2] is True
    assert pd.read_parquet(out).token_row[0].count("sound=dialogue") == 1


def test_cached_build_does_not_bypass_the_completion_check(root):
    build.write_tokens("f", root)
    assert build.write_tokens("f", root)[2] is False
    mf = root / "data/cache/p3c/runs/f_k1/manifest.json"
    for bad in ({"state": "running"}, {"state": "complete", "finished": "t", "coverage": {"kf_mid": 1, "kf_start": 6, "kf_end": 6}}):
        mf.write_text(json.dumps(bad))
        with pytest.raises(build.IncompletePhase3C):
            build.write_tokens("f", root)


@pytest.mark.parametrize("start_s,expected", [(3059.993, "051:00.0"), (59.96, "001:00.0"), (119.95, "002:00.0"), (0.0, "000:00.0"),
                                              (1334.467801, "022:14.5"), (4482.282282, "074:42.3"), (59.94, "000:59.9")])
def test_text_lines_never_prints_sixty_seconds(start_s, expected):
    df = pd.DataFrame({"shot_id": [7], "start_s": [start_s], "token_row": ["[x]"]})
    line = build.text_lines(df)[0]
    assert line == f"00007 {expected} [x]"
    assert int(line.split()[1].split(":")[1].split(".")[0]) < 60
