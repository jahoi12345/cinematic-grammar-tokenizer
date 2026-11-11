"""Headless test of the browse app with streamlit.testing on a synthetic token table (no real frames, no model)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from cgt.tokens.format import format_row

APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app/browse/app.py")


@pytest.fixture
def root(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        r = {"film_id": "synth", "shot_id": i, "start_s": 60.0 * i, "start_min": float(i), "len_s": 1.0 + i, "key": ["low-key", "mid-key"][i % 2],
             "dof": "deep", "people": i % 6, "sound": "score", "palette_id": i % 5, "move": None,
             "size": ["CU", "MS"][i % 2], "lens": "medium", "angle": "low", "framing": "single", "light_type": "daylight", "light_cond": "hard",
             "composition": "center"}
        for a in ["size", "lens", "angle", "framing", "light_type", "light_cond", "composition"]:
            r[f"{a}_p1"], r[f"{a}_margin"], r[f"{a}_consistent"] = 0.6, 0.3, bool(i % 4)
        r["n_kf_disagree"] = 7 if i % 4 == 0 else 0
        r["kf_all_agree"] = i % 4 != 0
        r["token_row"] = format_row(r)
        for k in ("kf_start", "kf_mid", "kf_end"):
            rel = f"data/interim/synth/keyframes/{i:05d}_{k[3:]}.jpg"
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).save(tmp_path / rel)
            r[k] = rel
        rows.append(r)
    (tmp_path / "data/processed").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "data/processed/synth_tokens.parquet", index=False)
    monkeypatch.setenv("CGT_BROWSE_ROOT", str(tmp_path))
    return tmp_path


def test_app_renders_banner_rows_and_paging(root):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("UNMEASURED" in e.value for e in at.error)
    assert len(at.code) == 8 and at.code[0].value.startswith("[size=")
    assert at.metric[0].value == "20 / 20"


def test_app_filters_toggle_and_random_sample(root):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.multiselect(key="f_size").set_value(["CU"]).run()
    assert at.metric[0].value == "10 / 20"
    at.sidebar.radio(key="f_kf").set_value("disagree").run()
    assert at.metric[0].value == "5 / 20"  # shots 0,4,8,12,16 disagree; all of them have size CU
    at.sidebar.multiselect(key="f_people").set_value(["4+"]).run()
    assert not at.exception
    at.sidebar.radio(key="f_kf").set_value("any").run()
    at.sidebar.multiselect(key="f_size").set_value([]).run()
    at.sidebar.multiselect(key="f_people").set_value([]).run()
    at.button(key="random").click().run()
    assert not at.exception and len(at.code) == 8


def test_app_offers_v1_table_with_version_switch(root):
    v0 = pd.read_parquet(root / "data/processed/synth_tokens.parquet")
    v1 = v0.assign(len_tok="2-4s", palette_tok="g03", tokens_version="v1")
    v1.iloc[:10].to_parquet(root / "data/processed/synth_tokens_v1.parquet", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and at.sidebar.radio(key="version").value == "v1"
    assert at.metric[0].value == "10 / 10" and any("soft vote" in i.value for i in at.info)
    at.sidebar.multiselect(key="f_len_tok").set_value(["2-4s"]).run()
    assert at.metric[0].value == "10 / 10"
    at.sidebar.radio(key="version").set_value("v0").run()
    assert at.metric[0].value == "20 / 20"


def test_app_marks_an_owned_film_as_local_only(root):
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs/films.yaml").write_text("films:\n  - id: synth\n    public_domain: false\n")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and any("OWNED FILM: LOCAL ONLY" in e.value for e in at.error)
    (root / "configs/films.yaml").write_text("films:\n  - id: synth\n    public_domain: true\n")
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not any("OWNED FILM" in e.value for e in at.error)


def test_app_filters_by_predicted_movement_atom(root):
    v0 = pd.read_parquet(root / "data/processed/synth_tokens.parquet")
    atoms = [["static"] if i % 2 else ["pan-left", "push-in"] for i in range(len(v0))]
    v0.assign(len_tok="2-4s", palette_tok="g03", tokens_version="v1", move_present=True, move_atoms=atoms).to_parquet(root / "data/processed/synth_tokens_v1.parquet", index=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception and at.metric[0].value == "20 / 20" and any("PREDICTION" in i.value for i in at.info)
    at.sidebar.multiselect(key="f_move").set_value(["push-in"]).run()
    assert at.metric[0].value == "10 / 20"
