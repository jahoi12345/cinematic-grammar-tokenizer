"""Independent token contract checks; synthetic fixtures from the existing harness."""
import json
from pathlib import Path

import numpy as np
import pytest
import test_tokens_build as build_fixtures
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from cgt.tokens import build
from cgt.tokens.format import parse_row
from cgt.tokens.palette import fit_palette_ids


@pytest.fixture
def build_root(tmp_path):
    return build_fixtures.root.__wrapped__(tmp_path)


@pytest.fixture
def root(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    import test_tokens_app as app_fixtures
    return app_fixtures.root.__wrapped__(tmp_path, monkeypatch)


def test_reused_outputs_still_require_a_complete_manifest(build_root):
    build.write_tokens("f", build_root)
    (build_root / "data/cache/p3c/runs/f_k1/manifest.json").write_text(json.dumps({"state": "running"}))
    with pytest.raises(build.IncompletePhase3C):
        build.write_tokens("f", build_root)


@pytest.mark.parametrize("state", ["running", "limited_complete"])
def test_incomplete_state_refuses_even_with_complete_coverage(build_root, state):
    path = build_root / "data/cache/p3c/runs/f_k1/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["state"] = state
    path.write_text(json.dumps(manifest))
    with pytest.raises(build.IncompletePhase3C):
        build.build_tokens("f", build_root)


def test_parse_rejects_duplicate_slot_even_if_dictionary_order_matches():
    valid = "[size=CU | lens=wide | angle=low | framing=single | light=daylight/hard | comp=center | key=mid-key | DoF=deep | people=0 | sound=score | len=1.0s | palette=p00 | move=n/a]"
    with pytest.raises(ValueError):
        parse_row(valid[:-1] + " | size=WS]")


def test_palette_uses_registered_seed_and_initializations():
    palettes = np.random.default_rng(19).normal(size=(70, 15))
    with threadpool_limits(limits=1):
        km = KMeans(n_clusters=32, random_state=0, n_init=10).fit(palettes)
        order = np.argsort(km.cluster_centers_[:, ::3].mean(axis=1), kind="stable")
        rank = np.empty(32, dtype=int)
        rank[order] = np.arange(32)
        np.testing.assert_array_equal(fit_palette_ids(palettes.tolist()), rank[km.labels_])


def test_app_page_filter_and_ordered_reset(root):
    from streamlit.testing.v1 import AppTest

    app = str(Path(__file__).resolve().parents[1] / "app/browse/app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    at.number_input(key="page").set_value(2).run()
    assert not at.exception and "start 8.0 min" in at.markdown[0].value
    assert len(at.code) == 8
    at.number_input(key="page").set_value(3).run()
    assert len(at.code) == 4
    at.sidebar.radio(key="f_kf").set_value("disagree").run()
    assert not at.exception and at.metric[0].value == "5 / 20"
    assert len(at.code) == 5 and all("*" in c.value for c in at.code)
    at.sidebar.radio(key="f_kf").set_value("agree").run()
    assert at.metric[0].value == "15 / 20" and all("*" not in c.value for c in at.code)
    at.button(key="random").click().run()
    at.button(key="ordered").click().run()
    assert not at.exception and at.session_state["seed"] is None
