"""Demo "Shots like this" panel (docs/similar_shots_spec.md, section Demo): caption, public-mode guard, keyframe guard, palette mode note. Synthetic tmp roots only."""
import numpy as np
import pandas as pd
import pytest
from test_demo_guard import CLIP, NOTLD, OWNED, SECRET, _texts, core, make_root, run_app, snapshot


@pytest.fixture
def root(tmp_path):
    return make_root(tmp_path)


def test_panel_caption_in_public_mode_and_no_owned_film(monkeypatch, root):  # D1, D2
    at = run_app(monkeypatch, root, True, "Shot explorer")
    text = _texts(at)
    assert "Shots like this" in text
    assert core.CAPTION_SIMILAR.replace("`", "")[:100] in text.replace("`", "")
    low = text.lower()
    for s in SECRET:
        assert s not in low
    assert f"film={NOTLD}" in text  # the panel's token rows are NOTLD rows
    heads = [m.value for m in at.markdown if " · similarity " in m.value]
    assert len(heads) == 8 * 5  # 8 shots on the page, 5 similar each


def test_panel_local_rows_only_from_selected_film(monkeypatch, root):  # D2
    at = run_app(monkeypatch, root, False, "Shot explorer", ex_film=NOTLD)
    text = _texts(at)
    assert f"film={OWNED}" not in text and f"film={CLIP}" not in text
    at = run_app(monkeypatch, root, False, "Shot explorer", ex_film=OWNED)
    text = _texts(at)
    assert core.BANNER_LOCAL_ONLY in text and f"film={NOTLD}" not in text


def test_panel_keyframes_go_through_frame_guard(monkeypatch, root):  # D3
    p = root / f"data/processed/{NOTLD}_tokens_v1.parquet"
    df = pd.read_parquet(p)
    df["kf_mid"] = f"data/interim/{OWNED}/keyframes/00000_mid.jpg"  # exists, but outside NOTLD's keyframe directory
    df.to_parquet(p)
    at = run_app(monkeypatch, root, True, "Shot explorer")
    assert [c.value for c in at.caption].count("similar shot: mid frame not available") == 8 * 5


def test_panel_palette_mode_note(monkeypatch, root):
    note = "palette slot is an exact id match here"
    assert note in _texts(run_app(monkeypatch, root, True, "Shot explorer"))
    p = root / core.CENTROIDS_REL
    p.parent.mkdir(parents=True)
    np.savez(p, centroids=np.arange(32 * 15, dtype=float).reshape(32, 15))
    assert note not in _texts(run_app(monkeypatch, root, True, "Shot explorer"))


def test_similar_table_refuses_owned_in_public(root):
    df = pd.read_parquet(root / f"data/processed/{OWNED}_tokens_v1.parquet")
    with pytest.raises(PermissionError):
        core.similar_table(root, OWNED, True, df)
    before = snapshot(root)
    t = core.similar_table(root, OWNED, False, df)
    assert len(t) == len(df) * 5 and snapshot(root) == before
