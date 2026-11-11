"""Tidy an earlier change (docs/tidy_w3_2026-10-15.md) items 2 and 3. Synthetic stages and a synthetic tmp root; no model, no film, no GPU."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from cgt.pipeline import orchestrator as orch
from cgt.pipeline.orchestrator import Ctx, Stage, Status

REPO = Path(__file__).resolve().parents[1]


# ---- item 2: lock_plan evaluates locks_gpu only for stages that will run ------------------------------------------------------------------------

def _stage(name, calls):
    def uses_gpu(ctx):
        calls.append(name)
        return True
    return Stage(name, False, lambda ctx: Status(True, "k", "", []), lambda ctx: {}, uses_gpu=uses_gpu)


def test_lock_plan_skips_the_gpu_check_for_cache_hits():
    calls = []
    a, b, c = (_stage(n, calls) for n in ("a", "b", "c"))
    ctx = Ctx("pd", {})
    assert orch.lock_plan(ctx, [{"stage": a, "state": "hit"}, {"stage": b, "state": "hit"}]) == []
    assert calls == []  # all hits: locks_gpu (p2: a torch import) is never evaluated
    assert orch.lock_plan(ctx, [{"stage": a, "state": "hit"}, {"stage": b, "state": "miss"}, {"stage": c, "state": "hit"}]) == ["b", "c"]
    assert calls == ["b", "c"]  # the miss and the hit downstream of it (upstream rerun) are checked; the leading hit is not


# ---- item 3: the arc comparison report says when the video runs past the analysed part ------------------------------------------------------------

def _script(monkeypatch, root):
    spec = importlib.util.spec_from_file_location("visual_arc_compare_under_test", REPO / "scripts/visual_arc_compare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "ROOT", root)
    return m


def _video(p: Path, seconds: int):
    cv2 = pytest.importorskip("cv2")
    p.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 1.0, (16, 16))
    frame = np.zeros((16, 16, 3), np.uint8)
    for _ in range(seconds):
        w.write(frame)
    w.release()


@pytest.fixture
def root(tmp_path):
    sys.path.insert(0, str(REPO / "tests"))
    from test_demo_guard import _table
    (tmp_path / "configs").mkdir()
    films = [{"id": "pd_film", "public_domain": True, "path": "data/raw/pd.avi"}, {"id": "own_film", "public_domain": False, "path": "data/raw/own.avi", "cut_threshold": 27.0},
             {"id": "nofile", "public_domain": True, "path": "data/raw/missing.avi"}]
    (tmp_path / "configs/films.yaml").write_text(yaml.safe_dump({"films": films}))
    (tmp_path / "data/processed").mkdir(parents=True)
    for i, f in enumerate(("pd_film", "own_film", "nofile")):
        df = _table(f, seed=i)
        df["palette_global"] = df["shot_id"] % 5
        df.to_parquet(tmp_path / f"data/processed/{f}_tokens_v1.parquet")
    return tmp_path


def test_tail_caveat_only_when_video_runs_more_than_5_min_past_the_last_shot(root, monkeypatch):
    m = _script(monkeypatch, root)
    _video(root / "data/raw/own.avi", 30 * 60)
    _video(root / "data/raw/pd.avi", 9 * 60)
    reg = {e["id"]: e for e in yaml.safe_load((root / "configs/films.yaml").read_text())["films"]}
    assert m.tail_caveat(root, reg["own_film"], 6.0) == "the analysed part ends at 6.0 of 30.0 min (the rest is not analysed)"
    assert m.tail_caveat(root, reg["pd_film"], 6.0) == ""  # 3 min of tail: within the 5 min allowance
    assert m.tail_caveat(root, reg["pd_film"], 3.9) == "the analysed part ends at 3.9 of 9.0 min (the rest is not analysed)"
    assert m.tail_caveat(root, reg["nofile"], 1.0) == ""  # missing video: no caveat, no crash
    assert m.tail_caveat(root, {"id": "x"}, 1.0) == ""  # no path in the entry


def test_report_carries_the_tail_caveat(root, monkeypatch):
    m = _script(monkeypatch, root)
    _video(root / "data/raw/own.avi", 30 * 60)
    _video(root / "data/raw/pd.avi", 7 * 60)
    monkeypatch.setattr(sys, "argv", ["visual_arc_compare.py", "--films", "pd_film", "own_film"])
    m.main()
    md = (root / "data/interim/arc_compare/arc_compare.md").read_text()
    import pandas as pd
    own_min = pd.read_parquet(root / "data/processed/own_film_tokens_v1.parquet").end_s.max() / 60.0
    assert f"own_film: the analysed part ends at {own_min:.1f} of 30.0 min (the rest is not analysed)." in md
    assert "pd_film: the analysed part" not in md
