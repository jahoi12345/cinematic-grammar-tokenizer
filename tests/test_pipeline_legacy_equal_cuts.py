# ruff: noqa: F811
"""Equal cut times cannot establish the origin of old images or color tables."""
import json

import pandas as pd
from _p3c_fixtures import KF, film_env  # noqa: F401

from cgt.p1_shots_color import detection, pipeline
from cgt.pipeline import stages
from cgt.pipeline.orchestrator import Ctx


def test_unknown_legacy_source_with_equal_cut_times_rebuilds_keyframes(film_env, monkeypatch):
    root, shots = film_env
    shots(KF.format(0), KF.format(1))
    video = root / "synthetic.mp4"
    video.write_bytes(b"replacement source: different pictures, same scene boundaries")
    cuts = root / "data/interim/synthetic/cuts_t27.json"
    boundaries = [[0., 1.], [1., 2.]]
    cuts.write_text(json.dumps(boundaries))
    monkeypatch.setattr(detection, "detect_shots", lambda *args: boundaries)
    calls = []

    def rebuild(film, threshold):
        assert not cuts.exists(), "unknown cut cache must be invalidated before rerunning"
        calls.append((film, threshold))
        cuts.write_text(json.dumps(boundaries))
        return pd.read_parquet(root / "data/processed/synthetic_shots.parquet")

    monkeypatch.setattr(pipeline, "run_film", rebuild)
    ctx = Ctx("synthetic", {"public_domain": True, "path": "synthetic.mp4"}, root=root)
    stages.p1_run(ctx)
    assert calls == [("synthetic", 27.)], "equal boundaries cannot certify that the old keyframes belong to this video"
