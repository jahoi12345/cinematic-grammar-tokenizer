"""Shared synthetic fixtures for the Phase 3C runner/guard tests: fake engine, tmp_path repo root, synthetic public-domain film."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cgt.p3b_vlm import guards
from cgt.p3c_tokens import runner
from cgt.p3c_tokens.vocab import ATTRS, labels

REPO = Path(__file__).resolve().parents[1]


def _record(sid):
    return {"shot_id": sid, **{a: [0.0] * len(labels(a)) for a in ATTRS}}


class FakeEngine:
    adapter_sha = "a" * 64

    def __init__(self):
        self.passes, self.images = 0, []

    def image_state(self, path):
        self.images.append(str(path))
        return path

    def letter_logits(self, state, user, n):
        self.passes += 1
        return np.arange(n, dtype=float)


@pytest.fixture
def film_env(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(guards, "require_clean_tree", lambda paths: {"commit": "c" * 40})
    monkeypatch.setattr(runner, "shrunk_path", lambda path, root: path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/films.yaml").write_text("films:\n  - id: synthetic\n    public_domain: true\n")
    (tmp_path / "data/processed").mkdir(parents=True)

    def shots(*relatives):
        for r in relatives:
            (tmp_path / r).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), (128, 128, 128)).save(tmp_path / r)
        n = len(relatives)
        pd.DataFrame({"film_id": ["synthetic"] * n, "shot_id": list(range(n)),
                      **{kf: list(relatives) for kf in runner.KEYFRAMES}}).to_parquet(tmp_path / "data/processed/synthetic_shots.parquet", index=False)
    return tmp_path, shots


def _run(root, **kw):
    return runner.run_film("synthetic", "D", root / "adapter", engine=FakeEngine(), log=lambda m: None, **kw)


KF = "data/interim/synthetic/keyframes/{:05d}.jpg"
