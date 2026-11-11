"""guard and helper checks for an earlier change error analysis (CPU, synthetic inputs only)."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cgt.error_analysis import core as E

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/error_analysis_4b.py"


def _script():
    spec = importlib.util.spec_from_file_location("ea4b", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_load_preds_refuses_wrong_sha(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"line": 0, "id": "x", "logits": [1, 0, 0, 0]}\n')
    m = _script()
    with pytest.raises(SystemExit):
        m.load_preds(p, "0" * 64)
    good = hashlib.sha256(p.read_bytes()).hexdigest()
    assert m.load_preds(p, good)[0]["id"] == "x"


def test_auc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 60).astype(float)
    pos = rng.random(60) < 0.4
    p, n = x[pos], x[~pos]
    ref = ((p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n))
    assert E.auc(x, pos) == pytest.approx(ref)


def test_gold_rank_tie_after_gold():
    assert E.gold_rank([1, 2, 2, 0], 1) == 1   # a later tie does not push the gold down
    assert E.gold_rank([1, 2, 2, 0], 2) == 2


def test_sharpness_is_laplacian_variance():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (12, 16, 3)).astype(np.uint8)
    f = img.astype(float) / 255
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    lap = y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] - 4 * y[1:-1, 1:-1]
    assert E.image_stats(img)["sharpness"] == pytest.approx(lap.var())
