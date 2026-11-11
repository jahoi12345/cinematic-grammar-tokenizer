"""Base-generation matched-budget experiment (docs/plans/base_gen_matched_budget.md): recipe parity, schedule position, reading
rule, image budget, and the Qwen2.5-VL leak probe (GPU, opt-in: CGT_GPU_TESTS=1, run under the GPU lock)."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgt.base_gen import config as C
from cgt.base_gen.reading import read
from cgt.p3b_vlm.images import budget_size
from cgt.stepc import config as SC


def _train_mod():
    pytest.importorskip("mlx_vlm")
    spec = importlib.util.spec_from_file_location("base_gen_train", ROOT / "scripts/base_gen_train.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_recipe_values_are_the_stepc_ones():
    for k in ("RANK", "ALPHA", "DROPOUT", "PEAK_LR", "END_LR", "WARMUP_FRAC", "ACCUM", "CLIP", "MAX_SEQ", "IMAGE_TOKENS",
              "ORDER_SEED", "MX_SEED", "RESUME_SEED_BASE", "STOP_AFTER", "STOP_WINDOW", "STOP_LOSS", "RESUME_EVERY", "MLX_ENV"):
        assert getattr(C, k) == getattr(SC, k), k
    assert (C.RANK, C.ALPHA, C.ACCUM, C.CLIP, C.RESUME_EVERY) == (16, 32.0, 16, 1.0, 25)
    assert C.TOTAL_UPDATES == -(-SC.N_TRAIN // SC.ACCUM) == 3195


def test_milestones_match_the_2b_run():
    m = _train_mod()
    assert m.milestones_for() == {799: 25, 1598: 50} == C.MILESTONES
    assert m.stop_update_for(25) == 799 and m.stop_update_for(50) == 1598
    with pytest.raises(ValueError):
        m.stop_update_for(100)


def test_image_budget_gives_at_most_144_qwen25_tokens():
    for w, h in ((1920, 1080), (1280, 544), (720, 576), (640, 480), (500, 2000), (4096, 1716)):
        nw, nh = budget_size(w, h, C.BUDGET_PX_3B, C.MULTIPLE_3B)
        assert nw % 28 == 0 and nh % 28 == 0 and nw * nh <= 112_896
        assert (nw // 28) * (nh // 28) <= 144     # one token per 28x28 cell (14-px patches, 2x2 merge)


def test_reading_rule():
    assert read(-2.0, -3.0, -0.5) == "NEWER_GENERATION_MATTERS"
    assert read(1.5, 0.2, 2.8) == "OLDER_3B_BETTER"
    assert read(-0.4, -1.3, 0.5) == "RECIPE_EXPLAINS"
    assert read(-1.2, -2.4, 0.1) == "INCONCLUSIVE"


@pytest.mark.ml
@pytest.mark.skipif(os.environ.get("CGT_GPU_TESTS") != "1", reason="GPU test: set CGT_GPU_TESTS=1 and hold /tmp/cgt_gpu.lock")
def test_qwen25vl_leak_probe_is_zero_in_our_trainer():
    """results/phase3b/benchmark.md note 2: mlx-vlm 0.7.4's padding mask leaked the answer for SmolVLM and Gemma 4. Our trainer calls
    the model with mask=None (causal_loss_fn); the probe on Qwen2.5-VL-3B must be exactly 0.0 that way (and is reported with the mask)."""
    m = _train_mod()
    from mlx_vlm.trainer.datasets import VisionDataset
    from mlx_vlm.trainer.utils import find_all_linear_names, get_peft_model
    from mlx_vlm.utils import load

    from cgt.p3b_vlm.loss import leakage_probe
    from cgt.p3b_vlm.model_path import pinned_model_path
    from cgt.stepc.data import train_stream

    model, processor = load(pinned_model_path(C.REPO_3B, C.REVISION_3B))
    model = get_peft_model(model, find_all_linear_names(model.language_model), rank=C.RANK, alpha=C.ALPHA, dropout=C.DROPOUT, verbose=False)
    cfg = model.config.__dict__
    for s in train_stream()[:3]:
        item = m.as_item(s)
        leak = leakage_probe(model, VisionDataset(m.ListView([item]), cfg, processor, None, True).process(item))
        print(s["id"], leak)
        assert leak["without_mask"] == 0.0
