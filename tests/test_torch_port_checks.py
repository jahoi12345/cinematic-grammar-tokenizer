"""an earlier review guard tests for cgt.p3b_vlm.torch_port: the parity aggregation (compare, log_softmax4) and lora_alpha at the real rank 16.
Synthetic numbers only (no model weights, no ShotQA)."""
import numpy as np
import pytest

from cgt.p3b_vlm import torch_port as tp


def test_log_softmax4_normalizes():
    x = np.array([[24.0, 20.875, 27.125, 22.75], [1.0, 1.0, 1.0, 1.0]])
    lp = tp.log_softmax4(x)
    np.testing.assert_allclose(np.exp(lp).sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(lp[1], np.log(0.25), rtol=1e-12)


def test_compare_hand_values():
    ref = [{"id": "a", "pred": "A", "logits": [2.0, 0.0, 0.0, 0.0]}, {"id": "b", "pred": "B", "logits": [0.0, 1.0, 0.0, 0.0]}]
    # second row: logits shifted by a constant (log-probs identical) but the prediction differs
    tor = [{"id": "a", "pred": "A", "logits": [2.0, 0.0, 0.0, 0.0]}, {"id": "b", "pred": "C", "logits": [5.0, 6.0, 5.0, 5.0]}]
    c = tp.compare(ref, tor)
    assert c["n"] == 2 and c["argmax_agree"] == 1 and c["argmax_agreement"] == 0.5
    assert c["logprob_absdiff_mean"] == pytest.approx(0.0, abs=1e-12)  # log-softmax is shift invariant; raw logits differ by 5
    assert c["logit_absdiff_mean"] == pytest.approx(2.5)
    assert c["mlx_top2_margin_of_disagreements"] == [1.0]
    with pytest.raises(ValueError):
        tp.compare(ref, tor[::-1])


def test_compare_logprob_diff_value():
    ref = [{"id": "a", "pred": "A", "logits": [1.0, 0.0, 0.0, 0.0]}]
    tor = [{"id": "a", "pred": "A", "logits": [0.0, 0.0, 0.0, 0.0]}]
    exp = np.abs(tp.log_softmax4([[1.0, 0, 0, 0]]) - np.log(0.25)).mean()
    assert tp.compare(ref, tor)["logprob_absdiff_mean"] == pytest.approx(float(exp), rel=1e-12)
    assert exp == pytest.approx(np.mean([abs(1 - np.log(np.e + 3) + np.log(4))] + 3 * [abs(-np.log(np.e + 3) + np.log(4))]))


def test_alpha_at_real_rank():
    cfg = tp.peft_config(["model.language_model.layers.0.self_attn.q_proj"], 16, 2.0)
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 32.0 and not cfg["use_rslora"] and not cfg["fan_in_fan_out"]
