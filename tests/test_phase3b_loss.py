"""CPU-only synthetic checks of the Phase3B written loss/probe contract."""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from cgt.p3b_vlm import guards
from cgt.p3b_vlm.loss import causal_loss_fn, leakage_probe


@pytest.fixture(autouse=True)
def cpu_only():
    # an earlier change owns the GPU after GPU-CHECKS-DONE. Never load real models here.
    mx.set_default_device(mx.cpu)
    assert mx.default_device() == mx.cpu


class ToyModel:
    def __init__(self, always_leaky=False, fixed=None):
        self.always_leaky = always_leaky
        self.fixed = fixed
        self.calls = []
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, ids, pixels, mask, **kwargs):
        if self.fixed is not None:
            logits = mx.array(self.fixed)
        else:
            context = mx.cumsum(ids.astype(mx.float32), axis=1)
            if mask is not None or self.always_leaky:
                context = mx.broadcast_to(mx.sum(ids, axis=1, keepdims=True), ids.shape)
            logits = context[:, :, None] * mx.arange(8)[None, None, :] / 50
        self.calls.append((mask, np.array(logits)))
        return SimpleNamespace(logits=logits)


def batch():
    return {'input_ids': mx.array([[1, 2, 3, 4, 5, 0]]),
            'attention_mask': mx.array([[1, 1, 1, 1, 1, 0]]),
            'completion_mask': mx.array([[0, 0, 0, 1, 1, 0]]),
            'pixel_values': mx.zeros((1, 2, 2, 3))}


def manual_ce(logits, targets):
    z = np.asarray(logits, dtype=np.float64)
    maximum = z.max(axis=-1)
    return maximum + np.log(np.exp(z - maximum[:, None]).sum(axis=-1)) - z[np.arange(len(targets)), targets]


def test_future_token_cannot_change_earlier_logits_or_loss_contributions():
    model = ToyModel()
    original = batch()
    original['completion_mask'] = mx.array([[0, 0, 0, 1, 0, 0]])
    changed = {**original, 'input_ids': mx.array([[1, 2, 3, 4, 7, 0]])}
    first = float(causal_loss_fn(model, original, train_on_completions=True))
    second = float(causal_loss_fn(model, changed, train_on_completions=True))
    assert all(mask is None for mask, _ in model.calls)
    np.testing.assert_array_equal(model.calls[0][1][:, :4], model.calls[1][1][:, :4])
    assert first == second


def test_completion_only_loss_matches_independent_ce_and_ignores_prompt():
    logits = np.arange(40, dtype=np.float32).reshape(1, 5, 8) / 17
    logits[0, 2, 4] = 3
    logits[0, 3, 5] = -2
    expected = manual_ce(logits[0, [2, 3]], [4, 5]).mean()
    actual = float(causal_loss_fn(ToyModel(fixed=logits), batch(), train_on_completions=True))
    assert actual == pytest.approx(expected, rel=1e-6)
    changed = logits.copy()
    changed[0, [0, 1, 4]] *= -100
    assert float(causal_loss_fn(ToyModel(fixed=changed), batch(), train_on_completions=True)) == actual


def test_missing_completion_mask_raises():
    item = batch()
    del item['completion_mask']
    with pytest.raises(ValueError, match='completion_mask'):
        causal_loss_fn(ToyModel(), item, train_on_completions=True)


def test_no_supervised_tokens_returns_finite_zero():
    item = batch()
    item['completion_mask'] = mx.zeros((1, 6), dtype=mx.int32)
    value = float(causal_loss_fn(ToyModel(), item, train_on_completions=True))
    assert np.isfinite(value) and value == 0


def test_probe_detects_mask_leak_and_safe_call_has_exact_zero():
    result = leakage_probe(ToyModel(), batch())
    assert set(result) == {'with_mask', 'without_mask'}
    assert result['with_mask'] > 0
    assert result['without_mask'] == 0


def test_deliberately_leaky_safe_call_fails_required_probe_gate():
    result = leakage_probe(ToyModel(always_leaky=True), batch())
    assert result['without_mask'] > 0
    with pytest.raises(guards.LeakProbeFailed) as error:
        guards.require_zero_leak(result)
    assert error.value.exit_code == 4


def test_probe_disables_dropout_and_restores_prior_training_mode():
    class DropoutSensitiveModel(ToyModel):
        def __call__(self, *args, **kwargs):
            assert self.training is False, 'probe must disable stochastic training behavior'
            return super().__call__(*args, **kwargs)
    for training in (False, True):
        model = DropoutSensitiveModel()
        model.training = training
        assert leakage_probe(model, batch())['without_mask'] == 0
        assert model.training is training
