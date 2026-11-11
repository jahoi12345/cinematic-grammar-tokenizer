"""Written scoring API on synthetic processors/logits; CPU only, no models."""
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip('mlx.core')
from cgt.p3b_vlm import score


@pytest.fixture(autouse=True)
def cpu_only():
    mx.set_default_device(mx.cpu)


class Processor:
    def __init__(self, ids, leading_space=False, mismatch=False):
        self.ids = ids
        self.leading_space = leading_space
        self.mismatch = mismatch
        self.tokenizer = self

    def apply_chat_template(self, messages, add_generation_prompt):
        assert messages[0]['content'][0]['type'] == 'image'
        assert messages[0]['content'][1]['text'] == 'Q?'
        if add_generation_prompt:
            return 'prompt'
        return messages[1]['content'][0]['text']

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        if text == 'prompt':
            return [10, 20, 30]
        return [10, 20, 99 if self.mismatch else 30, self.ids[text], 900]

    def decode(self, ids):
        letter = next(c for c, i in self.ids.items() if i == ids[0])
        return (' ' if self.leading_space else '') + letter


def test_answer_tokens_come_from_each_models_template():
    for leading, ids in ((True, dict(zip('ABCD', [330, 389, 340, 422], strict=True))),
                         (False, dict(zip('ABCD', [32, 33, 34, 35], strict=True)))):
        assert score.answer_token_ids(Processor(ids, leading)) == ids
    custom = dict(zip('ABCD', [82, 19, 51, 73], strict=True))
    assert score.answer_token_ids(Processor(custom)) == custom


def test_template_prefix_mismatch_is_rejected():
    with pytest.raises(ValueError, match='extend'):
        score.answer_token_ids(Processor(dict(zip('ABCD', range(32, 36), strict=True)), mismatch=True))


class Model:
    def __init__(self):
        self.calls = []

    def __call__(self, ids, pixels, mask, **kwargs):
        self.calls.append((np.array(ids), mask, kwargs))
        logits = np.full((1, ids.shape[1], 40), -100, dtype=np.float32)
        logits[0, -1, 32:36] = [.5, 2, 2, -4]
        logits[0, -1, 9] = 10000  # unrestricted argmax is outside A-D
        return SimpleNamespace(logits=mx.array(logits))


def processed():
    return {'input_ids': np.array([[7, 8, 33, 39]]),
            'completion_mask': np.array([[0, 0, 1, 1]]),
            'attention_mask': np.ones((1, 4)), 'pixel_values': np.zeros((1, 2, 2, 3)),
            'image_grid_thw': np.array([[1, 1, 1]])}


def test_letter_logits_use_last_prompt_position_without_answer_tokens():
    model = Model()
    four, gold = score.letter_logits(model, processed(), dict(zip('ABCD', range(32, 36), strict=True)))
    np.testing.assert_array_equal(four, [.5, 2, 2, -4])
    assert four.shape == (4,) and gold == 33
    ids, mask, kwargs = model.calls[0]
    np.testing.assert_array_equal(ids, [[7, 8]])
    assert mask is None
    np.testing.assert_array_equal(np.array(kwargs['image_grid_thw']), [[1, 1, 1]])
    item = processed()
    item['completion_mask'][:] = 0
    with pytest.raises(IndexError):
        score.letter_logits(model, item, dict(zip('ABCD', range(32, 36), strict=True)))


def sample():
    return {'id': 'synthetic#0', 'attribute': 'light_cond', 'answer': 'B',
            'images': ['synthetic.png'], 'messages': [],
            'options': {'A': 'Backlight', 'B': 'Soft light, Side light', 'C': 'Hard light', 'D': 'Top light'}}


def test_evaluation_restricts_to_letters_breaks_ties_and_preserves_gold():
    ds = SimpleNamespace(process=lambda _: processed())
    result = score.evaluate_samples(Model(), ds, [sample()], dict(zip('ABCD', range(32, 36), strict=True)))
    assert result == [{'id': 'synthetic#0', 'attribute': 'light_cond', 'gold': 'B', 'pred': 'B',
                       'compound': True, 'logits': [.5, 2., 2., -4.]}]


def test_tokenized_gold_must_match_template_answer_token():
    ds = SimpleNamespace(process=lambda _: processed())
    wrong = sample()
    wrong['answer'] = 'C'
    with pytest.raises(ValueError) as error:
        score.evaluate_samples(Model(), ds, [wrong], dict(zip('ABCD', range(32, 36), strict=True)))
    assert wrong['id'] in str(error.value)
    assert 'C' not in str(error.value)
    assert '34' not in str(error.value)


def test_summary_uses_unweighted_attribute_mean_and_answer_strata():
    rows = [{'attribute': 'size', 'gold': 'A', 'pred': 'A', 'compound': False},
            {'attribute': 'size', 'gold': 'B', 'pred': 'A', 'compound': True},
            {'attribute': 'size', 'gold': 'C', 'pred': 'A', 'compound': True},
            {'attribute': 'lens', 'gold': 'D', 'pred': 'D', 'compound': False}]
    summary = score.summarize(rows)
    assert summary['n'] == 4
    assert summary['mean_acc'] == pytest.approx((1/3 + 1)/2)
    assert summary['per_attribute']['size'] == {'n': 3, 'correct': 1, 'acc': 1/3,
                                              'n_compound': 2, 'acc_single': 1., 'acc_compound': 0.}
    assert summary['per_attribute']['lens']['acc_compound'] is None
