"""Ordinal loss contract on synthetic logits; no training or weights needed."""
import pytest
import torch

from cgt.p3_static_attrs.heads import coral_loss


def test_ordinal_loss_increases_with_distance_and_approaches_zero():
    # Correct class 2: cumulative targets 1,1,0,0. Finite BCE has no exact zero.
    y = torch.tensor([2])
    losses = []
    for prediction in (2, 3, 4):
        logits = torch.tensor([[40. if k < prediction else -40. for k in range(4)]])
        losses.append(float(coral_loss(logits, y)))
    assert losses[0] == pytest.approx(0., abs=1e-15)
    assert losses[0] < losses[1] < losses[2]
    assert losses[1] == pytest.approx(10.)
    assert losses[2] == pytest.approx(20.)
