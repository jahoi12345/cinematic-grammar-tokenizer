"""CPU tests of the style feasibility pieces on tiny synthetic data (seconds)."""
import numpy as np
import pytest
import torch

from cgt.tokens import style


def test_ridge_residual_is_orthogonal_to_the_regressors_and_removes_a_linear_signal():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(500, 6))
    x = t @ rng.normal(size=(6, 12)) + 0.01 * rng.normal(size=(500, 12)) + 3.0
    w, b = style.ridge_fit(t, x, alpha=1e-6)
    r = style.residual(t, x, w, b)
    assert np.abs(r).max() < 0.1 and np.abs(t.T @ r).max() < 1.0  # linear part removed
    t2 = rng.normal(size=(5, 6))
    assert style.residual(t2, t2 @ w + b, w, b).shape == (5, 12)


def test_one_hot_and_usage_stats():
    oh = style.one_hot(np.array([0, 2, 1]), 3)
    assert oh.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    codes = np.stack([np.arange(100) % 4, np.zeros(100, dtype=int), np.arange(100) % 100], 1)
    u = style.usage_stats(codes, k=256)["levels"]
    assert u[0]["used"] == 4 and u[0]["perplexity"] == pytest.approx(4.0) and u[0]["max_share"] == 0.25
    assert u[1]["used"] == 1 and u[1]["perplexity"] == pytest.approx(1.0) and u[1]["max_share"] == 1.0
    assert u[2]["used"] == 100


def test_same_shot_agreement_and_chance():
    a = np.stack([np.arange(200) % 7] * 2, 1)
    same = {"kf_start": a, "kf_mid": a, "kf_end": a}
    r = style.same_shot_agreement(same)
    assert r["all_three"] == 1.0 and r["chance_all_three_shuffled"] < 0.2
    diff = {"kf_start": a, "kf_mid": (a + 1) % 7, "kf_end": a}
    assert style.same_shot_agreement(diff)["all_three"] == 0.0


def test_rqvae_trains_reconstructs_better_than_the_mean_and_is_deterministic():
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(8, 16)).astype(np.float32)
    x = (centers[rng.integers(0, 8, 600)] + 0.05 * rng.normal(size=(600, 16))).astype(np.float32)
    kw = {"epochs": 25, "batch": 64, "lr": 3e-3, "log": lambda m: None, "hidden": 32, "d_lat": 8, "levels": 2, "codes": 16}
    m, info = style.train_rqvae(x, **kw)
    assert info["epochs_run"] == 25 and info["loss"][-1] < info["loss"][0]
    codes = m.encode(x)
    assert codes.shape == (600, 2) and codes.max() < 16
    assert style.usage_stats(codes, k=16)["levels"][0]["used"] >= 4
    m.eval()
    rec, _, _ = m(torch.from_numpy(x))
    assert float(((rec.detach() - torch.from_numpy(x)) ** 2).mean()) < float(((x - x.mean(0)) ** 2).mean())
    m2, _ = style.train_rqvae(x, **kw)
    assert (m2.encode(x) == codes).all()  # seeded: identical codes


def test_time_bound_stops_training_early():
    x = np.random.default_rng(0).normal(size=(200, 8)).astype(np.float32)
    _, info = style.train_rqvae(x, epochs=1000, batch=50, max_seconds=0.0, log=lambda m: None, hidden=8, d_lat=4, levels=2, codes=8)
    assert info["epochs_run"] == 1
