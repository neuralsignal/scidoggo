"""Tests for the RankConstraint estimator."""

import numpy as np
from sklearn.metrics import r2_score

from scidoggo import RankConstraint


def _low_rank_problem(rng, n_samples=200, n_features=8, n_targets=5, rank=2):
    """Build y = X @ W with W of exactly ``rank``, derived from the inputs."""
    X = rng.standard_normal((n_samples, n_features))
    G = rng.standard_normal((n_features, rank))
    H = rng.standard_normal((n_targets, rank))
    W = G @ H.T  # (n_features, n_targets), rank == rank
    Y = X @ W
    return X, Y, W


def test_predict_shape_single_output(rng):
    """Single-target predict returns shape (n_samples,) per sklearn contract."""
    X = rng.standard_normal((30, 8))
    w = rng.standard_normal(8)
    y = X @ w
    model = RankConstraint(rank=1).fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert model.coef_.shape == (X.shape[1],)


def test_predict_shape_multi_output(rng):
    """Multi-target predict returns shape (n_samples, n_targets)."""
    X, Y, W = _low_rank_problem(rng)
    model = RankConstraint(rank=2).fit(X, Y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0], Y.shape[1])
    assert model.coef_.shape == (Y.shape[1], X.shape[1])


def test_recovers_low_rank_solution(rng):
    """A rank-recoverable problem is fit well and the coef rank is constrained."""
    X, Y, W = _low_rank_problem(rng, rank=2)
    model = RankConstraint(rank=2, n_iter=3000).fit(X, Y)
    y_pred = model.predict(X)

    # the target is exactly representable, so the fit should be strong
    assert r2_score(Y, y_pred) > 0.9
    # the learned weight matrix respects the rank constraint
    assert np.linalg.matrix_rank(model.coef_, tol=1e-6) <= 2
    # G/H factors carry the requested rank
    assert model.G_.shape == (X.shape[1], 2)
    assert model.H_.shape == (Y.shape[1], 2)
