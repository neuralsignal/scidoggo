"""Tests for Rank1PlusSparse (requires the [sparse] extra: cvxpy)."""

import numpy as np
import pytest

pytest.importorskip("cvxpy")

from scidoggo.linear_model import Rank1PlusSparse  # noqa: E402

pytestmark = pytest.mark.sparse


def _rank1_problem(seed=10, n=4, m=4, s=100):
    rng = np.random.RandomState(seed)
    X = rng.normal(0, 1, (s, n))
    w1 = rng.normal(0, 1, n)
    w1 /= np.sum(np.abs(w1))
    w2 = -rng.random(m)
    W = np.outer(w1, w2)
    Y = X @ W
    return X, Y


def test_fit_runs_and_predicts():
    X, Y = _rank1_problem()
    model = Rank1PlusSparse(max_iter=2000).fit(X, Y)
    assert model.coef_.shape == (Y.shape[1], X.shape[1])
    y_pred = model.predict(X)
    assert y_pred.shape == Y.shape


def test_regression_no_first_iteration_indexerror():
    """Regression guard: the fix for the 3 bugs means the first iteration

    converges without raising ``IndexError`` (the loss array is sized
    ``max_iter`` and ``loss[i + 1:]`` indexing must be safe) and the fitted
    factors are populated.
    """
    X, Y = _rank1_problem()
    model = Rank1PlusSparse(max_iter=2000).fit(X, Y)
    # convergence populates these attributes
    assert model.w1_ is not None
    assert model.w2_ is not None
    assert np.isfinite(model.loss_).all()
    # the recovered coefficients fit the noiseless problem well
    np.testing.assert_allclose(X @ model.coef_.T, Y, atol=1e-3)


def test_sparse_component_mask():
    """With an S_mask the sparse component respects the support constraint."""
    rng = np.random.RandomState(10)
    n, m, s = 4, 4, 100
    X = rng.normal(0, 1, (s, n))
    w1 = rng.normal(0, 1, n)
    w1 /= np.sum(np.abs(w1))
    w2 = -rng.random(m)
    W = np.outer(w1, w2)
    S = np.roll(np.eye(n, m), 2, axis=1)
    Y = X @ (W - S)
    model = Rank1PlusSparse(S_mask=S, S_sign=-1, w2_sign=-1, max_iter=2000)
    model.fit(X, Y)
    # entries outside the mask must be ~0 in the fitted sparse matrix
    S_fit = np.asarray(model.S_)
    np.testing.assert_allclose(S_fit[S == 0], 0.0, atol=1e-6)
