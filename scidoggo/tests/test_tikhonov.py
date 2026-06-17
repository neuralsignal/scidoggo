"""Tests for the Tikhonov estimator and its solver helpers."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from scidoggo import Tikhonov
from scidoggo.linear_model.tikhonov_solver import (
    analytic_tikhonov,
    find_tikhonov_from_covariance,
    to_general_form,
    to_standard_form,
)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 2.5, 10.0])
def test_tikhonov_no_L_equals_ridge(regression_data, alpha):
    """Tikhonov(L=None) is identical to Ridge(fit_intercept=False)."""
    X, y = regression_data
    tik = Tikhonov(alpha=alpha, L=None).fit(X, y)
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)

    np.testing.assert_allclose(tik.coef_, ridge.coef_)
    np.testing.assert_allclose(tik.predict(X), ridge.predict(X))


def test_tikhonov_with_L_fits(regression_data, rng):
    """With a small alpha and an SPD L, the model fits with low error."""
    X, y = regression_data
    n_features = X.shape[1]
    L = rng.random((n_features, n_features))
    L = (L.T @ L + np.eye(n_features)) / 2  # symmetric positive definite

    model = Tikhonov(alpha=1e-3, L=L).fit(X, y)
    y_pred = model.predict(X)
    assert mean_squared_error(y, y_pred) < 1.0


def test_analytic_tikhonov_shape():
    """analytic_tikhonov returns coefficients of shape (n_features,)."""
    X, y = make_regression(n_samples=60, n_features=6, random_state=0)
    beta = analytic_tikhonov(X, y, alpha=1.0, sigma=None)
    assert beta.shape == (X.shape[1],)


def test_find_tikhonov_from_covariance_roundtrip():
    """find_tikhonov_from_covariance recovers L with L.T @ L ~= inv(cov)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    cov = np.cov(X.T)
    L = find_tikhonov_from_covariance(cov, cutoff=1e-10, eps=1e-12)
    # L.T @ L should approximate the inverse covariance
    np.testing.assert_allclose(L.T @ L, np.linalg.inv(cov), atol=1e-6)


def test_find_tikhonov_from_covariance_rejects_asymmetric():
    """A non-symmetric input is rejected."""
    bad = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        find_tikhonov_from_covariance(bad, cutoff=1e-8, eps=1e-12)


def test_standard_general_form_roundtrip():
    """At negligible regularisation the standard<->general round trip is OLS.

    Rotating ``X``/``y`` to standard form, fitting an (essentially
    unregularised) ridge there and rotating the coefficients back must recover
    the generating weights, confirming ``to_standard_form`` and
    ``to_general_form`` are inverses of one another.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((80, 6))
    w_true = rng.standard_normal(6)
    y = X @ w_true  # noiseless so OLS recovers w_true exactly
    cov = np.cov(X.T)
    L = find_tikhonov_from_covariance(cov, cutoff=1e-10, eps=1e-12)

    x_std, y_std = to_standard_form(X, y, L)
    coef_std = Ridge(alpha=1e-10, fit_intercept=False).fit(x_std, y_std).coef_
    coef_gen = to_general_form(coef_std.T, X, y, L)

    np.testing.assert_allclose(np.ravel(coef_gen), w_true, atol=1e-6)
