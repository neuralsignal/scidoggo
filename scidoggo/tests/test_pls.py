"""Tests for the custom PLS functionality (demean + extra methods)."""

import numpy as np
import pytest

from scidoggo import CCA, PLSRegression


@pytest.fixture
def xy(rng):
    """Multi-target data for PLS."""
    X = rng.standard_normal((50, 6))
    Y = rng.standard_normal((50, 3))
    return X, Y


def test_demean_default_centers(xy):
    """With demean=True (default) the intercept absorbs the means."""
    X, Y = xy
    pls = PLSRegression(n_components=2).fit(X, Y)
    # default centering means a non-zero intercept for non-centered targets
    assert not np.allclose(pls.intercept_, 0.0)


def test_demean_false_zero_intercept(xy):
    """With demean=False no centering happens, so the intercept is zero."""
    X, Y = xy
    pls = PLSRegression(n_components=2, demean=False).fit(X, Y)
    assert np.allclose(pls.intercept_, 0.0)


def test_demean_changes_fit(xy):
    """demean=False produces a different coefficient matrix than the default."""
    X, Y = xy
    coef_demean = PLSRegression(n_components=2).fit(X, Y).coef_
    coef_raw = PLSRegression(n_components=2, demean=False).fit(X, Y).coef_
    assert coef_demean.shape == coef_raw.shape == (Y.shape[1], X.shape[1])
    assert not np.allclose(coef_demean, coef_raw)


def test_kth_coef_shapes_and_full(xy):
    """kth_coef has the coef_ shape; k=None / k=n_components reproduce coef_."""
    X, Y = xy
    n_comp = 3
    pls = PLSRegression(n_components=n_comp).fit(X, Y)
    expected_shape = (Y.shape[1], X.shape[1])

    assert pls.kth_coef(2).shape == expected_shape
    # k=None returns the full fitted coefficients
    np.testing.assert_allclose(pls.kth_coef(None), pls.coef_)
    # using all components reconstructs the full coefficients
    np.testing.assert_allclose(pls.kth_coef(n_comp), pls.coef_)


def test_kth_rotations_shapes(xy):
    """kth_rotations slices the rotation matrices to k columns."""
    X, Y = xy
    pls = PLSRegression(n_components=3).fit(X, Y)
    x_rot, y_rot = pls.kth_rotations(2)
    assert x_rot.shape == (X.shape[1], 2)
    assert y_rot.shape == (Y.shape[1], 2)

    # k=None returns the full fitted rotations
    x_full, y_full = pls.kth_rotations(None)
    np.testing.assert_allclose(x_full, pls.x_rotations_)
    np.testing.assert_allclose(y_full, pls.y_rotations_)


def test_decompose_coef_orthogonal_and_reconstructs(xy):
    """decompose_coef yields orthonormal X rotations that reconstruct kth_coef."""
    X, Y = xy
    k = 3
    pls = PLSRegression(n_components=k, scale=False).fit(X, Y)
    x_rot_orth, y_load_orth = pls.decompose_coef(k)

    assert x_rot_orth.shape == (X.shape[1], k)
    assert y_load_orth.shape == (Y.shape[1], k)

    # the X rotations are orthonormal
    np.testing.assert_allclose(x_rot_orth.T @ x_rot_orth, np.eye(k), atol=1e-8)

    # round trip: (x_rot_orth @ y_load_orth.T) * y_std == kth_coef(k).T
    recon = (x_rot_orth @ y_load_orth.T) * pls._y_std
    np.testing.assert_allclose(recon, pls.kth_coef(k).T, atol=1e-8)


def test_cca_fit_transform(xy):
    """CCA basic fit then transform returns the right scores shape."""
    X, Y = xy
    cca = CCA(n_components=1).fit(X, Y)
    x_scores = cca.transform(X)
    assert x_scores.shape == (X.shape[0], 1)

    x_scores2, y_scores = cca.transform(X, Y)
    assert x_scores2.shape == (X.shape[0], 1)
    assert y_scores.shape == (X.shape[0], 1)
