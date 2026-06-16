"""Solver and helper functions for Tikhonov regression.

Based on the implementation by Stout and Kalivas, 2006, Journal of Chemometrics.
L2-regularized regression using a non-diagonal regularization matrix.

This can be done in two ways: by setting the original problem into "standard
space", such that regular ridge regression can be employed, or by solving the
equation in original space. As the number of features increases, rotating the
original problem should be faster.

Modified from Jeff Chiang's code <jeff.njchiang@gmail.com>.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.linear_model import Ridge

__all__ = [
    "analytic_tikhonov",
    "find_tikhonov_from_covariance",
    "to_standard_form",
    "to_general_form",
    "fit_learner",
]


def _qr(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the QR factorization of a matrix.

    Parameters
    ----------
    x : numpy.ndarray
        Input matrix of shape ``(n_samples, n_features)``.

    Returns
    -------
    qp : numpy.ndarray
        First block matrix of the QR factorization of ``x`` with shape
        ``(n_samples, n_features)``.
    qo : numpy.ndarray
        Second block matrix of the QR factorization of ``x`` with shape
        ``(n_samples, n_regularizers)``. In the degenerate case where there is
        no lower block, a 0-d array equal to ``1.0`` is returned.
    rp : numpy.ndarray
        Upper triangular matrix of the QR factorization of ``x`` with shape
        ``(n_features, n_features)``.
    """
    m, n = x.shape
    q, r = np.linalg.qr(x, mode="complete")
    rp = r[:n, :]
    qp = q[:, :n]
    qo = q[:, n:]
    # check for degenerate case
    if qo.shape[1] == 0:
        qo = np.array(1.0)
    return qp, qo, rp


def analytic_tikhonov(
    x: np.ndarray, y: np.ndarray, alpha: float, sigma: np.ndarray | None
) -> np.ndarray:
    """Solve the Tikhonov problem with the weight covariance as a prior.

    Parameters
    ----------
    x : numpy.ndarray
        Training data of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Target values of shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    alpha : float
        Regularization parameter.
    sigma : numpy.ndarray or None
        Covariance matrix of the prior with shape ``(n_features, n_features)``.
        If ``None``, the identity matrix is used.

    Returns
    -------
    numpy.ndarray
        Beta weight estimates of shape ``(n_features,)`` or
        ``(n_features, n_targets)``.
    """
    if sigma is None:
        sigma = np.eye(x.shape[1])
    return np.dot(
        np.linalg.pinv(np.dot(x.T, x) + np.linalg.pinv(sigma) * alpha),
        np.dot(x.T, y),
    )


def find_tikhonov_from_covariance(
    x: np.ndarray, cutoff: float, eps: float
) -> np.ndarray:
    """Use truncated SVD to find a Tikhonov matrix.

    Parameters
    ----------
    x : numpy.ndarray
        Feature-by-feature covariance matrix of shape
        ``(n_features, n_features)``. This is used to find a Tikhonov matrix
        ``L`` such that ``inv(x) = L.T @ L``.
    cutoff : float
        Cutoff value for singular value magnitude. If it is too low, the rank
        will suffer.
    eps : float
        Tolerance for singular values to be considered non-zero.

    Returns
    -------
    numpy.ndarray
        The Tikhonov matrix of shape ``(n_features, n_regularizers)``.

    Raises
    ------
    ValueError
        If ``x`` is not symmetric (and therefore not a covariance matrix).
    """
    if not np.allclose(x.T, x):
        raise ValueError(
            "Input matrix is not symmetric. Are you sure it is covariance?"
        )
    _, s, vt = np.linalg.svd(x)
    return np.dot(np.diag(1 / np.sqrt(s[s > cutoff])), vt[s > cutoff])


def _standardize_params(
    x: np.ndarray, L: np.ndarray
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Compute parameters for rotating the data to standard form.

    Parameters
    ----------
    x : numpy.ndarray
        Training data of shape ``(n_samples, n_features)``.
    L : numpy.ndarray
        Tikhonov matrix of shape ``(n_features, n_regularizers)``.

    Returns
    -------
    hq : numpy.ndarray
        First block matrix of the QR factorization of ``np.dot(x, ko)``.
    kp : numpy.ndarray
        First block matrix of the QR factorization of ``L.T``;
        ``kp @ pinv(rp.T)`` is ``inv(L)``.
    rp : numpy.ndarray
        Upper triangular matrix of the QR factorization of ``L.T``.
    ko : numpy.ndarray or None
        If ``L`` is not square, the matrix used to transform ``y`` to standard
        form; otherwise ``None``.
    ho : numpy.ndarray or None
        If ``L`` is not square, the matrix used to transform ``y`` to standard
        form; otherwise ``None``.
    to : numpy.ndarray or None
        If ``L`` is not square, the matrix used to transform ``y`` to standard
        form; otherwise ``None``.
    """
    kp, ko, rp = _qr(L.T)
    if ko is None:  # there is no lower part of matrix
        ho, hq, to = np.ones(1), np.ones(1), np.ones(1)
    else:
        ho, hq, to = _qr(np.dot(x, ko))
    if hq.shape == ():  # special case where L is square (saves time later)
        ko, to, ho = None, None, None
    return hq, kp, rp, ko, ho, to


def to_standard_form(
    x: np.ndarray, y: np.ndarray, L: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``x`` and ``y`` into standard form.

    This efficiently sets up the Tikhonov regression problem so that ordinary
    ridge regression can be applied.

    Parameters
    ----------
    x : numpy.ndarray
        Training data of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Target values of shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    L : numpy.ndarray
        Generally, ``L.T @ L`` is the inverse covariance matrix of the data.
        Shape is ``(n_features, n_regularizers)``.

    Returns
    -------
    x_new : numpy.ndarray
        The data rotated to standard form.
    y_new : numpy.ndarray
        The target array rotated to standard form.
    """
    # this is derived by doing a bit of algebra:
    # x_new = hq.T * x * kp * inv(rp).T
    hq, kp, rp, _, _, _ = _standardize_params(x, L)
    x_new = solve_triangular(rp, np.dot(kp.T, np.dot(x.T, hq))).T
    y_new = np.dot(hq.T, y)
    return x_new, y_new


def to_general_form(
    b: np.ndarray, x: np.ndarray, y: np.ndarray, L: np.ndarray
) -> np.ndarray:
    """Convert weights back into general form space.

    Parameters
    ----------
    b : numpy.ndarray
        Regression coefficients of shape ``(n_features,)`` or
        ``(n_features, n_targets)``.
    x : numpy.ndarray
        Training data of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Target values of shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    L : numpy.ndarray
        Generally, ``L.T @ L`` is the inverse covariance matrix of the data.
        Shape is ``(n_features, n_regularizers)``.

    Returns
    -------
    numpy.ndarray
        Ridge coefficients rotated back to original space of shape
        ``(n_features,)`` or ``(n_features, n_targets)``.
    """
    hq, kp, rp, ko, ho, to = _standardize_params(x, L)

    if ko is to is ho is None:
        L_inv = np.dot(kp, np.linalg.pinv(rp.T))
        return np.dot(L_inv, b)
    L_inv = np.dot(kp, np.linalg.pinv(rp.T))
    kth = np.dot(ko, np.dot(np.linalg.pinv(to), ho.T))
    resid = y - np.dot(x, np.dot(L_inv, b))
    return np.dot(L_inv, b) + np.dot(kth, resid)


def fit_learner(
    x: np.ndarray, y: np.ndarray, L: np.ndarray, ridge: Ridge | None
) -> Ridge:
    """Return a trained ridge model fit optimally in standard form.

    The returned model behaves exactly like :class:`~sklearn.linear_model.Ridge`
    but its coefficients are computed in standard form and rotated back.

    Parameters
    ----------
    x : numpy.ndarray
        Training data of shape ``(n_samples, n_features)``.
    y : numpy.ndarray
        Target values of shape ``(n_samples,)`` or ``(n_samples, n_targets)``.
    L : numpy.ndarray
        Generally, ``L.T @ L`` is the inverse covariance matrix of the data.
        Shape is ``(n_features, n_regularizers)``.
    ridge : sklearn.linear_model.Ridge or None
        A ridge object used to fit the transformed data. If ``None``, a new
        ``Ridge(fit_intercept=False)`` instance is created.

    Returns
    -------
    sklearn.linear_model.Ridge
        A trained ridge object with optimized coefficients.
    """
    if ridge is None:
        ridge = Ridge(fit_intercept=False)
    x_new, y_new = to_standard_form(x, y, L)
    ta_est_standard = ridge.fit(x_new, y_new).coef_
    ta_est = to_general_form(ta_est_standard.T, x, y, L)
    ridge.coef_ = ta_est.T
    return ridge
