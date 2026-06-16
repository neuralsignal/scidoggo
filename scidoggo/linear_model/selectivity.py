"""Hybrid linear / nonlinear exponential-basis selectivity regression."""

from __future__ import annotations

from itertools import product
from numbers import Number

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data


class SelectivityModel(RegressorMixin, BaseEstimator):
    """Hybrid linear / nonlinear exponential-basis regression model.

    The model fits a linear regression to a subset of the features and uses a
    selective (exponential-basis) regression to fit nonlinear relationships
    between the remaining features and the target. A grid search over
    ``alphas`` and ``kappas`` selects the combination with the best
    :math:`R^2`.

    Parameters
    ----------
    kappas : array-like or None, default=None
        The range of kappa values to search over. When ``None`` (the default)
        a logarithmically spaced grid ``np.logspace(-2, 1, 19)`` is used and
        stored in ``kappas_`` during :term:`fit`.

    alphas : array-like or None, default=None
        The range of alpha values to search over. When ``None`` (the default)
        a logarithmically spaced grid ``np.logspace(-1, 1, 13)`` is used and
        stored in ``alphas_`` during :term:`fit`.

    linear_features : int, list, or None, default=None
        The indices of features to be treated as linear. If an integer is
        given, that many features starting from the first are linear. If a
        list is given, the listed indices are linear. If ``None``, all
        features are nonlinear.

    eps : float, default=1e-5
        A small positive constant added to kappa to avoid division by zero in
        the selective regression term.

    Attributes
    ----------
    alphas_ : ndarray
        The resolved grid of alpha values searched during :term:`fit`.

    kappas_ : ndarray
        The resolved grid of kappa values searched during :term:`fit`.

    lin_model_ : sklearn.linear_model.LinearRegression
        The linear regression model used to fit the linear features.

    w_linear_ : ndarray of shape (n_features,)
        The coefficients learned by ``lin_model_``, used as the optimisation
        warm start.

    linear_features_ : list of int
        The resolved list of linear feature indices.

    n_linear_ : int
        The number of linear features.

    coef_ : ndarray of shape (n_features,)
        The learned coefficients for the best ``(alpha, kappa)`` pair.

    alpha_ : float
        The optimal alpha value found during fitting.

    kappa_ : float
        The optimal kappa value found during fitting.

    r2s_ : ndarray
        The :math:`R^2` score for each ``(alpha, kappa)`` combination tried.

    params_ : list of [ndarray, float, float]
        The coefficients, alpha, and kappa for each combination tried.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    >>> model = SelectivityModel().fit(X, y)
    >>> y_pred = model.predict(X)
    """

    def __init__(
        self,
        kappas: ArrayLike | None = None,
        alphas: ArrayLike | None = None,
        linear_features: int | list[int] | None = None,
        eps: float = 1e-5,
    ) -> None:
        self.eps = eps
        self.kappas = kappas
        self.alphas = alphas
        self.linear_features = linear_features

    def _selective_regression(
        self,
        X_normalized: NDArray[np.float64],
        X_norm: NDArray[np.float64],
        w_normalized: NDArray[np.float64],
        w_norm: float,
        alpha: float,
        kappa: float,
    ) -> NDArray[np.float64]:
        """Evaluate the exponential-basis selective regression term."""
        kappa = kappa + np.sign(kappa) * self.eps
        angle = X_normalized @ w_normalized
        return X_norm**alpha * w_norm * (np.exp(kappa * angle) - 1) / kappa

    def _model(
        self,
        theta: NDArray[np.float64],
        X: NDArray[np.float64],
        X_norm: NDArray[np.float64],
        a: float,
        k: float,
    ) -> NDArray[np.float64]:
        """Compute the model prediction for coefficients ``theta``."""
        w = theta[self.n_linear_:]
        w_norm = np.linalg.norm(w)
        if np.isclose(w_norm, 0):
            w_normalized = w
        else:
            w_normalized = w / w_norm
        y_pred = self._selective_regression(
            X[:, self.n_linear_:], X_norm, w_normalized, w_norm, a, k
        ) + X[:, : self.n_linear_] @ theta[: self.n_linear_]
        return y_pred

    def _residuals(
        self,
        theta: NDArray[np.float64],
        X: NDArray[np.float64],
        X_norm: NDArray[np.float64],
        y: NDArray[np.float64],
        a: float,
        k: float,
        sample_weight: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute weighted residuals used by :func:`scipy.optimize.least_squares`."""
        y_pred = self._model(theta, X, X_norm, a, k)
        return np.sqrt(sample_weight) * (y - y_pred)

    def _check_and_assign_params(self) -> None:
        """Resolve ``linear_features`` and the alpha/kappa grids into fitted attrs."""
        if self.linear_features is None:
            self.linear_features_ = []
            self.n_linear_ = 0
        else:
            self.linear_features_ = (
                [self.linear_features]
                if isinstance(self.linear_features, Number)
                else list(self.linear_features)
            )
            self.n_linear_ = len(self.linear_features_)

        self.alphas_ = (
            np.logspace(-1, 1, 13) if self.alphas is None else np.asarray(self.alphas)
        )
        self.kappas_ = (
            np.logspace(-2, 1, 19) if self.kappas is None else np.asarray(self.kappas)
        )

    def _transformX(
        self, X: NDArray[np.float64], return_nonnormalized: bool = False
    ):
        """Split features into linear / nonlinear blocks and normalise the latter."""
        idcs = list(range(X.shape[1]))
        idcs = list(set(idcs) - set(self.linear_features_))
        Xlinear = X[:, self.linear_features_]
        X = X[:, idcs]
        if return_nonnormalized:
            return np.hstack([Xlinear, X])

        xnorm = np.linalg.norm(X, axis=-1, ord=2)
        X = X.astype(np.float64, copy=True)
        X[xnorm != 0] /= xnorm[xnorm != 0, None]
        X[~np.isfinite(X)] = 0

        X = np.hstack([Xlinear, X])
        return X, xnorm

    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> "SelectivityModel":
        """Fit the selectivity model to ``X`` and ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : SelectivityModel
            The fitted estimator.
        """
        X, y = validate_data(self, X, y, y_numeric=True)
        sample_weight = _check_sample_weight(sample_weight, X)

        self._check_and_assign_params()

        X4linear = self._transformX(X, return_nonnormalized=True)
        lin_model = LinearRegression(fit_intercept=False)
        lin_model.fit(X4linear, y, sample_weight=sample_weight)
        self.lin_model_ = lin_model
        self.w_linear_ = lin_model.coef_

        X, X_norm = self._transformX(X)

        params = []
        r2s = []
        for a, k in product(self.alphas_, self.kappas_):
            result = least_squares(
                self._residuals,
                self.w_linear_,
                args=(X, X_norm, y, a, k, sample_weight),
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            test_y = self._model(result.x, X, X_norm, a, k)
            r2s.append(r2_score(y, test_y, sample_weight=sample_weight))
            params.append([result.x, a, k])

        self.r2s_ = np.array(r2s)
        self.params_ = params

        self.coef_, self.alpha_, self.kappa_ = params[int(np.argmax(r2s))]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict target values for ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X, X_norm = self._transformX(X)
        return self._model(self.coef_, X, X_norm, self.alpha_, self.kappa_)
