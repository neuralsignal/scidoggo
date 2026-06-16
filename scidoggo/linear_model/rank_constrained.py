"""Rank-constrained least-squares linear regression."""

from __future__ import annotations

import numpy as np
from numpy import linalg as LA
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted, validate_data


class RankConstraint(RegressorMixin, MultiOutputMixin, BaseEstimator):
    """Least-squares linear regression with a rank constraint on the weights.

    The weight matrix ``W`` is factorised as ``W = G @ H.T`` where ``G`` and
    ``H`` each have ``rank`` columns, so the effective rank of ``W`` is at most
    ``rank``. The factors are refined by gradient descent until the change in
    loss falls below ``tol`` or ``n_iter`` iterations have elapsed.

    Parameters
    ----------
    n_iter : int, default=1000
        The maximum number of gradient-descent iterations to perform.

    rank : int, default=1
        The rank constraint on the weight matrix ``W``.

    alpha : float, default=1e-3
        The learning rate for the gradient-descent updates.

    tol : float, default=1e-6
        Convergence tolerance. Iteration stops early when the absolute change
        in the loss between two consecutive steps is below ``tol``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_targets, n_features) or (n_features,)
        The learned weight matrix ``(G @ H.T).T``, matching the layout of
        :class:`sklearn.linear_model.LinearRegression`.

    G_ : ndarray of shape (n_features, rank)
        The left factor of the rank-constrained weight matrix.

    H_ : ndarray of shape (n_targets, rank)
        The right factor of the rank-constrained weight matrix.

    losses_ : list of float
        The loss value recorded at each iteration during training.

    n_features_in_ : int
        Number of features seen during :term:`fit`.
    """

    def __init__(
        self,
        n_iter: int = 1000,
        rank: int = 1,
        alpha: float = 1e-3,
        tol: float = 1e-6,
    ) -> None:
        self.n_iter = n_iter
        self.rank = rank
        self.alpha = alpha
        self.tol = tol

    def _init_GH(
        self, W_full_rank: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Initialise the factors ``G`` and ``H`` from a full-rank ``W``."""
        rank = min(self.rank, *W_full_rank.shape)
        pca = PCA(n_components=rank, random_state=0)
        G = pca.fit_transform(W_full_rank)
        H = pca.components_.T
        return G, H

    @staticmethod
    def _compute_loss(
        G: NDArray[np.float64],
        H: NDArray[np.float64],
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Compute the residual Frobenius-norm loss for the current factors."""
        y_pred = X @ (G @ H.T)
        return float(LA.norm(y_pred - y))

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RankConstraint":
        """Fit the rank-constrained regressor to ``X`` and ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : RankConstraint
            The fitted estimator.
        """
        X, y = validate_data(self, X, y, multi_output=True, y_numeric=True)
        single_output = y.ndim == 1
        Y = y.reshape(-1, 1) if single_output else y

        reg = LinearRegression(fit_intercept=False).fit(X, Y)
        W_full_rank = reg.coef_.T

        G, H = self._init_GH(W_full_rank)

        losses: list[float] = []
        prev_loss = self._compute_loss(G, H, X, Y)
        for _ in range(self.n_iter):
            y_pred = X @ (G @ H.T)
            resid = y_pred - Y

            grad_G = X.T @ (resid @ (H @ (G.T @ G)))
            grad_H = resid.T @ (X @ (G @ (H.T @ H)))

            new_G = G - self.alpha * grad_G
            new_H = H - self.alpha * grad_H

            loss = self._compute_loss(new_G, new_H, X, Y)
            # Guard against divergence from the fixed learning rate: keep the
            # previous factors if a step blows up or increases the loss.
            if not np.isfinite(loss) or loss > prev_loss:
                break
            G, H = new_G, new_H
            losses.append(loss)
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        self.G_ = G
        self.H_ = H
        self.losses_ = losses
        coef = (G @ H.T).T
        self.coef_ = coef.ravel() if single_output else coef
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict target values for ``X`` using the fitted factors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred = X @ (self.G_ @ self.H_.T)
        if self.coef_.ndim == 1:
            return y_pred.ravel()
        return y_pred
