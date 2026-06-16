"""Rank-1-plus-sparse matrix linear regression model.

This module implements :class:`Rank1PlusSparse`, which decomposes the
coefficient matrix as a rank-1 outer product plus a (masked) sparse matrix
and solves the resulting problem via alternating convex minimisation.

Requires the optional ``[sparse]`` extra (``cvxpy``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel

from scidoggo._validation import check_xy
from scidoggo.exceptions import ConvergenceError, MissingDependencyError

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover - exercised only without cvxpy
    raise MissingDependencyError(
        'Rank1PlusSparse requires the [sparse] extra. Install with: pip install "scidoggo[sparse]"'
    ) from exc


class Rank1PlusSparse(MultiOutputMixin, RegressorMixin, LinearModel):
    """Rank-1-constrained and sparse-matrix-constrained linear regression.

    The coefficient matrix is decomposed as::

        W = w1 @ w2 + S
        Y = X @ W

    where ``w1`` is a column vector, ``w2`` is a row vector (their outer
    product forms the rank-1 component) and ``S`` is an optional sparse matrix
    whose support is fixed by ``S_mask``. The problem is solved by alternating
    convex minimisation over ``w1`` and ``w2``.

    Parameters
    ----------
    normalize_w1 : bool, default=True
        Whether to normalize the ``w1`` vector by its l1-norm.
    normalize_w2 : bool, default=False
        Whether to normalize the ``w2`` vector by its l1-norm.
    S_mask : numpy.ndarray of shape (n_features, n_targets), optional
        Boolean mask for the sparse matrix. Entries that are ``False`` (zero)
        are constrained to zero. If ``None``, no sparse component is fitted.
    w1_sign : {-1, 1}, optional
        If given, constrain ``w1`` to have this sign (all positive or all
        negative values).
    w2_sign : {-1, 1}, optional
        If given, constrain ``w2`` to have this sign.
    S_sign : {-1, 1}, optional
        If given, constrain ``S`` to have this sign.
    atol : float, default=1e-8
        Absolute tolerance for terminating the fit, comparing the loss to the
        loss of the previous iteration.
    rtol : float, default=1e-5
        Relative tolerance for terminating the fit, comparing the loss to the
        loss of the previous iteration.
    max_iter : int, default=1000
        Maximum number of alternating-minimisation iterations.
    cp_kwargs : dict, optional
        Extra keyword arguments forwarded to :meth:`cvxpy.Problem.solve`.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_targets, n_features)
        Estimated coefficient matrix ``(w1 @ w2 + S).T``.
    intercept_ : float
        Always ``0.0``; this estimator does not fit an intercept.
    w1_ : numpy.ndarray of shape (n_features, 1)
        Fitted ``w1`` column vector.
    w2_ : numpy.ndarray of shape (1, n_targets)
        Fitted ``w2`` row vector.
    S_ : numpy.ndarray or float
        Fitted sparse matrix, or ``0.0`` when ``S_mask`` is ``None``.
    loss_ : numpy.ndarray of shape (max_iter,)
        Loss value recorded at each iteration.

    Examples
    --------
    >>> import numpy as np
    >>> n, m, s = 4, 4, 100
    >>> rng = np.random.RandomState(10)
    >>> X = rng.normal(0, 1, (s, n))
    >>> w1 = rng.normal(0, 1, n)
    >>> w1 /= np.sum(np.abs(w1))
    >>> w2 = -rng.random(m)
    >>> W = np.outer(w1, w2)
    >>> S = np.roll(np.eye(n, m), 2, axis=1)
    >>> Y = X @ (W - S)
    >>> y = Y + rng.normal(0, 0.01, Y.shape)
    >>> model = Rank1PlusSparse(S_mask=S, S_sign=-1, w2_sign=-1)
    >>> model.fit(X, y)  # doctest: +SKIP
    Rank1PlusSparse(...)
    """

    def __init__(
        self,
        *,
        normalize_w1: bool = True,
        normalize_w2: bool = False,
        S_mask: npt.ArrayLike | None = None,
        w1_sign: int | None = None,
        w2_sign: int | None = None,
        S_sign: int | None = None,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        max_iter: int = 1000,
        cp_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.normalize_w1 = normalize_w1
        self.normalize_w2 = normalize_w2
        self.S_mask = S_mask
        self.w1_sign = w1_sign
        self.w2_sign = w2_sign
        self.S_sign = S_sign
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter
        self.cp_kwargs = cp_kwargs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Rank1PlusSparse:
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples, n_targets)
            Target values. Will be cast to ``X``'s dtype if necessary.

        Returns
        -------
        self : Rank1PlusSparse
            Fitted estimator.

        Raises
        ------
        ConvergenceError
            If the alternating minimisation does not converge within
            ``max_iter`` iterations.
        """
        X, y = check_xy(
            X,
            y,
            accept_sparse=False,
            y_numeric=True,
            multi_output=True,
        )
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        cp_kwargs = {} if self.cp_kwargs is None else self.cp_kwargs

        n = X.shape[1]
        m = y.shape[1]

        constraints: list[Any] = []

        S_sign = 1.0 if self.S_sign is None else self.S_sign
        w1_sign = 1.0 if self.w1_sign is None else self.w1_sign
        w2_sign = 1.0 if self.w2_sign is None else self.w2_sign

        w1_ = np.random.random((n, 1)) * w1_sign
        # normalize w1
        if self.normalize_w1:
            w1_ /= np.sum(np.abs(w1_))
        w2_ = np.random.random((1, m)) * w2_sign
        # normalize w2
        if self.normalize_w2:
            w2_ /= np.sum(np.abs(w2_))
        if self.S_mask is None:
            S_ = 0.0
        else:
            S = np.asarray(self.S_mask).astype(float)
            if S.shape != (n, m):
                raise ValueError(f"S_mask has shape {S.shape}, expected {(n, m)}.")
            S_ = cp.Variable(S.shape, name="S", pos=(self.S_sign is not None))
            S_ = cp.multiply(S_, S_sign)
            constraints.append(S_[S == 0] == 0)

        loss = np.zeros(self.max_iter)

        converged = False
        for i in range(self.max_iter):
            one = i % 2

            if one:
                w1_ = cp.Variable((n, 1), name="w1", pos=(self.w1_sign is not None))
                w1_ = cp.multiply(w1_, w1_sign)
            else:
                w2_ = cp.Variable((1, m), name="w2", pos=(self.w2_sign is not None))
                w2_ = cp.multiply(w2_, w2_sign)

            W = cp.multiply(w1_, w2_) + S_
            ypred = X @ W
            obj = cp.Minimize(0.5 * cp.sum_squares(y - ypred))
            prob = cp.Problem(obj, constraints)
            loss[i] = prob.solve(**cp_kwargs)

            if one:
                w1_ = w1_.value
                # normalize w1
                if self.normalize_w1:
                    w1_ /= np.sum(np.abs(w1_))
            else:
                w2_ = w2_.value
                # normalize w2
                if self.normalize_w2:
                    w2_ /= np.sum(np.abs(w2_))

            # recalculate loss due to normalization step
            if self.normalize_w1 or self.normalize_w2:
                W = w1_ * w2_
                if self.S_mask is not None:
                    W += S_.value * S_sign
                ypred = X @ W
                loss[i] = 0.5 * np.sum(np.square(y - ypred))

            if i > 0 and np.isclose(loss[i - 1], loss[i], atol=self.atol, rtol=self.rtol):
                loss[i + 1 :] = loss[i]
                if self.S_mask is not None:
                    S_ = S_.value
                converged = True
                break

        if not converged:
            if self.S_mask is not None:
                S_ = S_.value
            raise ConvergenceError(
                "Maximum number of iterations reached, "
                "consider increasing `max_iter`, or changing `rtol`/`atol`."
            )

        self.intercept_ = 0.0
        self.w1_ = w1_
        self.w2_ = w2_
        self.S_ = S_
        self.loss_ = loss
        self.coef_ = (w1_ * w2_ + S_).T
        return self
