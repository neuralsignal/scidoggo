"""Tikhonov regularization estimator.

Based on the implementation by Stout and Kalivas, 2006, Journal of Chemometrics.
L2-regularized regression using a non-diagonal regularization matrix.

Modified from Jeff Chiang's code <jeff.njchiang@gmail.com>.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.validation import validate_data

from scidoggo.exceptions import InvalidParameterError

from .tikhonov_solver import to_general_form, to_standard_form

__all__ = ["Tikhonov"]


class Tikhonov(Ridge):
    """Tikhonov regularization estimator.

    This estimator extends :class:`~sklearn.linear_model.Ridge` by allowing the
    user to provide a regularization matrix ``L``, which is used to compute the
    coefficients of the regression problem. The ``L`` matrix generalizes the
    ridge penalty so that ``L.T @ L`` plays the role of the inverse covariance
    matrix of the coefficients.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength. Larger values specify stronger regularization.
    L : numpy.ndarray or None, default=None
        Tikhonov regularization matrix of shape
        ``(n_features, n_regularizers)``. If ``None``, ordinary ridge
        regression is performed.
    fit_intercept : bool, default=False
        Whether to calculate the intercept for this model. Fitting an intercept
        is not yet implemented, so this must be ``False``.
    copy_X : bool, default=True
        If ``True``, ``X`` will be copied; else, it may be overwritten.
    max_iter : int or None, default=None
        Maximum number of iterations for the conjugate gradient solver.
        ``None`` means the solver is iterated until convergence. Only used by
        the conjugate gradient solver.
    tol : float, default=1e-4
        Precision of the solution. Only used by the conjugate gradient solver.
    solver : str, default='auto'
        Solver to use in the computational routines. One of ``'auto'``,
        ``'svd'``, ``'cholesky'``, ``'lsqr'``, ``'sparse_cg'``, ``'sag'``,
        ``'saga'``.
    positive : bool, default=False
        When ``True``, forces the coefficients to be positive. Only ``'lbfgs'``
        solver is supported in this case.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling data. Pass an int for
        reproducible results across multiple function calls.

    Attributes
    ----------
    coef_ : numpy.ndarray of shape (n_features,) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
    intercept_ : float or numpy.ndarray of shape (n_targets,)
        Independent term in the linear model. Set to ``0.0`` when
        ``fit_intercept=False``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        L: np.ndarray | None = None,
        *,
        fit_intercept: bool = False,
        copy_X: bool = True,
        max_iter: int | None = None,
        tol: float = 1e-4,
        solver: str = "auto",
        positive: bool = False,
        random_state=None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            positive=positive,
            max_iter=max_iter,
            tol=tol,
            solver=solver,
            random_state=random_state,
        )
        self.L = L

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight=None
    ) -> "Tikhonov":
        """Fit the Tikhonov regression model.

        Parameters
        ----------
        X : numpy.ndarray
            Training data of shape ``(n_samples, n_features)``.
        y : numpy.ndarray
            Target values of shape ``(n_samples,)`` or
            ``(n_samples, n_targets)``.
        sample_weight : numpy.ndarray or None, default=None
            Individual weights for each sample.

        Returns
        -------
        Tikhonov
            The fitted estimator.

        Raises
        ------
        InvalidParameterError
            If ``fit_intercept`` is ``True``, which is not yet supported.
        """
        if self.fit_intercept:
            raise InvalidParameterError(
                "fit_intercept=True is not yet implemented for Tikhonov."
            )

        # Choose dtype based on the solver: SGD-based solvers require float64.
        if self.solver in ("sag", "saga"):
            dtype = np.float64
        else:
            dtype = [np.float64, np.float32]

        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=["csr", "csc", "coo"],
            dtype=dtype,
            multi_output=True,
            y_numeric=True,
        )

        if self.L is not None:
            # Rotate into standard form to fit, then rotate the coefficients
            # back using the *original* X and y (matching ``fit_learner``).
            x_std, y_std = to_standard_form(X, y, self.L)
            super().fit(x_std, y_std, sample_weight=sample_weight)
            self.coef_ = to_general_form(self.coef_.T, X, y, self.L).T
        else:
            super().fit(X, y, sample_weight=sample_weight)
        return self
