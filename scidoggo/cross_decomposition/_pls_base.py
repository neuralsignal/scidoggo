"""Base class and helpers for Partial Least Squares (PLS).

This module is forked from scikit-learn with tweaks that allow disabling the
centering (``demean``) of the datasets and that expose helpers to decompose the
fitted coefficients into orthogonal components.

The :mod:`sklearn.pls` module implements Partial Least Squares (PLS).
"""

# Author: Edouard Duchesnay <edouard.duchesnay@cea.fr>
# License: BSD 3 clause

import numbers
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import pinv as pinv2
from scipy.linalg import svd
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import Tags
from sklearn.utils.validation import (
    FLOAT_DTYPES,
    check_array,
    check_consistent_length,
    check_is_fitted,
    check_scalar,
    validate_data,
)


def _pinv2_old(a: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute a pseudo-inverse mimicking the legacy ``scipy.linalg.pinv2``.

    Parameters
    ----------
    a : ndarray of shape (M, N)
        The matrix to pseudo-invert.

    Returns
    -------
    a_pinv : ndarray of shape (N, M)
        The pseudo-inverse of `a`.
    """
    # Used previous scipy pinv2 that was updated in:
    # https://github.com/scipy/scipy/pull/10067
    # We can not set `cond` or `rcond` for pinv2 in scipy >= 1.3 to keep the
    # same behavior of pinv2 for scipy < 1.3, because the condition used to
    # determine the rank is dependent on the output of svd.
    u, s, vh = svd(a, full_matrices=False, check_finite=False)

    t = u.dtype.char.lower()
    factor = {"f": 1e3, "d": 1e6}
    cond = np.max(s) * factor[t] * np.finfo(t).eps
    rank = np.sum(s > cond)

    u = u[:, :rank]
    u /= s[:rank]
    return np.transpose(np.conjugate(np.dot(u, vh[:rank])))


def _get_first_singular_vectors_power_method(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    mode: str,
    max_iter: int,
    tol: float,
    norm_y_weights: bool,
) -> tuple[NDArray[np.floating], NDArray[np.floating], int]:
    """Return the first left and right singular vectors of X'Y.

    Provides an alternative to the ``svd(X'Y)`` and uses the power method
    instead. With `norm_y_weights` set to True and in mode A, this corresponds
    to the algorithm of section 11.3 of the Wegelin's review, except this
    starts at the "update saliences" part.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The (deflated) predictor block.

    Y : ndarray of shape (n_samples, n_targets)
        The (deflated) response block.

    mode : {'A', 'B'}
        The PLS mode. Mode 'B' corresponds to CCA and precomputes the
        pseudo-inverses of `X` and `Y`.

    max_iter : int
        The maximum number of iterations of the power method.

    tol : float
        The convergence tolerance on the squared norm of the weight update.

    norm_y_weights : bool
        Whether to normalize the `Y` weights to unit norm.

    Returns
    -------
    x_weights : ndarray of shape (n_features,)
        The first left singular vector.

    y_weights : ndarray of shape (n_targets,)
        The first right singular vector.

    n_iter : int
        The number of iterations performed.
    """
    eps = np.finfo(X.dtype).eps
    try:
        y_score = next(col for col in Y.T if np.any(np.abs(col) > eps))
    except StopIteration as e:
        raise StopIteration("Y residual is constant") from e

    x_weights_old = 100  # init to big value for first convergence check

    if mode == "B":
        # Precompute pseudo inverse matrices
        # Basically: X_pinv = (X.T X)^-1 X.T
        # Which requires inverting a (n_features, n_features) matrix.
        # As a result, and as detailed in the Wegelin's review, CCA (i.e. mode
        # B) will be unstable if n_features > n_samples or n_targets >
        # n_samples
        X_pinv, Y_pinv = _pinv2_old(X), _pinv2_old(Y)

    i = 0
    for i in range(max_iter):  # noqa: B007 - i is read after the loop for n_iter
        if mode == "B":
            x_weights = np.dot(X_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score, y_score)

        x_weights /= np.sqrt(np.dot(x_weights, x_weights)) + eps
        x_score = np.dot(X, x_weights)

        if mode == "B":
            y_weights = np.dot(Y_pinv, x_score)
        else:
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)

        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights, y_weights)) + eps

        y_score = np.dot(Y, y_weights) / (np.dot(y_weights, y_weights) + eps)

        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        x_weights_old = x_weights

    n_iter = i + 1
    if n_iter == max_iter:
        warnings.warn(
            "Maximum number of iterations reached",
            ConvergenceWarning,
            stacklevel=2,
        )

    return x_weights, y_weights, n_iter


def _get_first_singular_vectors_svd(
    X: NDArray[np.floating], Y: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return the first left and right singular vectors of X'Y.

    Here the whole SVD is computed.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The (deflated) predictor block.

    Y : ndarray of shape (n_samples, n_targets)
        The (deflated) response block.

    Returns
    -------
    x_weights : ndarray of shape (n_features,)
        The first left singular vector.

    y_weights : ndarray of shape (n_targets,)
        The first right singular vector.
    """
    C = np.dot(X.T, Y)
    U, _, Vt = svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]


def _center_scale_xy(
    X: NDArray[np.floating], Y: NDArray[np.floating], scale: bool, demean: bool
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Center `X`, `Y` and scale them if requested.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The predictor block, modified in place.

    Y : ndarray of shape (n_samples, n_targets)
        The response block, modified in place.

    scale : bool
        Whether to scale `X` and `Y` to unit standard deviation.

    demean : bool
        Whether to center `X` and `Y` to zero mean.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The centered/scaled predictor block.

    Y : ndarray of shape (n_samples, n_targets)
        The centered/scaled response block.

    x_mean : ndarray of shape (n_features,)
        The per-feature mean of `X` (zeros if `demean` is False).

    y_mean : ndarray of shape (n_targets,)
        The per-target mean of `Y` (zeros if `demean` is False).

    x_std : ndarray of shape (n_features,)
        The per-feature standard deviation of `X` (ones if `scale` is False).

    y_std : ndarray of shape (n_targets,)
        The per-target standard deviation of `Y` (ones if `scale` is False).
    """
    # center
    if demean:
        x_mean = X.mean(axis=0)
        X -= x_mean
        y_mean = Y.mean(axis=0)
        Y -= y_mean
    else:
        x_mean = np.zeros(X.shape[1])
        y_mean = np.zeros(Y.shape[1])
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


def _svd_flip_1d(u: NDArray[np.floating], v: NDArray[np.floating]) -> None:
    """Flip the signs of 1d singular vectors in place for consistency.

    Same as :func:`sklearn.utils.extmath.svd_flip` but works on 1d arrays and
    operates in place.

    Parameters
    ----------
    u : ndarray of shape (n_features,)
        The left singular vector, modified in place.

    v : ndarray of shape (n_targets,)
        The right singular vector, modified in place.

    Returns
    -------
    None
    """
    # svd_flip would force us to convert to 2d array and would also return 2d
    # arrays. We don't want that.
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign


class _PLS(
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
    BaseEstimator,
    metaclass=ABCMeta,
):
    """Partial Least Squares (PLS).

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://www.stat.washington.edu/research/reports/2000/tr371.pdf
    """

    @abstractmethod
    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        demean: bool = True,
        deflation_mode: str = "regression",
        mode: str = "A",
        algorithm: str = "nipals",
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.demean = demean
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X: NDArray, Y: NDArray) -> "_PLS":
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted model.
        """
        check_consistent_length(X, Y)
        X = validate_data(self, X, dtype=np.float64, copy=self.copy, ensure_min_samples=2)
        Y = check_array(Y, input_name="Y", dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        n_components = self.n_components
        if self.deflation_mode == "regression":
            # With PLSRegression n_components is bounded by the rank of (X.T X)
            # see Wegelin page 25
            rank_upper_bound = p
            check_scalar(
                n_components,
                "n_components",
                numbers.Integral,
                min_val=1,
                max_val=rank_upper_bound,
            )
        else:
            # With CCA and PLSCanonical, n_components is bounded by the rank of
            # X and the rank of Y: see Wegelin page 12
            rank_upper_bound = min(n, p, q)
            check_scalar(
                n_components,
                "n_components",
                numbers.Integral,
                min_val=1,
                max_val=rank_upper_bound,
            )

        if self.algorithm not in ("svd", "nipals"):
            raise ValueError(f"algorithm should be 'svd' or 'nipals', got {self.algorithm}.")

        self._norm_y_weights = self.deflation_mode == "canonical"  # 1.1
        norm_y_weights = self._norm_y_weights

        # Scale (in place)
        Xk, Yk, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, Y, self.scale, self.demean
        )

        self.x_weights_ = np.zeros((p, n_components))  # U
        self.y_weights_ = np.zeros((q, n_components))  # V
        self._x_scores = np.zeros((n, n_components))  # Xi
        self._y_scores = np.zeros((n, n_components))  # Omega
        self.x_loadings_ = np.zeros((p, n_components))  # Gamma
        self.y_loadings_ = np.zeros((q, n_components))  # Delta
        self.n_iter_ = []

        # This whole thing corresponds to the algorithm in section 4.1 of the
        # review from Wegelin. See above for a notation mapping from code to
        # paper.
        Y_eps = np.finfo(Yk.dtype).eps
        for k in range(n_components):
            # Find first left and right singular vectors of the X.T.dot(Y)
            # cross-covariance matrix.
            if self.algorithm == "nipals":
                # Replace columns that are all close to zero with zeros
                Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                Yk[:, Yk_mask] = 0.0

                try:
                    (
                        x_weights,
                        y_weights,
                        n_iter_,
                    ) = _get_first_singular_vectors_power_method(
                        Xk,
                        Yk,
                        mode=self.mode,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        norm_y_weights=norm_y_weights,
                    )
                except StopIteration as e:
                    if str(e) != "Y residual is constant":
                        raise
                    warnings.warn(
                        f"Y residual is constant at iteration {k}",
                        stacklevel=2,
                    )
                    break

                self.n_iter_.append(n_iter_)

            elif self.algorithm == "svd":
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, Yk)

            # inplace sign flip for consistency across solvers and archs
            _svd_flip_1d(x_weights, y_weights)

            # compute scores, i.e. the projections of X and Y
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss

            # Deflation: subtract rank-one approx to obtain Xk+1 and Yk+1
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)

            if self.deflation_mode == "canonical":
                # regress Yk on y_score
                y_loadings = np.dot(y_scores, Yk) / np.dot(y_scores, y_scores)
                Yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == "regression":
                # regress Yk on x_score
                y_loadings = np.dot(x_scores, Yk) / np.dot(x_scores, x_scores)
                Yk -= np.outer(x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings

        # X was approximated as Xi . Gamma.T + X_(R+1)
        # Xi . Gamma.T is a sum of n_components rank-1 matrices. X_(R+1) is
        # whatever is left to fully reconstruct X, and can be 0 if X is of rank
        # n_components.
        # Similarly, Y was approximated as Omega . Delta.T + Y_(R+1)

        # Compute transformation matrices (rotations_). See User Guide.
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False),
        )
        self.y_rotations_ = np.dot(
            self.y_weights_,
            pinv2(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False),
        )
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = (self.coef_ * self._y_std).T
        self.intercept_ = self._y_mean
        self._n_features_out = self.x_rotations_.shape[1]
        return self

    # --- #

    # additional functionalities added to
    # orthogonalize components and
    # take the kth coefs/rotations
    def decompose_coef(
        self, k: int | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Decompose the fitted coefficients into orthogonal components.

        The coefficients are decomposed into orthogonal `X` rotations and their
        corresponding `Y` loadings.

        Parameters
        ----------
        k : int or None, default=None
            The number of components to keep. If None, all fitted components
            are used.

        Returns
        -------
        x_rotations_orth : ndarray of shape (n_features, n_components)
            The orthogonalized `X` rotation matrix.

        y_loadings_orth : ndarray of shape (n_targets, n_components)
            The `Y` loadings expressed in the orthogonal basis.
        """
        x_rotations_, _ = self.kth_rotations(k)

        x_rotations_orth, E, Vt = svd(x_rotations_, full_matrices=False)
        y_loadings_orth = (np.diag(E) @ Vt @ self.y_loadings_.T[:k]).T
        return x_rotations_orth, y_loadings_orth

    def kth_coef(self, k: int | None = None) -> NDArray[np.floating]:
        """Return the linear coefficients using only the first `k` components.

        Parameters
        ----------
        k : int or None, default=None
            The number of components to use. If None, the full fitted
            coefficient matrix is returned.

        Returns
        -------
        coef : ndarray of shape (n_targets, n_features)
            The linear coefficients computed from the first `k` components.
        """
        if k is None:
            return self.coef_

        x_rotations_, _ = self.kth_rotations(k)
        coef = np.dot(x_rotations_, self.y_loadings_.T[:k])
        return (coef * self._y_std).T

    def kth_rotations(
        self, k: int | None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the `X` and `Y` rotations using only the first `k` components.

        Parameters
        ----------
        k : int or None, default=None
            The number of components to use. If None, the full fitted rotation
            matrices are returned.

        Returns
        -------
        x_rotations_ : ndarray of shape (n_features, n_components)
            The `X` rotation matrix from the first `k` components.

        y_rotations_ : ndarray of shape (n_targets, n_components)
            The `Y` rotation matrix from the first `k` components.
        """
        if k is None:
            return self.x_rotations_, self.y_rotations_
        x_rotations_ = np.dot(
            self.x_weights_[:, :k],
            pinv2(
                np.dot(self.x_loadings_[:, :k].T, self.x_weights_[:, :k]),
                check_finite=False,
            ),
        )
        y_rotations_ = np.dot(
            self.y_weights_[:, :k],
            pinv2(
                np.dot(self.y_loadings_[:, :k].T, self.y_weights_[:, :k]),
                check_finite=False,
            ),
        )
        return x_rotations_, y_rotations_

    # --- #

    def transform(
        self,
        X: NDArray,
        Y: NDArray | None = None,
        k: int | None = None,
        copy: bool = True,
    ):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        k : int or None, default=None
            The number of components to use for the rotation. If None, all
            fitted components are used.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        x_scores, y_scores : array-like or tuple of array-like
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)`
            otherwise.
        """
        check_is_fitted(self)
        X = validate_data(self, X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        x_rotations_, y_rotations_ = self.kth_rotations(k)
        # Normalize
        X -= self._x_mean
        X /= self._x_std
        # Apply rotation
        x_scores = np.dot(X, x_rotations_)
        if Y is not None:
            Y = check_array(Y, input_name="Y", ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self._y_mean
            Y /= self._y_std
            y_scores = np.dot(Y, y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X: NDArray, Y: NDArray | None = None):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Y : array-like of shape (n_samples, n_components)
            New target, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
            Return the reconstructed `X` data.

        Y_reconstructed : ndarray of shape (n_samples, n_targets)
            Return the reconstructed `X` target. Only returned when `Y` is
            given.

        Notes
        -----
        This transformation will only be exact if `n_components=n_features`.
        """
        check_is_fitted(self)
        X = check_array(X, input_name="X", dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T[: X.shape[1]])
        # Denormalize
        X_reconstructed *= self._x_std
        X_reconstructed += self._x_mean

        if Y is not None:
            Y = check_array(Y, input_name="Y", dtype=FLOAT_DTYPES)
            # From pls space to original space
            Y_reconstructed = np.matmul(Y, self.y_loadings_.T[: Y.shape[1]])
            # Denormalize
            Y_reconstructed *= self._y_std
            Y_reconstructed += self._y_mean
            return X_reconstructed, Y_reconstructed

        return X_reconstructed

    def predict(self, X: NDArray, copy: bool = True, k: int | None = None) -> NDArray[np.floating]:
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        k : int or None, default=None
            The number of components to use for the coefficients. If None, all
            fitted components are used.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.

        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        check_is_fitted(self)
        X = validate_data(self, X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Normalize
        X -= self._x_mean
        X /= self._x_std
        coef = self.kth_coef(k)
        Ypred = X @ coef.T
        return Ypred + self.intercept_

    def fit_transform(self, X: NDArray, y: NDArray | None = None, k: int | None = None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        k : int or None, default=None
            The number of components to use for the rotation. If None, all
            fitted components are used.

        Returns
        -------
        self : ndarray of shape (n_samples, n_components)
            Return `x_scores` if `Y` is not given, `(x_scores, y_scores)`
            otherwise.
        """
        return self.fit(X, y).transform(X, y, k=k)  # type: ignore[arg-type]

    def __sklearn_tags__(self) -> Tags:
        """Return the estimator tags.

        Returns
        -------
        tags : Tags
            The estimator tags with `poor_score` enabled and `requires_y`
            disabled.
        """
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        if tags.regressor_tags is not None:
            tags.regressor_tags.poor_score = True
        return tags
