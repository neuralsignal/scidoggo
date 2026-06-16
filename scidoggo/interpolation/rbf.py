"""Radial basis function (RBF) interpolation estimator."""
from __future__ import annotations

import warnings
from itertools import combinations_with_replacement
from typing import Optional, Union

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import KDTree
from scipy.special import comb
from scipy.linalg.lapack import dgesv  # type: ignore[attr-defined]

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

# ``_rbf_kernels_pythran`` is a Pythran source module. When the extension has
# been compiled, ``import`` resolves to the fast compiled module (same module
# name, ``.so`` shadows ``.py``); otherwise it resolves to the pure-Python/numpy
# implementation in ``_rbf_kernels_pythran.py``. Either way the public helpers
# below are available, so the package works without a compiled extension.
try:
    from ._rbf_kernels_pythran import (
        _build_system,
        _build_evaluation_coefficients,
        _polynomial_matrix,
    )
except ImportError as exc:  # pragma: no cover - defensive only
    raise ImportError(
        "Could not import RBF kernel helpers from "
        "'scidoggo.interpolation._rbf_kernels_pythran'."
    ) from exc


__all__ = ["RbfRegression"]


# These RBFs are implemented.
_AVAILABLE = {
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
    "gaussian",
}


# The shape parameter does not need to be specified when using these RBFs.
_SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}


# For RBFs that are conditionally positive definite of order m, the interpolant
# should include polynomial terms with degree >= m - 1. Define the minimum
# degrees here. These values are from Chapter 8 of Fasshauer's "Meshfree
# Approximation Methods with MATLAB". The RBFs that are not in this dictionary
# are positive definite and do not need polynomial terms.
_NAME_TO_MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2,
}


def _monomial_powers(ndim: int, degree: int, bias: bool) -> NDArray[np.int_]:
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.
    bias : bool
        If ``False``, the constant (all-zero powers) monomial is removed.

    Returns
    -------
    out : (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = comb(degree + ndim, ndim, exact=True)
    out = np.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out[count, var] += 1

            count += 1

    if not bias:
        bias_row = np.all(out == 0, axis=-1)
        out = out[~bias_row].copy()

    return out


def _build_and_solve_system(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    smoothing: NDArray[np.float64],
    kernel: str,
    epsilon: float,
    powers: NDArray[np.int_],
    normalize: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build and solve the RBF interpolation system of equations.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    normalize : str
        Strategy used to shift and scale the polynomial domain.

    Returns
    -------
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.
    coeffs : (P + R, S) float ndarray
        Coefficients for each RBF and monomial.

    """
    lhs, rhs, shift, scale = _build_system(
        y, d, smoothing, kernel, epsilon, powers, normalize
    )
    _, _, coeffs, info = dgesv(lhs, rhs, overwrite_a=True, overwrite_b=True)
    if info < 0:
        raise ValueError(f"The {-info}-th argument had an illegal value.")
    elif info > 0:
        msg = "Singular matrix."
        nmonos = powers.shape[0]
        if nmonos > 0:
            pmat = _polynomial_matrix((y - shift) / scale, powers)
            rank = np.linalg.matrix_rank(pmat)
            if rank < nmonos:
                msg = (
                    "Singular matrix. The matrix of monomials evaluated at "
                    "the data point coordinates does not have full column "
                    f"rank ({rank}/{nmonos})."
                )

        raise LinAlgError(msg)

    return shift, scale, coeffs


class RbfRegression(RegressorMixin, BaseEstimator):
    """Radial basis function (RBF) interpolation regressor.

    Parameters
    ----------
    neighbors : int, optional
        If specified, the value of the interpolant at each evaluation point
        will be computed using only this many nearest data points. If
        ``None``, all data points are used.
    smoothing : float or (n_samples,) array-like, default=1.0
        Smoothing parameter. Larger values produce smoother interpolants.
    kernel : str, default="thin_plate_spline"
        Name of the radial basis function.
    epsilon : float, optional
        Shape parameter. Required for kernels that are not scale invariant.
    degree : int, optional
        Degree of the added polynomial. ``None`` selects a sensible minimum
        for the chosen kernel.
    normalize : str, default="scale"
        Strategy for shifting and scaling the polynomial domain. One of
        ``"scale"``, ``"zero"``, or any other value to disable normalization.
    bias : bool, default=True
        Whether to include the constant polynomial term.

    Attributes
    ----------
    coef_ : (n_samples + n_monomials, n_outputs) float ndarray
        Solved interpolation coefficients (only when ``neighbors`` is None).
    powers_ : (n_monomials, n_features) int ndarray
        Exponents for each monomial in the polynomial term.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    """

    def __init__(
        self,
        neighbors: Optional[int] = None,
        smoothing: Union[float, ArrayLike] = 1.0,
        kernel: str = "thin_plate_spline",
        epsilon: Optional[float] = None,
        degree: Optional[int] = None,
        normalize: str = "scale",
        bias: bool = True,
    ) -> None:
        self.neighbors = neighbors
        self.bias = bias
        self.degree = degree
        self.normalize = normalize
        self.kernel = kernel
        self.smoothing = smoothing
        self.epsilon = epsilon

    def __sklearn_tags__(self):
        """Return scikit-learn estimator tags.

        Returns
        -------
        tags : sklearn.utils.Tags
            The estimator tags with multi-output support enabled.

        """
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "RbfRegression":
        """Fit the RBF interpolant.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data point coordinates.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. Each smoothing value is divided by the
            corresponding weight.

        Returns
        -------
        self : RbfRegression
            The fitted estimator.

        """
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            multi_output=True,
            order="C",
            dtype=np.float64,
            y_numeric=True,
        )
        # ``validate_data``'s ``dtype`` only coerces X; an integer y is left as
        # int by ``y_numeric``. The pythran kernel is typed for C-contiguous
        # float arrays, so force both dtype and memory layout here.
        y = np.ascontiguousarray(y, dtype=np.float64)

        if sample_weight is None:
            smoothing = self.smoothing
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if np.all(sample_weight == 0.0):
                raise ValueError(
                    "Cannot fit with all sample weights equal to zero."
                )
            smoothing = self.smoothing / sample_weight

        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"RbfRegression requires at least 2 samples; got n_samples = "
                f"{n_samples} (1 sample is not enough)."
            )
        y_shape = y.shape[1:]
        y = y.reshape((n_samples, -1))

        if np.isscalar(smoothing):
            smoothing = np.full(n_samples, smoothing, dtype=float)
        else:
            smoothing = np.asarray(smoothing, dtype=float, order="C")
            if smoothing.shape != (n_samples,):
                raise ValueError(
                    "Expected `smoothing` to be a "
                    "scalar or have shape "
                    f"({n_samples},)."
                )

        kernel = self.kernel.lower()
        if kernel not in _AVAILABLE:
            raise ValueError(f"`kernel` must be one of {_AVAILABLE}.")

        epsilon = self.epsilon
        if epsilon is None:
            if kernel in _SCALE_INVARIANT:
                epsilon = 1.0
            else:
                raise ValueError(
                    "`epsilon` must be specified if `kernel` is not one of "
                    f"{_SCALE_INVARIANT}."
                )
        else:
            epsilon = float(epsilon)

        min_degree = _NAME_TO_MIN_DEGREE.get(kernel, -1)
        degree = self.degree
        if degree is None:
            degree = max(min_degree, 0)
        else:
            degree = int(degree)
            if degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif degree < min_degree:
                warnings.warn(
                    f"`degree` should not be below {min_degree} when `kernel` "
                    f"is '{kernel}'. The interpolant may not be uniquely "
                    "solvable, and the smoothing parameter may have an "
                    "unintuitive effect.",
                    UserWarning,
                )

        neighbors = self.neighbors
        if neighbors is None:
            nobs = n_samples
        else:
            # Make sure the number of nearest neighbors used for interpolation
            # does not exceed the number of observations.
            neighbors = int(min(neighbors, n_samples))
            nobs = neighbors

        powers = _monomial_powers(n_features, degree, self.bias)
        # The polynomial matrix must have full column rank in order for the
        # interpolant to be well-posed, which is not possible if there are
        # fewer observations than monomials.
        if powers.shape[0] > nobs:
            raise ValueError(
                f"At least {powers.shape[0]} data points are required when "
                f"`degree` is {degree} and the number of dimensions is "
                f"{n_features}."
            )

        if neighbors is None:
            shift, scale, coeffs = _build_and_solve_system(
                X, y, smoothing, kernel, epsilon, powers, self.normalize
            )

            self.shift_ = shift
            self.scale_ = scale
            self.coef_ = coeffs
        else:
            # The tree is queried with evaluation coordinates in `predict`, so
            # it must be built from the data point coordinates `X`, not `y`.
            self.tree_ = KDTree(X)

        self.Xfit_ = X
        self.y_shape_ = y_shape
        self.yfit_ = y
        self.powers_ = powers
        self.smoothing_ = smoothing
        self.neighbors_ = neighbors
        self.kernel_ = kernel
        self.epsilon_ = epsilon
        self.n_samples_ = n_samples
        return self

    def _chunk_evaluator(
        self,
        X: NDArray[np.float64],
        Xfit: NDArray[np.float64],
        shift: NDArray[np.float64],
        scale: NDArray[np.float64],
        coeffs: NDArray[np.float64],
        memory_budget: int,
    ) -> NDArray[np.float64]:
        """Evaluate the interpolant while controlling memory consumption.

        The input is chunked if evaluating it all at once would require more
        memory than ``memory_budget``.

        Parameters
        ----------
        X : (Q, N) float ndarray
            Array of points on which to evaluate.
        Xfit : (P, N) float ndarray
            Array of points on which we know function values.
        shift : (N,) float ndarray
            Domain shift used to create the polynomial matrix.
        scale : (N,) float ndarray
            Domain scaling used to create the polynomial matrix.
        coeffs : (P + R, S) float ndarray
            Coefficients in front of basis functions.
        memory_budget : int
            Total amount of memory (in units of ``sizeof(float)``) we wish to
            devote to storing the array of coefficients for interpolated
            points. If we need more memory than that, we chunk the input.

        Returns
        -------
        out : (Q, S) float ndarray
            Interpolated array.

        """
        n_samples, _ = X.shape
        if self.neighbors_ is None:
            nnei = self.n_samples_
        else:
            nnei = self.neighbors_
        # in each chunk we consume the same space we already occupy
        chunksize = memory_budget // (self.powers_.shape[0] + nnei) + 1
        if chunksize <= n_samples:
            out = np.empty((n_samples, self.yfit_.shape[1]), dtype=float)
            for i in range(0, n_samples, chunksize):
                vec = _build_evaluation_coefficients(
                    X[i:i + chunksize, :],
                    Xfit,
                    self.kernel_,
                    self.epsilon_,
                    self.powers_,
                    shift,
                    scale,
                )
                out[i:i + chunksize, :] = np.dot(vec, coeffs)
        else:
            vec = _build_evaluation_coefficients(
                X,
                Xfit,
                self.kernel_,
                self.epsilon_,
                self.powers_,
                shift,
                scale,
            )
            out = np.dot(vec, coeffs)
        return out

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the interpolant at ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Points to predict.

        Returns
        -------
        ypred : numpy.ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted outputs.

        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
            reset=False,
        )

        n_samples, _ = X.shape

        # Our memory budget for storing RBF coefficients is based on how many
        # floats in memory we already occupy. If this number is below 1e6 we
        # just use 1e6. This memory budget is used to decide how we chunk the
        # inputs.
        memory_budget = max(
            X.size + self.Xfit_.size + self.yfit_.size, 1000000
        )

        if self.neighbors is None:
            out = self._chunk_evaluator(
                X,
                self.Xfit_,
                self.shift_,
                self.scale_,
                self.coef_,
                memory_budget,
            )
        else:
            # Get the indices of the k nearest observation points to each
            # evaluation point.
            _, xindices = self.tree_.query(X, self.neighbors_)
            if self.neighbors_ == 1:
                # `KDTree` squeezes the output when neighbors=1.
                xindices = xindices[:, None]

            # Multiple evaluation points may have the same neighborhood of
            # observation points. Make the neighborhoods unique so that we only
            # compute the interpolation coefficients once for each
            # neighborhood.
            xindices = np.sort(xindices, axis=1)
            xindices, inv = np.unique(xindices, return_inverse=True, axis=0)
            # NumPy 2.x returns `inv` with the shape of the input; flatten it
            # so indexing below works regardless of NumPy version.
            inv = inv.reshape(-1)
            # `inv` tells us which neighborhood will be used by each evaluation
            # point. Now we find which evaluation points will be using each
            # neighborhood.
            xindices_new: list[list[int]] = [[] for _ in range(len(xindices))]
            for i, j in enumerate(inv):
                xindices_new[j].append(i)

            out = np.empty((n_samples, self.yfit_.shape[1]), dtype=float)
            for xidx, yidx in zip(xindices_new, xindices):
                # `yidx` are the indices of the observations in this
                # neighborhood. `xidx` are the indices of the evaluation points
                # that are using this neighborhood.
                xnbr = X[xidx]
                ynbr = self.Xfit_[yidx]
                dnbr = self.yfit_[yidx]
                snbr = self.smoothing_[yidx]
                shift, scale, coeffs = _build_and_solve_system(
                    ynbr,
                    dnbr,
                    snbr,
                    self.kernel_,
                    self.epsilon_,
                    self.powers_,
                    self.normalize,
                )
                out[xidx] = self._chunk_evaluator(
                    xnbr,
                    ynbr,
                    shift,
                    scale,
                    coeffs,
                    memory_budget,
                )

        ypred = out.reshape((n_samples,) + self.y_shape_)

        return ypred
