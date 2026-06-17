"""Public Partial Least Squares (PLS) estimators.

This module is forked from scikit-learn with tweaks that allow disabling the
centering (``demean``) of the datasets.
"""

# Author: Edouard Duchesnay <edouard.duchesnay@cea.fr>
# License: BSD 3 clause

import numbers

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    check_scalar,
    validate_data,
)

from ._pls_base import _PLS, _center_scale_xy

__all__ = ["PLSCanonical", "PLSRegression", "CCA", "PLSSVD"]


class PLSRegression(_PLS):
    """PLS regression.

    PLSRegression is also known as PLS2 or PLS1, depending on the number of
    targets.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    demean : bool, default=True
        Whether to center `X` and `Y` to zero mean before fitting.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in :term:`fit` before applying centering,
        and potentially scaling. If `False`, these operations will be done
        inplace, modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training samples.

    y_scores_ : ndarray of shape (n_samples, n_components)
        The transformed training targets.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.

    Examples
    --------
    >>> from scidoggo.cross_decomposition.pls import PLSRegression
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> pls2 = PLSRegression(n_components=2)
    >>> pls2.fit(X, Y)
    PLSRegression()
    >>> Y_pred = pls2.predict(X)
    """

    # This implementation provides the same results that 3 PLS packages
    # provided in the R language (R-project):
    #     - "mixOmics" with function pls(X, Y, mode = "regression")
    #     - "plspm " with function plsreg2(X, Y)
    #     - "pls" with function oscorespls.fit(X, Y)

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        demean: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            scale=scale,
            demean=demean,
            deflation_mode="regression",
            mode="A",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )

    def fit(self, X: NDArray, Y: NDArray, **kwargs) -> "PLSRegression":
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        **kwargs : dict
            Additional keyword arguments forwarded to the base ``fit``.

        Returns
        -------
        self : object
            Fitted model.
        """
        super().fit(X, Y, **kwargs)
        # expose the fitted attributes `x_scores_` and `y_scores_`
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        return self


class PLSCanonical(_PLS):
    """Partial Least Squares transformer and regressor.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    demean : bool, default=True
        Whether to center `X` and `Y` to zero mean before fitting.

    algorithm : {'nipals', 'svd'}, default='nipals'
        The algorithm used to estimate the first singular vectors of the
        cross-covariance matrix. 'nipals' uses the power method while 'svd'
        will compute the whole SVD.

    max_iter : int, default=500
        The maximum number of iterations of the power method when
        `algorithm='nipals'`. Ignored otherwise.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component. Empty if `algorithm='svd'`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    CCA : Canonical Correlation Analysis.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from scidoggo.cross_decomposition.pls import PLSCanonical
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [2.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> plsca = PLSCanonical(n_components=2)
    >>> plsca.fit(X, Y)
    PLSCanonical()
    >>> X_c, Y_c = plsca.transform(X, Y)
    """

    # This implementation provides the same results that the "plspm" package
    # provided in the R language (R-project), using the function plsca(X, Y).
    # Results are equal or collinear with the function
    # ``pls(..., mode = "canonical")`` of the "mixOmics" package. The
    # difference relies in the fact that mixOmics implementation does not
    # exactly implement the Wold algorithm since it does not normalize
    # y_weights to one.

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        demean: bool = True,
        algorithm: str = "nipals",
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            scale=scale,
            demean=demean,
            deflation_mode="canonical",
            mode="A",
            algorithm=algorithm,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )


class CCA(_PLS):
    """Canonical Correlation Analysis, also known as "Mode B" PLS.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    demean : bool, default=True
        Whether to center `X` and `Y` to zero mean before fitting.

    max_iter : int, default=500
        The maximum number of iterations of the power method.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If False, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the cross-covariance matrices of each
        iteration.

    y_weights_ : ndarray of shape (n_targets, n_components)
        The right singular vectors of the cross-covariance matrices of each
        iteration.

    x_loadings_ : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    y_loadings_ : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    x_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `X`.

    y_rotations_ : ndarray of shape (n_features, n_components)
        The projection matrix used to transform `Y`.

    coef_ : ndarray of shape (n_targets, n_features)
        The coefficients of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

    intercept_ : ndarray of shape (n_targets,)
        The intercepts of the linear model such that `Y` is approximated as
        `Y = X @ coef_.T + intercept_`.

        .. versionadded:: 1.1

    n_iter_ : list of shape (n_components,)
        Number of iterations of the power method, for each
        component.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    PLSSVD : Partial Least Square SVD.

    Examples
    --------
    >>> from scidoggo.cross_decomposition.pls import CCA
    >>> X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    >>> Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    >>> cca = CCA(n_components=1)
    >>> cca.fit(X, Y)
    CCA(n_components=1)
    >>> X_c, Y_c = cca.transform(X, Y)
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        demean: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            scale=scale,
            demean=demean,
            deflation_mode="canonical",
            mode="B",
            algorithm="nipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )


class PLSSVD(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Partial Least Square SVD.

    This transformer simply performs a SVD on the cross-covariance matrix
    `X'Y`. It is able to project both the training data `X` and the targets
    `Y`. The training data `X` is projected on the left singular vectors, while
    the targets are projected on the right singular vectors.

    Read more in the :ref:`User Guide <cross_decomposition>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep. Should be in `[1,
        min(n_samples, n_features, n_targets)]`.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    demean : bool, default=True
        Whether to center `X` and `Y` to zero mean before fitting.

    copy : bool, default=True
        Whether to copy `X` and `Y` in fit before applying centering, and
        potentially scaling. If `False`, these operations will be done inplace,
        modifying both arrays.

    Attributes
    ----------
    x_weights_ : ndarray of shape (n_features, n_components)
        The left singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    y_weights_ : ndarray of (n_targets, n_components)
        The right singular vectors of the SVD of the cross-covariance matrix.
        Used to project `X` in :meth:`transform`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    PLSCanonical : Partial Least Squares transformer and regressor.
    CCA : Canonical Correlation Analysis.

    Examples
    --------
    >>> import numpy as np
    >>> from scidoggo.cross_decomposition.pls import PLSSVD
    >>> X = np.array([[0., 0., 1.],
    ...               [1., 0., 0.],
    ...               [2., 2., 2.],
    ...               [2., 5., 4.]])
    >>> Y = np.array([[0.1, -0.2],
    ...               [0.9, 1.1],
    ...               [6.2, 5.9],
    ...               [11.9, 12.3]])
    >>> pls = PLSSVD(n_components=2).fit(X, Y)
    >>> X_c, Y_c = pls.transform(X, Y)
    >>> X_c.shape, Y_c.shape
    ((4, 2), (4, 2))
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        demean: bool = True,
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.demean = demean
        self.copy = copy

    def fit(self, X: NDArray, Y: NDArray) -> "PLSSVD":
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        check_consistent_length(X, Y)
        X = validate_data(self, X, dtype=np.float64, copy=self.copy, ensure_min_samples=2)
        Y = check_array(Y, input_name="Y", dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # we'll compute the SVD of the cross-covariance matrix = X.T.dot(Y)
        # This matrix rank is at most min(n_samples, n_features, n_targets) so
        # n_components cannot be bigger than that.
        n_components = self.n_components
        rank_upper_bound = min(X.shape[0], X.shape[1], Y.shape[1])
        check_scalar(
            n_components,
            "n_components",
            numbers.Integral,
            min_val=1,
            max_val=rank_upper_bound,
        )

        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, Y, self.scale, self.demean
        )

        # Compute SVD of cross-covariance matrix
        C = np.dot(X.T, Y)
        U, s, Vt = svd(C, full_matrices=False)
        U = U[:, :n_components]
        Vt = Vt[:n_components]
        U, Vt = svd_flip(U, Vt)
        V = Vt.T

        self.x_weights_ = U
        self.y_weights_ = V
        self._n_features_out = self.x_weights_.shape[1]
        return self

    def transform(self, X: NDArray, Y: NDArray | None = None):
        """Apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to be transformed.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        x_scores : array-like or tuple of array-like
            The transformed data `X_tranformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        check_is_fitted(self)
        X = validate_data(self, X, dtype=np.float64, reset=False)
        Xr = (X - self._x_mean) / self._x_std
        x_scores = np.dot(Xr, self.x_weights_)
        if Y is not None:
            Y = check_array(Y, input_name="Y", ensure_2d=False, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Yr = (Y - self._y_mean) / self._y_std
            y_scores = np.dot(Yr, self.y_weights_)
            return x_scores, y_scores
        return x_scores

    def fit_transform(self, X: NDArray, y: NDArray | None = None):
        """Learn and apply the dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets), \
                default=None
            Targets.

        Returns
        -------
        out : array-like or tuple of array-like
            The transformed data `X_tranformed` if `Y is not None`,
            `(X_transformed, Y_transformed)` otherwise.
        """
        return self.fit(X, y).transform(X, y)  # type: ignore[arg-type]
