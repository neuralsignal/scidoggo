"""Two-layer encoding network.

This module defines a bias-free two-layer neural network (:class:`TwoLayerNoBias`),
a scaling-aware mean-squared-error loss (:class:`ScaledMSE`), and helper routines
to compute the optimal linear readout weights and optional output thresholding.
A convenience constructor (:func:`create_model`) wraps the network in a
scikit-learn-compatible :class:`skorch.NeuralNetRegressor`.

This is a leaf module of the optional ``[neural]`` extra and therefore imports
its heavy third-party dependencies (``torch`` and ``skorch``) directly at module
top. Import errors are surfaced with an install hint by ``scidoggo.neural``.
"""

from typing import Any

import numpy as np
import skorch
import torch
from scipy.linalg import lstsq
from scipy.optimize import curve_fit, nnls
from sklearn.linear_model import LinearRegression
from torch import cosine_similarity, nn
from torch.nn.functional import normalize

__all__ = [
    "NONLINS",
    "TwoLayerNoBias",
    "ScaledMSE",
    "optimal_input_scaling",
    "scale_threshold_function",
    "optimal_scale_and_thresholding",
    "create_model",
]


NONLINS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "identity": nn.Identity,
    "relu": nn.ReLU,
}


def optimal_input_scaling(
    X: np.ndarray,
    Y: np.ndarray,
    mask_zero: bool,
    nonneg: bool,
) -> np.ndarray:
    """Compute the optimal linear readout weights mapping ``X`` onto ``Y``.

    Parameters
    ----------
    X : numpy.ndarray
        Input array of shape ``(n_samples, n_dim)``.
    Y : numpy.ndarray
        Target array of shape ``(n_samples, n_neurons)``.
    mask_zero : bool
        If ``True``, fit each output column independently using only the rows
        where the corresponding target value is non-zero. If ``False``, fit all
        columns jointly with a single least-squares (or non-negative
        least-squares) solve.
    nonneg : bool
        If ``True``, constrain the weights to be non-negative.

    Returns
    -------
    numpy.ndarray
        Weight matrix of shape ``(n_dim, n_neurons)`` mapping ``X`` to ``Y``.
    """
    # TODO efficiency?
    if not mask_zero:
        linear_model = LinearRegression(fit_intercept=False, copy_X=False, positive=nonneg)
        return linear_model.fit(X, Y).coef_.T

    W = np.zeros((X.shape[1], Y.shape[1]))
    for idx, y_ in enumerate(Y.T):
        nonzero = y_ != 0
        if not np.any(nonzero):
            continue
        y_ = y_[nonzero]
        X_ = X[nonzero]

        if nonneg:
            W[:, idx] = nnls(X_, y_)[0]
        else:
            W[:, idx] = lstsq(X_, y_)[0]
    return W


def scale_threshold_function(
    ypred: np.ndarray,
    a: float,
    ymin: float | None,
    ymax: float | None,
) -> np.ndarray:
    """Scale predictions by ``a`` and clip them to a scaled range.

    Parameters
    ----------
    ypred : numpy.ndarray
        Predicted values to scale and threshold.
    a : float
        Multiplicative scaling factor applied to ``ypred`` (and to the bounds).
    ymin : float or None
        Lower bound (before scaling). If ``None``, no lower clipping is applied.
    ymax : float or None
        Upper bound (before scaling). If ``None``, no upper clipping is applied.

    Returns
    -------
    numpy.ndarray
        The scaled and thresholded predictions.
    """
    ypred = a * ypred
    if ymin is not None:
        ypred[ypred < a * ymin] = a * ymin
    if ymax is not None:
        ypred[ypred > a * ymax] = a * ymax
    return ypred


def optimal_scale_and_thresholding(
    X: np.ndarray,
    Y: np.ndarray,
    mask_zero: bool,
    nonneg: bool,
    fit_ymin: bool,
    fit_ymax: bool,
) -> tuple[np.ndarray, Any, Any, Any]:
    """Compute optimal readout weights together with per-output scale and bounds.

    First the optimal linear readout weights are computed via
    :func:`optimal_input_scaling`. Then, for each output column, an optional
    scaling factor and lower/upper threshold are fit with
    :func:`scipy.optimize.curve_fit`.

    Parameters
    ----------
    X : numpy.ndarray
        Input array of shape ``(n_samples, n_dim)``.
    Y : numpy.ndarray
        Target array of shape ``(n_samples, n_neurons)``.
    mask_zero : bool
        If ``True``, ignore rows where the target is zero when fitting both the
        readout weights and the per-output scaling/thresholding.
    nonneg : bool
        If ``True``, constrain the readout weights to be non-negative.
    fit_ymin : bool
        If ``True``, fit a per-output lower threshold.
    fit_ymax : bool
        If ``True``, fit a per-output upper threshold.

    Returns
    -------
    W : numpy.ndarray
        Weight matrix of shape ``(n_dim, n_neurons)``.
    scalings : numpy.ndarray or int
        Per-output scaling factors of shape ``(n_neurons,)``. Returns the scalar
        ``1`` when neither threshold is fit.
    lbs : numpy.ndarray or float
        Per-output lower bounds of shape ``(n_neurons,)``. Returns ``-inf`` when
        neither threshold is fit.
    ubs : numpy.ndarray or float
        Per-output upper bounds of shape ``(n_neurons,)``. Returns ``inf`` when
        neither threshold is fit.
    """
    # scaling and thresholding
    W = optimal_input_scaling(X, Y, mask_zero=mask_zero, nonneg=nonneg)
    if not fit_ymin and not fit_ymax:
        return W, 1, -np.inf, np.inf

    Ypred = X @ W

    if not fit_ymin:
        curve_function = lambda ypred, a, ymax: scale_threshold_function(
            ypred, a=a, ymin=None, ymax=ymax
        )
    else:
        curve_function = lambda ypred, a, ymin, ymax: scale_threshold_function(
            ypred, a=a, ymin=ymin, ymax=ymax
        )

    scalings = np.ones(Y.shape[1])
    lbs = scalings * -np.inf
    ubs = scalings * np.inf

    for idx, (ypred, y) in enumerate(zip(Ypred.T, Y.T)):
        if mask_zero:
            nonzero = y != 0
            if not np.any(nonzero):
                continue
            y = y[nonzero]
            ypred = ypred[nonzero]

        p0 = [1.0]
        lb = [0.0]
        ub = [np.inf]

        ymin = np.min([y, ypred])
        ymax = np.max([y, ypred])

        if fit_ymin:
            p0.append(ymin)
            lb.append(-np.inf)
            ub.append(ymax)

        if fit_ymax:
            p0.append(ymax)
            lb.append(ymin)
            ub.append(np.inf)

        popt, _ = curve_fit(curve_function, ypred, y, p0=p0, bounds=(lb, ub))

        scalings[idx] = popt[0]
        if fit_ymin:
            lbs[idx] = popt[1]
            if fit_ymax:
                ubs[idx] = popt[2]
        else:
            ubs[idx] = popt[1]

    return W, scalings, lbs, ubs


# -- Loss functions


class ScaledMSE(nn.MSELoss):
    """Mean-squared-error loss with an adaptive linear readout and orthogonality penalty.

    Before computing the MSE, the network output is projected through the
    optimal (adaptively recomputed, gradient-free) linear readout weights, with
    optional per-output scaling and thresholding. An optional penalty on the
    pairwise cosine similarity between latent dimensions encourages orthogonal
    representations.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`torch.nn.MSELoss`.
    mask_zero : bool, default=False
        If ``True``, ignore zero-valued targets when fitting the readout and
        when computing the loss.
    nonneg : bool, default=False
        If ``True``, constrain the fitted readout weights to be non-negative.
    alpha : float, default=0
        Weight of the pairwise cosine-similarity orthogonality penalty. The
        penalty is only applied when ``alpha`` is truthy and the latent space
        has more than one dimension.
    fit_ymin : bool, default=False
        If ``True``, fit a per-output lower threshold for the readout.
    fit_ymax : bool, default=False
        If ``True``, fit a per-output upper threshold for the readout.
    **kwargs
        Keyword arguments forwarded to :class:`torch.nn.MSELoss`.
    """

    def __init__(
        self,
        *args: Any,
        mask_zero: bool = False,
        nonneg: bool = False,
        alpha: float = 0,
        fit_ymin: bool = False,
        fit_ymax: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mask_zero = mask_zero
        self.nonneg = nonneg
        self.alpha = alpha
        self.fit_ymin = fit_ymin
        self.fit_ymax = fit_ymax

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the scaled MSE plus the orthogonality penalty.

        Parameters
        ----------
        input : torch.Tensor
            Latent representation of shape ``(n_samples, n_dim)``. The last
            dimension is assumed to be a linear combination of the targets.
        target : torch.Tensor
            Target tensor of shape ``(n_samples, n_neurons)``.

        Returns
        -------
        torch.Tensor
            The scalar loss value (MSE plus the optional orthogonality penalty).
        """
        # input (n_samples x n_dim), target (n_samples x n_neurons)
        # assuming the last dimensions should just best a linear combinations
        ndim = input.shape[1]
        if self.alpha and ndim > 1:
            # TODO efficiency?
            sims = cosine_similarity(input[:, None, :], input[:, :, None], dim=0)
            # must be orthogonal!
            sims = torch.abs(sims)
            # sum up all pairwise cosine similarities
            reg = torch.triu(sims, diagonal=1).sum()
        else:
            reg = 0

        # no gradients here - this is adaptively adjusted
        if not self.fit_ymin and not self.fit_ymax:
            W = optimal_input_scaling(
                input.detach().numpy(),
                target.detach().numpy(),
                mask_zero=self.mask_zero,
                nonneg=self.nonneg,
            )
            W = torch.tensor(W, dtype=torch.float32)
            input = input @ W
        else:
            W, scalings, lbs, ubs = optimal_scale_and_thresholding(
                input.detach().numpy(),
                target.detach().numpy(),
                mask_zero=self.mask_zero,
                nonneg=self.nonneg,
                fit_ymin=self.fit_ymin,
                fit_ymax=self.fit_ymax,
            )
            W = torch.tensor(W, dtype=torch.float32)
            scalings = torch.tensor(scalings, dtype=torch.float32)
            lbs = torch.tensor(lbs, dtype=torch.float32)
            ubs = torch.tensor(ubs, dtype=torch.float32)

            input = (input @ W) * scalings
            input = input.clamp(scalings * lbs, scalings * ubs)

        if self.mask_zero:
            ybool = target != 0
            input = input * ybool.float()
        return super().forward(input, target) + reg


class TwoLayerNoBias(nn.Module):
    """Two-layer fully connected network without bias terms.

    The network applies a hidden linear layer with a configurable
    non-linearity, followed by an output linear layer with its own
    non-linearity. Both layers omit bias terms and optionally row-normalize
    their weights at forward time. Layer weights can be initialized from given
    arrays and optionally frozen.

    Parameters
    ----------
    n_in : int
        Number of input features.
    n_out : int
        Number of output features.
    n_hidden : int, default=1
        Number of hidden units.
    nonlin : str, default='relu'
        Name of the hidden-layer non-linearity. One of the keys of
        :data:`NONLINS`.
    output_nonlin : str, default='identity'
        Name of the output-layer non-linearity. One of the keys of
        :data:`NONLINS`.
    linear_weights : array-like or None, default=None
        Initial weights for the hidden linear layer. If ``None``, default
        initialization is used.
    linear_init : bool, default=False
        Whether the hidden-layer weights require gradients when initialized from
        ``linear_weights``.
    output_weights : array-like or None, default=None
        Initial weights for the output linear layer. If ``None``, default
        initialization is used.
    output_init : bool, default=False
        Whether the output-layer weights require gradients when initialized from
        ``output_weights``.
    linear_norm : bool, default=False
        If ``True``, row-normalize the hidden-layer weights at forward time.
    output_norm : bool, default=False
        If ``True``, row-normalize the output-layer weights at forward time.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: int = 1,
        nonlin: str = "relu",
        output_nonlin: str = "identity",
        linear_weights: np.ndarray | None = None,
        linear_init: bool = False,
        output_weights: np.ndarray | None = None,
        output_init: bool = False,
        linear_norm: bool = False,
        output_norm: bool = False,
    ) -> None:
        super().__init__()

        self.linear_norm = linear_norm
        self.output_norm = output_norm

        self.linear = nn.Linear(n_in, n_hidden, bias=False)
        self.nonlin = NONLINS[nonlin]()

        self.output = nn.Linear(n_hidden, n_out, bias=False)
        self.output_nonlin = NONLINS[output_nonlin]()

        self._modify_weights(self.linear.weight, linear_weights, grad=linear_init)

        self._modify_weights(self.output.weight, output_weights, grad=output_init)

        self.output_nonlin = NONLINS[output_nonlin]()

    @staticmethod
    def _modify_weights(
        param: nn.Parameter,
        weights: np.ndarray | None,
        grad: bool,
    ) -> None:
        """Optionally overwrite a parameter's data and gradient requirement.

        Parameters
        ----------
        param : torch.nn.Parameter
            The parameter to (possibly) overwrite.
        weights : array-like or None
            Replacement weights. If ``None``, the parameter is left unchanged.
        grad : bool
            Whether the parameter should require gradients after being set.
        """
        if weights is None:
            pass
        else:
            param.data = torch.tensor(weights, dtype=torch.float32)
            param.requires_grad = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass through both layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(n_samples, n_in)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(n_samples, n_out)``.
        """
        if self.linear_norm:
            weight = normalize(self.linear.weight, dim=1)
            y = self.nonlin(torch.mm(x, weight.t()))
        else:
            y = self.nonlin(self.linear(x))
        if self.output_norm:
            weight = normalize(self.output.weight, dim=1)
            return self.output_nonlin(torch.mm(y, weight.t()))
        else:
            return self.output_nonlin(self.output(y))


def create_model(
    module: type[nn.Module] = TwoLayerNoBias,
    lr: float = 0.001,
    criterion: type[nn.Module] = ScaledMSE,
    criterion__nonneg: bool = True,
    criterion__reduction: str = "sum",
    criterion__mask_zero: bool = True,
    optimizer__momentum: float = 0.95,
    max_epochs: int = 1000,
    batch_size: int = 50,
    predict_nonlinearity: Any | None = None,
    module__nonlin: str = "relu",
    train_split: Any | None = None,
    verbose: bool = True,
    module__n_in: int = 4,
    module__n_out: int = 1,
    module__n_hidden: int = 2,
    module__output_weights: np.ndarray = np.array([[1, -1]]),
    **kwargs: Any,
) -> "skorch.NeuralNetRegressor":
    """Create a scikit-learn-compatible estimator wrapping the two-layer network.

    Parameters
    ----------
    module : type[torch.nn.Module], default=TwoLayerNoBias
        The network module class to train.
    lr : float, default=0.001
        Learning rate.
    criterion : type[torch.nn.Module], default=ScaledMSE
        The loss-function class.
    criterion__nonneg : bool, default=True
        Value forwarded to ``ScaledMSE.nonneg``.
    criterion__reduction : str, default='sum'
        Reduction mode forwarded to the criterion.
    criterion__mask_zero : bool, default=True
        Value forwarded to ``ScaledMSE.mask_zero``.
    optimizer__momentum : float, default=0.95
        Momentum forwarded to the optimizer.
    max_epochs : int, default=1000
        Maximum number of training epochs.
    batch_size : int, default=50
        Mini-batch size.
    predict_nonlinearity : callable or None, default=None
        Non-linearity applied at prediction time (forwarded to skorch).
    module__nonlin : str, default='relu'
        Hidden-layer non-linearity for the module.
    train_split : callable or None, default=None
        Train/validation split callable (forwarded to skorch).
    verbose : bool, default=True
        Whether skorch prints training progress.
    module__n_in : int, default=4
        Number of input features for the module.
    module__n_out : int, default=1
        Number of output features for the module.
    module__n_hidden : int, default=2
        Number of hidden units for the module.
    module__output_weights : numpy.ndarray, default=numpy.array([[1, -1]])
        Initial output-layer weights for the module.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`skorch.NeuralNetRegressor`.

    Returns
    -------
    skorch.NeuralNetRegressor
        The configured estimator.
    """
    return skorch.NeuralNetRegressor(
        module,
        lr=lr,
        criterion=criterion,
        criterion__nonneg=criterion__nonneg,
        criterion__reduction=criterion__reduction,
        criterion__mask_zero=criterion__mask_zero,
        optimizer__momentum=optimizer__momentum,
        max_epochs=max_epochs,
        batch_size=batch_size,
        predict_nonlinearity=predict_nonlinearity,
        module__nonlin=module__nonlin,
        train_split=train_split,
        verbose=verbose,
        module__n_in=module__n_in,
        module__n_out=module__n_out,
        module__n_hidden=module__n_hidden,
        module__output_weights=module__output_weights,
        **kwargs,
    )
