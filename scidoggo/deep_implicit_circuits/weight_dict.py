"""Sparse matrix/vector parameterizations backed by Python dictionaries.

This module holds the conversion helpers that translate dense matrices and
vectors into the sparse ``dict`` format consumed by the circuit models, along
with the :class:`WeightDict` and :class:`ValuesDict` containers that turn those
sparse specifications into trainable parameter tensors.

The ``dict`` format keys are either single indices / index pairs (for a single
entry) or tuples of indices (for an entry shared by several connections). The
values are either a fixed :class:`numbers.Number` or a ``(lower, upper)`` /
``(lower, upper, init)`` tuple describing a trainable parameter.
"""

from numbers import Number

import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from torch import nn

__all__ = [
    "matrix_to_weightdict",
    "values_to_dict",
    "WeightDict",
    "ValuesDict",
]


def matrix_to_weightdict(
    W: np.ndarray | torch.Tensor | pd.DataFrame | list[list],
) -> dict[tuple[int, int], float]:
    """Convert a dense 2-D weight matrix into the sparse weight-dict format.

    Parameters
    ----------
    W : numpy.ndarray or torch.Tensor or pandas.DataFrame or list of list
        Dense matrix indexed as ``W[ipre, ipost]``.

    Returns
    -------
    dict of (int, int) to float
        Mapping from ``(ipre, ipost)`` index pairs to the corresponding value.
    """
    if isinstance(W, pd.DataFrame):
        W = W.values
    elif isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()

    weight_dict = {}
    for ipre, values in enumerate(W):
        for ipost, value in enumerate(values):
            weight_dict[(ipre, ipost)] = value

    return weight_dict


def values_to_dict(
    values: np.ndarray | torch.Tensor | pd.Series | list,
) -> dict[int, float]:
    """Convert a dense 1-D vector into the sparse values-dict format.

    Parameters
    ----------
    values : numpy.ndarray or torch.Tensor or pandas.Series or list
        Dense 1-D sequence of values.

    Returns
    -------
    dict of int to float
        Mapping from index to value.
    """
    if isinstance(values, pd.Series):
        values = values.values
    elif isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    values_dict = {}
    for i, value in enumerate(values):
        values_dict[i] = value

    return values_dict


class WeightDict:
    """Sparse parameterization of a 2-D weight matrix.

    Parses a weight-dict specification into the index bookkeeping needed to map
    a flat trainable parameter vector back onto a dense matrix, while keeping
    fixed entries constant.

    Parameters
    ----------
    weight_dict : dict
        Mapping from index pairs (or tuples of index pairs) to either a fixed
        :class:`numbers.Number` or a ``(lower, upper)`` / ``(lower, upper,
        init)`` bound specification.
    shape : tuple
        Shape ``(n_pre, n_post)`` of the dense matrix produced by
        :meth:`get_weights`.
    """

    def __init__(self, weight_dict: dict, shape: tuple):
        self.shape = shape

        fixed_idcs = []
        fixed_jdcs = []
        fixed_value = []
        idcs = []
        jdcs = []
        vidcs = []
        lower = []
        upper = []
        inits = []
        labels = []  # labels for unfixed
        n_params = 0
        for k, v in weight_dict.items():
            # fixed values
            if isinstance(v, Number) and isinstance(k[0], Number):
                fixed_value.append(v)
                fixed_idcs.append(k[0])
                fixed_jdcs.append(k[1])
                continue
            elif isinstance(v, Number):
                fixed_value.extend([v] * len(k[0]))
                fixed_idcs.extend(k[0])
                fixed_jdcs.extend(k[1])
                continue

            if isinstance(k[0], Number):
                idcs.append(k[0])
                jdcs.append(k[1])
                vidcs.append(n_params)

            else:
                idcs.extend(k[0])
                jdcs.extend(k[1])
                vidcs.extend([n_params] * len(k[0]))

            lower.append(v[0])
            upper.append(v[1])
            # add inits
            if len(v) == 3:
                inits.append(v[2])
            else:
                inits.append(np.nan)

            labels.append(k)
            n_params += 1

        self.fixed_idcs = np.array(fixed_idcs)
        self.fixed_jdcs = np.array(fixed_jdcs)
        self.fixed_value = np.array(fixed_value)

        self.idcs = np.array(idcs)
        self.jdcs = np.array(jdcs)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.inits = np.array(inits)
        self.labels = labels
        self.vidcs = np.array(vidcs)

        self.n_params = len(self.lower)

        assert len(idcs) == len(vidcs)

    def get_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Scatter a flat parameter vector into a dense weight matrix.

        Parameters
        ----------
        w : torch.Tensor
            Flat vector of the trainable parameter values.

        Returns
        -------
        torch.Tensor
            Dense matrix of shape ``self.shape`` with fixed and trainable
            entries filled in.
        """
        weights = torch.zeros(self.shape).to(w)
        if len(self.fixed_idcs):
            weights[self.fixed_idcs, self.fixed_jdcs] = torch.tensor(self.fixed_value).to(w)
        if len(self.idcs):
            weights[self.idcs, self.jdcs] = w[self.vidcs]
        return weights

    def clip_values(self, w: nn.Parameter) -> nn.Parameter:
        """Clip parameter values in-place to their ``[lower, upper]`` bounds.

        Parameters
        ----------
        w : torch.nn.Parameter
            Parameter vector to clip in-place.

        Returns
        -------
        torch.nn.Parameter
            The clipped parameter (same object as ``w``).
        """
        with torch.no_grad():
            w.clip_(torch.tensor(self.lower).to(w), torch.tensor(self.upper).to(w))
        return w

    def sample_values(self, w: nn.Parameter) -> nn.Parameter:
        """Initialize parameters from bounds or explicit init values in-place.

        Entries without an explicit init are drawn uniformly from
        ``[lower, upper]``; entries with an init are set to it.

        Parameters
        ----------
        w : torch.nn.Parameter
            Parameter vector to initialize in-place.

        Returns
        -------
        torch.nn.Parameter
            The initialized parameter (same object as ``w``).
        """
        isnull = np.isnan(self.inits)
        with torch.no_grad():
            w[isnull] = dist.Uniform(
                torch.tensor(self.lower[isnull]).to(w),
                torch.tensor(self.upper[isnull]).to(w),
            ).sample()
            w[~isnull] = torch.tensor(self.inits[~isnull]).to(w)
            return w


class ValuesDict:
    """Sparse parameterization of a 1-D values vector.

    Like :class:`WeightDict` but for a 1-D vector (e.g. per-neuron gains or
    offsets), with a configurable default for entries not present in the dict.

    Parameters
    ----------
    values_dict : dict
        Mapping from index (or tuple of indices) to either a fixed
        :class:`numbers.Number` or a ``(lower, upper)`` / ``(lower, upper,
        init)`` bound specification.
    length : int
        Length of the dense vector produced by :meth:`get_values`.
    default : float, optional
        Value used for indices absent from ``values_dict``. Defaults to ``0.0``.
    """

    def __init__(self, values_dict: dict, length: int, default: float = 0.0):
        self.length = length
        self.default = default

        fixed_idcs = []
        fixed_value = []
        idcs = []
        vidcs = []
        lower = []
        upper = []
        inits = []
        labels = []

        n_params = 0
        for k, v in values_dict.items():
            # fixed values
            if isinstance(v, Number) and isinstance(k, Number):
                fixed_value.append(v)
                fixed_idcs.append(k)
                continue
            elif isinstance(v, Number):
                fixed_value.extend([v] * len(k))
                fixed_idcs.extend(k)
                continue

            # non fixed values
            if isinstance(k, Number):
                idcs.append(k)
                vidcs.append(n_params)
            else:
                # same parameter for multiple neurons
                idcs.extend(k)
                vidcs.extend([n_params] * len(k))

            lower.append(v[0])
            upper.append(v[1])
            if len(v) == 3:
                inits.append(v[2])
            else:
                inits.append(np.nan)

            labels.append(k)
            n_params += 1

        assert len(lower) == n_params

        self.fixed_idcs = np.array(fixed_idcs)
        self.fixed_value = np.array(fixed_value)

        self.idcs = np.array(idcs)
        self.vidcs = np.array(vidcs)
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.inits = np.array(inits)
        self.labels = labels
        self.n_params = len(self.lower)

        assert len(idcs) == len(vidcs)

    def get_values(self, w: torch.Tensor) -> torch.Tensor:
        """Scatter a flat parameter vector into a dense values vector.

        Parameters
        ----------
        w : torch.Tensor
            Flat vector of the trainable parameter values.

        Returns
        -------
        torch.Tensor
            Dense vector of length ``self.length`` with the default, fixed, and
            trainable entries filled in.
        """
        values = torch.ones(self.length).to(w) * self.default
        if len(self.fixed_idcs):
            values[self.fixed_idcs] = torch.tensor(self.fixed_value).to(w)
        if len(self.idcs):
            values[self.idcs] = w[self.vidcs]
        return values

    def clip_values(self, w: nn.Parameter) -> nn.Parameter:
        """Clip parameter values in-place to their ``[lower, upper]`` bounds.

        Parameters
        ----------
        w : torch.nn.Parameter
            Parameter vector to clip in-place.

        Returns
        -------
        torch.nn.Parameter
            The clipped parameter (same object as ``w``).
        """
        with torch.no_grad():
            w.clip_(torch.tensor(self.lower).to(w), torch.tensor(self.upper).to(w))
        return w

    def sample_values(self, w: nn.Parameter) -> nn.Parameter:
        """Initialize parameters from bounds or explicit init values in-place.

        Entries without an explicit init are drawn uniformly from
        ``[lower, upper]``; entries with an init are set to it.

        Parameters
        ----------
        w : torch.nn.Parameter
            Parameter vector to initialize in-place.

        Returns
        -------
        torch.nn.Parameter
            The initialized parameter (same object as ``w``).
        """
        isnull = np.isnan(self.inits)
        with torch.no_grad():
            w[isnull] = dist.Uniform(
                torch.tensor(self.lower[isnull]).to(w),
                torch.tensor(self.upper[isnull]).to(w),
            ).sample()
            w[~isnull] = torch.tensor(self.inits[~isnull]).to(w)
            return w
