"""Loss functions and nonlinearities for implicit circuit models.

This module collects the differentiable nonlinearity used by the recurrent
circuits (:func:`tanh_like` and its trainable :class:`TanhLike` module) along
with the masked, optionally weighted, mean-squared-error loss
(:func:`mse_loss`) used during training.
"""

import torch
from torch import nn

from .weight_dict import ValuesDict, values_to_dict

__all__ = [
    "tanh_like",
    "mse_loss",
    "TanhLike",
]


def tanh_like(
    x: torch.Tensor,
    r0: float = 0,
    a: float = 1,
    b: float = 1,
) -> torch.Tensor:
    """tanh-like nonlinearity from Rajan, Abbott, Sompolinsky (2010).

    A saturating nonlinearity with separate slopes for negative and positive
    inputs controlled by ``r0``, scaled by gain ``a`` and input scale ``b``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    r0 : float, optional
        Asymmetry parameter controlling the difference between the negative and
        positive branches. Defaults to ``0`` (symmetric ``tanh``).
    a : float, optional
        Output scale (gain). Defaults to ``1``.
    b : float, optional
        Input scale. Defaults to ``1``.

    Returns
    -------
    torch.Tensor
        Element-wise nonlinearity applied to ``x``.
    """
    # tanh = getattr(module, 'tanh')
    # zeros_like = getattr(module, 'zeros_like')
    y = torch.zeros_like(x)
    x = b * x

    y1 = (1 - r0) * torch.tanh(x / (1 - r0 + 1e-6))
    y2 = (1 + r0) * torch.tanh(x / (1 + r0 + 1e-6))

    y[x <= 0] = y1[x <= 0]
    y[x > 0] = y2[x > 0]

    return a * y


def mse_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-squared-error loss with optional masking and weighting.

    Parameters
    ----------
    input : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values.
    mask : torch.Tensor, optional
        Boolean mask selecting the entries to include in the loss. When given,
        ``input``, ``target`` and ``weight`` are restricted to the masked
        entries.
    weight : torch.Tensor, optional
        Per-entry weights applied to the squared residuals.

    Returns
    -------
    torch.Tensor
        Scalar mean of the (optionally weighted) squared residuals.
    """
    if mask is not None:
        input = input[mask]
        target = target[mask]
        if weight is not None:
            weight = weight[mask]
    if weight is None:
        res_squared = (input - target) ** 2
    else:
        res_squared = weight * (input - target) ** 2
    return res_squared.mean()


class TanhLike(nn.Module):
    """Trainable :func:`tanh_like` nonlinearity with per-neuron parameters.

    The ``r0``, ``a`` and ``b`` parameters of :func:`tanh_like` are each
    parameterized as sparse :class:`~.weight_dict.ValuesDict` vectors so that
    individual neurons can have fixed or trainable values.

    Parameters
    ----------
    length : int
        Number of neurons (length of each parameter vector).
    r0_dict : dict, optional
        Values-dict specification for the ``r0`` parameter. Defaults to ``{}``.
    a_dict : dict, optional
        Values-dict specification for the ``a`` parameter. Defaults to ``{}``.
    b_dict : dict, optional
        Values-dict specification for the ``b`` parameter. Defaults to ``{}``.
    device : torch.device, optional
        Device for the parameter tensors.
    dtype : torch.dtype, optional
        Dtype for the parameter tensors.
    """

    def __init__(
        self,
        length: int,
        r0_dict: dict = {},
        a_dict: dict = {},
        b_dict: dict = {},
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.length = length
        self.r0_dict = r0_dict
        self.a_dict = a_dict
        self.b_dict = b_dict
        self.r0_dict_obj = ValuesDict(r0_dict, length, default=0.0)
        self.a_dict_obj = ValuesDict(a_dict, length, default=1.0)
        self.b_dict_obj = ValuesDict(b_dict, length, default=1.0)

        factory_kwargs = dict(dtype=dtype, device=device)
        self._r0 = nn.Parameter(torch.empty(self.r0_dict_obj.n_params, **factory_kwargs))
        self._a = nn.Parameter(torch.empty(self.a_dict_obj.n_params, **factory_kwargs))
        self._b = nn.Parameter(torch.empty(self.b_dict_obj.n_params, **factory_kwargs))

        self.reset_parameters()

        self.num_params = sum(param.numel() for param in self.parameters())

    @property
    def r0(self) -> torch.Tensor:
        """torch.Tensor: Dense per-neuron ``r0`` values."""
        return self.r0_dict_obj.get_values(self._r0)

    @property
    def a(self) -> torch.Tensor:
        """torch.Tensor: Dense per-neuron ``a`` (gain) values."""
        return self.a_dict_obj.get_values(self._a)

    @property
    def b(self) -> torch.Tensor:
        """torch.Tensor: Dense per-neuron ``b`` (input scale) values."""
        return self.b_dict_obj.get_values(self._b)

    def reset_parameters(self) -> None:
        """Re-initialize all trainable parameters in-place."""
        self.r0_dict_obj.sample_values(self._r0)
        self.a_dict_obj.sample_values(self._a)
        self.b_dict_obj.sample_values(self._b)

    def clip_parameters(self) -> None:
        """Clip all trainable parameters to their bounds in-place."""
        self.r0_dict_obj.clip_values(self._r0)
        self.a_dict_obj.clip_values(self._a)
        self.b_dict_obj.clip_values(self._b)

    def get_dict(self) -> dict:
        """Return the current parameters as values-dict mappings.

        Returns
        -------
        dict
            Mapping with keys ``"r0_dict"``, ``"a_dict"`` and ``"b_dict"``, each
            a dense values-dict for the corresponding parameter.
        """
        return {
            "r0_dict": values_to_dict(self.r0),
            "a_dict": values_to_dict(self.a),
            "b_dict": values_to_dict(self.b),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the trainable :func:`tanh_like` nonlinearity.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise nonlinearity applied to ``x``.
        """
        return tanh_like(x, r0=self.r0, a=self.a, b=self.b)
