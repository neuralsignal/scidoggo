"""Deep implicit recurrent circuit model and Lightning training wrapper.

This module defines the core :class:`Circuit` model -- a recurrent neural
circuit whose steady state is found by an Anderson-accelerated fixed-point
solver (:func:`anderson`) of the single-step update :func:`step_forward` -- and
the :class:`LitModel` PyTorch Lightning wrapper used to train it.

Sparse parameterizations live in :mod:`.weight_dict` and the nonlinearities /
losses live in :mod:`.losses`.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.autograd as autograd
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import r2_score

from .losses import mse_loss
from .weight_dict import (
    ValuesDict,
    WeightDict,
    matrix_to_weightdict,
    values_to_dict,
)

__all__ = [
    "step_forward",
    "anderson",
    "Circuit",
    "LitModel",
]


def step_forward(
    X: torch.Tensor,  # inputs to recurrent circuit
    Yt: torch.Tensor,  # state of recurrent circuit Y(t-1)
    Wi: torch.Tensor,  # input weights
    Wr: torch.Tensor,  # recurrent weights
    offset: torch.Tensor,  # offset
    gain: torch.Tensor,  # gain
    nonlin: Callable,  # nonlinearity
) -> torch.Tensor:
    """Apply one recurrent update step of the circuit dynamics.

    Parameters
    ----------
    X : torch.Tensor
        Inputs to the recurrent circuit.
    Yt : torch.Tensor
        Current state ``Y(t-1)`` of the recurrent circuit.
    Wi : torch.Tensor
        Input weight matrix.
    Wr : torch.Tensor
        Recurrent weight matrix.
    offset : torch.Tensor
        Per-neuron offset applied before the nonlinearity.
    gain : torch.Tensor
        Per-neuron gain applied before the nonlinearity.
    nonlin : callable
        Element-wise nonlinearity.

    Returns
    -------
    torch.Tensor
        Updated state ``Y(t)``.
    """
    Yt = torch.matmul(X, Wi) + torch.matmul(Yt, Wr)
    Yt = nonlin(gain * Yt - offset) + nonlin(offset)
    return Yt


class Circuit(nn.Module):
    """Recurrent circuit solved to its fixed point with Anderson acceleration.

    The circuit applies :func:`step_forward` repeatedly to find the steady-state
    activity for a given input, using :func:`anderson` for the forward solve and
    an implicit-differentiation backward hook for gradients.

    Parameters
    ----------
    n_neurons : int
        Number of recurrent neurons.
    n_inputs : int
        Number of input dimensions.
    winp_dict : dict
        Weight-dict specification for the input weights (shape
        ``(n_inputs, n_neurons)``).
    wrec_dict : dict
        Weight-dict specification for the recurrent weights (shape
        ``(n_neurons, n_neurons)``).
    offset_dict : dict, optional
        Values-dict specification for the per-neuron offsets. Defaults to ``{}``.
    gain_dict : dict, optional
        Values-dict specification for the per-neuron gains. Defaults to ``{}``.
    output_gain_dict : dict, optional
        Values-dict specification for the per-neuron output gains. Defaults to
        ``{}``.
    nonlin : callable, optional
        Element-wise nonlinearity. Defaults to :func:`torch.tanh`.
    normalize_weights : bool, optional
        Whether to L1-normalize the stacked input/recurrent weights per column.
        Defaults to ``True``.
    device : torch.device, optional
        Device for the parameter tensors.
    dtype : torch.dtype, optional
        Dtype for the parameter tensors.
    anderson_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`anderson`.
    """

    def __init__(
        self,
        n_neurons: int,
        n_inputs: int,
        winp_dict: dict,
        wrec_dict: dict,
        offset_dict: dict = {},
        gain_dict: dict = {},
        output_gain_dict: dict = {},
        nonlin: Callable = torch.tanh,
        normalize_weights: bool = True,
        device=None,
        dtype=None,
        anderson_kwargs: dict | None = None,
    ):
        super().__init__()
        self.normalize_weights = normalize_weights
        self.winp_dict = winp_dict
        self.wrec_dict = wrec_dict
        self.gain_dict = gain_dict
        self.output_gain_dict = output_gain_dict
        self.offset_dict = offset_dict
        self.anderson_kwargs = anderson_kwargs or {}
        # pre x post
        self.winp_dict_obj = WeightDict(winp_dict, (n_inputs, n_neurons))
        self.wrec_dict_obj = WeightDict(wrec_dict, (n_neurons, n_neurons))
        self.gain_dict_obj = ValuesDict(gain_dict, n_neurons, default=1.0)
        self.output_gain_dict_obj = ValuesDict(output_gain_dict, n_neurons, default=1.0)
        self.offset_dict_obj = ValuesDict(offset_dict, n_neurons, default=0.0)
        self.nonlin = nonlin
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

        self.hook = None

        # parameter objects
        factory_kwargs = dict(dtype=dtype, device=device)
        self._wi = nn.Parameter(torch.empty(self.winp_dict_obj.n_params, **factory_kwargs))
        self._wr = nn.Parameter(torch.empty(self.wrec_dict_obj.n_params, **factory_kwargs))
        self._gain = nn.Parameter(torch.empty(self.gain_dict_obj.n_params, **factory_kwargs))
        self._output_gain = nn.Parameter(
            torch.empty(self.output_gain_dict_obj.n_params, **factory_kwargs)
        )
        self._offset = nn.Parameter(torch.empty(self.offset_dict_obj.n_params, **factory_kwargs))

        self.reset_parameters()

        self.num_params = sum(param.numel() for param in self.parameters())

    @property
    def Wi(self) -> torch.Tensor:
        """torch.Tensor: Dense input weight matrix."""
        return self.winp_dict_obj.get_weights(self._wi)

    @property
    def Wr(self) -> torch.Tensor:
        """torch.Tensor: Dense recurrent weight matrix."""
        return self.wrec_dict_obj.get_weights(self._wr)

    @property
    def offset(self) -> torch.Tensor:
        """torch.Tensor: Dense per-neuron offsets."""
        return self.offset_dict_obj.get_values(self._offset)

    @property
    def gain(self) -> torch.Tensor:
        """torch.Tensor: Dense per-neuron gains."""
        return self.gain_dict_obj.get_values(self._gain)

    @property
    def output_gain(self) -> torch.Tensor:
        """torch.Tensor: Dense per-neuron output gains."""
        return self.output_gain_dict_obj.get_values(self._output_gain)

    def reset_parameters(self) -> None:
        """Re-initialize all trainable parameters in-place."""
        self.winp_dict_obj.sample_values(self._wi)
        self.wrec_dict_obj.sample_values(self._wr)
        self.offset_dict_obj.sample_values(self._offset)
        self.gain_dict_obj.sample_values(self._gain)
        self.output_gain_dict_obj.sample_values(self._output_gain)

    def clip_parameters(self) -> None:
        """Clip all trainable parameters (and the nonlinearity) to bounds."""
        self.winp_dict_obj.clip_values(self._wi)
        self.wrec_dict_obj.clip_values(self._wr)
        self.offset_dict_obj.clip_values(self._offset)
        self.gain_dict_obj.clip_values(self._gain)
        self.output_gain_dict_obj.clip_values(self._output_gain)
        if hasattr(self.nonlin, "clip_parameters"):
            self.nonlin.clip_parameters()

    def get_dict(self) -> dict:
        """Return the current parameters as weight/values-dict mappings.

        Returns
        -------
        dict
            Mapping with keys ``"winp_dict"``, ``"wrec_dict"``,
            ``"offset_dict"``, ``"gain_dict"`` and ``"output_gain_dict"`` (and
            ``"nonlin"`` if the nonlinearity exposes ``get_dict``).
        """
        d = {
            "winp_dict": matrix_to_weightdict(self.Wi),
            "wrec_dict": matrix_to_weightdict(self.Wr),
            "offset_dict": values_to_dict(self.offset),
            "gain_dict": values_to_dict(self.gain),
            "output_gain_dict": values_to_dict(self.output_gain),
        }
        if hasattr(self.nonlin, "get_dict"):
            d["nonlin"] = self.nonlin.get_dict()
        return d

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor | None = None,
        silence_inputs: list | None = None,
        silence_recs: list | None = None,
    ) -> torch.Tensor:
        """Solve the circuit to its fixed point for the given inputs.

        Parameters
        ----------
        X : torch.Tensor
            Batch of inputs, shape ``(batch, n_inputs)``.
        Y : torch.Tensor, optional
            Initial state. Defaults to zeros of shape ``(batch, n_neurons)``.
        silence_inputs : list of int, optional
            Input indices whose weights are zeroed before the solve.
        silence_recs : list of int, optional
            Recurrent indices whose weights are zeroed before the solve.

        Returns
        -------
        torch.Tensor
            Steady-state activity scaled by ``output_gain``.
        """
        if Y is None:
            Y = torch.zeros((X.shape[0], self.n_neurons)).to(X)

        Wi, Wr = self.Wi, self.Wr
        if silence_inputs is not None:
            Wi.index_fill_(0, torch.tensor(silence_inputs), 0.0)
        if silence_recs is not None:
            Wr.index_fill_(0, torch.tensor(silence_recs), 0.0)

        if self.normalize_weights:
            norm = torch.linalg.norm(torch.vstack([Wi, Wr]), ord=1, axis=0)
            Wi = Wi / norm
            Wr = Wr / norm

        offset = self.offset
        gain = self.gain
        nonlin = self.nonlin

        with torch.no_grad():
            Y = anderson(
                lambda Y: step_forward(
                    X,
                    Y,
                    Wi,
                    Wr,
                    offset=offset,
                    gain=gain,
                    nonlin=nonlin,
                ),
                Y,
                **self.anderson_kwargs,
            )["result"]
        z = step_forward(
            X,
            Y,
            Wi,
            Wr,
            offset=offset,
            gain=gain,
            nonlin=nonlin,
        )
        if self.num_params:
            # set up Jacobian vector product (without additional forward calls)
            z0 = z.requires_grad_()
            f0 = step_forward(
                X,
                z0,
                Wi,
                Wr,
                offset=offset,
                gain=gain,
                nonlin=nonlin,
            )

            if self.training:

                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                    new_grad = anderson(
                        lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                        torch.zeros_like(grad),
                        **self.anderson_kwargs,
                    )["result"]
                    return new_grad

                self.hook = z.register_hook(backward_hook)
        return z * self.output_gain


def anderson(
    f: Callable,
    x0: torch.Tensor,
    m: int = 6,
    lam: float = 1e-4,
    threshold: int = 50,
    eps: float = 1e-3,
    stop_mode: str = "rel",
    beta: float = 1.0,
    **kwargs,
) -> dict:
    """Anderson acceleration for fixed point iteration.

    Parameters
    ----------
    f : callable
        Fixed-point map; ``f(x)`` should return a tensor shaped like ``x``.
    x0 : torch.Tensor
        Initial guess, shape ``(batch, length)``.
    m : int, optional
        History window size. Defaults to ``6``.
    lam : float, optional
        Regularization added to the normal equations. Defaults to ``1e-4``.
    threshold : int, optional
        Maximum number of iterations. Defaults to ``50``.
    eps : float, optional
        Convergence tolerance on the ``stop_mode`` residual. Defaults to
        ``1e-3``.
    stop_mode : str, optional
        Residual mode used for the stopping criterion, ``"rel"`` or ``"abs"``.
        Defaults to ``"rel"``.
    beta : float, optional
        Mixing parameter. Defaults to ``1.0``.
    **kwargs
        Ignored extra keyword arguments (for call-site convenience).

    Returns
    -------
    dict
        Result dictionary with keys ``"result"``, ``"lowest"``, ``"nstep"``,
        ``"prot_break"``, ``"abs_trace"``, ``"rel_trace"``, ``"eps"`` and
        ``"threshold"``.
    """
    bsz, L = x0.shape
    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, L, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = (F[:, k % m] - X[:, k % m]).view_as(x0)
        abs_diff = gx.norm().item()
        rel_diff = abs_diff / (1e-5 + F[:, k % m].norm().item())
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode]:
                if mode == stop_mode:
                    lowest_xest = X[:, k % m].view_as(x0).clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(threshold - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {
        "result": lowest_xest,
        "lowest": lowest_dict[stop_mode],
        "nstep": lowest_step_dict[stop_mode],
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "threshold": threshold,
    }
    X = F = None
    return out


class LitModel(LightningModule):
    """PyTorch Lightning module for training a :class:`Circuit` model.

    Parameters
    ----------
    loss : callable, optional
        Loss function called as ``loss(Ypred, Y, mask=..., weight=...)``.
        Defaults to :func:`~.losses.mse_loss`.
    learning_rate : float, optional
        Optimizer learning rate. Defaults to ``1e-3``.
    optimizer_type : str, optional
        One of ``"adam"``, ``"sgd"`` or ``"adamw"``. Defaults to ``"adamw"``.
    opt_args : dict, optional
        Extra keyword arguments forwarded to the optimizer.
    schedule : str, optional
        Learning-rate schedule identifier. Defaults to ``"linear"``.
    warmup_steps : int, optional
        Number of warmup steps. Defaults to ``0``.
    total_steps : int, optional
        Total number of training steps. Defaults to ``1000``.
    model_kwargs : dict, optional
        Keyword arguments forwarded to the wrapped model constructor.
    """

    model_class = Circuit

    def __init__(
        self,
        loss: Callable = mse_loss,
        learning_rate: float = 1e-3,
        optimizer_type: str = "adamw",
        opt_args: dict | None = None,
        schedule: str = "linear",
        warmup_steps: int = 0,
        total_steps: int = 1000,
        model_kwargs: dict = {},
    ):
        super().__init__()
        self.loss = loss
        self.model_kwargs = model_kwargs
        self.model = self.model_class(**model_kwargs)
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.opt_args = opt_args if opt_args is not None else {}
        self.save_hyperparameters(ignore=["model", "loss_function"])

    def forward(self, X, Y, W=None):
        """Compute the loss and predictions for a batch.

        Parameters
        ----------
        X : torch.Tensor
            Inputs.
        Y : torch.Tensor
            Targets (non-finite entries are masked out of the loss).
        W : torch.Tensor, optional
            Per-entry sample weights.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            The scalar loss and the model predictions.
        """
        Ypred = self.model(X)
        loss = self.loss(Ypred, Y, mask=torch.isfinite(Y), weight=W)
        return loss, Ypred

    def log_r2s(self, Ypred, Y, W, prefix="train"):
        """Log per-target and mean weighted R^2 scores.

        Parameters
        ----------
        Ypred : torch.Tensor
            Model predictions.
        Y : torch.Tensor
            Targets.
        W : torch.Tensor
            Per-entry sample weights.
        prefix : str, optional
            Prefix for the logged metric names. Defaults to ``"train"``.
        """
        r2s = 0.0
        for i, (y, ypred, w) in enumerate(zip(Y.T, Ypred.T, W.T)):
            isfinite = torch.isfinite(y)
            if isfinite.sum() < 2:
                continue
            w = torch.sqrt(w[isfinite])
            r2 = r2_score(ypred[isfinite] * w, y[isfinite] * w)
            r2s += r2
            self.log(f"{prefix}_r2_{i}", r2)
        self.log(f"{prefix}_r2", r2s / Y.shape[-1])

    def training_step(self, batch, batch_idx):
        """Run a single training step, clip parameters and log metrics."""
        if len(batch) == 2:
            X, Y = batch
            W = None
        else:
            X, Y, W = batch

        loss, Ypred = self.forward(X, Y, W)

        self.model.clip_parameters()

        self.log("train_loss", loss)
        self.log_r2s(Ypred, Y, W, prefix="train")

        return loss

    def validation_step(self, batch, batch_idx):
        """Run a single validation step and log metrics."""
        if len(batch) == 2:
            X, Y = batch
            W = None
        else:
            X, Y, W = batch

        loss, Ypred = self.forward(X, Y, W)

        self.log("val_loss", loss)
        self.log_r2s(Ypred, Y, W, prefix="val")
        return loss

    def test_step(self, batch, batch_idx):
        """Run a single test step and log metrics."""
        if len(batch) == 2:
            X, Y = batch
            W = None
        else:
            X, Y, W = batch

        loss, Ypred = self.forward(X, Y, W)

        self.log("test_loss", loss)
        self.log_r2s(Ypred, Y, W, prefix="test")
        return loss

    def configure_optimizers(self):
        """Build the optimizer from ``optimizer_type``.

        Returns
        -------
        torch.optim.Optimizer
            The configured optimizer.

        Raises
        ------
        ValueError
            If ``optimizer_type`` is not recognized.
        """
        if self.optimizer_type == "adam":
            optimizer = Adam(self.parameters(), lr=self.learning_rate, **self.opt_args)
        elif self.optimizer_type == "sgd":
            optimizer = SGD(self.parameters(), lr=self.learning_rate, **self.opt_args)
        elif self.optimizer_type == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.learning_rate, **self.opt_args)
        else:
            raise ValueError("Optimizer type not recognized.")

        # TODO add scheduler for training
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     opt, max_epochs*len(train_loader), eta_min=1e-6
        # )
        return optimizer


if __name__ == "__main__":
    import time

    n = 64 * 4
    m = 4
    l = 6
    X = torch.randn((n, m))
    winp_dict = {(0, 1): 0.5, (1, 2): 0.5, (2, 4): 0.5, (3, 5): 0.5}
    wrec_dict = {
        (0, 1): -0.1,
        (1, 0): 0.1,
        (2, 3): -0.1,
        (3, 2): -0.1,
        (4, 5): -0.1,
        (5, 4): 0.1,
    }

    now = time.time()
    circuit = Circuit(l, m, winp_dict=winp_dict, wrec_dict=wrec_dict)

    Y = circuit.forward(X)

    print(circuit.parameters())
    print(time.time() - now)
    print(Y.shape)
    print(np.linalg.norm(Y))
    # print(circuit.forward_res[-1])

    wrec_dict = {
        (0, 1): (-1, 1),
        (1, 0): (-1, 1),
        (2, 3): (-1, 1),
        (3, 2): (-1, 1),
        (4, 5): (-1, 1),
        (5, 4): (-1, 1),
    }
    model_kwargs = dict(n_neurons=l, n_inputs=m, winp_dict=winp_dict, wrec_dict=wrec_dict)

    model = LitModel(model_kwargs=model_kwargs, learning_rate=5e-2)

    training_data = TensorDataset(X, Y)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    trainer = Trainer(max_epochs=100, log_every_n_steps=1)
    trainer.fit(model, train_dataloader)
