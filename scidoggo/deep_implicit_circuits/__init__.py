"""Deep implicit recurrent circuit models (the optional ``[deep]`` extra).

This subpackage requires the optional ``[deep]`` dependencies (``torch``,
``pytorch-lightning`` and ``torcheval``). Importing it without those installed
raises :class:`~scidoggo.exceptions.MissingDependencyError` with an install
hint instead of a raw :class:`ImportError`.
"""

from scidoggo.exceptions import MissingDependencyError

try:
    from .implicit_circuit import Circuit, LitModel, anderson, step_forward
    from .losses import TanhLike, mse_loss, tanh_like
    from .weight_dict import (
        ValuesDict,
        WeightDict,
        matrix_to_weightdict,
        values_to_dict,
    )
except ImportError as exc:
    raise MissingDependencyError(
        'deep_implicit_circuits requires the [deep] extra. '
        'Install with: pip install "scidoggo[deep]"'
    ) from exc

__all__ = [
    "Circuit",
    "LitModel",
    "anderson",
    "step_forward",
    "TanhLike",
    "mse_loss",
    "tanh_like",
    "WeightDict",
    "ValuesDict",
    "matrix_to_weightdict",
    "values_to_dict",
]
