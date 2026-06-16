"""Two-layer neural encoding model (optional ``[neural]`` extra).

This subpackage requires the heavy optional dependencies ``torch`` and
``skorch``. They are imported lazily here: importing this package attempts to
load the leaf module :mod:`scidoggo.neural.two_layer`, and if the optional
dependencies are missing the resulting :class:`ImportError` is translated into a
:class:`~scidoggo.exceptions.MissingDependencyError` carrying a clear install
hint.
"""

from scidoggo.exceptions import MissingDependencyError

try:
    from .two_layer import (
        NONLINS,
        ScaledMSE,
        TwoLayerNoBias,
        create_model,
        optimal_input_scaling,
        optimal_scale_and_thresholding,
        scale_threshold_function,
    )
except ImportError as exc:
    raise MissingDependencyError(
        'The neural model requires the [neural] extra. '
        'Install with: pip install "scidoggo[neural]"'
    ) from exc

__all__ = [
    "NONLINS",
    "TwoLayerNoBias",
    "ScaledMSE",
    "optimal_input_scaling",
    "scale_threshold_function",
    "optimal_scale_and_thresholding",
    "create_model",
]
