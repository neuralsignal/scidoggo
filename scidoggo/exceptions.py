"""Custom exception classes for scidoggo.

These provide explicit, catchable error types so callers never have to rely on
bare ``Exception`` and so optional-dependency failures surface a clear install
hint instead of an opaque ``ImportError``.
"""

__all__ = [
    "ScidoggoError",
    "MissingDependencyError",
    "ConvergenceError",
    "InvalidParameterError",
]


class ScidoggoError(Exception):
    """Base class for all scidoggo-specific errors."""


class MissingDependencyError(ScidoggoError, ImportError):
    """Raised when an optional extra is required but not installed.

    The message should name the extra to install, e.g.
    ``pip install "scidoggo[circuits]"``.
    """


class ConvergenceError(ScidoggoError):
    """Raised when an iterative solver fails to converge within ``max_iter``."""


class InvalidParameterError(ScidoggoError, ValueError):
    """Raised for invalid non-array hyperparameters.

    Used for configuration validation where scikit-learn's array validation
    machinery does not apply. Array/shape validation should continue to raise
    the ``ValueError`` produced by ``sklearn.utils.validation``.
    """
