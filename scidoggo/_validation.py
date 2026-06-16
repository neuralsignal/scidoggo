"""Typed thin wrappers around scikit-learn's validation helpers.

scikit-learn types ``validate_data``/``check_X_y``/``check_array`` as returning a
broad ``ArrayLike`` union, which forces every downstream array attribute access
(``.ndim``, ``.shape``, indexing, arithmetic) to fail static type checking. In
practice these helpers always return concrete arrays for our usage, so these
wrappers narrow the return type to :class:`numpy.ndarray` at the single
validation boundary (DRY) rather than casting at every call site.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.utils.validation import check_X_y as _check_X_y
from sklearn.utils.validation import validate_data as _validate_data

__all__ = ["validate_xy", "validate_x", "check_xy"]


def validate_xy(estimator: Any, X: Any, y: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``X`` and ``y`` via ``validate_data``, returning ndarrays."""
    return cast(
        "tuple[np.ndarray, np.ndarray]",
        _validate_data(estimator, X, y, **kwargs),
    )


def validate_x(estimator: Any, X: Any, **kwargs: Any) -> np.ndarray:
    """Validate ``X`` via ``validate_data``, returning an ndarray."""
    return cast("np.ndarray", _validate_data(estimator, X, **kwargs))


def check_xy(X: Any, y: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``X`` and ``y`` via ``check_X_y``, returning ndarrays."""
    return cast("tuple[np.ndarray, np.ndarray]", _check_X_y(X, y, **kwargs))
