"""Linear models with a scikit-learn API.

``Rank1PlusSparse`` requires the optional ``[sparse]`` extra (cvxpy) and is
exposed lazily so importing this subpackage never pulls in cvxpy.
"""

from typing import Any

from .rank_constrained import RankConstraint
from .selectivity import SelectivityModel
from .tikhonov import Tikhonov

__all__ = ["Tikhonov", "RankConstraint", "SelectivityModel", "Rank1PlusSparse"]


def __getattr__(name: str) -> Any:
    if name == "Rank1PlusSparse":
        from .rank_plus_sparse import Rank1PlusSparse

        return Rank1PlusSparse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
