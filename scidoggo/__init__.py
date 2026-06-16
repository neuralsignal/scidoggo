"""scidoggo: a unified collection of scientific models with a scikit-learn API.

The top-level namespace exposes only the lightweight estimators and utilities
that depend solely on numpy/scipy/scikit-learn. Heavier models live in optional
subpackages installed via extras and raise a clear ``MissingDependencyError``
when their dependency is absent:

* ``scidoggo.neural`` -- ``pip install "scidoggo[neural]"`` (torch + skorch)
* ``scidoggo.circuit_models`` -- ``pip install "scidoggo[circuits]"`` (pyro)
* ``scidoggo.deep_implicit_circuits`` -- ``pip install "scidoggo[deep]"``
* ``scidoggo.linear_model.Rank1PlusSparse`` -- ``pip install "scidoggo[sparse]"`` (cvxpy)
"""

from ._version import __version__
from .cross_decomposition import CCA, PLSSVD, PLSCanonical, PLSRegression
from .interpolation import RbfRegression
from .linear_model import RankConstraint, SelectivityModel, Tikhonov
from .resampling import bs_cis, draw_bs_replicates, sig_directional, sig_overlap

__all__ = [
    "PLSRegression",
    "PLSCanonical",
    "CCA",
    "PLSSVD",
    "Tikhonov",
    "RankConstraint",
    "SelectivityModel",
    "RbfRegression",
    "draw_bs_replicates",
    "bs_cis",
    "sig_directional",
    "sig_overlap",
    "__version__",
]
