"""Verify heavy subpackages raise MissingDependencyError when deps are absent.

Each case is only asserted when the underlying dependency is actually missing,
so the test is correct in both environments (with and without the extras).
"""

import importlib

import pytest

from scidoggo.exceptions import MissingDependencyError


def _dep_present(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# (subpackage to import, dependency that gates it)
CASES = [
    ("scidoggo.neural", "torch"),
    ("scidoggo.circuit_models", "pyro"),
    ("scidoggo.deep_implicit_circuits", "pytorch_lightning"),
    ("scidoggo.linear_model.rank_plus_sparse", "cvxpy"),
]


@pytest.mark.parametrize("subpackage,dependency", CASES)
def test_missing_dependency_raises(subpackage, dependency):
    if _dep_present(dependency):
        pytest.skip(f"{dependency} is installed; MissingDependencyError not expected")
    with pytest.raises(MissingDependencyError):
        importlib.import_module(subpackage)
