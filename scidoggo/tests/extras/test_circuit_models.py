"""Tests for the circuit_models subpackage (requires the [circuits] extra)."""

import pytest

pytest.importorskip("pyro")

import scidoggo.circuit_models as cm  # noqa: E402

pytestmark = pytest.mark.circuits


def test_smoke_construct_model():
    """Importing the subpackage and constructing one model works."""
    # Tanh is a simple PyroModule with all-default args
    model = cm.Tanh()
    assert model is not None
