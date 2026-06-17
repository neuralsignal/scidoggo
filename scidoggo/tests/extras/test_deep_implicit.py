"""Tests for the deep_implicit_circuits subpackage (requires the [deep] extra)."""

import pytest

pytest.importorskip("pytorch_lightning")

import scidoggo.deep_implicit_circuits as dic  # noqa: E402

pytestmark = pytest.mark.deep


def test_import_exposes_public_api():
    """The subpackage imports and exposes its documented public symbols."""
    for name in ("Circuit", "LitModel", "WeightDict", "mse_loss"):
        assert hasattr(dic, name)
