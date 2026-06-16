"""Shared pytest fixtures for the scidoggo test suite."""

import numpy as np
import pytest
from sklearn.datasets import make_regression

RANDOM_STATE = 42


@pytest.fixture
def regression_data():
    """Return a fixed (X, y) regression dataset.

    Single-target, well-conditioned, low-noise so estimators can fit it.
    """
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        noise=0.1,
        random_state=RANDOM_STATE,
    )
    return X, y


@pytest.fixture
def rng():
    """Return a deterministic numpy random Generator."""
    return np.random.default_rng(RANDOM_STATE)


@pytest.fixture
def seeded_legacy_rng():
    """Seed numpy's legacy global RNG and yield.

    The bootstrap functions use ``np.random.randint`` (the legacy global
    state), so seeding it makes those functions deterministic for a test.
    """
    np.random.seed(RANDOM_STATE)
    yield
