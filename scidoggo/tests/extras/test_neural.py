"""Tests for the neural two-layer model (requires the [neural] extra)."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("skorch")

from scidoggo.neural import create_model  # noqa: E402

pytestmark = pytest.mark.neural


def test_create_model_smoke_fit():
    """A tiny create_model instance fits without error and predicts."""
    rng = np.random.default_rng(0)
    n_samples, n_in = 40, 4
    X = rng.standard_normal((n_samples, n_in)).astype(np.float32)
    y = rng.standard_normal((n_samples, 1)).astype(np.float32)

    model = create_model(
        max_epochs=2,
        batch_size=10,
        verbose=False,
        module__n_in=n_in,
        module__n_out=1,
        module__n_hidden=2,
        train_split=None,
    )
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape[0] == n_samples
