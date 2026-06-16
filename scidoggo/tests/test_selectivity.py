"""Tests for the SelectivityModel estimator."""

from sklearn.base import clone
from sklearn.metrics import r2_score

from scidoggo import SelectivityModel


def _data(rng, n_samples=80, n_features=4):
    X = rng.standard_normal((n_samples, n_features))
    w = rng.standard_normal(n_features)
    y = X @ w  # linearly learnable target
    return X, y


def test_fit_returns_self(rng):
    X, y = _data(rng)
    model = SelectivityModel()
    assert model.fit(X, y) is model


def test_predict_shape(rng):
    X, y = _data(rng)
    model = SelectivityModel().fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_clone(rng):
    """clone() reproduces an unfitted estimator with the same params."""
    model = SelectivityModel(eps=1e-4, linear_features=2)
    cloned = clone(model)
    assert isinstance(cloned, SelectivityModel)
    assert cloned.get_params() == model.get_params()


def test_r2_sane_on_learnable_target(rng):
    """On a learnable target the model should achieve a sane R^2."""
    X, y = _data(rng)
    # treat all features as linear so the target is exactly representable
    model = SelectivityModel(linear_features=list(range(X.shape[1]))).fit(X, y)
    assert r2_score(y, model.predict(X)) > 0.9
