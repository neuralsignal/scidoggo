"""
==========================================
Tikhonov regression on synthetic data
==========================================

This example fits :class:`scidoggo.Tikhonov` to a synthetic regression problem
and compares the recovered coefficients to the ground-truth ones. ``Tikhonov``
extends scikit-learn's ``Ridge`` and, with ``L=None``, reduces to ordinary
ridge regression. We show how increasing the regularization strength ``alpha``
shrinks the coefficient estimates.
"""

import matplotlib

matplotlib.use("Agg")  # run headless

import matplotlib.pyplot as plt
import numpy as np

from scidoggo import Tikhonov

# %%
# Generate synthetic data
# -----------------------
# A linear problem with a sparse-ish ground-truth coefficient vector and a bit
# of observation noise.
rng = np.random.default_rng(0)
n_samples, n_features = 200, 8
X = rng.standard_normal((n_samples, n_features))
true_coef = np.array([4.0, -3.0, 0.0, 2.0, 0.0, -1.5, 0.0, 0.5])
y = X @ true_coef + 0.5 * rng.standard_normal(n_samples)

# %%
# Fit Tikhonov (ridge) regression for a few values of alpha
# ---------------------------------------------------------
alphas = [0.1, 1.0, 10.0]
models = {alpha: Tikhonov(alpha=alpha).fit(X, y) for alpha in alphas}

for alpha, model in models.items():
    y_pred = model.predict(X)
    r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"alpha={alpha:>5}: R^2 = {r2:.3f}")

# %%
# Plot the recovered coefficients
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
idx = np.arange(n_features)
width = 0.2

ax.bar(idx - 1.5 * width, true_coef, width, label="true", color="black")
for i, (alpha, model) in enumerate(models.items()):
    ax.bar(
        idx + (i - 0.5) * width,
        np.ravel(model.coef_),
        width,
        label=f"alpha={alpha}",
    )

ax.set_xlabel("feature index")
ax.set_ylabel("coefficient value")
ax.set_title("Tikhonov coefficients vs. ground truth")
ax.axhline(0.0, color="gray", linewidth=0.8)
ax.legend()
fig.tight_layout()

plt.show()
