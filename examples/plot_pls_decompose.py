"""
=================================================
Decomposing PLS regression coefficients
=================================================

:class:`scidoggo.PLSRegression` adds a few helpers over the scikit-learn
estimator for inspecting a fitted model component by component. This example
fits a PLS regression on synthetic multi-output data and demonstrates:

* :meth:`~scidoggo.PLSRegression.kth_coef` -- the coefficient matrix
  reconstructed from only the first ``k`` latent components, and
* :meth:`~scidoggo.PLSRegression.decompose_coef` -- an orthogonal decomposition
  of the fitted coefficients into ``X`` rotations and ``Y`` loadings.
"""

import matplotlib

matplotlib.use("Agg")  # run headless

import matplotlib.pyplot as plt
import numpy as np

from scidoggo import PLSRegression

# %%
# Generate synthetic low-rank multi-output data
# ---------------------------------------------
# We build ``Y`` from ``X`` through a coefficient matrix that is the sum of a
# small number of rank-one terms, so that a low-component PLS model can recover
# most of the signal.
rng = np.random.default_rng(0)
n_samples, n_features, n_targets = 300, 10, 4

X = rng.standard_normal((n_samples, n_features))
# two dominant directions
u = rng.standard_normal((n_features, 2))
v = rng.standard_normal((n_targets, 2))
true_coef = (u @ v.T).T  # shape (n_targets, n_features)
Y = X @ true_coef.T + 0.1 * rng.standard_normal((n_samples, n_targets))

# %%
# Fit PLS regression and reconstruct coefficients per component
# -------------------------------------------------------------
pls = PLSRegression(n_components=4).fit(X, Y)

full_coef = pls.coef_
coef_1 = pls.kth_coef(1)  # using only the first component
coef_2 = pls.kth_coef(2)  # using the first two components

print("coef_ shape:", full_coef.shape)
print("kth_coef(1) shape:", coef_1.shape)


def rel_error(approx):
    return np.linalg.norm(approx - true_coef) / np.linalg.norm(true_coef)


for k in range(1, 5):
    print(f"k={k}: relative error to truth = {rel_error(pls.kth_coef(k)):.3f}")

# %%
# Orthogonal decomposition of the coefficients
# --------------------------------------------
# ``decompose_coef`` returns orthonormal X rotations and the corresponding
# Y loadings.
x_rot, y_load = pls.decompose_coef()
print("x_rotations_orth shape:", x_rot.shape)
print("y_loadings_orth shape:", y_load.shape)

# %%
# Visualise how reconstruction improves with more components
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
vmax = np.abs(true_coef).max()
kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

axes[0].imshow(true_coef, **kw)
axes[0].set_title("true coef")
axes[1].imshow(coef_1, **kw)
axes[1].set_title("kth_coef(1)")
im = axes[2].imshow(full_coef, **kw)
axes[2].set_title("full coef_")
for ax in axes:
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
fig.colorbar(im, ax=axes, shrink=0.8)

plt.show()
