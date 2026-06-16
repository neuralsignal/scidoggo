.. title:: User guide : contents

.. _user_guide:

==========
User Guide
==========

``scidoggo`` bundles a number of models and utilities that share a
scikit-learn API. This guide gives a short overview of the model families and
when to reach for each one.

Cross decomposition
===================

The :mod:`scidoggo.cross_decomposition` subpackage provides the Partial Least
Squares family (:class:`~scidoggo.PLSRegression`,
:class:`~scidoggo.PLSCanonical`, :class:`~scidoggo.CCA` and
:class:`~scidoggo.PLSSVD`), forked from scikit-learn. They are appropriate when
predictors are highly collinear or when you want a low-rank relationship
between a multivariate ``X`` and a multivariate ``Y``.

Two additions over the scikit-learn versions are worth highlighting:

* A ``demean`` option that lets you disable centering of ``X`` and ``Y`` before
  fitting (useful when the offset carries meaning).
* Helper methods on the regression estimators to inspect the contribution of
  each latent component:
  :meth:`~scidoggo.PLSRegression.kth_rotations`,
  :meth:`~scidoggo.PLSRegression.kth_coef` (the coefficients reconstructed from
  the first ``k`` components) and
  :meth:`~scidoggo.PLSRegression.decompose_coef` (an orthogonal decomposition
  of the fitted coefficients into ``X`` rotations and ``Y`` loadings).

Linear models
=============

The :mod:`scidoggo.linear_model` subpackage collects regularized linear models:

* :class:`~scidoggo.Tikhonov` -- generalized ridge regression. It extends
  scikit-learn's ``Ridge`` with a user-supplied regularization matrix ``L``,
  so the penalty ``L.T @ L`` can encode a non-diagonal prior covariance on the
  coefficients. Use it when you have structured prior knowledge about the
  coefficients (e.g. smoothness). The functional building blocks live in
  :mod:`scidoggo.linear_model.tikhonov_solver`
  (``analytic_tikhonov``, ``find_tikhonov_from_covariance``,
  ``to_standard_form`` / ``to_general_form``).
* :class:`~scidoggo.RankConstraint` -- a linear model with an explicit rank
  constraint on the coefficient matrix, an alternative to PLS for low-rank
  multi-output regression.
* :class:`~scidoggo.SelectivityModel` -- a model that non-linearly integrates
  input features to produce concave and convex isoresponse surfaces.

Interpolation
=============

:class:`~scidoggo.RbfRegression` is a scikit-learn-compatible wrapper around
``scipy.interpolate.RBFInterpolator`` for smooth radial-basis-function
interpolation/regression on scattered data.

Resampling
==========

The :mod:`scidoggo.resampling` subpackage provides bootstrap utilities:
:func:`~scidoggo.draw_bs_replicates` (bootstrap replicates of an estimator),
:func:`~scidoggo.bs_cis` (bootstrap confidence intervals), and the significance
helpers :func:`~scidoggo.sig_directional` and :func:`~scidoggo.sig_overlap`.

Optional extras
===============

Heavier, dependency-laden models are kept out of the core import path. They
live in optional subpackages and are pulled in via extras:

============================================  ===================================  ===============================
Subpackage                                    Install                              Dependencies
============================================  ===================================  ===============================
``scidoggo.neural``                           ``pip install "scidoggo[neural]"``   torch + skorch
``scidoggo.circuit_models``                   ``pip install "scidoggo[circuits]"`` pyro
``scidoggo.deep_implicit_circuits``           ``pip install "scidoggo[deep]"``     torch + pytorch-lightning
``scidoggo.linear_model.Rank1PlusSparse``     ``pip install "scidoggo[sparse]"``   cvxpy
============================================  ===================================  ===============================

Importing or using one of these without its dependency installed raises a
``scidoggo.exceptions.MissingDependencyError`` telling you which extra to
install. ``pip install "scidoggo[all]"`` installs all of them.
