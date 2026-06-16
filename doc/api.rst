###############
scidoggo API
###############

This is the public API of :mod:`scidoggo`. The objects below are importable
directly from the top-level ``scidoggo`` namespace and depend only on the core
dependencies (numpy / scipy / scikit-learn).

.. currentmodule:: scidoggo

Cross decomposition
===================

Partial Least Squares family forked from scikit-learn, with optional
de-meaning (``demean``) and helpers to decompose the regression coefficients
into orthogonal components.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   PLSRegression
   PLSCanonical
   CCA
   PLSSVD

Linear models
=============

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Tikhonov
   RankConstraint
   SelectivityModel

Tikhonov solver helpers
-----------------------

Functional building blocks for Tikhonov / generalized ridge regression.

.. currentmodule:: scidoggo.linear_model.tikhonov_solver

.. autosummary::
   :toctree: generated/
   :template: function.rst

   analytic_tikhonov
   find_tikhonov_from_covariance
   to_standard_form
   to_general_form
   fit_learner

Interpolation
=============

.. currentmodule:: scidoggo

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RbfRegression

Resampling
==========

Bootstrap utilities for confidence intervals and significance testing.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   draw_bs_replicates
   bs_cis
   sig_directional
   sig_overlap

Optional extras
===============

The heavier models live in optional subpackages. Each is installed via an
extra and raises :class:`scidoggo.exceptions.MissingDependencyError` if its
dependency is missing. They are **not** imported by the core package and are
therefore not auto-documented here.

* :mod:`scidoggo.neural` -- two-layer encoding model.
  Install with ``pip install "scidoggo[neural]"`` (torch + skorch).
* :mod:`scidoggo.circuit_models` -- probabilistic circuit / chromatic models.
  Install with ``pip install "scidoggo[circuits]"`` (pyro).
* :mod:`scidoggo.deep_implicit_circuits` -- implicit steady-state circuit
  models. Install with ``pip install "scidoggo[deep]"``
  (torch + pytorch-lightning).
* ``scidoggo.linear_model.Rank1PlusSparse`` -- rank-1-plus-sparse linear model.
  Install with ``pip install "scidoggo[sparse]"`` (cvxpy).

Install everything at once with ``pip install "scidoggo[all]"``.
