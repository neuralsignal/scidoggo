.. -*- mode: rst -*-

scidoggo: a collection of shallow and deep scientific models
============================================================

``scidoggo`` is a collection of models and data tools, gathered and modified
across different research projects, all exposing a familiar scikit-learn API
(``fit`` / ``predict`` / ``transform``). The core package depends only on
numpy, scipy and scikit-learn; heavier models are available through optional
extras.

Installation
============

Core install (numpy / scipy / scikit-learn only)::

    pip install scidoggo

Optional extras pull in the heavier dependencies. Each optional model raises a
clear ``MissingDependencyError`` if used without its extra installed::

    pip install "scidoggo[neural]"     # torch + skorch (two-layer encoding model)
    pip install "scidoggo[circuits]"   # pyro (probabilistic circuit models)
    pip install "scidoggo[deep]"       # torch + pytorch-lightning (implicit circuit models)
    pip install "scidoggo[sparse]"     # cvxpy (Rank1PlusSparse)

Install everything at once::

    pip install "scidoggo[all]"

Models
======

Core models (importable directly from ``scidoggo``):

* **Cross decomposition** (``scidoggo.cross_decomposition``): ``PLSRegression``,
  ``PLSCanonical``, ``CCA`` and ``PLSSVD`` -- the Partial Least Squares family
  forked from scikit-learn with an optional bias-removal (``demean``) flag and
  helpers (``decompose_coef``, ``kth_coef``, ``kth_rotations``) to orthogonalize
  and inspect the components.
* **Linear models** (``scidoggo.linear_model``): ``Tikhonov`` (generalized
  ridge regression with a non-diagonal regularization matrix, plus solver
  helpers in ``scidoggo.linear_model.tikhonov_solver``), ``RankConstraint``
  (rank-constrained linear model, an alternative to PLS) and
  ``SelectivityModel`` (non-linear integration of input features producing
  concave/convex isoresponse surfaces).
* **Interpolation** (``scidoggo.interpolation``): ``RbfRegression`` -- a
  scikit-learn-compatible wrapper around ``scipy.interpolate.RBFInterpolator``.
* **Resampling** (``scidoggo.resampling``): bootstrap utilities
  ``draw_bs_replicates``, ``bs_cis``, ``sig_directional`` and ``sig_overlap``.

Optional models (each behind an extra):

* **Two-layer encoding model** (``scidoggo.neural``, ``[neural]``).
* **Probabilistic circuit / chromatic models** (``scidoggo.circuit_models``,
  ``[circuits]``) -- model circuits while incorporating anatomical constraints
  as priors.
* **Deep implicit circuit models** (``scidoggo.deep_implicit_circuits``,
  ``[deep]``) -- constrained circuit models assuming the observed responses are
  at steady-state.
* **Rank1PlusSparse** (``scidoggo.linear_model.Rank1PlusSparse``, ``[sparse]``)
  -- linear model with a rank-one constraint plus an added sparse weight matrix.

Documentation
=============

The documentation is built with Sphinx, numpydoc and sphinx-gallery::

    pip install "scidoggo[docs]"
    cd doc
    make html

Acknowledgments
===============

The PLS estimators are forked from scikit-learn. The original package layout
was bootstrapped from the scikit-learn-contrib project template.
