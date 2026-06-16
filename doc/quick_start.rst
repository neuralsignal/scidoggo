###########
Quick Start
###########

``scidoggo`` is a collection of scientific models exposing a familiar
scikit-learn API (``fit`` / ``predict`` / ``transform``).

Installation
============

Core install
------------

The core package depends only on numpy, scipy and scikit-learn::

    $ pip install scidoggo

Optional extras
---------------

Heavier models live in optional subpackages and are installed via extras.
Each raises a clear ``MissingDependencyError`` if used without its
dependency::

    $ pip install "scidoggo[neural]"     # torch + skorch (two-layer encoding)
    $ pip install "scidoggo[circuits]"   # pyro (probabilistic circuit models)
    $ pip install "scidoggo[deep]"       # torch + pytorch-lightning (implicit circuits)
    $ pip install "scidoggo[sparse]"     # cvxpy (Rank1PlusSparse)

Install everything at once::

    $ pip install "scidoggo[all]"

Building the documentation
--------------------------

The documentation is built with Sphinx, numpydoc and sphinx-gallery::

    $ pip install "scidoggo[docs]"
    $ cd doc
    $ make html

Usage
=====

Tikhonov regression
-------------------

:class:`~scidoggo.Tikhonov` extends scikit-learn's ``Ridge`` with an optional
Tikhonov regularization matrix ``L`` (``L.T @ L`` plays the role of the inverse
prior covariance of the coefficients). With ``L=None`` it is ordinary ridge::

    >>> import numpy as np
    >>> from scidoggo import Tikhonov
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 5))
    >>> true_coef = np.array([3.0, -2.0, 0.0, 1.0, 0.0])
    >>> y = X @ true_coef + 0.1 * rng.standard_normal(100)
    >>> model = Tikhonov(alpha=1.0).fit(X, y)
    >>> y_pred = model.predict(X)
    >>> model.coef_.shape
    (5,)

Partial Least Squares
---------------------

:class:`~scidoggo.PLSRegression` works like the scikit-learn estimator but adds
a ``demean`` option and methods to inspect the model component by component
(:meth:`~scidoggo.PLSRegression.decompose_coef`,
:meth:`~scidoggo.PLSRegression.kth_coef`)::

    >>> from scidoggo import PLSRegression
    >>> Y = np.column_stack([y, X @ np.array([0., 1., 1., 0., -1.])])
    >>> pls = PLSRegression(n_components=3).fit(X, Y)
    >>> Y_pred = pls.predict(X)
    >>> # coefficients reconstructed from only the first component
    >>> coef_1 = pls.kth_coef(1)
    >>> coef_1.shape
    (2, 5)

See the :ref:`User Guide <user_guide>` and the
:ref:`examples gallery <general_examples>` for more.
