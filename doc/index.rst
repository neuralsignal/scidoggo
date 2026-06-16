.. scidoggo documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scidoggo's documentation!
====================================

``scidoggo`` is a collection of shallow and deep scientific models and data
tools, all exposing a familiar scikit-learn API. The core package depends only
on numpy, scipy and scikit-learn; heavier models (neural encoding,
probabilistic circuits, deep implicit circuits, rank-1-plus-sparse) are
available through optional extras.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

`Getting started <quick_start.html>`_
-------------------------------------

Installation instructions (core and extras) and short runnable usage snippets.

`User Guide <user_guide.html>`_
-------------------------------

Narrative documentation describing the model families and the extras mechanism.

`API Documentation <api.html>`_
-------------------------------

The full public API, grouped by subpackage.

`Examples <auto_examples/index.html>`_
--------------------------------------

A gallery of runnable examples that complements the
`User Guide <user_guide.html>`_.
