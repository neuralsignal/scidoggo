"""Hypothesis property tests for the bootstrap resampling functions.

All arguments are passed explicitly (these functions have no defaults). The
legacy global numpy RNG is seeded inside each example for determinism (a
function-scoped fixture cannot be used with ``@given``) and ``nboots`` is kept
small for speed.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from scidoggo import bs_cis, draw_bs_replicates, sig_directional, sig_overlap

NBOOTS = 100

finite_floats = st.floats(
    allow_nan=False, allow_infinity=False, width=64, min_value=-1e6, max_value=1e6
)


def finite_1d_arrays(min_size=2, max_size=30):
    return hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=finite_floats,
    )


@settings(max_examples=30, deadline=None)
@given(data=finite_1d_arrays())
def test_draw_bs_replicates_shape(data):
    np.random.seed(0)
    reps = draw_bs_replicates(data, np.mean, NBOOTS, None)
    assert reps.shape[0] == NBOOTS


@settings(max_examples=30, deadline=None)
@given(data=finite_1d_arrays())
def test_bs_cis_lower_le_upper(data):
    np.random.seed(0)
    cis = bs_cis(data, 0.05, np.mean, NBOOTS, None)
    lower, upper = cis[0], cis[1]
    assert np.all(lower <= upper)


@settings(max_examples=30, deadline=None)
@given(data=finite_1d_arrays())
def test_sig_directional_in_set(data):
    np.random.seed(0)
    result = sig_directional(data, None, 0.05, np.mean, NBOOTS)
    assert np.all(np.isin(result, (-1, 0, 1)))


@settings(max_examples=30, deadline=None)
@given(data=finite_1d_arrays())
def test_sig_overlap_identical_is_true(data):
    np.random.seed(0)
    # ``sig_overlap`` uses strict < / > comparisons, so a zero-variance sample
    # produces zero-width point intervals that do not "overlap" themselves.
    # Restrict to samples with some spread, where overlap is well-defined.
    from hypothesis import assume

    assume(np.ptp(data) > 0)
    # identical inputs must produce overlapping confidence intervals
    result = sig_overlap(data, data, 0.05, np.mean, NBOOTS, None)
    assert np.all(result)


def test_string_estimator_matches_callable():
    """A string estimator resolves to the matching numpy callable."""
    data = np.arange(10.0)
    np.random.seed(0)
    a = draw_bs_replicates(data, "mean", NBOOTS, None)
    np.random.seed(0)
    b = draw_bs_replicates(data, np.mean, NBOOTS, None)
    np.testing.assert_allclose(a, b)


def test_axis_reduces_dimension():
    """With an explicit axis the replicates keep the remaining dimensions."""
    np.random.seed(0)
    data = np.arange(40.0).reshape(8, 5)
    reps = draw_bs_replicates(data, np.mean, NBOOTS, 0)
    assert reps.shape == (NBOOTS, 5)
