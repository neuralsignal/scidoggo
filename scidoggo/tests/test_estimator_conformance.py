"""scikit-learn estimator conformance checks for the core estimators.

Each core estimator is run through ``check_estimator`` with the verified
``expected_failed_checks`` / ``n_components`` declarations.
"""

import pytest
from sklearn.utils.estimator_checks import check_estimator

from scidoggo import (
    CCA,
    PLSCanonical,
    PLSRegression,
    PLSSVD,
    RankConstraint,
    RbfRegression,
    SelectivityModel,
    Tikhonov,
)

SW = "check_sample_weight_equivalence_on_dense_data"
SW_SPARSE = "check_sample_weight_equivalence_on_sparse_data"
N_ITER = "check_non_transformer_estimators_n_iter"

# Tikhonov subclasses sklearn's Ridge. Under scikit-learn 1.9.0 a bare
# ``Ridge(fit_intercept=False)`` itself fails these same three checks via
# ``check_estimator`` (verified directly), so they are inherited stock-Ridge
# behaviour, not scidoggo bugs:
#   * the two sample_weight-equivalence checks (Ridge(fit_intercept=False) does
#     not give weight==repeat equivalence), and
#   * the n_iter check (the direct solver leaves ``n_iter_`` as ``None``).
TIKHONOV_EXPECTED = {
    SW: "Ridge(fit_intercept=False) intrinsic",
    SW_SPARSE: "Ridge(fit_intercept=False) intrinsic (sparse variant)",
    N_ITER: "inherited Ridge: direct solver leaves n_iter_=None",
}

# (id, estimator, expected_failed_checks)
ESTIMATORS = [
    ("PLSRegression", PLSRegression(), None),
    # default n_components=2 fails like stock sklearn, so use 1
    ("PLSCanonical", PLSCanonical(n_components=1), None),
    ("PLSSVD", PLSSVD(n_components=1), None),
    ("CCA", CCA(n_components=1), None),
    ("Tikhonov", Tikhonov(), TIKHONOV_EXPECTED),
    ("RankConstraint", RankConstraint(), None),
    (
        "SelectivityModel",
        SelectivityModel(),
        {SW: "grid-search + nonlinear least_squares"},
    ),
    (
        "RbfRegression",
        RbfRegression(),
        {SW: "sample_weight scales smoothing, not frequency"},
    ),
]


@pytest.mark.parametrize(
    "estimator,expected_failed_checks",
    [(est, efc) for _, est, efc in ESTIMATORS],
    ids=[name for name, _, _ in ESTIMATORS],
)
def test_estimator_conformance(estimator, expected_failed_checks):
    if expected_failed_checks is None:
        check_estimator(estimator)
    else:
        check_estimator(estimator, expected_failed_checks=expected_failed_checks)
