"""
Bootstrapping tools.

These are pure functions: every value is supplied by the caller and there are
no default arguments.
"""

from collections.abc import Callable

import numpy as np


def draw_bs_replicates(
    data: np.ndarray,
    estimator: str | Callable[..., np.ndarray],
    nboots: int,
    axis: int | None,
) -> np.ndarray:
    """
    Create bootstrap samples, compute replicates, and return them.

    Parameters
    ----------
    data : np.ndarray
        The input data.
    estimator : str or callable
        The estimator used to calculate the bootstrap replicates. If a string,
        it is resolved to the corresponding attribute of :mod:`numpy`
        (e.g. ``'mean'`` -> ``numpy.mean``).
    nboots : int
        The number of bootstrapped samples to generate.
    axis : int or None
        The axis along which to draw the bootstrap samples. If ``None`` the
        array is raveled first.

    Returns
    -------
    np.ndarray
        An array of bootstrapped replicates.

    Examples
    --------
    Drawing replicates of the mean of a 1-D array with 10000 bootstrap
    resamples and no axis::

        draw_bs_replicates(np.array([1, 2, 3, 4, 5]), np.mean, 10000, None)

    returns a 1-D array of length 10000 whose values cluster around the sample
    mean of 3.0. Because the resampling is stochastic, the exact values differ
    on each call.
    """
    # Create an empty array to store replicates
    if axis is None:
        bs_replicates = np.empty(nboots)
        length = data.size
        data = data.ravel()
    else:
        shape = list(data.shape)
        shape.pop(axis)
        bs_replicates = np.empty((nboots,) + tuple(shape))
        length = data.shape[axis]

    func: Callable[..., np.ndarray] = (
        getattr(np, estimator) if isinstance(estimator, str) else estimator
    )

    # Create bootstrap replicates as much as size
    for i in range(nboots):
        # Create a bootstrap sample
        idcs = np.random.randint(0, length, size=length)
        bs_sample = np.take(data, idcs, axis=axis)
        # Get bootstrap replicate and append to bs_replicates
        bs_replicates[i] = func(bs_sample, axis=axis)

    return bs_replicates


def bs_cis(
    data: np.ndarray,
    alpha: float,
    estimator: str | Callable[..., np.ndarray],
    nboots: int,
    axis: int | None,
) -> np.ndarray:
    """
    Compute bootstrap confidence intervals.

    Parameters
    ----------
    data : np.ndarray
        The data to compute bootstrap confidence intervals for.
    alpha : float
        The significance level.
    estimator : str or callable
        The estimator used to compute the replicates. If a string, it is
        resolved to the corresponding attribute of :mod:`numpy`.
    nboots : int
        The number of bootstrap replicates to compute.
    axis : int or None
        The axis along which to compute the bootstrap confidence intervals.
        If ``None`` the array is raveled first.

    Returns
    -------
    np.ndarray
        A 2D array containing the lower and upper bounds of the bootstrap
        confidence intervals.

    See Also
    --------
    draw_bs_replicates : Creates bootstrap replicates.

    Examples
    --------
    For a 1-D array such as ``np.array([1, 2, 3, 4, 5])`` with
    ``alpha=0.05``, ``estimator=np.mean``, ``nboots=1000`` and ``axis=None``,
    the result is a two-element array giving the lower and upper bounds of the
    95% confidence interval for the mean (roughly ``[1.6, 4.4]``). For a 2-D
    array evaluated with ``axis=1`` the result has one lower/upper pair per
    row. The exact bounds vary between calls because the resampling is
    stochastic.
    """
    samples = draw_bs_replicates(data, estimator, nboots, axis)
    return np.percentile(samples, axis=0, q=[alpha / 2 * 100, (1 - alpha / 2) * 100])


def sig_directional(
    data: np.ndarray,
    axis: int | None,
    alpha: float,
    estimator: str | Callable[..., np.ndarray],
    nboots: int,
) -> np.ndarray:
    """
    Determine if the data is significantly positive or negative.

    This function returns an integer array indicating whether the data is
    significantly positive or negative based on bootstrap confidence
    intervals.

    Parameters
    ----------
    data : np.ndarray
        The data to test.
    axis : int or None
        The axis along which to perform the test. If ``None`` the array is
        raveled first.
    alpha : float
        The significance level.
    estimator : str or callable
        The estimator used to compute the replicates. If a string, it is
        resolved to the corresponding attribute of :mod:`numpy`.
    nboots : int
        The number of bootstrap replicates to compute.

    Returns
    -------
    np.ndarray
        An integer array indicating whether the data is significantly positive
        (``1``), significantly negative (``-1``) or neither (``0``).

    See Also
    --------
    bs_cis : Computes bootstrap confidence intervals.
    """
    cis = bs_cis(data, alpha, estimator, nboots, axis)
    pos = (cis > 0).all(0).astype(int)
    neg = (cis < 0).all(0).astype(int)
    return pos - neg


def sig_overlap(
    data1: np.ndarray,
    data2: np.ndarray,
    alpha: float,
    estimator: str | Callable[..., np.ndarray],
    nboots: int,
    axis: int | None,
) -> np.ndarray:
    """
    Determine if two sets of bootstrapped confidence intervals overlap.

    Parameters
    ----------
    data1 : np.ndarray
        The first set of data to compute bootstrapped confidence intervals
        for.
    data2 : np.ndarray
        The second set of data to compute bootstrapped confidence intervals
        for.
    alpha : float
        The significance level passed to :func:`bs_cis`.
    estimator : str or callable
        The estimator used to compute the replicates. If a string, it is
        resolved to the corresponding attribute of :mod:`numpy`.
    nboots : int
        The number of bootstrap replicates to compute.
    axis : int or None
        The axis along which to compute the confidence intervals. If ``None``
        the arrays are raveled first.

    Returns
    -------
    np.ndarray
        A boolean array indicating whether the confidence intervals for
        ``data1`` and ``data2`` overlap (``True``) or not (``False``).

    See Also
    --------
    bs_cis : Computes bootstrap confidence intervals.

    Examples
    --------
    Comparing two identical samples such as ``np.array([1, 2, 3, 4, 5])``
    against itself yields overlapping intervals (``True``), whereas comparing
    ``np.array([1, 2, 3, 4, 5])`` against ``np.array([6, 7, 8, 9, 10])``
    yields non-overlapping intervals (``False``). The exact outcome depends on
    the stochastic resampling for borderline cases.
    """
    cis1 = bs_cis(data1, alpha, estimator, nboots, axis)
    cis2 = bs_cis(data2, alpha, estimator, nboots, axis)
    return (cis1[0] < cis2[1]) & (cis1[1] > cis2[0])
