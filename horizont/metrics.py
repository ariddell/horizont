"""
The :mod:`horizont.metrics` submodule provides functions that calculate
pairwise distances.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import itertools

import numpy as np


def kl_div(X):
    """
    Considering the rows of X as discrete probability distributions, calculate
    the Kullback-Leibler divergence between each pair of vectors.

    The Kullback-Leibler (KL) divergence between two discrete probability
    distributions, ``p`` and ``q`` is given by

        :math:`D_KL(p||q) = \sum_i ln(p[i]/q[i])*p[i]

    Parameters
    ----------
    X : array, shape = [n, p]

    Returns
    -------
    distances : array, shape = [n, n]

    Examples
    --------
    >>> from horizont.metrics import kl_div
    >>> X = [[0.7, 0.3], [0.5, 0.5]]
    >>> # distance between rows of X
    >>> kl_div(X)
    array([[ 0.,  0.082282878505051782],
           [ 0.087176693572388914,  0.]])
    """

    # FIXME: look into scipy.spatial.distance.pdist
    X = np.asarray(X, dtype=float)
    n = len(X)
    distances = np.empty((n, n))
    for i, j in itertools.product(range(n), range(n)):
        distances[i, j] = 0 if i == j else np.sum(X[i]*np.log(X[i]/X[j]))
    return distances


def js_div(X):
    """Calculate Jensen–Shannon divergence. See `kl_div` for details."""
    return 0.5*(kl_div(X)+kl_div(X).T)
