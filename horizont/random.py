"""
Random number generation routines
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from sklearn.utils import check_random_state


def sample_index(p, random_state=None):
    """
    Draw an integer index with probability specified by `p`.

    The integer is between ``0`` and ``len(p) - 1``.

    `p` need not be be a normalized probability distribution
    """
    # NB: the following is much faster than np.random.choice
    # and slightly faster than np.random.multinomial(1, probs).argmax()
    random_state = check_random_state(random_state)
    dist_sum = np.cumsum(p)
    r = random_state.rand() * dist_sum[-1]
    return np.searchsorted(dist_sum, r)
