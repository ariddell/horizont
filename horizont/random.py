import numpy as np

from horizont import _random


MAX_UINT = 2**32 - 1


def pg(b, z, seed=None):
    if seed is None:
        seed = np.random.randint(MAX_UINT)
    return _random.pg(b, z, seed)
