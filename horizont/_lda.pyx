cimport cython
cimport numpy as np
cimport libc.math as math

import numpy as np
import horizont._utils


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _sample_topics(np.ndarray[np.int_t] WS, np.ndarray[np.int_t] DS, np.ndarray[np.int_t] ZS,
                   np.ndarray[np.int_t, ndim=2] nzw, np.ndarray[np.int_t, ndim=2] ndz, np.ndarray[np.int_t] nz,
                   np.ndarray[np.float_t] alpha, np.ndarray[np.float_t] eta, np.ndarray[np.float_t] rands):
    cdef unsigned int i, w, d, z, z_new, n_topics
    cdef unsigned int n_rand = len(rands)
    cdef double r, dist_cum
    cdef np.ndarray[np.float_t] dist_sum
    n_topics = len(nzw)
    cdef double alpha_sum = sum(alpha)
    cdef double eta_sum = sum(eta)

    dist_sum = np.empty(n_topics)

    for i in range(len(WS)):
        w = WS[i]
        d = DS[i]
        z = ZS[i]

        nzw[z, w] -= 1
        ndz[d, z] -= 1
        nz[z] -= 1

        dist_cum = 0
        for k in range(n_topics):
            # eta is a double so cdivision yields a double
            dist_cum += (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (ndz[d, k] + alpha[k])
            dist_sum[k] = dist_cum

        r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
        # XXX: using cimport and a pxd file for searchsorted might be faster
        z_new = horizont._utils.searchsorted(dist_sum, r)

        ZS[i] = z_new
        nzw[z_new, w] += 1
        ndz[d, z_new] += 1
        nz[z_new] += 1
