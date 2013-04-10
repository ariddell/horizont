cimport cython
cimport numpy as np
cimport libc.math as math

import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _sample_topics(np.ndarray[np.int_t] WS, np.ndarray[np.int_t] DS, np.ndarray[np.int_t] ZS, 
                   np.ndarray[np.int_t, ndim=2] nzw, np.ndarray[np.int_t, ndim=2] ndz, np.ndarray[np.int_t] nz,
                   np.ndarray[np.float_t] alpha, float alpha_sum,
                   np.ndarray[np.float_t] eta, float eta_sum,
                   np.ndarray[np.float_t] rands, int init):
    cdef unsigned int i, w, d, z, z_new, n_topics
    cdef unsigned int n_rand = len(rands)
    cdef double r, dist_cum
    cdef unsigned int imin, imax, imid
    cdef np.ndarray[np.float_t] dist, dist_sum
    n_topics = len(nzw)

    dist = np.empty(n_topics)
    dist_sum = np.empty(n_topics)

    for i in range(len(WS)):
        w = WS[i]
        d = DS[i]
        z = ZS[i]

        if init == 0:
            nzw[z, w] -= 1
            ndz[d, z] -= 1
            nz[z] -= 1

        dist_cum = 0
        for k in range(n_topics):
            dist[k] = (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (ndz[d, k] + alpha[k])
            dist_cum = dist_cum + dist[k]
            dist_sum[k] = dist_cum

        r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]

        # numpy.searchsorted (bisection search) implemented in C/Cython
        imin = 0
        imax = n_topics
        while imin < imax:
            imid = imin + ((imax - imin) >> 2)
            if r > dist_sum[imid]:
                imin = imid + 1
            else:
                imax = imid
        z_new = imin

        ZS[i] = z_new
        nzw[z_new, w] += 1
        ndz[d, z_new] += 1
        nz[z_new] += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _log_prob(np.ndarray[np.int_t] WS, np.ndarray[np.int_t] DS, np.ndarray[np.int_t] ZS, 
              np.ndarray[np.int_t, ndim=2] nzw, np.ndarray[np.int_t, ndim=2] ndz, np.ndarray[np.int_t] nz,
              np.ndarray[np.float_t] alpha, float alpha_sum,
              np.ndarray[np.float_t] eta, float eta_sum):
    cdef unsigned int i, n, d, w, z, d_last
    cdef double lp = 0.0

    nzw.fill(0)
    nz.fill(0)
    ndz.fill(0)

    n = 0
    d_last = -1
    for i in range(len(DS)):
        d, w, z = DS[i], WS[i], ZS[i]
        n = n + 1 if d == d_last else 0
        lp += math.log((nzw[z, w] + eta[w]) / (nz[z] + eta_sum) *
                        ndz[d, z] + alpha[z] / (n + alpha_sum))
        nzw[z, w] += 1
        nz[z] += 1
        ndz[d, z] += 1
        d_last = d

    return lp
