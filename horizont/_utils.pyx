import numpy as np
cimport numpy as np

cpdef searchsorted(np.ndarray[np.float_t] arr, double v):
    """
    Find index where an element should be inserted to maintain order.
    
    This is a Cython version of numpy.searchsorted (bisection search).
    """
    cdef unsigned int imin, imax, imid
    imin = 0
    imax = len(arr)
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if v > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


cpdef choice(np.ndarray[np.float_t] p, double r):
    """
    Cython version of numpy.random.choice where uniform random variate r must
    be provided.
    """
    cdef double dist_cum = 0
    cdef int K = len(p)
    cdef np.ndarray[np.float_t] dist_sum = np.empty(K)
    for k in range(K):
        dist_cum += p[k]
        dist_sum[k] = dist_cum
    return searchsorted(dist_sum, dist_cum * r)
