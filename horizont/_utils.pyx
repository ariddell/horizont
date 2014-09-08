#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

from libc.stdlib cimport malloc, free

cpdef int searchsorted(double[:] arr, double v):
    """
    Find index where an element should be inserted to maintain order.
    
    This is a Cython version of numpy.searchsorted (bisection search).
    """
    cdef int imin, imax, imid
    imin = 0
    imax = len(arr)
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if v > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


cpdef int choice(double[:] p, double r):
    """
    Cython version of numpy.random.choice where uniform random variate r must
    be provided.
    """
    cdef int k, z
    cdef int K = len(p)
    cdef double dist_cum = 0
    cdef double * dist_sum_ptr = <double*> malloc(sizeof(double) * K)
    cdef double[:] dist_sum = <double [:K]> dist_sum_ptr
    for k in range(K):
        dist_cum += p[k]
        dist_sum[k] = dist_cum
    z = searchsorted(dist_sum, dist_cum * r)
    free(<void *> dist_sum_ptr)
    return z
