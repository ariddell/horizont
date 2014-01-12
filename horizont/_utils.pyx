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
