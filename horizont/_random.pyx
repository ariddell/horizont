# distutils: language = c++
from cython.operator cimport dereference as deref
from libc.math cimport sqrt

import numpy as np

cdef extern from "RNG/GRNG.hpp":
    cdef cppclass BasicRNG:
        BasicRNG()
        double norm(double mean, double sd)
        void set(unsigned long seed)

cdef extern from "RNG/RNG.hpp":
    cdef cppclass RNG(BasicRNG):
        RNG()

cdef extern from "BayesLogit/Code/C/PolyaGamma.h":
    cdef cppclass PolyaGamma:
        double pg_m1(double b, double z)
        double pg_m2(double b, double z)
        double draw(int n, double z, RNG& r)
        double draw_sum_of_gammas(double n, double z, RNG& r)

cdef extern from "BayesLogit/Code/C/PolyaGammaAlt.h":
    cdef cppclass PolyaGammaAlt:
        double draw(double h, double z, RNG& r)

cdef extern from "BayesLogit/Code/C/PolyaGammaSP.h":
    cdef cppclass PolyaGammaSP:
        int draw(double& d, double h, double z, RNG& r)

# REF: BayesLogit/Code/C/LogitWrapper.cpp
# XXX: there is now a parallel implementation see ParallelWrapper.cpp
cdef rpg_hybrid(double *x, double *h, double *z, int* num, unsigned long seed):
    cdef:
        RNG r = RNG()
        PolyaGamma dv
        PolyaGammaAlt alt
        PolyaGammaSP sp
        int i
        double b, c, m, v

    # works more readily than inheriting constructor from BasicRNG
    r.set(seed)

    # Cython optimizes this to a pure C loop
    for i in range(deref(num)):
        b = h[i]
        c = z[i]
        if b > 170:
            m = dv.pg_m1(b, c)
            v = dv.pg_m2(b, c) - m*m
            x[i] = r.norm(m, sqrt(v))
        elif b > 13:
            sp.draw(x[i], b, c, r)
        elif  b == 1 or b == 2:
            x[i] = dv.draw(<int>b, c, r)
        elif b > 1:
            x[i] = alt.draw(b, c, r);
        elif b > 0:
            x[i] = dv.draw_sum_of_gammas(b, c, r)
        else:
            x[i] = 0.0


def pg(double b, double z, unsigned long seed):
    """Draw once from a Pólya-Gamma distribution"""

    cdef:
        int size = 1
        double x
    rpg_hybrid(&x, &b, &z, &size, seed)
    return x


def pg_m1(double b, double z):
    """First moment of PG(b, z)"""
    cdef:
        PolyaGamma dv
    return dv.pg_m1(b, z)


def pg_m2(double b, double z):
    """Second moment of PG(b, z)"""
    cdef:
        PolyaGamma dv
    return dv.pg_m2(b, z)
