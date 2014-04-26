import unittest

import numpy as np

import horizont
import horizont.random
import horizont._random
import horizont.utils


class TestPolyaGamma(unittest.TestCase):
    """Test the different PG random variable generation routines"""

    def setUp(self):
        np.random.seed(5)

    def test_rng(self):
        b, c = 200, 0
        r1 = horizont.random.pg(b, c)
        r2 = horizont.random.pg(b, c)
        self.assertNotEqual(r1, r2)

    def test_moments(self):
        # test that moments are positive
        b, c = 1400, -3.7
        m1 = horizont._random.pg_m1(b, c)
        self.assertGreater(m1, 1)
        m2 = horizont._random.pg_m2(b, c)
        self.assertGreater(m2, 1)

        b, c = 1400, -2.62
        m1 = horizont._random.pg_m1(b, c)
        self.assertGreater(m1, 1)
        m2 = horizont._random.pg_m2(b, c)
        self.assertGreater(m2, 1)

    def test_neg_param(self):
        b, c = 1400, -3.78
        r = horizont.random.pg(b, c)
        self.assertGreater(r, 0)

        b, c = 1400, -2.62
        m1 = horizont._random.pg_m1(b, c)
        self.assertGreater(m1, 1)
        m2 = horizont._random.pg_m2(b, c)
        self.assertGreater(m2, 1)
        self.assertGreater(m2 - m1*m1, 1)
        r = horizont.random.pg(b, c)
        self.assertGreater(r, 0)

    def test_rng_with_seed(self):
        b, c = 200, 0
        seed = 5
        r1 = horizont.random.pg(b, c, seed=seed)
        r2 = horizont.random.pg(b, c, seed=seed)
        self.assertEqual(r1, r2)

        np.random.seed(seed)
        r1 = horizont.random.pg(b, c)
        np.random.seed(seed)
        r2 = horizont.random.pg(b, c)
        self.assertEqual(r1, r2)

    def test_polya_gamma_norm(self):
        # with a high b, a normal approximation is used
        b, c = 200, 0
        S = int(1e4)
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.03)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.03)

        b, c = 200, 3.5
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.02)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.02)

    def test_polya_gamma(self):
        b, c = 2, 0
        S = int(1e4)
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.05)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.05)

        b, c = 2, 3.5
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.05)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.05)

    def test_polya_gamma_alt(self):
        b, c = 9, 0
        S = int(1e4)
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.02)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.02)

        b, c = 9, 3.5
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.02)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.02)

    def test_polya_gamma_sp(self):
        b, c = 20, 0
        S = int(1e4)
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.02)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.02)

        b, c = 20, 3.5
        rvs = [horizont.random.pg(b, c) for _ in range(S)]
        m = horizont._random.pg_m1(b, c)
        v = horizont._random.pg_m2(b, c) - m*m
        np.testing.assert_allclose(np.mean(rvs), m, rtol=0.02)
        np.testing.assert_allclose(np.var(rvs, ddof=1), v, rtol=0.02)
