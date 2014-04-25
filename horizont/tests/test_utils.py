from __future__ import absolute_import, division, print_function, unicode_literals

import io
import os
import unittest

import numpy as np

import horizont.utils as utils
import horizont._utils as _utils


class TestUtils(unittest.TestCase):

    np.random.seed(99)

    D = 100
    W = 50
    N_WORDS_PER_DOC = 500
    N = W * N_WORDS_PER_DOC
    dtm = np.zeros((D, W), dtype=int)
    for d in range(D):
        dtm[d] = np.random.multinomial(N_WORDS_PER_DOC, np.ones(W) * 1/W)
    N_BY_W = np.sum(dtm, axis=0)
    N_BY_D = np.sum(dtm, axis=1)

    def test_setup(self):
        dtm, D, N_WORDS_PER_DOC = self.dtm, self.D, self.N_WORDS_PER_DOC
        self.assertEqual(np.sum(dtm), D * N_WORDS_PER_DOC)

    def test_matrix_to_lists(self):
        dtm, D, N_WORDS_PER_DOC = self.dtm, self.D, self.N_WORDS_PER_DOC
        N_BY_D, N_BY_W = self.N_BY_D, self.N_BY_W
        WS, DS = utils.matrix_to_lists(dtm)
        self.assertEqual(len(WS), D * N_WORDS_PER_DOC)
        self.assertEqual(len(WS), len(DS))
        self.assertEqual(dtm.shape, (max(DS) + 1, max(WS) + 1))
        self.assertTrue(all(DS == sorted(DS)))
        self.assertTrue(np.all(np.bincount(DS) == N_BY_D))
        self.assertTrue(np.all(np.bincount(WS) == N_BY_W))

    def test_lists_to_matrix(self):
        dtm = self.dtm
        WS, DS = utils.matrix_to_lists(dtm)
        dtm_new = utils.lists_to_matrix(WS, DS)
        self.assertTrue(np.all(dtm == dtm_new))

    def test_ldac2dtm_offset(self):
        test_dir = os.path.dirname(__file__)
        ap_ldac_fn = os.path.join(test_dir, 'ap.dat')
        with self.assertRaises(ValueError):
            utils.ldac2dtm(open(ap_ldac_fn), offset=1)

    def test_ldac_conversion(self):
        dtm = self.dtm
        N, V = dtm.shape
        doclines = list(utils.dtm2ldac(self.dtm))
        nd_unique = np.sum(dtm > 0, axis=1)
        for n, docline in zip(nd_unique, doclines):
            self.assertEqual(n, int(docline.split(' ')[0]))
        self.assertEqual(len(doclines), N)
        f = io.StringIO('\n'.join(doclines))
        dtm_new = utils.ldac2dtm(f)
        self.assertTrue(np.all(dtm == dtm_new))

    def test_searchsorted(self):
        for _ in range(10000):
            arr = np.sort(np.random.rand(10))
            r = np.random.rand()
            self.assertEqual(np.searchsorted(arr, r), _utils.searchsorted(arr, r))
        r = -1
        self.assertEqual(np.searchsorted(arr, r), _utils.searchsorted(arr, r))
        r = 10
        self.assertEqual(np.searchsorted(arr, r), _utils.searchsorted(arr, r))

    def test_choice(self):
        r = 0.9851367976958045
        probz = np.array([0.14800396, 0.29600989, 0.46001528])
        z = _utils.choice(probz, r)
        self.assertEqual(z, len(probz) - 1)

        N = 10000
        K = 100
        samples = []
        random_state = np.random.RandomState(1)
        for _ in range(N):
            # unnormalized probability distribution
            dist = random_state.randint(10, 1000, size=K).astype(float)
            samples.append(_utils.choice(dist, random_state.rand()))
        self.assertGreaterEqual(max(samples), 0)
        self.assertLessEqual(max(samples), K-1)

        p = 1/K
        bound = 3*np.sqrt(N*p*(1-p))
        counts = np.bincount(samples)
        for k in range(K):
            self.assertLess(abs(counts[k] - N*p), bound)

    def test_choice_consistency(self):
        N = 1000
        K = 100
        samples = []
        random_state = np.random.RandomState(1)
        for _ in range(N):
            # unnormalized probability distribution
            dist = random_state.randint(10, 1000, size=K).astype(float)
            samples.append(_utils.choice(dist, random_state.rand()))

        samples2 = []
        random_state = np.random.RandomState(1)
        for _ in range(N):
            # unnormalized probability distribution
            dist = random_state.randint(10, 1000, size=K).astype(float)
            samples2.append(_utils.choice(dist, random_state.rand()))
        self.assertEqual(samples, samples2)
