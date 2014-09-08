from __future__ import absolute_import, division, print_function, unicode_literals
import os
import unittest

import numpy as np

import horizont


class TestLDANews(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_dir = os.path.dirname(__file__)
        ap_ldac_fn = os.path.join(test_dir, 'ap.dat')
        cls.dtm = dtm = horizont.utils.ldac2dtm(open(ap_ldac_fn), offset=0)
        cls.model = model = horizont.LDA(n_topics=10, n_iter=2, random_state=1)
        cls.doctopic = model.fit_transform(dtm)

    def test_lda_news(self):
        dtm = self.dtm
        doctopic = self.doctopic
        self.assertEqual(len(doctopic), len(dtm))

    def test_lda_attributes(self):
        dtm = self.dtm
        doctopic = self.doctopic
        model = self.model

        # check dims
        N = dtm.sum()
        D, V = dtm.shape
        _, K = doctopic.shape
        self.assertEqual(model.theta_.shape, doctopic.shape)
        np.testing.assert_array_equal(model.theta_, doctopic)
        self.assertEqual(model.theta_.shape, (D, K))
        self.assertEqual(model.ndz_.shape, (D, K))
        self.assertEqual(model.phi_.shape, (K, V))
        self.assertEqual(model.nzw_.shape, (K, V))

        # check contents
        self.assertAlmostEqual(model.nzw_.sum(), N)
        self.assertAlmostEqual(model.ndz_.sum(), N)
        self.assertAlmostEqual(model.nz_.sum(), N)
        self.assertAlmostEqual(model.theta_.sum(), D)
        self.assertAlmostEqual(model.phi_.sum(), K)
        np.testing.assert_array_equal(model.ndz_.sum(axis=0), model.nz_)

        # check distributions sum to one
        np.testing.assert_array_almost_equal(model.theta_.sum(axis=1), np.ones(D))
        np.testing.assert_array_almost_equal(model.phi_.sum(axis=1), np.ones(K))
