from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np

import horizont.random as random


class TestRandom(unittest.TestCase):

    np.random.seed(1)

    def test_sample_index(self):
        N = 2
        probs = [1/N] * N
        x = [random.sample_index(probs) for _ in range(10000)]
        hist = np.bincount(x)
        probs_sample = hist / np.sum(hist)
        np.testing.assert_allclose(probs_sample, 1/N, atol=0.02)

        N = 15
        probs = [1/N] * N
        x = [random.sample_index(probs) for _ in range(100000)]
        hist = np.bincount(x)
        probs_sample = hist / np.sum(hist)
        np.testing.assert_allclose(probs_sample, 1/N, atol=0.02)

    def test_sample_index_unnormalized(self):
        N = 2
        probs = [N] * N
        x = [random.sample_index(probs) for _ in range(10000)]
        hist = np.bincount(x)
        probs_sample = hist / np.sum(hist)
        np.testing.assert_allclose(probs_sample, 1/N, atol=0.02)

        N = 15
        probs = [N] * N
        x = [random.sample_index(probs) for _ in range(100000)]
        hist = np.bincount(x)
        probs_sample = hist / np.sum(hist)
        np.testing.assert_allclose(probs_sample, 1/N, atol=0.02)

    def test_sample_index_nonuniform(self):
        probs = [1/3, 2/3]
        x = [random.sample_index(probs) for _ in range(10000)]
        hist = np.bincount(x)
        probs_sample = hist / np.sum(hist)
        np.testing.assert_allclose(probs_sample, probs, atol=0.02)

        probs = np.array([1/5, 1/5, 3/5]) * 1000  # unnormalized
        x = [random.sample_index(probs) for _ in range(10000)]
        hist = np.bincount(x)
        probs_sample = hist / np.sum(hist)
        np.testing.assert_allclose(probs_sample, probs/sum(probs), atol=0.02)

    def test_sample_sparse(self):
        probs = [0, .5, 0, .5]
        x = [random.sample_index(probs) for _ in range(10000)]
        hist = np.bincount(x)
        self.assertEqual(hist[0], 0)
        self.assertEqual(hist[2], 0)
