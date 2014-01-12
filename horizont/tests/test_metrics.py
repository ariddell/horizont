from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np

np.random.seed(1)

from horizont.metrics import kl_div, js_div


class TestMetrics(unittest.TestCase):

    def test_kl_div(self):
        X = [[0.7, 0.3], [0.5, 0.5]]
        dist = kl_div(X)
        desired = np.array([[0., 0.082282878505051782],
                            [0.087176693572388914, 0.]])
        np.testing.assert_allclose(dist, desired)

    def test_js_div(self):
        X = [[0.7, 0.3], [0.5, 0.5]]
        dist = js_div(X)
        desired = np.array([[0., 0.082282878505051782],
                            [0.087176693572388914, 0.]])
        desired = 0.5 * (desired + desired.T)
        np.testing.assert_allclose(dist, desired)
