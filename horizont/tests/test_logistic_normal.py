import unittest

import numpy as np
import scipy.misc
import scipy.special
from scipy.stats.mstats import mquantiles

import horizont
import horizont.random
import horizont.utils

trigamma = lambda x: scipy.special.polygamma(1, x)


def softmax(beta):
    return np.exp(beta - scipy.misc.logsumexp(beta))


class TestLogisticNormal(unittest.TestCase):

    def test_logistic_normal_multinomial(self):
        """Verify posterior resembles a posterior with a Dirichlet prior"""

        random_state = np.random.RandomState(5)
        nz = np.array([400, 700, 100, 200])
        J = len(nz)
        N = sum(nz)
        n_iter = 1000
        alpha = 1
        m0 = np.repeat(0, J)
        P0 = np.diag(np.repeat(1/trigamma(alpha), J))

        # summary statistics
        kappa = nz - np.sum(nz)/2

        # initialize
        beta = random_state.multivariate_normal(m0, np.linalg.inv(P0))
        beta[-1] = 0
        omega = np.zeros(J)
        c = np.zeros(J)

        # variables to hold results
        THETA = np.empty((n_iter, J))

        # sampling loop
        for it in range(n_iter):

            for j in range(J):
                c_j = scipy.misc.logsumexp(np.delete(beta, j))
                c[j] = c_j
                eta_j = beta[j] - c_j
                # sample omega[j]
                omega[j] = horizont.random.pg(N, eta_j)
                self.assertGreater(omega[j], 0)

            # sample beta
            P1 = np.diag(omega) + P0
            V1 = np.linalg.inv(P1)
            m1 = np.dot(V1, (kappa + omega*c + np.dot(P0, m0)))
            # this may generate an annoying deprecation warning
            beta = random_state.multivariate_normal(m1, V1)
            beta[-1] = 0

            # save scan
            THETA[it] = softmax(beta)

        burnin = 100
        theta_means = np.mean(THETA[burnin:, :], axis=0)
        dirichlet_post = random_state.dirichlet(alpha + nz, size=2000)
        dirichlet_quantiles = mquantiles(dirichlet_post, prob=[0.25, 0.75], axis=0)
        for i, quantiles in enumerate(dirichlet_quantiles.T):
            self.assertTrue(quantiles[0] < theta_means[i] < quantiles[1])

    def test_logistic_normal_multinomial_wishart(self):
        """Verify posterior resembles a posterior with a Dirichlet prior"""

        random_state = np.random.RandomState(5)
        nz = np.array([400, 700, 100, 200])
        J = len(nz)
        N = sum(nz)
        n_iter = 1000
        alpha = 1
        m0 = np.repeat(0, J)
        P0 = np.diag(np.repeat(1/trigamma(alpha), J))

        # summary statistics
        kappa = nz - np.sum(nz)/2

        # initialize
        beta = random_state.multivariate_normal(m0, np.linalg.inv(P0))
        beta[-1] = 0
        omega = np.zeros(J)
        c = np.zeros(J)

        # variables to hold results
        THETA = np.empty((n_iter, J))
        SIGMA = np.empty((n_iter, J, J))

        # sampling loop
        for it in range(n_iter):

            for j in range(J):
                c_j = scipy.misc.logsumexp(np.delete(beta, j))
                c[j] = c_j
                eta_j = beta[j] - c_j
                # sample omega[j]
                omega[j] = horizont.random.pg(N, eta_j)
                self.assertGreater(omega[j], 0)

            # sample beta
            P1 = np.diag(omega) + P0
            V1 = np.linalg.inv(P1)
            m1 = np.dot(V1, (kappa + omega*c + np.dot(P0, m0)))
            # this may generate an annoying deprecation warning
            beta = random_state.multivariate_normal(m1, V1)
            beta[-1] = 0

            # save scan
            THETA[it] = softmax(beta)

        burnin = 100
        theta_means = np.mean(THETA[burnin:, :], axis=0)
        dirichlet_post = random_state.dirichlet(alpha + nz, size=2000)
        dirichlet_quantiles = mquantiles(dirichlet_post, prob=[0.25, 0.75], axis=0)
        for i, quantiles in enumerate(dirichlet_quantiles.T):
            self.assertTrue(quantiles[0] < theta_means[i] < quantiles[1])
