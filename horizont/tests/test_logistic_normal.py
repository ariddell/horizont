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
        m0 = 0
        P0 = 1 / trigamma(alpha)

        # summary statistics
        kappa = nz[:-1] - np.sum(nz)/2

        # initialize
        beta = random_state.multivariate_normal(np.repeat(m0, J), np.diag(np.repeat(1/P0, J)))
        beta[-1] = 0
        omega = np.array([horizont.random.pg(np.sum(nz)//J, 0) for _ in range(J)])

        THETA = np.empty((n_iter, J))
        for it in range(n_iter):
            for j in range(J-1):
                c_j = scipy.misc.logsumexp(np.delete(beta, j))
                eta_j = beta[j] - c_j
                # sample omega[j]
                omega[j] = horizont.random.pg(N, eta_j)
                self.assertGreater(omega[j], 0)
                # sample beta[j]
                P1 = omega[j] + P0
                V1 = 1/P1
                m1 = V1 * (kappa[j] + omega[j]*c_j + m0*P0)
                beta[j] = random_state.normal(m1, np.sqrt(V1))

            # save scan
            THETA[it] = softmax(beta)

        burnin = 100
        theta_means = np.mean(THETA[burnin:, :], axis=0)
        dirichlet_post = random_state.dirichlet(alpha + nz, size=2000)
        dirichlet_quantiles = mquantiles(dirichlet_post, prob=[0.25, 0.75], axis=0)
        for i, quantiles in enumerate(dirichlet_quantiles.T):
            self.assertTrue(quantiles[0] < theta_means[i] < quantiles[1])
