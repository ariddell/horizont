# coding=utf-8
"""
Latent Dirichlet Allocation with Gibbs sampling

Copyright (C) 2013 Allen B. Riddell (abr@ariddell.org)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
from __future__ import division
import logging

import numpy as np
from scipy.special import gammaln

from sklearn.base import BaseEstimator, TransformerMixin

import horizont.utils as utils
import horizont._lda as _lda


class LDA(BaseEstimator, TransformerMixin):
    """Latent Dirichlet allocation using Gibbs sampling

    Parameters
    ----------
    n_topics : int
        Number of topics

    n_iter : int, default 1000
        Number of sampling iterations

    alpha : float, default 0.1
        Dirichlet parameter for distribution over topics

    eta : double, default 0.01
        Dirichlet parameter for distribution over words

    random_state : numpy.RandomState | int, optional
        The generator used for the initial topics. Default: numpy.random

    Attributes
    ----------
    components_ : array, shape = [n_components, n_features]
        Matrix of counts recording topic-word assignments

    Examples
    --------

    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> from horizont import LDA
    >>> model = LDA(n_topics=2, random_state=0, n_iter=100)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LDA(alpha=...
    >>> model.components_
    array([[ 0.85714286,  0.14285714],
           [ 0.45      ,  0.55      ]])
    >>> model.loglikelihood() #doctest: +ELLIPSIS
    -40.395...

    References
    ----------
    Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent Dirichlet
    Allocation." Journal of Machine Learning Research 3 (2003): 993–1022.

    Griffiths, Thomas L., and Mark Steyvers. "Finding Scientific Topics."
    Proceedings of the National Academy of Sciences 101 (2004): 5228–5235.
    doi:10.1073/pnas.0307752101.

    Wallach, Hanna, David Mimno, and Andrew McCallum. "Rethinking LDA: Why
    Priors Matter." In Advances in Neural Information Processing Systems 22,
    edited by Y.  Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, and A.
    Culotta, 1973–1981, 2009.
    """

    def __init__(self, n_topics, n_iter=1000, alpha=0.1, eta=0.01,
                 random_state=None):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha_param = alpha
        self.eta_param = eta
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(seed=random_state)
        else:
            self.random_state = random_state

    def fit(self, X, **params):
        """Learn an LDA model for the document-term matrix X

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Document-term matrix

        Returns
        -------
        self
        """
        self.X = X
        self.D, self.W = D, W = X.shape
        self.N = N = np.sum(X)
        n_topics = self.n_topics
        n_iter = self.n_iter
        logging.info("num_documents: {}".format(D))
        logging.info("num_vocab: {}".format(W))
        logging.info("num_words: {}".format(N))
        logging.info("num_topics: {}".format(n_topics))
        logging.info("num_iter: {}".format(n_iter))

        self.alpha = np.ones(n_topics, dtype=float) * self.alpha_param
        self.alpha_sum = np.sum(self.alpha)
        self.eta = np.ones(W, dtype=float) * self.eta_param
        self.eta_sum = np.sum(self.eta)

        self.nzw = np.zeros((n_topics, W), dtype=int)
        self.ndz = np.zeros((D, n_topics), dtype=int)
        self.nz = np.zeros(n_topics, dtype=int)

        self.nd = np.sum(X, axis=1)

        self.WS, self.DS = WS, DS = utils.matrix_to_lists(X)
        self.ZS = ZS = np.zeros_like(self.WS, dtype=int)

        self.n_rand = n_rand = 1000
        self.rands = self.random_state.random_sample(size=n_rand)

        self._sample_topics(init=True)

        for it in range(1, n_iter + 1):
            if it % 10 == 0:
                ll_per_word = self.loglikelihood() / N
                msg = "<{}> loglikelihood / N: {}".format(it, ll_per_word)
                logging.info(msg)
            self._sample_topics()

        self.components_ = self.nzw / np.sum(self.nzw, axis=1, keepdims=True)

        return self

    def fit_transform(self, X):
        """Learn an LDA model for the document-term matrix X and return a matrix
        with document-term shares.


        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Document-term matrix

        Returns
        -------
        doctopic: array, [n_samples, n_topics]
            Matrix containing document-topic shares
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        # hack to allow `transform` to work on the same matrix that was used for fit
        if self.X is not None and self.X == X:
            DS, ZS = self.DS, self.ZS
            return utils.lists_to_matrix(ZS, DS)
        else:
            raise NotImplementedError("Transform not implemented yet.")

    def loglikelihood(self):
        """
        Calculate the "complete" log likelihood:

        log p(w,z) = log p(w|z) + log p(z)

        """

        D, W, N = self.D, self.W, self.N
        WS, DS, ZS = self.WS, self.DS, self.ZS
        nzw, ndz, nz = self.nzw, self.ndz, self.nz
        eta, eta_sum = self.eta, self.eta_sum
        alpha, alpha_sum = self.alpha, self.alpha_sum

        n_topics = self.n_topics
        nd = self.nd

        ll = 0.0

        # calculate log p(w|z)
        gammaln_eta = gammaln(eta)
        gammaln_alpha = gammaln(alpha)

        ll += n_topics * gammaln(eta_sum)
        for k in range(n_topics):
            ll -= gammaln(eta_sum + nz[k])
            for w in range(W):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += gammaln(eta[w] + nzw[k, w]) - gammaln_eta[w]

        # calculate log p(z)
        for d in range(D):
            ll += gammaln(alpha_sum) - gammaln(alpha_sum + nd[d])
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += gammaln(alpha[k] + ndz[d, k]) - gammaln_alpha[k]
        return ll

    def _sample_topics(self, init=False):
        """Sample new topic assignments"""
        WS, DS, ZS = self.WS, self.DS, self.ZS
        nzw, ndz, nz = self.nzw, self.ndz, self.nz

        eta, eta_sum = self.eta, self.eta_sum
        alpha, alpha_sum = self.alpha, self.alpha_sum

        rands = self.rands
        # shuffle the randoms to avoid starting with the same one every time
        self.random_state.shuffle(rands)

        _lda._sample_topics(WS, DS, ZS, nzw, ndz, nz, alpha, alpha_sum,
                            eta, eta_sum, rands, init)
