import io
import logging
import os
import tarfile
import time
import unittest

import numpy as np

import horizont
import horizont.utils

logging.basicConfig(level=logging.DEBUG)


class TestLDA(unittest.TestCase):
    ap_tarfile_path = os.path.join(os.path.dirname(__file__), 'ap.tgz')
    ldac_fp = tarfile.open(ap_tarfile_path).extractfile('ap/ap.dat')
    ldac = io.StringIO(ldac_fp.read().decode('ascii'))
    dtm = horizont.utils.ldac2dtm(ldac)
    K = 20

    vocab_fp = tarfile.open(ap_tarfile_path).extractfile('ap/vocab.txt')
    vocab = np.array([v for v in vocab_fp.read().decode('ascii').split('\n')])

    def test_dtm(self):
        dtm = self.dtm
        self.assertEqual(dtm.shape, (2246, 10473))

    def test_ldac_conversion(self):
        dtm = self.dtm
        self.assertEqual(dtm.shape, (2246, 10473))
        ldac = io.StringIO(u'\n'.join(list(horizont.utils.dtm2ldac(dtm))))
        self.assertTrue(np.all(horizont.utils.ldac2dtm(ldac) == dtm))

    def test_lda_vs_mallet(self):
        """Compare LDA with MALLET"""
        dtm, K = self.dtm, self.K
        clf = horizont.LDA(n_topics=K, n_iter=5, random_state=1)
        clf.fit(dtm)
        # MALLET at 50 iterations: <50> LL/token: -8.79553
        mallet_at_50 = -8.79553 * np.sum(dtm)
        self.assertLess(clf.loglikelihood(), mallet_at_50)
        clf = horizont.LDA(n_topics=K, n_iter=70, random_state=1)
        clf.fit(dtm)
        self.assertGreater(clf.loglikelihood(), mallet_at_50)

    def test_lda_words(self):
        dtm, K = self.dtm, self.K
        vocab = self.vocab
        t0 = time.time()
        n_iter = 100
        t0 = time.time()
        fit = horizont.LDA(n_topics=K, n_iter=n_iter, random_state=1).fit(dtm)
        print("fit done in %0.3fs." % (time.time() - t0))
        n_top_words = 20
        topic_words = []
        for topic in fit.components_:
            top_words = vocab[topic.argsort()[:-n_top_words - 1:-1]]
            topic_words.append(top_words)

        def verify_same_topic(word1, word2):
            """If `word1` is in list then `word2` must be as well"""
            for t in topic_words:
                if word1 in t:
                    self.assertIn(word2, t)

        verify_same_topic("party", "soviet")
        verify_same_topic("dukakis", "bush")
        verify_same_topic("medical", "health")
        verify_same_topic("yen", "dollar")
