from __future__ import absolute_import, division, print_function, unicode_literals

import concurrent.futures as futures
import os
import shutil
import tempfile
import unittest

import numpy as np
import scipy.misc

NUM_TOPICS = 10
DOC_LENGTH = 100
NUM_DOCS = 1000

np.random.seed(1)

from horizont.random import sample_index
from horizont.metrics import kl_div
from horizont import utils, LDA


def _loglikelihood_conditional(X, THETA, PHI):
    """
    Calculate the conditional loglikelihood under LDA of the document-term
    matrix X given \Theta and \Phi

    For example, for a single word w
    p(w|\Theta, \Phi) = \sum_k p(w|z = k, \Phi) p(z = k|\Theta)
    """
    ll = 0
    np.testing.assert_equal(len(X), len(THETA))
    LOGTHETA = np.log(THETA)
    LOGPHI = np.log(PHI)
    for d, doc in enumerate(X):
        for v, cnt in enumerate(doc):
            if cnt != 0:
                ll += cnt * scipy.misc.logsumexp(LOGPHI[:, v] + LOGTHETA[d, :])
    return ll


def make_topic(width, index, fill_col=False):
    """
    Return topic distribution whose "words" form a vertical or horizontal stripe.
    """
    topic = np.zeros((width, width), dtype=float)
    if fill_col:
        topic[:, index] = 1
    else:
        topic[index, :] = 1
    topic = topic.flatten()
    return topic / np.sum(topic)


def make_topics(num_topics):
    """
    Make topic distributions for the corpus.
    """
    width = num_topics // 2
    vocab_size = width ** 2
    doctopic = np.zeros((num_topics, vocab_size))
    for i in range(width):
        doctopic[i] = make_topic(width, i)
    for i in range(width):
        doctopic[width + i] = make_topic(width, i, fill_col=True)
    return doctopic


def print_topic(topic):
    """
    Print a topic to the console
    """
    vocab_size = len(topic)
    width = int(np.sqrt(vocab_size))
    topic = np.round(topic, 3)
    for i in range(width):
        startidx = i * width
        stopidx = (i+1) * width
        row = topic[startidx:stopidx]
        print(' '.join('{:.2f}'.format(v) for v in row))


def print_topics(topics):
    for i, t in enumerate(topics):
        print("TOPIC {}:".format(i))
        print_topic(t)


def make_document(topics, doc_length, alpha=0.1):
    num_topics, vocab_size = topics.shape
    doctopic = np.random.dirichlet([alpha] * num_topics)
    document = np.zeros(vocab_size, dtype=int)
    for _ in range(doc_length):
        z = sample_index(doctopic)
        w = sample_index(topics[z])
        document[w] += 1
    return document


def make_corpus(topics, num_docs, doc_length, alpha=0.1):
    """
    Returns a document-term matrix of counts.
    """
    num_topics, vocab_size = topics.shape
    dtm = np.zeros((num_docs, vocab_size), dtype=int)
    for i in range(num_docs):
        dtm[i] = make_document(topics, doc_length, alpha)
    return dtm


def identify_topic(unknown_topic, topics):
    """
    Returns row index of closest topic in `topics`
    """
    # pending an improved implementation of kl_div
    X = np.vstack([topics, unknown_topic])
    np.testing.assert_equal(len(X), len(topics) + 1)
    dist = kl_div(X+1e-7)  # avoid np.log(0)
    closest_topic_index = np.delete(dist[-1], -1).argmin()
    return closest_topic_index


def order_topics(unordered, reference):
    """
    Sort ``unordered`` by matching it to closest rows in ``reference``.
    """
    np.testing.assert_equal(unordered.shape, reference.shape)
    row_map = []
    for row in unordered:
        row_map.append(identify_topic(row, reference))
    if len(set(row_map)) != len(row_map):
        raise ValueError("Unable to uniquely sort topics")
    return unordered[np.argsort(row_map)]


def sparsify_topics(topics):
    """
    Bluntly sparsify topic distributions.

    Take the top ``width`` values and make everything else zero.
    """
    num_topics = len(topics)
    width = num_topics // 2
    topics = topics.copy()
    for row in topics:
        lowest_indexes = row.argsort()[:-width]
        row[lowest_indexes] = 0
        row /= sum(row)
    return topics


def save_document_image(filename, doc, zoom=2):
    """
    Save document as an image.

    doc must be a square matrix
    """
    height, width = doc.shape
    zoom = np.ones((width*zoom, width*zoom))
    # imsave scales pixels between 0 and 255 automatically
    scipy.misc.imsave(filename, np.kron(doc, zoom))


class TestLDAStripes(unittest.TestCase):

    topics = make_topics(NUM_TOPICS)
    dtm = make_corpus(topics, NUM_DOCS, DOC_LENGTH)
    tempdir = tempfile.mkdtemp()
    with open(os.path.join(tempdir, 'strips.ldac'), 'w') as f:
        f.write('\n'.join(utils.dtm2ldac(dtm)))
        f.write('\n')

    def _make_picture(self, clf):
        topics = self.topics
        ll = clf.loglikelihood()
        inferred = clf.components_
        print("making picture for loglike: {}".format(ll))
        try:
            inferred = order_topics(sparsify_topics(inferred), topics)
        except ValueError:
            return None
        print("max abs difference: ", np.max(np.abs(topics - inferred)))
        np.testing.assert_allclose(topics, inferred, atol=0.05)
        tmpdir = '/tmp/horizont-topic-images'
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.mkdir(tmpdir)
        for z, topic in enumerate(inferred):
            width = 5
            phi = topic + clf.eta
            phi /= np.sum(phi)
            fn = "{}/topic{:02d}-iter{:d}-ll{:.0f}.png".format(tmpdir, z, clf.n_iter, ll)
            save_document_image(fn, phi.reshape(width, -1))

    def test_topics(self):
        topics = self.topics
        width = NUM_TOPICS // 2
        np.testing.assert_equal(topics.shape, (NUM_TOPICS, width**2))
        np.testing.assert_almost_equal(np.sum(topics), NUM_TOPICS)
        self.assertTrue(all(topics.sum(axis=1) == 1))
        self.assertTrue(all(topics.sum(axis=0) == 0.4))

    def test_make_document(self):
        topics = self.topics
        num_topics, vocab_size = topics.shape
        doc = make_document(topics, DOC_LENGTH)
        self.assertEqual(sum(doc), DOC_LENGTH)

        docs = [make_document(topics, DOC_LENGTH) for _ in range(1000)]
        np.testing.assert_allclose(sum(docs) / len(docs), DOC_LENGTH / vocab_size, 0.2)

    def test_make_corpus(self):
        topics = self.topics
        dtm = self.dtm
        num_topics, vocab_size = topics.shape
        self.assertEqual(dtm.shape, (NUM_DOCS, vocab_size))
        np.testing.assert_equal(np.sum(dtm, axis=1), DOC_LENGTH)
        np.testing.assert_allclose(np.sum(dtm, axis=0) / len(dtm), DOC_LENGTH / vocab_size, 0.2)

    def test_identify_topic(self):
        topics = self.topics
        num_topics, vocab_size = topics.shape
        width = num_topics // 2
        unknown_topic = np.zeros(vocab_size)
        unknown_topic[width:2*width] = 1  # so topic #1 (0-indexing)
        unknown_topic /= sum(unknown_topic)
        self.assertEqual(identify_topic(unknown_topic, topics), 1)

    def test_order_topics(self):
        topics = self.topics
        unordered = topics.copy()
        self.assertTrue((topics == unordered).all())
        np.random.shuffle(unordered)
        self.assertFalse((topics == unordered).all())
        ordered = order_topics(unordered, topics)
        self.assertTrue((ordered == topics).all())

    def test_sparsify_topics(self):
        topics = self.topics
        np.testing.assert_allclose(topics, sparsify_topics(topics))

        unsparse = topics + 1e-7
        sparse = sparsify_topics(unsparse)
        np.testing.assert_allclose(topics, sparse)

    def test_dtm_conversion(self):
        topics = self.topics
        dtm = self.dtm
        num_topics, vocab_size = topics.shape
        WS, DS = utils.matrix_to_lists(dtm)
        self.assertEqual(max(WS) + 1, vocab_size)
        self.assertEqual(max(DS) + 1, NUM_DOCS)
        dtm_recovered = utils.lists_to_matrix(WS, DS)
        np.testing.assert_allclose(dtm, dtm_recovered)

    def test_LDA_random_seed(self):
        """
        Make sure we get to the same place starting from the same random seed.
        """
        dtm = self.dtm
        n_iter = 2
        random_state = 5
        fits, lls = [], []
        for _ in range(2):
            clf = LDA(n_topics=NUM_TOPICS, n_iter=n_iter, random_state=random_state)
            clf.fit(dtm)
            fits.append(clf)
            lls.append(clf.loglikelihood())
        np.testing.assert_array_equal(fits[0].nzw, fits[1].nzw)
        self.assertEqual(lls[0], lls[1])

    def test_LDA(self):
        dtm = self.dtm
        n_words = np.sum(dtm)
        n_iter = 50
        # sometimes the sampler gets stuck so we adopt the following procedure
        # run the sampler with 4 random initializations and check the following:
        # 1. all samplers have log likelihood above some (lower) threshold
        # 2. at least one sampler has a log likelihood above a higher threshold
        lls = []
        with futures.ProcessPoolExecutor(max_workers=None) as executor:
            futs = []
            for seed in range(4):
                clf = LDA(n_topics=NUM_TOPICS, n_iter=n_iter, random_state=seed)
                fut = executor.submit(clf.fit, dtm)
                futs.append(fut)
            for fut in futures.as_completed(futs):
                clf = fut.result()
                ll = clf.loglikelihood()
                self._make_picture(clf)
                lls.append(ll)
        for ll in lls:
            # LDA after 20 iterations should be -266000
            self.assertGreater(ll / n_words, -267000 / 1e5)
        # LDA after 100 iterations should be around -255000
        self.assertGreater(max(lls) / n_words, -255000 / 1e5)

    def test_LDA_loglikelihood(self):
        """
        Test one loglikelihood calculation against another
        """
        dtm = self.dtm
        n_topics_true, vocab_size = self.topics.shape
        n_iter = 35
        random_state = 5
        clf = LDA(n_topics=n_topics_true, n_iter=n_iter, random_state=random_state)
        clf.fit(dtm)
        theta, phi = clf.theta_, clf.phi_
        ll = clf.loglikelihood()
        ll_cond = _loglikelihood_conditional(dtm, theta, phi)
        self.assertGreater(-240000, ll_cond)
        self.assertGreater(ll_cond, ll)

    def test_LDA_held_out(self):
        """
        Use ad-hoc marginal log probability estimation.
        """
        hold_prop = 0.2
        n_iter = 100
        n_topics_true, vocab_size = self.topics.shape
        dtm = self.dtm
        num_test = int(hold_prop * len(dtm))
        num_train = len(dtm) - num_test
        random_state = np.random.RandomState(5)
        fit = LDA(n_topics=n_topics_true, n_iter=n_iter, random_state=random_state).fit(dtm)
        ll_target = _loglikelihood_conditional(dtm[num_train:, :], fit.theta_[num_train:, :], fit.phi_)
        dtm_train = dtm[0:num_train]
        dtm_test = dtm[num_train:]
        fit = LDA(n_topics=n_topics_true, n_iter=n_iter, random_state=random_state).fit(dtm_train)
        ndz_test = fit.transform(dtm_test)
        theta = ndz_test + fit.alpha
        theta /= np.sum(theta, axis=1, keepdims=True)
        ll = _loglikelihood_conditional(dtm_test, theta, fit.phi_)
        self.assertGreater(ll, ll_target - 3000)  # ll_target is around -49000
