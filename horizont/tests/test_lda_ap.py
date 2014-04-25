from __future__ import absolute_import, division, print_function, unicode_literals
import os
import unittest

from horizont import LDA, utils


class TestLDANews(unittest.TestCase):

    test_dir = os.path.dirname(__file__)
    ap_ldac_fn = os.path.join(test_dir, 'ap.dat')
    dtm = utils.ldac2dtm(open(ap_ldac_fn), offset=0)

    def test_lda_news(self):
        dtm = self.dtm
        n_topics = 20
        n_iter = 10
        random_state = 1
        clf = LDA(n_topics=n_topics, n_iter=n_iter, random_state=random_state)
        doctopic = clf.fit_transform(dtm)
        self.assertEqual(len(doctopic), len(dtm))
