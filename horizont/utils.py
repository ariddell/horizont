from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import scipy.sparse

logger = logging.getLogger('horizont')


def matrix_to_lists(doc_word):
    """Convert a (sparse) matrix of counts into arrays of word and doc indices

    Parameters
    ----------
    doc_word : array (D, V)
        document-term matrix of counts

    Returns
    -------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    """
    if np.count_nonzero(doc_word.sum(axis=1)) != doc_word.shape[0]:
        logger.warning("all zero row in document-term matrix found")
    if np.count_nonzero(doc_word.sum(axis=0)) != doc_word.shape[1]:
        logger.warning("all zero column in document-term matrix found")
    try:
        doc_word = doc_word.tocoo()
    except AttributeError:
        doc_word = scipy.sparse.coo_matrix(doc_word)
    ii, jj, ss = doc_word.row, doc_word.col, doc_word.data
    n_tokens = int(doc_word.sum())
    DS = np.zeros(n_tokens, dtype=int)
    WS = np.zeros(n_tokens, dtype=int)
    startidx = 0
    for i, cnt in enumerate(ss):
        cnt = int(cnt)
        DS[startidx:startidx + cnt] = ii[i]
        WS[startidx:startidx + cnt] = jj[i]
        startidx += cnt
    return WS, DS


def lists_to_matrix(WS, DS):
    """Convert array of word (or topic) and document indices to doc-term array

    Parameters
    -----------
    (WS, DS) : tuple of two arrays
        WS[k] contains the kth word in the corpus
        DS[k] contains the document index for the kth word

    Returns
    -------
    doc_word : array (D, V)
        document-term array of counts

    """
    D = max(DS) + 1
    V = max(WS) + 1
    doc_word = np.zeros((D, V), dtype=int)
    for d in range(D):
        for v in range(V):
            doc_word[d, v] = np.count_nonzero(WS[DS == d] == v)
    return doc_word


def dtm2ldac(dtm):
    """Convert a document-term matrix into a "LDA-C" formatted file

    Parameters
    ----------
    dtm : array of shape N,V

    Returns
    -------
    doclines : iterable of LDA-C lines suitable for writing to file

    Note
    ----
    These particular LDA-C formatted files are offset 1.
    """
    try:
        dtm = dtm.tocsr()
    except AttributeError:
        pass
    num_rows = dtm.shape[0]
    for i, row in enumerate(dtm):
        try:
            row = row.toarray().squeeze()
        except AttributeError:
            pass
        unique_terms = np.count_nonzero(row)
        if unique_terms == 0:
            raise ValueError("dtm contains row with all zero entries.")
        term_cnt_pairs = [(i + 1, cnt) for i, cnt in enumerate(row) if cnt > 0]
        docline = str(unique_terms) + ' '
        docline += ' '.join(["{}:{}".format(i, cnt) for i, cnt in term_cnt_pairs])
        if (i + 1) % 1000 == 0:
            logger.info("dtm2ldac: on row {} of {}".format(i + 1, num_rows))
        yield docline


def ldac2dtm(stream, offset=1):
    """Convert lda-c formatted file to a document-term array

    Parameters
    ----------
    stream : file object
        Object that has a `read` method.

    Returns
    -------
    dtm : array of shape N,V

    Note
    ----
    These LDA-C formatted files are offset 1.
    """
    contents_bytes_maybe = stream.read()
    try:
        contents = contents_bytes_maybe.decode('utf-8')
    except AttributeError:
        contents = contents_bytes_maybe

    # This is a sparse matrix, so we're not particularly concerned about memory
    doclines = [docline for docline in contents.split('\n') if docline]

    # We need to figure out the dimensions of the dtm.
    # Finding N is easy; finding V takes a pass through the data.
    N = len(doclines)
    data = []
    for l in doclines:
        unique_terms = int(l.split(' ')[0])
        term_cnt_pairs = [s.split(':') for s in l.split(' ')[1:]]
        # check that format is indeed LDA-C with the appropriate offset
        for v, _ in term_cnt_pairs:
            if int(v) == 0 and offset == 1:
                raise ValueError("Indexes in LDA-C are offset 1")
        term_cnt_pairs = [(int(v) - offset, int(cnt)) for v, cnt in term_cnt_pairs]
        np.testing.assert_equal(unique_terms, len(term_cnt_pairs))
        data.append(term_cnt_pairs)
    V = -1
    for doc in data:
        vocab_indicies = [V] + [v for v, cnt in doc]
        V = max(vocab_indicies)
    V = V + 1
    dtm = np.zeros((N, V), dtype=int)
    for i, doc in enumerate(data):
        for v, cnt in doc:
            np.testing.assert_(dtm[i, v] == 0)
            dtm[i, v] = cnt
    return dtm
