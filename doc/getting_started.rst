=================
 Getting started
=================

Installation
============

Requirements
------------

Python 2.7 or Python 3.3+ is required. The following packages are required:

- numpy_
- scipy_
- scikit-learn_
- futures (Python 2.7 only)

`GSL <https://www.gnu.org/software/gsl/>`_ is required for random number
generation inside the Pólya-Gamma random variate generator. On Debian-based
sytems, GSL may be installed with the following command::

    sudo apt-get install libgsl0-dev
    
(``horizont`` looks for GSL headers and libraries in ``/usr/include`` and
``/usr/lib/`` respectively.)

Cython is needed if compiling from source.

Installation
------------

On Debian-based systems the following command should be sufficient to install
``horizont``:

    pip install horizont


Quickstart
==========

``horizont.LDA`` implements latent Dirichlet allocation (LDA) using Gibbs
sampling. The interface follows conventions in scikit-learn_.

.. code-block:: python

    >>> import numpy as np
    >>> from horizont import LDA
    >>> X = np.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> model = LDA(n_topics=2, random_state=0, n_iter=100)
    >>> doc_topic = model.fit_transform(X)  # estimate of document-topic distributions
    >>> model.components_  # estimate of topic-word distributions

Example
=======

The following demonstrates fitting a small corpus of newswire articles using
:ref:`horizont.LDA`.

First download the following files into the working directory:

- :download:`ch.ldac <_static/ch.ldac>` (word frequencies in a format similar to SVMLight_ and LDA-C_; indexes start with 1)
- :download:`ch.tokens <_static/ch.tokens>` (list of words)

::

    >>> import numpy as np
    >>> import horizont
    >>> dtm = horizont.utils.ldac2dtm(open('ch.ldac'))
    >>> vocabulary = np.array([word.strip() for word in open('ch.tokens')])
    >>> model = horizont.LDA(n_topics=10, random_state=0, n_iter=100)
    >>> doc_topic = model.fit_transform(dtm)  # estimate of document-topic distributions
    >>> model
    LDA(alpha=0.1, eta=0.01, n_iter=100, n_topics=10, random_state=0)

Having fit the model, the top words associated with each topic may be extracted
with the following lines of code::

    >>> for i, dist in enumerate(model.components_):  # model.components_ is an estimate of topic-word distributions
    >>>     top_words = vocabulary[dist.argsort()[::-1][0:10]]
    >>>     print("Topic {}: {}".format(i, ', '.join(top_words)))

The above should produce the following::

    Topic 0: charles, prince, king, diana, royal, queen, parker, bowles, camilla, marriage
    Topic 1: church, first, years, life, time, died, year, work, death, ceremony
    Topic 2: pope, mother, teresa, vatican, order, hospital, doctors, surgery, heart, pontiff
    Topic 3: city, against, made, told, century, world, state, million, country, second
    Topic 4: yeltsin, elvis, russian, russia, music, kremlin, president, heart, moscow, fans
    Topic 5: political, president, war, government, minister, leader, party, former, last, law
    Topic 6: harriman, u.s, clinton, churchill, paris, ambassador, france, president, american, british
    Topic 7: bernardin, police, east, catholic, told, peace, miami, versace, cunanan, home
    Topic 8: film, germany, people, german, simpson, letter, court, american, against, book
    Topic 9: church, people, say, last, public, visit, during, show, n't, first

We can inspect the document-specific topic distribution for document 173, which
has the title "RUSSIA: Yeltsin spends Russian Christmas in bed with cold" (see
:download:`ch.titles <_static/ch.titles>` for document titles).  We 
reasonably anticipate that the topic featuring the word "yeltsin" will be
prominent::

    >>> doc_topic[173, :]
    array([ 0.00041667,  0.12541667,  0.23375   ,  0.03375   ,  0.25041667,
            0.04625   ,  0.02958333,  0.00041667,  0.02958333,  0.25041667])
    >>> doc_topic[173, :].argmax()
    4

.. links

.. _Python: http://www.python.org/
.. _scikit-learn: http://scikit-learn.org
.. _MALLET: http://mallet.cs.umass.edu/
.. _numpy: http://www.numpy.org/
.. _scipy:  http://docs.scipy.org/doc/
.. _SVMLight: http://scikit-learn.org/stable/datasets/index.html#datasets-in-svmlight-libsvm-format
.. _LDA-C: http://www.cs.princeton.edu/~blei/lda-c/index.html
