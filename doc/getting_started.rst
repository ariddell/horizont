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

    >>> import numpy as np
    >>> import horizont
    >>> dtm = horizont.utils.ldac2dtm(open('ch.ldac'))
    >>> vocabulary = np.array([word.strip() for word in open('ch.tokens')])
    >>> model = horizont.LDA(n_topics=20, random_state=0, n_iter=500)
    >>> doc_topic = model.fit_transform(dtm)  # estimate of document-topic distributions

Having fit the model, the top words associated with each topic may be extracted
with the following lines of code

    >>> for i, dist in enumerate(model.components_):  # model.components_ is an estimate of topic-word distributions
    >>>     top_words = vocabulary[dist.argsort()[::-1][0:10]]
    >>>     print("Topic {}: {}".format(i, ', '.join(top_words)))

The above should produce the following::

    Topic 0: france, french, church, south, african, national, africa, buddhist, catholic, paris
    Topic 1: mother, teresa, order, heart, nuns, charity, calcutta, missionaries, sister, hospital
    Topic 2: bishop, bernardin, east, peace, catholic, cardinal, prize, timor, belo, indonesia
    Topic 3: visit, michael, romania, trip, king, last, poles, country, romanian, poland
    Topic 4: war, world, years, political, former, three, during, minister, country, leader
    Topic 5: died, life, president, clinton, church, service, funeral, white, family, house
    Topic 6: government, party, against, state, president, political, last, group, minister, parliament
    Topic 7: city, years, year, churches, quebec, million, percent, irish, set, opera
    Topic 8: pope, vatican, surgery, rome, hospital, pontiff, paul, roman, mass, john
    Topic 9: russia, russian, soviet, museum, art, moscow, lenin, stalin, church, century
    Topic 10: police, miami, versace, cunanan, home, family, city, york, beach, gay
    Topic 11: yeltsin, kremlin, president, operation, russian, heart, russia, surgery, chernomyrdin, doctors
    Topic 12: elvis, music, fans, king, concert, first, presley, every, death, stage
    Topic 13: british, million, churchill, sale, letters, london, papers, former, britain, estate
    Topic 14: charles, prince, diana, royal, king, queen, parker, bowles, camilla, marriage
    Topic 15: harriman, u.s, paris, ambassador, churchill, france, clinton, pamela, british, american
    Topic 16: against, city, bardot, salonika, cultural, animal, byzantine, second, works, off
    Topic 17: film, simpson, wright, star, life, show, people, festival, hollywood, catholic
    Topic 18: germany, german, nazi, letter, jews, scientology, israel, hitler, kohl, israeli
    Topic 19: church, year, first, people, told, years, say, time, n't, later


.. links

.. _Python: http://www.python.org/
.. _scikit-learn: http://scikit-learn.org
.. _MALLET: http://mallet.cs.umass.edu/
.. _numpy: http://www.numpy.org/
.. _scipy:  http://docs.scipy.org/doc/
.. _SVMLight: http://scikit-learn.org/stable/datasets/index.html#datasets-in-svmlight-libsvm-format
.. _LDA-C: http://www.cs.princeton.edu/~blei/lda-c/index.html
