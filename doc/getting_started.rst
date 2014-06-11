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
sampling. The interface follows conventions in scikit_learn_.

.. code-block:: python

    >>> import numpy as np
    >>> from horizont import LDA
    >>> X = np.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> model = LDA(n_topics=2, random_state=0, n_iter=100)
    >>> doc_topic = model.fit_transform(X)  # estimate of document-topic distributions
    >>> model.components_  # estimate of topic-word distributions


.. _Python: http://www.python.org/
.. _scikit-learn: http://scikit-learn.org
.. _MALLET: http://mallet.cs.umass.edu/
.. _numpy: http://www.numpy.org/
.. _scipy:  http://docs.scipy.org/doc/
