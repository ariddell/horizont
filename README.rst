horizont: Dynamic topic models
==============================

.. image:: https://travis-ci.org/ariddell/horizont.png
        :target: https://travis-ci.org/ariddell/horizont

horizont implements a number of topic models. Conventions from scikit-learn_ are
followed.

The following models are implemented using Gibbs sampling.

- Latent Dirichlet allocation (Blei et al., 2003; Pritchard et al., 2000)
- Logistic Normal topic model
- Dynamic topic model (Blei and Lafferty, 2006)

Getting started
---------------

``horizont.LDA`` implements latent Dirichlet allocation (LDA) using Gibbs
sampling. The interface follows conventions in scikit-learn_.

.. code-block:: python

    >>> import numpy as np
    >>> from horizont import LDA
    >>> X = np.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> model = LDA(n_topics=2, random_state=0, n_iter=100)
    >>> doc_topic = model.fit_transform(X)

Requirements
------------

Python 2.7 or Python 3.3+ is required. The following packages are also required:

- numpy_
- scipy_
- scikit-learn_

`GSL <https://www.gnu.org/software/gsl/>`_ is required for random number
generation inside the Pólya-Gamma random variate generator. On Debian-based
sytems, GSL may be installed with the command ``sudo apt-get install
libgsl0-dev``.  horizont looks for GSL headers and libraries in ``/usr/include``
and ``/usr/lib/`` respectively.

Cython is needed if compiling from source.

Important links
---------------

- Home page: https://github.com/ariddell/horizont/
- Documentation: http://horizont.readthedocs.org
- Repository: https://github.com/ariddell/horizont.git

License
-------

horizont is licensed under Version 3.0 of the GNU General Public License. See
LICENSE file for a text of the license or visit
http://www.gnu.org/copyleft/gpl.html.


.. _Python: http://www.python.org/
.. _scikit-learn: http://scikit-learn.org
.. _MALLET: http://mallet.cs.umass.edu/
.. _numpy: http://www.numpy.org/
.. _scipy:  http://docs.scipy.org/doc/
