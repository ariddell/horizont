**In haitus.** Currently evaluating best way to approach the DTM model. Using the Pólya-Gamma augmentation and the original DTM formulation is complicated and might not give better performance than simpler models (e.g., using truncated Pitman-Yor Processes).

NOTE: The implementation of LDA has been broken out (and refined) into `lda
<https://github.com/ariddell/lda>`_.

NOTE: If you're interested in implementing the dynamic topic model using Pólya-Gamma, most of the hard work has been done: https://github.com/HIPS/pgmult

horizont: Topic models in Python
================================

.. image:: https://travis-ci.org/ariddell/horizont.png
        :target: https://travis-ci.org/ariddell/horizont

horizont implements a number of topic models. Conventions from scikit-learn_ are
followed.

The following models are implemented using Gibbs sampling.

- Latent Dirichlet allocation (Blei et al., 2003; Pritchard et al., 2000)
- (Coming soon) Logistic normal topic model
- (Coming soon) Dynamic topic model (Blei and Lafferty, 2006)

Getting started
---------------

``horizont.LDA`` implements latent Dirichlet allocation (LDA) using Gibbs
sampling. The interface follows conventions in scikit-learn_.

.. code-block:: python

    >>> import numpy as np
    >>> from horizont import LDA
    >>> X = np.array([[1,1], [2, 1], [3, 1], [4, 1], [5, 8], [6, 1]])
    >>> model = LDA(n_topics=2, random_state=0, n_iter=100)
    >>> doc_topic = model.fit_transform(X)  # estimate of document-topic distributions
    >>> model.components_  # estimate of topic-word distributions

Requirements
------------

Python 2.7 or Python 3.3+ is required. The following packages are also required:

- numpy_
- scipy_
- scikit-learn_
- futures (Python 2.7 only)

`GSL <https://www.gnu.org/software/gsl/>`_ is required for random number
generation inside the Pólya-Gamma random variate generator. On Debian-based
sytems, GSL may be installed with the command ``sudo apt-get install
libgsl0-dev``.  horizont looks for GSL headers and libraries in ``/usr/include``
and ``/usr/lib/`` respectively.

Cython is needed if compiling from source.

Important links
---------------

- Documentation: http://pythonhosted.org/horizont
- Source code: https://github.com/ariddell/horizont/
- Issue tracker: https://github.com/ariddell/horizont/issues

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
