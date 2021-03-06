from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from horizont.lda import LDA


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('horizont')
logger.addHandler(logging.NullHandler())

# following PEP 386
__version__ = "0.0.5"
