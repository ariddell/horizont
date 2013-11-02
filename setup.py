#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

LONG_DESCRIPTION    = """Dynamic topic models"""
NAME                = 'horizont'
DESCRIPTION         = 'Dynamic topic models'
MAINTAINER          = 'Allen B. Riddell',
MAINTAINER_EMAIL    = 'abr@ariddell.org',
URL                 = 'https://github.com/ariddell/horizont'
LICENSE             = 'GPLv3'
CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Development Status :: 3 - Alpha',
    'Environment :: Console',
    'Operating System :: OS Independent',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Information Analysis'
]
REQUIRES = ['numpy', 'scipy', 'scikit-learn']
MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'

import os
from setuptools import setup, find_packages
from distutils.command.sdist import sdist
from distutils.extension import Extension

try:
    from Cython.Build import cythonize
    cython = True
except ImportError:
    cython = False


# use CheckSDist strategy from pandas
class CheckSDist(sdist):
    """Custom sdist that ensures Cython has compiled all pyx files to c."""

    _pyxfiles = ['horizont/_lda.pyx']

    def initialize_options(self):
        sdist.initialize_options(self)

        '''
        self._pyxfiles = []
        for root, dirs, files in os.walk('pandas'):
            for f in files:
                if f.endswith('.pyx'):
                    self._pyxfiles.append(pjoin(root, f))
        '''

    def run(self):
        if 'cython' in cmdclass:
            self.run_command('cython')
        else:
            for pyxfile in self._pyxfiles:
                cfile = pyxfile[:-3] + 'c'
                msg = "C-source file '%s' not found." % (cfile) +\
                    " Run 'setup.py cython' before sdist."
                assert os.path.isfile(cfile), msg
        sdist.run(self)

cmdclass = {'sdist': CheckSDist}

if cython:
    extensions = cythonize([Extension("horizont._lda", ["horizont/_lda.pyx"])])
else:
    extensions = [Extension("horizont._lda", ["horizont/_lda.c"])]

import numpy
include_dirs = [numpy.get_include()]

setup(install_requires=REQUIRES,
      name=NAME,
      version=FULLVERSION,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      packages=find_packages(),
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      cmdclass=cmdclass,
      url=URL,
      classifiers=CLASSIFIERS,
      ext_modules=extensions,
      package_data={'horizont.tests': ['ap.tgz']},
      include_dirs=include_dirs,
      platforms='any')
