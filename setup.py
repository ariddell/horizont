#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

NAME                = 'horizont'
DESCRIPTION         = 'Dynamic topic models'
LONG_DESCRIPTION    =  open('README.rst').read()
MAINTAINER          = 'Allen B. Riddell',
MAINTAINER_EMAIL    = 'abr@ariddell.org',
URL                 = 'https://github.com/ariddell/horizont'
LICENSE             = 'GPLv3'
CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
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
MICRO = 3
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'

import os
import sys
from setuptools import setup, find_packages
from distutils.command.sdist import sdist
from distutils.extension import Extension

PY2 = sys.version_info[0] == 2
if PY2:
    REQUIRES += ['futures']

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
                msg = "C-source file '%s' not found." % (cfile) + \
                    " Run 'setup.py cython' before sdist."
                assert os.path.isfile(cfile), msg
        sdist.run(self)

cmdclass = {'sdist': CheckSDist}


###########################################################################
# Cython extensions to compile
###########################################################################

random_sources = ["horizont/RNG/GRNG.cpp",
                  "horizont/RNG/RNG.cpp",
                  "horizont/BayesLogit/Code/C/PolyaGamma.cpp",
                  "horizont/BayesLogit/Code/C/PolyaGammaAlt.cpp",
                  "horizont/BayesLogit/Code/C/PolyaGammaSP.cpp",
                  "horizont/BayesLogit/Code/C/InvertY.cpp"]

include_gsl_dir = "/usr/include/"
lib_gsl_dir = "/usr/lib/"
random_include_dirs = ["horizont/BayesLogit/Code/C",
                       "horizont/RNG",
                       include_gsl_dir]
random_library_dirs = [lib_gsl_dir]
random_libraries = ['gsl', 'gslcblas']

# FIXME: this could be simplified, c.f. pandas
if cython:
    extensions = [Extension("horizont._lda", ["horizont/_lda.pyx"]),
                  Extension("horizont._random",
                            ["horizont/_random.pyx"] + random_sources,
                            include_dirs=random_include_dirs,
                            library_dirs=random_library_dirs,
                            libraries=random_libraries),
                  Extension("horizont._utils", ["horizont/_utils.pyx"])]
    extensions = cythonize(extensions)
else:
    extensions = [Extension("horizont._lda", ["horizont/_lda.c"]),
                  Extension("horizont._random",
                            ["horizont/_random.cpp"] + random_sources,
                            include_dirs=random_include_dirs,
                            library_dirs=random_library_dirs,
                            libraries=random_libraries),
                  Extension("horizont._utils", ["horizont/_utils.c"])]

import numpy
include_dirs = [numpy.get_include()]


###########################################################################
# Setup proper
###########################################################################

setup(install_requires=REQUIRES,
      name=NAME,
      version=FULLVERSION,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      packages=find_packages(),
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      license=LICENSE,
      url=URL,
      classifiers=CLASSIFIERS,
      ext_modules=extensions,
      include_dirs=include_dirs,
      package_data={'horizont.tests': ['ap.dat', 'ch.ldac']},
      platforms='any')
