from __future__ import absolute_import, division, print_function, unicode_literals

from vbench.benchmark import Benchmark

common_setup = """from horizont_vb_common import *
"""

#----------------------------------------------------------------------
# Series constructors

setup = common_setup + """
np.random.seed(99)
import horizont.random
N = 23  # just an arbitrary number
probs = [1/N] * N
S = 100000
"""

stmt = """
[horizont.random.sample_index(probs) for _ in range(S)]
"""

sample_index = Benchmark(stmt, name="sample_index", setup=setup)
