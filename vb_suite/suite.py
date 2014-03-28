from __future__ import absolute_import, division, print_function, unicode_literals

from vbench.api import Benchmark
from datetime import datetime

import os
import textwrap

modules = ['sample_index']

by_module = {}
benchmarks = []

for modname in modules:
    ref = __import__(modname)
    by_module[modname] = [v for v in ref.__dict__.values()
                          if isinstance(v, Benchmark)]
    benchmarks.extend(by_module[modname])

for bm in benchmarks:
    assert(bm.name is not None)

import getpass
import sys

USERNAME = getpass.getuser()

if sys.platform == 'darwin':
    HOME = '/Users/%s' % USERNAME
else:
    HOME = '/home/%s' % USERNAME

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

config = ConfigParser.ConfigParser()
config.readfp(open(os.path.expanduser('~/.vbenchcfg')))

REPO_PATH = config.get('setup', 'repo_path')
REPO_URL = config.get('setup', 'repo_url')
DB_PATH = config.get('setup', 'db_path')
TMP_DIR = config.get('setup', 'tmp_dir')

PREPARE = """
python setup.py clean
"""
BUILD = """
python setup.py build_ext --inplace
"""
dependencies = ['horizont_vb_common.py']

START_DATE = datetime(2011, 1, 12)

RST_BASE = 'source'


def generate_rst_files(benchmarks):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    vb_path = os.path.join(RST_BASE, 'vbench')
    fig_base_path = os.path.join(vb_path, 'figures')

    if not os.path.exists(vb_path):
        print('creating %s' % vb_path)
        os.makedirs(vb_path)

    if not os.path.exists(fig_base_path):
        print('creating %s' % fig_base_path)
        os.makedirs(fig_base_path)

    for bmk in benchmarks:
        print('Generating rst file for %s' % bmk.name)
        rst_path = os.path.join(RST_BASE, 'vbench/%s.txt' % bmk.name)

        fig_full_path = os.path.join(fig_base_path, '%s.png' % bmk.name)

        # make the figure
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        bmk.plot(DB_PATH, ax=ax)

        start, end = ax.get_xlim()

        plt.xlim([start - 30, end + 30])
        plt.savefig(fig_full_path, bbox_inches='tight')
        plt.close('all')

        fig_rel_path = 'vbench/figures/%s.png' % bmk.name
        rst_text = bmk.to_rst(image_path=fig_rel_path)
        with open(rst_path, 'w') as f:
            f.write(rst_text)

    with open(os.path.join(RST_BASE, 'index.rst'), 'w') as f:
        msg = """
            Performance Benchmarks
            ======================

            These historical benchmark graphs were produced with `vbench
            <http://github.com/pydata/vbench>`__.

            Produced on a machine with

            - Intel Core i5 2540 processor
            - (L)ubuntu Linux 13.10
            - Python 2.7.5 64-bit
            - NumPy 1.8

            .. toctree::
                :hidden:
                :maxdepth: 3
            """
        msg = textwrap.dedent(msg)
        f.write(msg)
        for modname, mod_bmks in sorted(by_module.items()):
            f.write('    vb_%s' % modname)
            modpath = os.path.join(RST_BASE, 'vb_%s.rst' % modname)
            with open(modpath, 'w') as mh:
                header = '%s\n%s\n\n' % (modname, '=' * len(modname))
                mh.write(header)

                for bmk in mod_bmks:
                    mh.write(bmk.name)
                    mh.write('-' * len(bmk.name))
                    mh.write('.. include:: vbench/%s.txt\n' % bmk.name)
