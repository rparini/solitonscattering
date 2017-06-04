#!/usr/bin/env python
import os
import sys
import shutil
import unittest
from distutils.core import setup, Command
from distutils.extension import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
import warnings
import numpy as np

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

print('use_cython =', use_cython)

cmdclass = { }

import os
path = os.path.dirname(os.path.abspath(__file__))

ext = '.pyx' if use_cython else '.cpp'

ext_modules = [Extension("SolitonScattering.CUtilities_caller", [path+"/SolitonScattering/CUtilities_caller"+ext], language = "c++", include_dirs = [np.get_include()])]


if use_cython:
    ext_modules = cythonize(ext_modules)
    cmdclass.update({ 'build_ext': build_ext })

packages = [
    'SolitonScattering',
]


# configure setup
setup(
    name = 'SolitonScattering',
    author = 'Robert Parini',
    ext_modules = ext_modules,
    packages = packages,
    platforms = ['all'],
)
