#!/usr/bin/env python
import os
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

from Cython.Distutils import build_ext
from Cython.Build import cythonize

import os
path = os.path.dirname(os.path.abspath(__file__))

ext_modules = [Extension("SolitonScattering.CUtilities_caller", [path+"/SolitonScattering/CUtilities_caller.pyx"], 
    language = "c++", 
    include_dirs = [np.get_include()],
)]

packages = [
    'SolitonScattering',
]

# configure setup
setup(
    name = 'SolitonScattering',
    author = 'Robert Parini',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    install_requires = ['cython', 'numpy', 'scipy', 'xarray'],
    packages = packages,
    platforms = ['all']
)
