#!/usr/bin/env python
import os
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

import os
path = os.path.dirname(os.path.abspath(__file__))

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    cmdclass = {'build_ext': build_ext}

    ext_modules = [Extension("SolitonScattering.CUtilities_caller", [path+"/SolitonScattering/CUtilities_caller.pyx"], 
        language = "c++", 
        include_dirs = [np.get_include()],
    )]
except ImportError:
    cmdclass = {}
    ext_modules = []

packages = [
    'SolitonScattering',
]

# configure setup
setup(
    name = 'SolitonScattering',
    author = 'Robert Parini',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    install_requires = ['numpy', 'scipy', 'xarray'],
    packages = packages,
    platforms = ['all']
)
