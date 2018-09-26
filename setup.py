#!/usr/bin/env python
import os
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

# get the version, this will assign __version__
exec(open('SolitonScattering/version.py').read())

import os
path = os.path.dirname(os.path.abspath(__file__))

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    print('Using cython')

    cmdclass = {'build_ext': build_ext}

    ext_modules = [Extension("SolitonScattering.CUtilities_caller", [path+"/SolitonScattering/CUtilities_caller.pyx"], 
        language = "c++", 
        include_dirs = [np.get_include()],
    )]
except ImportError:
    print('Not using cython')
    cmdclass = {}
    ext_modules = []

packages = [
    'SolitonScattering',
]

# configure setup
setup(
    name = 'SolitonScattering',
    version = __version__,
    author = 'Robert Parini',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    install_requires = ['numpy', 'scipy', 'xarray'],
    packages = packages,
    platforms = ['all']
)
