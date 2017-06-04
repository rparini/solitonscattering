# This will compile the cython script into something python can import
# Compile CUtilities_caller.cpp -> CUtilities_caller.so
# python CythonSetup.py build_ext --inplace -f

from distutils.core import setup, run_setup
from distutils.extension import Extension
# from Cython.Build import cythonize    This should work but I cant get it to
import numpy as np

print('Running CythonSetup.py')

USE_CYTHON = True
try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False


import os
path = os.path.dirname(os.path.abspath(__file__))

print('USE_CYTHON =', USE_CYTHON)

ext = '.pyx' if USE_CYTHON else '.cpp'

extensions = [Extension("CUtilities_caller", [path+"/CUtilities_caller"+ext], language = "c++", include_dirs = [np.get_include()])]

if USE_CYTHON:
    extensions = cythonize(extensions)
    # setup(name = 'RungeKutta',
    #   cmdclass = {'build_ext': build_ext},
    #   ext_modules = ext,
    #   include_dirs=[np.get_include()]
    #   )

setup(ext_modules = extensions)


# setup(name = 'RungeKutta',
#       cmdclass = {'build_ext': build_ext},
#       ext_modules = ext,
#       include_dirs=[np.get_include()]
#       )

