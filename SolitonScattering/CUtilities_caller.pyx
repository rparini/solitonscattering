# Import the Python-level symbols of numpy
import numpy as np
import cython

# Import the C-level symbols of numpy
cimport numpy as np
cimport cython


ctypedef double complex cx
cdef extern from "CUtilities.cpp":
    void RungeKutta(int M, int ySize, double h, cx* y, cx* A, cx* B)


@cython.boundscheck(False)
@cython.wraparound(False)
def CRungeKuttaArray(double h,
                np.ndarray[np.complex128_t, ndim=1, mode = 'c'] y0 not None,
                np.ndarray[np.complex128_t, ndim=3, mode = 'c'] A not None,
                np.ndarray[np.complex128_t, ndim=2, mode = 'c'] B = None):

    # Copy given y0 so that we don't overwrite it (to be consistant with Python implementations)
    cdef int ySize = len(y0)
    cdef np.ndarray[np.complex128_t, ndim=1] y = y0.copy()

    # Runge Kutta uses a midpoint so the first step of size h requires A[0] = A(a), A[1]=A(h/2), A[2]=A(h)
    # The number of steps to be taken based on the size of the supplied A is therefore M
    cdef int M = (A.shape[0]-1)//2

    # B is not given assume B=0
    if B is None:
        B = np.zeros((A.shape[0], ySize), dtype = 'complex')

    # Call the C++ function
    RungeKutta(M, ySize, h, &y[0], &A[0,0,0], &B[0,0])

    return y