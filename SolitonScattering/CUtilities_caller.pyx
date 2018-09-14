# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np
cimport cython

from libcpp.vector cimport vector
ctypedef double complex cx
ctypedef vector[complex] cxv

cdef extern from "CUtilities.cpp":
    void RungeKutta(int M, int ySize, double h, cx * y0, cx * A, cx * B)

# The ArrayWrapper class is largely copied from Gael Varoquaux, http://gael-varoquaux.info/blog/?p=157
from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# Since we are defining the array length at runtime
# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        """ Set the data of the array

        This cannot be done in the constructor as it must recieve C-level
        arguments.

        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data
        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_COMPLEX128, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.data_ptr)

# Previously used 'float h' which has half the precision of the python 'float' and was causing rounding issues, 'double h' seems to have fixed this
def CRungeKuttaArray(double h,
                np.ndarray[np.complex128_t, ndim=1, mode = 'c'] y not None,
                np.ndarray[np.complex128_t, ndim=3, mode = 'c'] A not None,
                np.ndarray[np.complex128_t, ndim=2, mode = 'c'] B = None):

    cdef int ySize = len(y)
    cdef np.ndarray ndarray

    # Runge Kutta uses a midpoint so the first step of size h requires A[0] = A(a), A[1]=A(h/2), A[2]=A(h)
    # The number of steps to be taken based on the size of the supplied A is therefore M
    cdef int M = (A.shape[0]-1)//2

    # if B is not given create the right size array of zeros
    if B is None:
        B = np.zeros((A.shape[0], ySize), dtype = 'complex')

    # Call the C++ function
    RungeKutta(M, ySize, h, &y[0], &A[0,0,0], &B[0,0])
    # y = RungeKutta(M, ySize, h, y0, A, B)
    # cdef cxv *yPoint = & y

    array_wrapper = ArrayWrapper()
    array_wrapper.set_data(ySize, <void*> y)
    ndarray = np.array(array_wrapper, copy=False)
    # Assign our object to the 'base' of the ndarray object
    ndarray.base = <PyObject*> array_wrapper
    # Increment the reference count, as the above assignement was done in
    # C, and Python does not know that there is this additional reference
    Py_INCREF(array_wrapper)

    return ndarray

